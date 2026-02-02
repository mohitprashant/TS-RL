import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import tracemalloc

from svssm import StochasticVolatilityModel

tf.random.set_seed(42)
np.random.seed(42)

DTYPE = tf.float32
# DTYPE = tf.float64
tfd = tfp.distributions


class ConditionalGNM(tf.keras.layers.Layer):
    """
    Conditional Gated Network Module (GNM).

    This layer implements a non-linear transformation conditioned on an 
    external context vector (the observation).
    """
    def __init__(self, embed_dim=32):
        """
        Initializes the GNM layer.

        Args:
            embed_dim (int): Dimensionality of the internal embedding space.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.activation = tf.nn.tanh 
        

    def build(self, input_dim):
        """Creates the trainable weight matrix and context projection network."""
        self.W = self.add_weight(shape=(1, self.embed_dim), initializer='glorot_normal', trainable=True)
        self.context = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(self.embed_dim + 1)
        ])
        

    def call(self, x, y):
        """
        Applies the conditional transformation.

        z = W^T * activation( beta * (W * x + b) )

        Args:
            x (tf.Tensor): Input particle states.
            y (tf.Tensor): Context/Observation vector.

        Returns:
            tf.Tensor: Transformed particle states.
        """
        params = self.context(y)
        b = params[:, :self.embed_dim]
        beta = tf.math.softplus(params[:, self.embed_dim:]) + 0.01              # Softplus ensures beta is positive, add a little bit to avoid dying

        z = tf.matmul(x, self.W) + b
        z = self.activation(z * beta)
        z = tf.matmul(z, self.W, transpose_b=True)
        return z


class CondGradNet(tf.keras.Model):
    """
    Conditional Gradient Network for Particle Transport.
    """
    def __init__(self, num_modules=4, embed_dim=32):
        """
        Initializes the transport network.

        Args:
            num_modules (int): Number of GNM layers to stack.
        """
        super().__init__()
        self.modules_list = [ConditionalGNM(embed_dim) for _ in range(num_modules)]
        self.alpha_net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(num_modules)])
        self.bias_net = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(1)])
        self.y_norm = tf.keras.layers.Normalization(axis=-1)

    def call(self, inputs):
        """
        Forward pass of the transport map.

        Args:
            inputs (tuple): (x, y) tuple where x is the particle set and y is the observation.

        Returns:
            tf.Tensor: The mapped particle locations.
        """
        x, y = inputs
        y_n = self.y_norm(y)
        z = x
        alphas = tf.math.softplus(self.alpha_net(y_n))
        
        for i, mod in enumerate(self.modules_list):
            z += alphas[:, i:i+1] * mod(x, y_n)
            
        return z + self.bias_net(y_n)


class GradNetParticleFilter:
    """
    Particle Filter using a Pre-trained Gradient Network for Transport.
    """
    def __init__(self, num_particles=100, lr=0.002):
        """
        Initializes the filter and the underlying neural network.

        Args:
            num_particles (int): Number of particles (N).
            lr (float): Learning rate for the Adam optimizer during pretraining.
        """
        self.num_particles = num_particles
        self.net = CondGradNet(num_modules=4)
        self.net([tf.zeros((1,1)), tf.zeros((1,1))])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)



    def pretrain(self, model, steps=2000, batch_size=64):
        """
        Pre-trains the transport network in a supervised manner.

        Args:
            model (StochasticVolatilityModel): The generative model.
            steps (int): Number of training steps.
            batch_size (int): Batch size for training.
        """
        print(f"Pretraining GradNetOT (Supervised Mode) - {steps} steps...")

        dummy_y = model.observation_dist(tf.random.normal((1000,))).sample()
        self.net.y_norm.adapt(tf.reshape(dummy_y, (-1, 1)))
        
        for step in range(steps):
            x_prev_true = tf.random.normal((batch_size, 1), 0, 2.5)                                    # Sample x_prev from stationary distribution approx
            x_true = model.transition_dist(x_prev_true).sample()
            y_obs = model.observation_dist(x_true).sample()

            x_prev_est = x_prev_true + tf.random.normal(shape=tf.shape(x_prev_true), stddev=0.5)       # Generate Proposal inputs (Simulating the filter state)
            x_prop = model.transition_dist(x_prev_est).sample()
            
            with tf.GradientTape() as tape:                                                            # Train Map
                x_mapped = self.net([x_prop, y_obs])
                loss = tf.reduce_mean(tf.square(x_mapped - x_true))
                
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
            
            if(step % 500 == 0):
                print(f"Step {step}: MSE Loss = {loss.numpy():.4f}")


###################################################################################################################################



    def run(self, model, observations):
        """
        Executes the filter using Gradient Flow and Importance Sampling.

        The importance weights are calculated using the Change of Variables formula:
        w = p(y|x') * p(x'|x_{t-1}) / q(x')
        where q(x') = p(x_prop|x_{t-1}) / |det(J)|.

        Args:
            model (StochasticVolatilityModel): The system dynamics model.
            observations (tf.Tensor): Sequence of observations.

        Returns:
            tf.Tensor: Estimated state trajectory.
        """
        T = observations.shape[0]
        N = self.num_particles
        particles = model.initial_dist().sample(N)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        estimates = []
        
        for t in range(T):
            x_prop = model.transition_dist(particles).sample()                          # Proposal Generation
            
            y_curr = observations[t]                                                    # Apply Transport Map (Neural Proposal)
            x_in = tf.reshape(x_prop, (N, 1))
            y_in = tf.repeat(y_curr, N)[:, None]

            with tf.GradientTape() as j_tape:
                j_tape.watch(x_in)
                x_map = self.net([x_in, y_in])
            
            jac = j_tape.gradient(x_map, x_in)                                           # Jacobian Determinant for change of variables
            jac = tf.clip_by_value(tf.abs(jac), 1e-4, 1e4)                               # Clip jacobian to prevent log(0) or log(inf)
            log_det = tf.reshape(tf.math.log(jac), (N,))
            
            particles_new = tf.reshape(x_map, (N,))
            log_lik = model.observation_dist(particles_new).log_prob(y_curr)
            log_trans_new = model.transition_dist(particles).log_prob(particles_new)
            log_trans_prop = model.transition_dist(particles).log_prob(x_prop)
            
            weight_update = log_lik + log_trans_new - log_trans_prop + log_det          # Importance Weight Update
            weight_update = tf.clip_by_value(weight_update, -100.0, 100.0)              # Clip weight update to prevent explosion (gradient stability)
            log_weights += weight_update
            
            w_norm = tf.nn.softmax(log_weights)                                          # Estimate
            est = tf.reduce_sum(particles_new * w_norm)
            estimates.append(est)

            indices = tf.random.categorical(tf.reshape(log_weights, (1, N)), N)[0]
            particles = tf.gather(particles_new, indices)
            log_weights = tf.fill([N], -tf.math.log(float(N)))
            
        return tf.stack(estimates)
    

    ##############################################################################################################

def run_comparison(num_particles):
    alpha, sigma, beta = 0.91, 1.0, 0.5
    T, N = 50, num_particles
    
    gt_model = StochasticVolatilityModel(alpha, sigma, beta)
    true_x, obs = gt_model.simulate(T)
    
    pre_model = StochasticVolatilityModel(alpha, sigma, beta)                # Pretrain GradNet to learn ot
    gradnet = GradNetParticleFilter(num_particles=N, lr=0.01)
    gradnet.pretrain(pre_model, steps=2000)
    
    methods = {
        # "Sinkhorn": SinkhornParticleFilter(num_particles=N, epsilon=0.5),
        "GradNetOT": gradnet
    }
    
    print(f'{N} Particles:')
    print(f"\n{'Method':<12} | {'RMSE':<8} | {'GradNorm':<10} | {'Time(s)':<8} | {'Mem(MB)':<8}")
    print("-" * 60)
    
    for name, pf in methods.items():
        a_var = tf.Variable(alpha, dtype=DTYPE)
        s_var = tf.Variable(sigma, dtype=DTYPE)
        b_var = tf.Variable(beta, dtype=DTYPE)
        
        tracemalloc.start()
        start_time = time.time()
        
        with tf.GradientTape() as tape:
            diff_model = StochasticVolatilityModel(a_var, s_var, b_var)
            est = pf.run(diff_model, obs)
            loss = tf.sqrt(tf.reduce_mean(tf.square(true_x - est)))
            
        grads = tape.gradient(loss, [a_var, s_var, b_var])
        grad_norm = tf.linalg.global_norm(grads).numpy() if all(g is not None for g in grads) else 0.0
        
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"{name:<12} | {loss:<8.4f} | {grad_norm:<10.4f} | {time.time()-start_time:<8.4f} | {peak_mem/1024**2:<8.2f}")


if __name__ == "__main__":
    run_comparison(50)
    run_comparison(100)
    run_comparison(150)
    run_comparison(200)