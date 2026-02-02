import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

# from gradnetot_analysis import ConditionalGNM
from svssm import StochasticVolatilityModel
from gradnetot_analysis import CondGradNet as RegularGradNet

tf.random.set_seed(42)
np.random.seed(42)

DTYPE = tf.float32
# DTYPE = tf.float64 
tfd = tfp.distributions


class GNM_Module(tf.keras.layers.Layer):
    """
    Trunk Network Module: Static Monotone Basis Function phi(x).
    """
    def __init__(self, embed_dim=32):
        """
        Initializes the GNM basis module.

        Args:
            embed_dim (int): Dimension of the internal embedding space.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.activation = tf.nn.tanh
        
    def build(self, input_dim):
        """Creates trainable weights for the basis function."""
        self.W = self.add_weight(shape=(1, self.embed_dim), initializer='glorot_normal', trainable=True)
        self.b = self.add_weight(shape=(self.embed_dim,), initializer='zeros', trainable=True)
        self.beta = self.add_weight(shape=(1,), initializer=tf.constant_initializer(1.0), trainable=True)
        
    def call(self, x):
        """
        Computes the basis function output.

        z = W^T * activation( beta * (W * x + b) )

        Args:
            x (tf.Tensor): Input particle states (batch_size, 1).

        Returns:
            tf.Tensor: Transformed output.
        """
        beta = tf.math.softplus(self.beta)
        z = tf.matmul(x, self.W) + self.b
        z = self.activation(z * beta)
        z = tf.matmul(z, self.W, transpose_b=True)
        return z


class DeepONetGradNet(tf.keras.Model):
    """
    Deep Operator Network (DeepONet) for Particle Transport.

    DeepONet approximates the optimal transport map as an operator.
    - **Trunk Net**: Learns a set of continuous basis functions phi_k(x) over the state space.
    - **Branch Net**: Learns coefficients c_k(y) conditioned on the observation.
    - **Output**: G(x, y) = Sum( c_k(y) * phi_k(x) ) + bias(y)
    """
    def __init__(self, num_basis=16, embed_dim=32):
        """
        Initializes the DeepONet architecture.

        Args:
            num_basis (int): Number of basis functions in the Trunk (width of the inner product).
            embed_dim (int): Internal dimension of each GNM basis module.
        """
        super().__init__()
        self.trunk_basis = [GNM_Module(embed_dim) for _ in range(num_basis)]
        self.branch_net = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_basis)
        ])
        self.bias_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.y_norm = tf.keras.layers.Normalization(axis=-1)

    def call(self, inputs):
        """
        Forward pass computing the inner product of Branch and Trunk outputs.

        Args:
            inputs (tuple): (x, y) where x is the particle set and y is the observation.

        Returns:
            tf.Tensor: Transported particle locations.
        """
        x, y = inputs
        y_n = self.y_norm(y)

        weights = tf.math.softplus(self.branch_net(y_n))  # Keep weights positive
        basis_outs = [mod(x) for mod in self.trunk_basis]
        trunk_out = tf.concat(basis_outs, axis=1)
        weighted_sum = tf.reduce_sum(weights * trunk_out, axis=1, keepdims=True)
        z = x + weighted_sum + self.bias_net(y_n)
        return z


class NeuralParticleFilter:
    """
    Particle Filter framework.
    """
    def __init__(self, net_type='deeponet', num_particles=100, lr=0.002):
        """
        Initializes the Neural Particle Filter.

        Args:
            net_type (str): 'deeponet' for operator learning, or 'regular' for standard GradNet.
            num_particles (int): Default number of particles for training/testing.
            lr (float): Learning rate for the optimizer.
        """
        self.default_N = num_particles
        
        if(net_type == 'deeponet'):
            self.net = DeepONetGradNet(num_basis=16, embed_dim=32)
        else:
            self.net = RegularGradNet(num_modules=4, embed_dim=32)
            
        self.net([tf.zeros((1,1)), tf.zeros((1,1))])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    def pretrain(self, model, steps=2500, batch_size=64):
        """
        Pre-trains the transport network using supervised learning.

        Trains the network to minimize the L2 distance between the mapped 
        proposal particles and the true posterior particles.

        Args:
            model (StochasticVolatilityModel): Generative model for data simulation.
            steps (int): Number of gradient descent steps.
            batch_size (int): Number of samples per batch.
        """
        print(f"Pretraining {self.net.__class__.__name__} ({steps} steps)...")
        dummy_y = model.observation_dist(tf.random.normal((100,))).sample()
        self.net.y_norm.adapt(tf.reshape(dummy_y, (-1, 1)))
        
        for step in range(steps):
            x_prev = tf.random.normal((batch_size, 1), 0, 2.5)
            x_true = model.transition_dist(x_prev).sample()
            y_obs = model.observation_dist(x_true).sample()
            x_prev_est = x_prev + tf.random.normal(tf.shape(x_prev), 0, 0.5)
            x_prop = model.transition_dist(x_prev_est).sample()
            
            with tf.GradientTape() as tape:
                x_map = self.net([x_prop, y_obs])
                loss = tf.reduce_mean(tf.square(x_map - x_true))
                
            grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.net.trainable_variables))
            if(step % 500 == 0):
                print(f"  Step {step}: Loss = {loss.numpy():.4f}")


#######################################################################################################################


    def run(self, model, observations, num_particles=None):
        """
        Runs the filter on a sequence of observations.

        If using DeepONet, `num_particles` can be changed at runtime (e.g., 
        trained on N=100, run on N=1000) without retraining.

        Args:
            model (StochasticVolatilityModel): System dynamics.
            observations (tf.Tensor): Observed data sequence.
            num_particles (int, optional): Number of particles to use. Defaults to self.default_N.

        Returns:
            tf.Tensor: Sequence of state estimates.
        """
        N = num_particles if num_particles is not None else self.default_N
        T = observations.shape[0]
        
        particles = model.initial_dist().sample(N)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        estimates = []
        
        for t in range(T):
            x_prop = model.transition_dist(particles).sample()          # Proposal
            y_curr = observations[t]                                    # Cur obs
            x_in = tf.reshape(x_prop, (N, 1))
            y_in = tf.repeat(y_curr, N)[:, None]
            
            with tf.GradientTape() as j_tape:
                j_tape.watch(x_in)
                x_map = self.net([x_in, y_in])
                
            jac = j_tape.gradient(x_map, x_in)
            log_det = tf.reshape(tf.math.log(tf.abs(jac) + 1e-6), (N,))
            particles_new = tf.reshape(x_map, (N,))

            lik = model.observation_dist(particles_new).log_prob(y_curr)
            tr_new = model.transition_dist(particles).log_prob(particles_new)
            tr_prop = model.transition_dist(particles).log_prob(x_prop)
            
            w = lik + tr_new - tr_prop + log_det                                     # Determine weights
            w = tf.clip_by_value(w, -100.0, 100.0)
            log_weights += w

            w_norm = tf.nn.softmax(log_weights)                                      # Make estimates
            est = tf.reduce_sum(particles_new * w_norm)
            estimates.append(est)

            indices = tf.random.categorical(tf.reshape(log_weights, (1, N)), N)[0]   # Resampling step
            particles = tf.gather(particles_new, indices)
            log_weights = tf.fill([N], -tf.math.log(float(N)))
            
        return tf.stack(estimates)


def run_comparison():
    alpha = 0.91
    sigma = 1.0
    beta = 0.5
    T = 100
    N_base = 100
    
    gt_model = StochasticVolatilityModel(alpha, sigma, beta)
    true_x, obs = gt_model.simulate(T)
    
    dummy_model = StochasticVolatilityModel(alpha, sigma, beta)
    reg_net = NeuralParticleFilter('regular', N_base)
    reg_net.pretrain(dummy_model, 100)
    deep_net = NeuralParticleFilter('deeponet', N_base)
    deep_net.pretrain(dummy_model, 100)

    experiments = []
    # experiments.append(("Sinkhorn (N=100)", SinkhornParticleFilter(100), 100))
    experiments.append(("GradNet (N=100)", reg_net, 100))
    deep_ns = [20, 50, 100, 200, 500, 1000]
    for n in deep_ns:
        experiments.append((f"DeepONet (N={n})", deep_net, n))
    
    print(f"\n{'Method':<25} | {'RMSE':<8} | {'GradNorm':<10} | {'Time(s)':<8} | {'Mem(MB)':<8}")
    print("-" * 75)
    
    results_est = {}
    
    for name, pf, N_run in experiments:
        a_var = tf.Variable(alpha, dtype=DTYPE)
        s_var = tf.Variable(sigma, dtype=DTYPE)
        b_var = tf.Variable(beta, dtype=DTYPE)
        
        tracemalloc.start()
        start_time = time.time()
        
        with tf.GradientTape() as tape:
            diff_model = StochasticVolatilityModel(a_var, s_var, b_var)
            est = pf.run(diff_model, obs, num_particles=N_run)
            rmse = tf.sqrt(tf.reduce_mean(tf.square(true_x - est)))
            
        grads = tape.gradient(rmse, [a_var, s_var, b_var])
        grad_norm = tf.linalg.global_norm(grads).numpy() if all(g is not None for g in grads) else 0.0
        
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"{name:<25} | {rmse:<8.4f} | {grad_norm:<10.4f} | {time.time()-start_time:<8.4f} | {peak_mem/1024**2:<8.2f}")
        
        if("DeepONet" in name or "Sinkhorn" in name):
            results_est[name] = est.numpy()

    plt.figure(figsize=(14, 7))
    plt.plot(true_x, 'k-', linewidth=2, label='True', alpha=0.7)
    # keys_to_plot = ["Sinkhorn (N=100)", "DeepONet (N=20)", "DeepONet (N=100)", "DeepONet (N=500)"]
    keys_to_plot = ["DeepONet (N=20)", "DeepONet (N=100)", "DeepONet (N=500)"]
    for k in keys_to_plot:
        if k in results_est:
            plt.plot(results_est[k], linestyle='--', label=k, alpha=0.8)
            
    plt.title("DeepONet Operator Generalization: Varying Particle Counts at Runtime")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_comparison()