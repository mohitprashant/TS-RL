import tensorflow as tf
import tensorflow_probability as tfp

from base_dpf import BaseParticleFilter

# DTYPE = tf.float64 
DTYPE = tf.float32
tfd = tfp.distributions


class CNFParticleFilter(BaseParticleFilter):
    """
    Conditional Normalizing Flow Particle Filter (CNF-PF).

    This filter enhances the standard particle filter by using neural networks to 
    parameterize a conditional normalizing flow.
    """
    def __init__(self, alpha, sigma, beta, num_particles=100):
        """
        Initializes the CNF Particle Filter with neural networks for flow parameters.

        Args:
            alpha (tf.Tensor): Persistence of the latent state.
            sigma (tf.Tensor): State noise standard deviation.
            beta (tf.Tensor): Observation volatility scaling.
            num_particles (int): Number of particles (N).
        """
        super().__init__(alpha, sigma, beta, num_particles)
        
        self.shift_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', input_shape=(1,)),
            tf.keras.layers.Dense(1, kernel_initializer='zeros') 
        ])
        
        self.scale_net = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='tanh', input_shape=(1,)),
            tf.keras.layers.Dense(1, kernel_initializer='zeros') 
        ])

        # self.shift_net = tf.keras.Sequential([
        #     tf.keras.layers.Dense(64, activation='tanh', input_shape=(1,)),
        #     tf.keras.layers.Dense(1, kernel_initializer='zeros') 
        # ])
        
        # self.scale_net = tf.keras.Sequential([
        #     tf.keras.layers.Dense(32, activation='tanh', input_shape=(1,)),
        #     tf.keras.layers.Dense(1, kernel_initializer='zeros') 
        # ])


    def get_flow_params(self, y_obs):
        """
        Passes the current observation through neural nets to get flow parameters.

        Args:
            y_obs (tf.Tensor): Current observation y_t.

        Returns:
            tuple: (shift, log_scale) where both are scalars conditioning the flow.
        """
        y_in = tf.reshape(y_obs, [1, 1]) 
        shift = self.shift_net(y_in)
        log_scale = self.scale_net(y_in)
        return tf.squeeze(shift), tf.squeeze(log_scale)


    def step(self, particles, log_weights, y_obs):
        """
        Performs one filtering step including the Neural Flow transformation.

        Args:
            particles (tf.Tensor): Particles from the previous timestep.
            log_weights (tf.Tensor): Log-weights from the previous timestep.
            y_obs (tf.Tensor): The current observation.

        Returns:
            tuple: (resampled_particles, resampled_log_weights, None)
        """
        hat_particles = self.transition(particles)         # Proposal basee
        shift, log_scale = self.get_flow_params(y_obs)     # Flow Transformation
        scale = tf.exp(log_scale)
        particles_new = hat_particles * scale + shift

        dist_dyn = tfd.Normal(loc=self.alpha * particles, scale=self.sigma)         # Importance Weight Update
        
        log_p_dyn_new = dist_dyn.log_prob(particles_new) 
        log_p_dyn_hat = dist_dyn.log_prob(hat_particles)
        log_lik = self.log_likelihood(y_obs, particles_new)
        
        log_weight_update = (log_p_dyn_new - log_p_dyn_hat) + log_lik + log_scale
        new_log_weights = log_weights + log_weight_update
        return self.resample_soft(particles_new, new_log_weights)
    

    def resample_soft(self, particles, log_weights):
        """
        Performs differentiable soft resampling to maintain gradient flow.

        Uses a mixture distribution (α=0.5) between the current weights and a 
        uniform distribution to prevent particle collapse and ensure that 
        model parameters can be learned via backpropagation.

        Args:
            particles (tf.Tensor): Particles after flow transformation.
            log_weights (tf.Tensor): Updated importance log-weights.

        Returns:
            tuple: (new_particles, new_log_weights, None)
        """
        N = self.num_particles
        particles = tf.reshape(particles, [N])
        log_weights = tf.reshape(log_weights, [N])
        
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        W_t = tf.exp(log_w_norm)
        
        alpha = 0.5
        uniform_w = 1.0 / float(N)
        W_tilde = alpha * W_t + (1.0 - alpha) * uniform_w
        
        indices = tf.random.categorical(tf.math.log(W_tilde[None, :] + 1e-9), N)[0]
        new_particles = tf.gather(particles, indices)
        
        W_t_sel = tf.gather(W_t, indices)
        W_tilde_sel = tf.gather(W_tilde, indices)
        new_log_weights = tf.math.log(W_t_sel / (W_tilde_sel + 1e-9) + 1e-9)
        
        return new_particles, new_log_weights, None
