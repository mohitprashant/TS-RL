import tensorflow as tf
import tensorflow_probability as tfp

from base_dpf import BaseParticleFilter

# DTYPE = tf.float64 
DTYPE = tf.float32
tfd = tfp.distributions



class SoftResamplingParticleFilter(BaseParticleFilter):
    """
    Particle Filter using Soft Resampling.
    """
    def __init__(self, alpha, sigma, beta, num_particles=100, soft_alpha=0.5):
        """
        Initializes the Soft Resampling Particle Filter.

        Args:
            alpha (tf.Tensor): Persistence of the latent state.
            sigma (tf.Tensor): State noise standard deviation.
            beta (tf.Tensor): Observation volatility scaling.
            num_particles (int): Number of particles (N).
            soft_alpha (float): The mixture parameter (0 <= α <= 1). 
                a=1 is standard multinomial resampling; 
                a=0 is sampling from a uniform distribution.
        """
        super().__init__(alpha, sigma, beta, num_particles)
        self.soft_alpha = tf.convert_to_tensor(soft_alpha, dtype=DTYPE)


    def step(self, particles, log_weights, y_obs):
        """
        Performs one full filtering step: Prediction, Weighting, and Soft Resampling.

        Args:
            particles (tf.Tensor): Particles from the previous timestep.
            log_weights (tf.Tensor): Log-weights from the previous timestep.
            y_obs (tf.Tensor): The current observation.

        Returns:
            tuple: (new_particles, new_log_weights, None)
        """
        particles = self.transition(particles)
        log_weights += self.log_likelihood(y_obs, particles)
        return self.resample(particles, log_weights)


    def resample(self, particles, log_weights):
        """
        Performs the Soft Resampling step with importance weight correction.

        This method generates a new particle set by sampling indices according 
        to a 'soft' proposal distribution and then updates the weights to 
        account for the discrepancy between the proposal and the target 
        filtering distribution.

        Args:
            particles (tf.Tensor): Current particle states.
            log_weights (tf.Tensor): Unnormalized log-weights.

        Returns:
            tuple:
                - new_particles (tf.Tensor): Resampled particle states.
                - new_log_weights (tf.Tensor): Corrected log-weights.
                - None: Placeholder for the transport matrix (unused here).
        """
        N = self.num_particles
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        W_t = tf.exp(log_w_norm)
        
        uniform_w = 1.0 / float(N)
        W_tilde = self.soft_alpha * W_t + (1.0 - self.soft_alpha) * uniform_w
        
        W_tilde = tf.reshape(W_tilde, [N])                                              # Resample indices based on W_tilde
        indices = tf.random.categorical(tf.math.log(W_tilde[None, :] + 1e-9), N)[0]
        new_particles = tf.gather(particles, indices)
        
        W_t_sel = tf.gather(W_t, indices)                                               # Importance Weight Correction
        W_tilde_sel = tf.gather(W_tilde, indices)
        new_log_weights = tf.math.log(W_t_sel / (W_tilde_sel + 1e-9) + 1e-9)
        new_log_weights = tf.reshape(new_log_weights, [N])
        
        return new_particles, new_log_weights, None
