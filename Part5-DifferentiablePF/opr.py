import tensorflow as tf
import tensorflow_probability as tfp

from base_dpf import BaseParticleFilter

# DTYPE = tf.float64 
DTYPE = tf.float32
tfd = tfp.distributions


class OptimalPlacementParticleFilter(BaseParticleFilter):
    """
    Differentiable Particle Filter using Optimal Placement (OPR) Resampling.
    """

    def __init__(self, alpha, sigma, beta, num_particles=100):
        """
        Initializes the OPR Particle Filter.

        Args:
            alpha (tf.Tensor): Persistence of the latent state.
            sigma (tf.Tensor): State noise standard deviation.
            beta (tf.Tensor): Observation volatility scaling.
            num_particles (int): Number of particles (N).
        """
        super().__init__(alpha, sigma, beta, num_particles)


    def step(self, particles, log_weights, y_obs):
        """
        Performs one full filtering step: Prediction, Weighting, and OPR Resampling.

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


    def inverse_cdf_transform(self, x_sorted, w_sorted):
        """
        Maps sorted particles to new positions using a differentiable inverse CDF.

        This function approximates the inverse of the empirical CDF through linear 
        interpolation.

        Args:
            x_sorted (tf.Tensor): Particle positions sorted in ascending order.
            w_sorted (tf.Tensor): Corresponding normalized weights, sorted by x.

        Returns:
            tf.Tensor: New particles 'placed' at uniform quantiles (1/2N, 3/2N, ...).
        """
        N = self.num_particles
        w_sorted = tf.maximum(w_sorted, 1e-9)
        w_sorted = w_sorted / tf.reduce_sum(w_sorted)
        
        w_cumsum = tf.cumsum(w_sorted)
        cdf_vals = w_cumsum - w_sorted / 2.0
        
        u_targets = (2.0 * tf.range(1, N + 1, dtype=DTYPE) - 1.0) / (2.0 * float(N))
        
        w_1, x_1 = w_sorted[0], x_sorted[0]
        w_N, x_N = w_sorted[-1], x_sorted[-1]
        
        left_mask = u_targets <= (w_1 / 2.0)
        right_mask = u_targets >= (1.0 - w_N / 2.0)
        
        x_left = x_1 + tf.math.log(2.0 * u_targets / w_1 + 1e-9)
        val_right = tf.maximum(2.0 * (1.0 - u_targets) / w_N, 1e-9)
        x_right = x_N - tf.math.log(val_right)
        
        indices = tf.clip_by_value(tf.searchsorted(cdf_vals, u_targets), 1, N - 1)                  # Interpolation step
        x_lo = tf.gather(x_sorted, indices - 1)
        x_hi = tf.gather(x_sorted, indices)
        c_lo = tf.gather(cdf_vals, indices - 1)
        c_hi = tf.gather(cdf_vals, indices)
        
        slope = (x_hi - x_lo) / tf.maximum(c_hi - c_lo, 1e-9)
        x_interp = x_lo + (u_targets - c_lo) * slope
        
        x_final = tf.where(right_mask, x_right, x_interp)
        x_final = tf.where(left_mask, x_left, x_final)
        return x_final


    def resample(self, particles, log_weights):
        """
        Deterministic, differentiable resampling via Optimal Placement.

        Args:
            particles (tf.Tensor): Current particle states.
            log_weights (tf.Tensor): Current unnormalized log-weights.

        Returns:
            tuple: (new_particles, new_log_weights, None)
        """
        N = self.num_particles
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        weights = tf.exp(log_w_norm)
        
        perm = tf.argsort(particles)                                                          # Sorting works like a pseudo-flow?
        x_sorted = tf.gather(particles, perm)
        w_sorted = tf.gather(weights, perm)
        
        new_particles = self.inverse_cdf_transform(x_sorted, w_sorted)
        new_log_weights = tf.fill([N], -tf.math.log(float(N)))
        return new_particles, new_log_weights, None
