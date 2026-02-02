import tensorflow as tf
import tensorflow_probability as tfp

from base_dpf import BaseParticleFilter

# DTYPE = tf.float64 
DTYPE = tf.float32
tfd = tfp.distributions


class SinkhornParticleFilter(BaseParticleFilter):
    """
    Differentiable Particle Filter using Sinkhorn (Entropy-Regularized OT) Resampling.
    """
    def __init__(self, alpha, sigma, beta, num_particles=100, epsilon=0.5, n_iter=20):
        """
        Initializes the Sinkhorn Particle Filter.

        Args:
            alpha (tf.Tensor): Persistence of the latent state.
            sigma (tf.Tensor): State noise standard deviation.
            beta (tf.Tensor): Observation volatility scaling.
            num_particles (int): Number of particles (N).
            epsilon (float): Regularization strength for the Sinkhorn algorithm. 
                Smaller values approach true OT; larger values are more stable.
            n_iter (int): Number of Sinkhorn iterations for convergence.
        """
        super().__init__(alpha, sigma, beta, num_particles)
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=DTYPE)
        self.n_iter = n_iter


    def step(self, particles, log_weights, y_obs):
        """
        Performs one full filtering step: Prediction, Weighting, and Sinkhorn Resampling.

        Args:
            particles (tf.Tensor): Particles from the previous timestep.
            log_weights (tf.Tensor): Log-weights from the previous timestep.
            y_obs (tf.Tensor): The current observation.

        Returns:
            tuple: (new_particles, new_log_weights, transport_matrix)
        """
        particles = self.transition(particles)
        log_weights += self.log_likelihood(y_obs, particles)
        return self.resample(particles, log_weights)


    def resample(self, particles, log_weights):
        """
        Performs soft-resampling via Entropy-Regularized Optimal Transport.

        Calculates a transport matrix P that maps the weighted empirical 
        distribution of particles to a uniform distribution. New particles 
        are then computed as the barycentric projection of the transport plan.

        Args:
            particles (tf.Tensor): Current particle states.
            log_weights (tf.Tensor): Unnormalized log-weights.

        Returns:
            tuple:
                - new_particles (tf.Tensor): Transported particles with uniform weights.
                - new_log_weights (tf.Tensor): Uniform log-weights (-log N).
                - P_clean (tf.Tensor): The computed NxN transport matrix.
        """
        N = self.num_particles
        log_weights = tf.maximum(log_weights, -1e9)                         # Normalize weights in log domain
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)
        
        diff = particles[:, None] - particles[None, :]                      # Cost matrix
        C = diff**2
        
        f = tf.zeros((N,), dtype=DTYPE)                                     # Initialize Sinkhorn Dual Potentials
        g = tf.zeros((N,), dtype=DTYPE)
        eps = self.epsilon
        log_b = tf.fill([N], -tf.math.log(float(N)))
        
        for _ in range(self.n_iter):
            tmp_f = log_b[None, :] + (g[None, :] - C) / eps
            f = 0.5 * (f - eps * tf.reduce_logsumexp(tmp_f, axis=1))
            tmp_g = log_w_norm[:, None] + (f[:, None] - C) / eps
            g = 0.5 * (g - eps * tf.reduce_logsumexp(tmp_g, axis=0))
        
        log_P = (f[:, None] + g[None, :] - C) / eps + log_w_norm[:, None] + log_b[None, :]         # Transport Matrix
        P = tf.exp(log_P)
        
        P_clean = tf.debugging.check_numerics(P, "Sinkhorn P has NaNs") 
        new_particles = float(N) * tf.linalg.matvec(tf.transpose(P_clean), particles)              # Apply Transport
        new_log_weights = log_b 
        return new_particles, new_log_weights, P_clean
