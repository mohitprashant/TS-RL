import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from flow_base import ParticleFlow

tfd = tfp.distributions


class PFPF_LEDHFilter(ParticleFlow):
    """
    Invertible Particle Flow Particle Filter (PF-PF) with LEDH Flow.
    
    - Uses Localized EDH flow for high-precision proposal construction.
    - Includes Importance Weighting and Resampling.
    """
    def __init__(self, model, num_particles=100, num_steps=30):
        """
        Initializes the PF-PF (LEDH) Filter.

        Args:
            model (Lorenz96Model): System model.
            num_particles (int): Number of particles.
            num_steps (int): Number of flow integration steps.
        """
        self.model = model
        self.N = num_particles
        self.steps = num_steps
        self.lambdas = tf.constant(np.linspace(0, 1, num_steps+1)**2, dtype=tf.float32)
        self.dlambdas = tf.constant(np.diff(np.linspace(0, 1, num_steps+1)**2), dtype=tf.float32)


    @tf.function(reduce_retracing=True)
    def run_step(self, particles, weights, m_ekf, P_ekf, y_curr):
        """
        Executes one time step of the PF-PF (LEDH) filter.

        Args:
            particles (tf.Tensor): Current particles (N, K).
            weights (tf.Tensor): Current particle weights (N,).
            m_ekf (tf.Tensor): Current EKF mean.
            P_ekf (tf.Tensor): Current EKF covariance.
            y_curr (tf.Tensor): Current observation.

        Returns:
            tuple: (particles, weights, m_upd, P_upd, est, ess, avg_cond)
        """
        m_pred = self.model.rk4_step(m_ekf)
        F = tf.squeeze(self.model.get_jacobian(tf.expand_dims(m_ekf, 0)))
        P_pred = F @ P_ekf @ tf.transpose(F) + self.model.Q
        
        particles_prop = self.model.transition(particles)
        log_det_J = tf.zeros(self.N)
        total_cond = 0.0
        
        for i in range(self.steps):
            lam = self.lambdas[i]
            dlam = self.dlambdas[i]
            
            A, b, cond = self.compute_Ab_and_cond(P_pred, y_curr, particles_prop, lam)
            total_cond += cond
            
            drift = tf.transpose(A @ tf.transpose(particles_prop)) + b
            particles_prop = particles_prop + dlam * drift
            
            det = tf.linalg.det(tf.eye(self.model.K) + dlam * A)
            log_det_J += tf.math.log(tf.abs(det) + 1e-12)
            
        obs_dist = tfd.MultivariateNormalDiag(loc=particles_prop, scale_diag=tf.sqrt(self.model.R_diag))
        log_lik = obs_dist.log_prob(y_curr)
        log_w = tf.math.log(weights + 1e-12) + log_lik + log_det_J
        
        w_unnorm = tf.exp(log_w - tf.reduce_max(log_w))
        weights = w_unnorm / tf.reduce_sum(w_unnorm)
        
        ess = 1.0 / tf.reduce_sum(weights**2)
        
        def resample_fn():
            indices = tf.random.categorical(tf.math.log([weights + 1e-12]), self.N)[0]
            new_parts = tf.gather(particles_prop, indices)
            new_weights = tf.fill([self.N], 1.0/self.N)
            return new_parts, new_weights

        def no_resample_fn():
            return particles_prop, weights

        particles, weights = tf.cond(ess < self.N / 2.0, resample_fn, no_resample_fn)
            
        y_res = y_curr - m_pred
        S = P_pred + self.model.R
        K_gain = P_pred @ tf.linalg.inv(S)
        m_upd = m_pred + tf.linalg.matvec(K_gain, y_res)
        P_upd = (tf.eye(self.model.K) - K_gain) @ P_pred
        
        est = tf.reduce_sum(particles * weights[:, None], axis=0)
        return particles, weights, m_upd, P_upd, est, ess, total_cond / self.steps

