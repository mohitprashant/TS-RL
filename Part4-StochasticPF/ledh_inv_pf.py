import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


def robust_pinv(A, rcond=1e-5):
    """
    Computes a robust pseudo-inverse on the CPU using SVD.
    
    This function forces execution on the CPU to avoid potential GPU solver crashes.

    Args:
        A (tf.Tensor): The input matrix or batch of matrices. Shape (..., M, N).
        rcond (float): The cutoff ratio for small singular values relative to the largest.

    Returns:
        tf.Tensor: The pseudo-inverse of A. Shape (..., N, M).
    """
    with tf.device("/CPU:0"):
        s, u, v = tf.linalg.svd(A)
        limit = rcond * tf.reduce_max(s)
        s_inv = tf.where(s > limit, 1.0 / s, tf.zeros_like(s))
        return tf.matmul(v, tf.matmul(tf.linalg.diag(s_inv), u, transpose_b=True))


def robust_svd_eig(A):
    """
    Approximates the Eigendecomposition of a symmetric matrix using SVD on the CPU.

    Args:
        A (tf.Tensor): The input symmetric matrix. Shape (..., N, N).

    Returns:
        tuple: (s, u)
            s (tf.Tensor): The singular values (eigenvalues).
            u (tf.Tensor): The left singular vectors (eigenvectors).
    """
    with tf.device("/CPU:0"):
        s, u, v = tf.linalg.svd(A)
    return s, u


class AuxiliaryEKF:
    """
    Auxiliary Extended Kalman Filter (EKF) used to guide the Particle Flow.
    """
    def __init__(self, model):
        """
        Initializes the Auxiliary EKF.

        Args:
            model: The dynamic system model (e.g., Lorenz96) containing:
                   - rk4_step(m): Deterministic state transition.
                   - get_jacobian(m): Method to compute the Jacobian F.
                   - Q: Process noise covariance.
                   - R: Measurement noise covariance.
                   - K: State dimension.
        """
        self.model = model
        self.K = model.K
        self.I = tf.eye(self.K)


    def predict(self, m, P):
        """
        Performs the EKF prediction step.

        Args:
            m (tf.Tensor): Mean at step k-1.
            P (tf.Tensor): Covariance at step k-1.

        Returns:
            tuple: (m_pred, P_pred) predicted mean and covariance.
        """
        m_pred = self.model.rk4_step(m)
        F = tf.squeeze(self.model.get_jacobian(tf.expand_dims(m, 0)))
        P_pred = F @ P @ tf.transpose(F) + self.model.Q
        
        P_pred = tf.where(tf.reduce_any(tf.math.is_nan(P_pred)), 10.0*self.I, P_pred)     # Guard: Reset covariance if it explodes
        return m_pred, P_pred
    

    def update(self, m_pred, P_pred, y):
        """
        Performs the EKF measurement update step.

        Args:
            m_pred (tf.Tensor): Predicted mean.
            P_pred (tf.Tensor): Predicted covariance.
            y (tf.Tensor): Observation vector.

        Returns:
            tuple: (m_upd, P_upd) updated mean and covariance.
        """
        y_res = y - m_pred
        S = P_pred + self.model.R
        S_inv = robust_pinv(S)
        K_gain = P_pred @ S_inv
        
        m_upd = m_pred + tf.linalg.matvec(K_gain, y_res)
        P_upd = (self.I - K_gain) @ P_pred
        return m_upd, P_upd



class LEDH_PFPF:
    """
    Localized Exact Daum-Huang Particle Flow Particle Filter (LEDH-PFPF).

    Attributes:
        model: The system model.
        N (int): Number of particles.
        steps (int): Number of pseudo-time integration steps (lambda steps).
        lambdas (tf.Tensor): Quadratic schedule for pseudo-time lambda (0 to 1).
    """
    def __init__(self, model, num_particles=100, num_steps=20):
        """
        Initializes the LEDH-PFPF filter.

        Args:
            model: The dynamic system model.
            num_particles (int): Number of particles.
            num_steps (int): Number of flow steps.
        """
        self.model = model
        self.N = num_particles
        self.steps = num_steps
        self.lambdas = tf.constant(np.linspace(0, 1, num_steps+1)**2, dtype=tf.float64)
        self.dlambdas = self.lambdas[1:] - self.lambdas[:-1]


    @tf.function
    def run_step(self, particles, weights, m_ekf, P_ekf, y_curr):
        """
        Executes a single filtering step (Prediction + Measurement Update).
        Args:
            particles (tf.Tensor): Current particle cloud (N, K).
            weights (tf.Tensor): Current particle weights (N,).
            m_ekf (tf.Tensor): Auxiliary EKF mean (K,).
            P_ekf (tf.Tensor): Auxiliary EKF covariance (K, K).
            y_curr (tf.Tensor): Current observation (K,).

        Returns:
            tuple:
                - particles_final: Updated particles.
                - weights_final: Updated weights.
                - m_upd, P_upd: Updated EKF state.
                - est: Estimated state mean (weighted average of particles).
                - ess: Effective Sample Size.
                - avg_cond: Average condition number during the flow (stability metric).
        """
        ekf = AuxiliaryEKF(self.model)                              # EKF Prediction
        m_pred, P_pred = ekf.predict(m_ekf, P_ekf)
        particles_prop = self.model.transition(particles)
        
        H0 = robust_pinv(P_pred)
        Hh = self.model.R_inv
        I = tf.eye(self.model.K)
        log_det_J = tf.zeros(self.N)
        total_cond = 0.0
        
        R_inv_y = tf.linalg.matvec(Hh, y_curr)
        
        for k in tf.range(self.steps):                             # Particle Propagation and flow
            lam = self.lambdas[k]
            dlam = self.dlambdas[k]
            
            S = lam * P_pred + self.model.R
            S_inv = robust_pinv(S)
            A = -0.5 * P_pred @ S_inv
            
            P_R_inv_y = tf.linalg.matvec(P_pred, R_inv_y)
            term1 = tf.linalg.matvec((I + lam * A), P_R_inv_y)
            term2 = tf.transpose(tf.matmul(A, particles_prop, transpose_b=True))
            b_vecs = tf.transpose(tf.matmul(I + 2*lam*A, term1 + term2, transpose_b=True))
            
            drift = tf.transpose(tf.matmul(A, particles_prop, transpose_b=True)) + b_vecs
            particles_prop = particles_prop + dlam * drift
            
            with tf.device("/CPU:0"):
                det = tf.linalg.det(I + dlam * A)
                log_det_J += tf.math.log(tf.abs(det) + 1e-12)
                M = (1.0 - lam) * H0 + lam * Hh
                s = tf.linalg.svd(M, compute_uv=False)
                total_cond += tf.reduce_max(s) / (tf.reduce_min(s) + 1e-9)
            
        obs_dist = tfd.MultivariateNormalDiag(loc=particles_prop, scale_diag=tf.sqrt(self.model.R_diag))
        log_lik = obs_dist.log_prob(y_curr)
        log_w = tf.math.log(weights + 1e-12) + log_lik + log_det_J
        log_w = tf.where(tf.math.is_finite(log_w), log_w, -1e9 * tf.ones_like(log_w))
        
        w_unnorm = tf.exp(log_w - tf.reduce_max(log_w))                              # Weight updates
        weights = w_unnorm / (tf.reduce_sum(w_unnorm) + 1e-12)
        ess = 1.0 / (tf.reduce_sum(weights**2) + 1e-9)
        
        def resample():                                                              # Resampling helper, use if needed :(
            idxs = tf.random.categorical(tf.reshape(log_w, [1, -1]), self.N)[0]
            return tf.gather(particles_prop, idxs), tf.fill([self.N], 1.0/float(self.N))
            
        particles_final, weights_final = tf.cond(ess < self.N/2.0, resample, lambda: (particles_prop, weights))
        m_upd, P_upd = ekf.update(m_pred, P_pred, y_curr)                             # EKF updates
        est = tf.reduce_sum(particles_final * weights_final[:, None], axis=0)
        
        return particles_final, weights_final, m_upd, P_upd, est, ess, total_cond / float(self.steps)

