import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
from svssm import StochasticVolatilityModel

tfd = tfp.distributions


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) for the Stochastic Volatility Model.
    
    Uses the Unscented Transform to propagate mean and covariance through non-linearities
    without explicit Jacobian calculation.

    Attributes:
        Wm (tf.Tensor): Weights for mean calculation.
        Wc (tf.Tensor): Weights for covariance calculation.
        lam (float): Scaling parameter lambda for sigma point spread.
    """
    def __init__(self, alpha, sigma, beta):
        """
        Initializes the UKF with model parameters and sigma point weights.

        Args:
            alpha (float): Autoregression coefficient.
            sigma (float): State noise standard deviation.
            beta (float): Observation scaling factor.
        """
        self.dtype = tf.float32
        self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype)
        self.sigma = tf.convert_to_tensor(sigma, dtype=self.dtype)
        self.beta = tf.convert_to_tensor(beta, dtype=self.dtype)
        self.Q = self.sigma**2
        self.R = tf.constant((np.pi**2) / 2.0, dtype=self.dtype)
        self.obs_offset = tf.math.log(self.beta**2) - 1.2704

        self.n = 1.0 
        self.alpha_ukf = 1e-3
        self.beta_ukf = 2.0
        self.kappa = 0.0
        self.lam = self.alpha_ukf**2 * (self.n + self.kappa) - self.n
        
        w_m0 = self.lam / (self.n + self.lam)
        w_c0 = w_m0 + (1 - self.alpha_ukf**2 + self.beta_ukf)
        w_i = 1.0 / (2 * (self.n + self.lam))
        self.Wm = tf.constant([w_m0, w_i, w_i], dtype=self.dtype)
        self.Wc = tf.constant([w_c0, w_i, w_i], dtype=self.dtype)


    def generate_sigma_points(self, x, P):
        """
        Generates 2n+1 sigma points based on current mean and covariance.

        Args:
            x (tf.Tensor): Current state mean.
            P (tf.Tensor): Current state covariance.

        Returns:
            tf.Tensor: Stack of sigma points [2n+1].
        """
        sigma = tf.sqrt(P * (self.n + self.lam))
        return tf.stack([x, x + sigma, x - sigma])


    def f(self, x):
        """State transition function applied to sigma points."""
        return self.alpha * x


    def h(self, x):
        """Measurement function applied to sigma points (log-squared domain)."""
        return x + self.obs_offset


    def run_filter(self, observations, true_states=None):
        """
        Runs the UKF over a sequence of observations.

        Args:
            observations (tf.Tensor): Tensor of shape (T,) containing observed values y_n.
            true_states (tf.Tensor, optional): Ground truth states for RMSE calculation.

        Returns:
            dict: Performance metrics including RMSE, runtime, memory, and estimates.
        """
        z_obs = tf.math.log(observations**2 + 1e-8)
        T = z_obs.shape[0]
        x = tf.zeros([], dtype=self.dtype)
        P = self.Q / (1 - self.alpha**2)
        estimates = []

        tracemalloc.start()
        start_time = time.time()

        for t in range(T):
            sig_pts = self.generate_sigma_points(x, P)                      # Predict
            sig_pts_pred = self.f(sig_pts)
            x_pred = tf.reduce_sum(self.Wm * sig_pts_pred)
            P_pred = tf.reduce_sum(self.Wc * ((sig_pts_pred - x_pred)**2)) + self.Q
            
            z = z_obs[t]                                                    # Update
            sig_pts_update = self.generate_sigma_points(x_pred, P_pred)
            obs_pts = self.h(sig_pts_update)
            y_pred_mean = tf.reduce_sum(self.Wm * obs_pts)
            S = tf.reduce_sum(self.Wc * ((obs_pts - y_pred_mean)**2)) + self.R
            Pxy = tf.reduce_sum(self.Wc * (sig_pts_update - x_pred) * (obs_pts - y_pred_mean))
            
            K = Pxy / S
            x = x_pred + K * (z - y_pred_mean)
            P = P_pred - K * S * K
            estimates.append(x)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time
        
        estimates_tensor = tf.stack(estimates)
        rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2)) if true_states is not None else 0.0

        return {'label': 'UKF', 'rmse': rmse, 'time': total_time, 'mem': peak_mem, 'estimates': estimates_tensor}