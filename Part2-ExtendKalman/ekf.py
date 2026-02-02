import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
from svssm import StochasticVolatilityModel

tfd = tfp.distributions


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for the Stochastic Volatility Model.
    
    This filter linearizes the non-linear measurement equation using the log-squared 
    transformation of the observations:
        z_n = log(y_n^2) = x_n + log(beta^2) + log(w_n^2)
        
    It uses automatic differentiation (tf.GradientTape) to compute Jacobians.

    Attributes:
        Q (tf.Tensor): Process noise covariance (sigma^2).
        R (tf.Tensor): Measurement noise covariance (approx pi^2 / 2).
        obs_offset (tf.Tensor): Mean offset for the log-chi-squared noise (approx -1.27).
    """
    def __init__(self, alpha, sigma, beta):
        """
        Initializes the EKF with model parameters.

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


    def f(self, x):
        """State transition function: x_{k|k-1} = f(x_{k-1})."""
        return self.alpha * x


    def h(self, x):
        """Measurement function in log-squared domain: z_k = h(x_k)."""
        return x + self.obs_offset


    def get_jacobian(self, func, x):
        """
        Computes the Jacobian of a function at x using Auto-Differentiation.
        
        Args:
            func (callable): The function to differentiate (f or h).
            x (tf.Tensor): The point at which to evaluate the Jacobian.
            
        Returns:
            tf.Tensor: The gradient (Jacobian) dy/dx.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = func(x)
        return tape.gradient(y, x)


    def run_filter(self, observations, true_states=None):
        """
        Runs the EKF over a sequence of observations.

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
            x_pred = self.f(x)                            # Predict
            F = self.get_jacobian(self.f, x)
            P_pred = (F**2) * P + self.Q
            
            z = z_obs[t]                                  # Update
            H = self.get_jacobian(self.h, x_pred)
            S = (H**2) * P_pred + self.R
            K = P_pred * H / S
            x = x_pred + K * (z - self.h(x_pred))
            P = (1.0 - K * H) * P_pred
            estimates.append(x)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time
        
        estimates_tensor = tf.stack(estimates)
        rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2)) if true_states is not None else 0.0

        return {'label': 'EKF', 'rmse': rmse, 'time': total_time, 'mem': peak_mem, 'estimates': estimates_tensor}
