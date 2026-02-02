import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

DTYPE = tf.float32
tfd = tfp.distributions


class MultivariateStochasticVolatilityModel:
    """
    Multivariate Stochastic Volatility (MSV) Model.

    System Dynamics:
        x_{t+1} = diag(phi) * x_t + eta_t
        y_t     = beta * exp(x_t / 2) * eps_t
    """
    def __init__(self, phi, sigma_matrix, beta):
        """
        Initializes the Multivariate SV parameters.

        Args:
            phi (tf.Tensor): Auto-regressive coefficients for the latent log-volatility. 
                Shape (p,).
            sigma_matrix (tf.Tensor): Joint covariance matrix of size (2p, 2p).
                The top-left (p,p) block represents observation noise (epsilon).
                The bottom-right (p,p) block represents state noise (eta).
                Off-diagonal blocks capture cross-correlations.
            beta (tf.Tensor): Scaling coefficients for the observations. Shape (p,).
        """
        self.phi = tf.convert_to_tensor(phi, dtype=DTYPE)
        self.beta = tf.convert_to_tensor(beta, dtype=DTYPE)
        self.p = int(self.phi.shape[0])
        
        full_sigma = tf.convert_to_tensor(sigma_matrix, dtype=DTYPE)
        self.sigma_eps = full_sigma[:self.p, :self.p]
        self.sigma_eta = full_sigma[self.p:, self.p:]
        
        self.chol_eps = tf.linalg.cholesky(self.sigma_eps)
        self.chol_eta = tf.linalg.cholesky(self.sigma_eta)
        self.chol_full = tf.linalg.cholesky(full_sigma)
        self.precision_eta = tf.linalg.inv(self.sigma_eta + 1e-5 * tf.eye(self.p))


    def initial_dist(self):
        """
        Returns the stationary distribution of the latent state process.

        Approximates the initial state distribution X_0 assuming stationarity.
        Variance is computed diagonally: sigma_eta_ii / (1 - phi_ii^2).

        Returns:
            tfd.MultivariateNormalDiag: The initial state distribution.
        """
        vars = tf.linalg.diag_part(self.sigma_eta) / (1.0 - self.phi**2 + 1e-4)
        return tfd.MultivariateNormalDiag(loc=tf.zeros(self.p), scale_diag=tf.sqrt(vars))


    def transition_dist(self, x_prev):
        """
        Computes the transition distribution p(x_t | x_{t-1}).

        Args:
            x_prev (tf.Tensor): Previous latent states. Shape (batch, p).

        Returns:
            tfd.MultivariateNormalTriL: The conditional distribution for the next state.
        """
        return tfd.MultivariateNormalTriL(loc=self.phi * x_prev, scale_tril=self.chol_eta)


    def observation_dist(self, x_curr):
        """
        Computes the observation distribution p(y_t | x_t).

        The covariance of y_t is constructed by scaling the base Cholesky 
        of epsilon by the volatility term exp(x_t / 2).

        Args:
            x_curr (tf.Tensor): Current latent states. Shape (batch, p).

        Returns:
            tfd.MultivariateNormalTriL: The likelihood distribution for observations.
        """
        x_safe = tf.clip_by_value(x_curr, -15.0, 15.0)
        scales = self.beta * tf.exp(x_safe / 2.0)
        scale_tril = scales[..., None] * self.chol_eps[None, ...]
        return tfd.MultivariateNormalTriL(loc=tf.zeros(self.p), scale_tril=scale_tril)


    def simulate(self, T):
        """
        Generates a synthetic trajectory from the joint model.

        Args:
            T (int): Number of time steps to simulate.

        Returns:
            tuple:
                - x_hist (tf.Tensor): Simulated latent states (T, p).
                - y_hist (tf.Tensor): Simulated observations (T, p).
        """
        x_hist, y_hist = [], []
        x = tf.zeros(self.p, dtype=DTYPE)
        noise_dist = tfd.MultivariateNormalTriL(loc=tf.zeros(2 * self.p), scale_tril=self.chol_full)
        
        for _ in range(T):
            joint_noise = noise_dist.sample()
            eps_t = joint_noise[:self.p]
            eta_t = joint_noise[self.p:]
            
            y = (self.beta * tf.exp(x / 2.0)) * eps_t
            x_next = self.phi * x + eta_t
            
            x_hist.append(x)
            y_hist.append(y)
            x = x_next
            
        return tf.stack(x_hist), tf.stack(y_hist)