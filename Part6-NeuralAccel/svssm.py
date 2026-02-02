import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

class StochasticVolatilityModel:
    """
    Implements the Stochastic Volatility Model described in Example 4.
    
    This class serves as the 'Ground Truth' system generator. It defines the 
    state-space equations used to simulate synthetic data for testing the filters.

    State Equation:
        X_n = alpha * X_{n-1} + sigma * V_n,  where V_n ~ N(0, 1)
        
    Observation Equation:
        Y_n = beta * exp(X_n / 2) * W_n,      where W_n ~ N(0, 1)
        
    Attributes:
        alpha (tf.Tensor): Autoregression coefficient (0 < alpha < 1).
        sigma (tf.Tensor): State noise standard deviation.
        beta (tf.Tensor): Observation scaling factor.
    """
    def __init__(self, alpha, sigma, beta):
        """
        Initialize the model parameters.

        Args:
            alpha (float): Autoregression coefficient (e.g., 0.91).
            sigma (float): State noise std dev (e.g., 1.0).
            beta (float): Observation scale (e.g., 0.5).
        """
        self.dtype = tf.float32
        self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype)
        self.sigma = tf.convert_to_tensor(sigma, dtype=self.dtype)
        self.beta = tf.convert_to_tensor(beta, dtype=self.dtype)

    def initial_dist(self):
        """
        Returns the initial stationary distribution for X_1.
        X_1 ~ N(0, sigma^2 / (1-alpha^2)).
        
        Returns:
            tfp.distributions.Normal: The distribution object for X_1.
        """
        variance = self.sigma**2 / (1.0 - self.alpha**2)
        return tfd.Normal(loc=0.0, scale=tf.sqrt(variance))

    def transition_dist(self, x_prev):
        """
        Returns the state transition distribution p(x_n | x_{n-1}).
        
        Args:
            x_prev (tf.Tensor): The state at time n-1.
            
        Returns:
            tfp.distributions.Normal: The distribution p(x_n | x_{n-1}).
        """
        return tfd.Normal(loc=self.alpha * x_prev, scale=self.sigma)

    def observation_dist(self, x_curr):
        """
        Returns the observation distribution p(y_n | x_n).
        
        Args:
            x_curr (tf.Tensor): The state at time n.
            
        Returns:
            tfp.distributions.Normal: The distribution p(y_n | x_n).
        """
        scale = self.beta * tf.exp(x_curr / 2.0)
        return tfd.Normal(loc=0.0, scale=scale)

    def simulate(self, T):
        """
        Simulates the Stochastic Volatility system for T time steps.

        Args:
            T (int): Number of time steps to simulate.

        Returns:
            tuple: (states, observations) containing:
                - states (tf.Tensor): The latent volatility path X_{1:T}.
                - observations (tf.Tensor): The observed returns Y_{1:T}.
        """
        x_hist, y_hist = [], []
        x = self.initial_dist().sample()
        y = self.observation_dist(x).sample()
        x_hist.append(x); y_hist.append(y)
        
        for _ in range(1, T):
            x = self.transition_dist(x).sample()
            y = self.observation_dist(x).sample()
            x_hist.append(x); y_hist.append(y)
            
        return tf.stack(x_hist), tf.stack(y_hist)
    

#####################################################################################################


######################################################################################################


def main():
    alpha = 0.91
    sigma = 1.0
    beta = 0.5
    T = 100    # Timesteps

    sv_model = StochasticVolatilityModel(alpha, sigma, beta)
    
    print(f"Simulating Stochastic Volatility Model (T={T})...")
    states, observations = sv_model.simulate(T)

    states_np = states.numpy()
    obs_np = observations.numpy()
    time_steps = np.arange(T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, states_np, color='blue', linewidth=1, label='Volatility (State X)')
    plt.scatter(time_steps, obs_np, color='red', s=10, marker='*', label='Observations (Y)')
    plt.title(f'Simulated Volatility Sequence (alpha={alpha}, sigma={sigma}, beta={beta})')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()