import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
from svssm import StochasticVolatilityModel

tfd = tfp.distributions
class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for the Stochastic Volatility Model.
    
    Implements the Sequential Importance Resampling (SIR) algorithm with adaptive resampling.
    
    Attributes:
        num_particles (int): Number of particles (N).
        ess_threshold (float): Threshold for resampling (ratio * N).
        ratio (float): Ratio used for labeling purposes.
    """
    def __init__(self, alpha, sigma, beta, num_particles=1000, resample_threshold_ratio=0.5):
        """
        Initializes the Particle Filter.

        Args:
            alpha (float): Autoregression coefficient.
            sigma (float): State noise standard deviation.
            beta (float): Observation scaling factor.
            num_particles (int): Number of particles to use.
            resample_threshold_ratio (float): Fraction of N below which resampling occurs.
        """
        self.dtype = tf.float32
        self.alpha = tf.convert_to_tensor(alpha, dtype=self.dtype)
        self.sigma = tf.convert_to_tensor(sigma, dtype=self.dtype)
        self.beta = tf.convert_to_tensor(beta, dtype=self.dtype)
        self.num_particles = num_particles
        self.ess_threshold = num_particles * resample_threshold_ratio
        self.ratio = resample_threshold_ratio


    def initial_dist(self):
        """Returns the initial proposal distribution for particles at t=1."""
        variance = self.sigma**2 / (1.0 - self.alpha**2)
        return tfd.Normal(loc=0.0, scale=tf.sqrt(variance))


    def transition(self, particles):
        """
        Propagates particles forward one step using the transition model p(x_n | x_{n-1}).
        
        Args:
            particles (tf.Tensor): Current particles of shape (N,).
        Returns:
            tf.Tensor: Propagated particles of shape (N,).
        """
        return tfd.Normal(loc=self.alpha * particles, scale=self.sigma).sample()


    def log_likelihood(self, y_obs, particles):
        """
        Computes the log-likelihood weights: log p(y_n | x_n).
        
        Args:
            y_obs (tf.Tensor): Current observation (scalar).
            particles (tf.Tensor): Current particles (N,).
        Returns:
            tf.Tensor: Log weights (N,).
        """
        scale = self.beta * tf.exp(particles / 2.0)
        return tfd.Normal(loc=0.0, scale=scale).log_prob(y_obs)


    def resample(self, particles, log_weights):
        """
        Performs Systematic Resampling of particles.

        Args:
            particles (tf.Tensor): Current particles.
            log_weights (tf.Tensor): Unnormalized log weights.
            
        Returns:
            tuple: (resampled_particles, new_log_weights)
        """
        log_weights_norm = log_weights - tf.reduce_logsumexp(log_weights)
        indices = tf.random.categorical(tf.expand_dims(log_weights_norm, 0), self.num_particles)
        resampled_particles = tf.gather(particles, tf.reshape(indices, [-1]))
        new_log_weights = tf.fill([self.num_particles], -tf.math.log(float(self.num_particles)))
        return resampled_particles, new_log_weights


    def run_filter(self, observations, true_states=None):
        """
        Runs the Particle Filter over a sequence of observations.

        Args:
            observations (tf.Tensor): Tensor of shape (T,) containing observed values y_n.
            true_states (tf.Tensor, optional): Ground truth states for RMSE calculation.

        Returns:
            dict: Performance metrics including RMSE, runtime, memory, ESS history, and estimates.
        """
        T = observations.shape[0]
        particles = self.initial_dist().sample(self.num_particles)
        log_weights = tf.fill([self.num_particles], -tf.math.log(float(self.num_particles)))
        estimates = []
        ess_history = []
        
        tracemalloc.start()                                                  # Track memory and runtime
        start_time = time.time()

        for t in range(T):
            particles = self.transition(particles)
            log_weights += self.log_likelihood(observations[t], particles)   # Weighting
            
            log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)      # ESS Calc
            w_norm = tf.exp(log_w_norm)
            ess = 1.0 / tf.reduce_sum(w_norm**2)
            ess_history.append(ess)
            
            if(ess < self.ess_threshold):                                    # Resample if ESS below threshold
                particles, log_weights = self.resample(particles, log_weights)
                w_norm = tf.fill([self.num_particles], 1.0/self.num_particles) 

            estimates.append(tf.reduce_sum(w_norm * particles))              # Get Estimate

        _, peak_mem = tracemalloc.get_traced_memory()                        # Finish Tracking mem and runtime
        tracemalloc.stop()
        total_time = time.time() - start_time
        
        # Process Output
        estimates_tensor = tf.stack(estimates)
        rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2)) if true_states is not None else 0.0

        return {
            'label': f'PF (N={self.num_particles}, Th={self.ratio:.1f})',
            'rmse': rmse, 
            'time': total_time, 
            'mem': peak_mem, 
            'estimates': estimates_tensor,
            'ess_avg': np.mean(ess_history),
            'particles': self.num_particles,
            'threshold_ratio': self.ratio
        }
    
  