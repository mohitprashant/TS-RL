# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 20:12:22 2025

@author: 18moh
"""


import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
from svssm import SVSSM
import matplotlib.pyplot as plt

tfd = tfp.distributions


class SV_ParticleFilter():
    """
    Particle Filter for the Stochastic Volatility Model
    """
    def __init__(self, alpha : float, sigma : float, beta : float, num_particles=1000, mu_0=0.0, P_0=1.0):
        self.alpha = tf.constant(alpha, dtype=tf.float32) # Persistence parameter
        self.sigma = tf.constant(sigma, dtype=tf.float32) # State noise std dev
        self.beta = tf.constant(beta, dtype=tf.float32) # Observation scaling factor
        self.N = num_particles
        
        # Initialize Particles from the prior
        self.particles = tf.random.normal(
            shape=[self.N], 
            mean=mu_0, 
            stddev=tf.sqrt(P_0)
        )
        
        # Weights (Log space for stability)
        self.log_weights = tf.zeros(shape=[self.N], dtype=tf.float32) - tf.math.log(float(self.N))
        
        self.mu_history = []
        self.P_history = []
        self.degen_history = []
        


    def _gaussian_log_likelihood(self, y, particles):
        """
        Helper function to calculates p(y|x) for the SV model
        """
        # Variance of observation given state
        obs_variance = (self.beta**2) * tf.math.exp(particles)
        
        # Gaussian Log PDF: -0.5 * log(2*pi*var) - (y-mean)^2 / (2*var)
        term1 = -0.5 * tf.math.log(2.0 * np.pi * obs_variance)
        term2 = -0.5 * (y**2) / obs_variance
        return term1 + term2



    def filter(self, observations):
        """
        Applies the particle filter to a sequence of observations
        
        Args:
            observations (tf.Tensor): A tensor of observations y[0], y[1], ..., y[T] 
                                      of shape (T+1, dim).
        Returns:
            (tf.Tensor, tf.Tensor): Tensors containing the filtered mean mu[t|t] 
                                    and covariance P[t|t] history.
        """
        
        # Reset
        self.mu_history = []
        self.P_history = []
        current_particles = self.particles
        

        for t in range(tf.shape(observations)[0]):
            y_obs = observations[t]
            
            process_noise = tf.random.normal(shape=[self.N], mean=0.0, stddev=1.0)
            pred_particles = (self.alpha * current_particles) + (self.sigma * process_noise)
            
            # Weights
            log_likelihoods = self._gaussian_log_likelihood(y_obs, pred_particles)
            log_weights_unnorm = log_likelihoods
            weights = tf.nn.softmax(log_weights_unnorm)
            
            # Get degen
            n_eff = 1.0 / tf.reduce_sum(tf.square(weights))
            self.degen_history.append(n_eff)
            
            # Estimate 
            mu_est = tf.reduce_sum(weights * pred_particles)
            var_est = tf.reduce_sum(weights * tf.square(pred_particles - mu_est)) # Variance = sum(w * (x - mean)^2)
            
            self.mu_history.append(mu_est)
            self.P_history.append(var_est)
            
            # Resample
            indices = tf.random.categorical(tf.expand_dims(tf.math.log(weights), 0), self.N)
            indices = tf.reshape(indices, [-1])
            current_particles = tf.gather(pred_particles, indices)
            
        return tf.stack(self.mu_history), tf.stack(self.P_history)
    
    
    
    
    
    
        
    ##########################################################################################################


if __name__ == "__main__":
    STEPS = 100
    sv_model = SVSSM(alpha=0.91, sigma=0.5, beta=0.5, seed=42)
    observations, true_states = sv_model.generate_data(STEPS)
    
    pf = SV_ParticleFilter(alpha=0.91, sigma=0.5, beta=0.5, num_particles=1000)
    pf_mu, pf_cov = pf.filter(observations)
    
    time_steps = np.arange(STEPS + 1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, true_states.numpy(), 'k-', label='True State', linewidth=2)
    plt.plot(time_steps, pf_mu.numpy(), 'g--', label='Particle Filter Estimate', linewidth=2)
    
    
    std_dev = tf.sqrt(pf_cov).numpy() # Confident Intervals
    plt.fill_between(time_steps, 
                     pf_mu.numpy() - 2*std_dev, 
                     pf_mu.numpy() + 2*std_dev, 
                     color='green', alpha=0.2, label='PF 95% CI')
    
    plt.title('Particle Filter for SV Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    pf_rmse = tf.sqrt(tf.reduce_mean(tf.square(pf_mu - true_states)))
    print(f"Particle Filter RMSE: {pf_rmse.numpy():.4f}")
    
    
    
    
    