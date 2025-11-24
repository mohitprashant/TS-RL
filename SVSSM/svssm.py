# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 18:28:05 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions



class SVSSM:
    """
    Stochastic Volatility model from Example 4, Doucet.
    
    State Equation:
        X[n] = alpha * X[n-1] + sigma * V[n]
        where V[n] ~ N(0, 1)
        
    Observation Equation:
        Y[n] = beta * exp(X[n] / 2) * W[n]
        where W[n] ~ N(0, 1)
        
    Initialization:
        X[1] ~ N(0, sigma^2 / (1 - alpha^2))
    """
    def __init__(self, alpha=0.91, sigma=1.0, beta=0.5, seed=None):
        # Set parameters
        self.alpha = tf.constant(alpha, dtype=tf.float32) # Persistence parameter
        self.sigma = tf.constant(sigma, dtype=tf.float32) # Standard deviation of the state transition noise
        self.beta  = tf.constant(beta, dtype=tf.float32) # Scaling factor for the observation
        self.state_history = []
        self.observation_history = []
        
        
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            

        self.stationary_std = self.sigma / tf.sqrt(1.0 - self.alpha**2)
        self.x = tf.random.normal(shape=[], mean=0.0, stddev=self.stationary_std)
        w_noise = tf.random.normal(shape=[], mean=0.0, stddev=1.0)
        self.y = self.beta * tf.math.exp(self.x / 2.0) * w_noise
        
        self.state_history.append(self.x)
        self.observation_history.append(self.y)



    def generate_next(self):
        """
        Simulates the next time step (n -> n+1).
        
        Returns:
            The system observation
        """
        
        v_n = tf.random.normal(shape=[], mean=0.0, stddev=1.0) # State noise
        w_n = tf.random.normal(shape=[], mean=0.0, stddev=1.0) # Observation noise
        
        # X_n = alpha * X_{n-1} + sigma * V_n
        self.x = (self.alpha * self.x) + (self.sigma * v_n)
        
        # Y_n = beta * exp(X_n / 2) * W_n
        self.y = self.beta * tf.math.exp(self.x / 2.0) * w_n
        
        self.state_history.append(self.x)
        self.observation_history.append(self.y)
        
        return self.y
    

    def generate_data(self, steps):
        """
        Generates a sequence of data for a given number of steps.
        
        Returns:
            y_data (tf.Tensor): Tensor of observations (shape: [steps+1])
            x_data (tf.Tensor): Tensor of latent states (shape: [steps+1])
        """
        
        for _ in range(steps):
            self.generate_next()
            
        return tf.stack(self.observation_history), tf.stack(self.state_history)



if __name__ == "__main__":
    sv_model = SVSSM(alpha=0.91, sigma=1.0, beta=0.5, seed=42)
    y_data, x_data = sv_model.generate_data(steps=100)
    
    print("First 5 Latent States (X):", x_data[:5].numpy())
    print("First 5 Observations (Y):", y_data[:5].numpy())





