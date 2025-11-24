# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:38:04 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
from svssm import SVSSM

tfd = tfp.distributions


class SV_UKF():
    """
    Unscented Kalman Filter for the Stochastic Volatility Model.
    """
    def __init__(self, alpha : float, sigma : float, beta : float, mu_0 : tf.Tensor, P_0 : tf.Tensor):
        self.alpha = tf.constant(alpha, dtype=tf.float32) # Persistence parameter
        self.sigma = tf.constant(sigma, dtype=tf.float32) # State noise std dev
        self.beta = tf.constant(beta, dtype=tf.float32) # Observation scaling factor

        self.mu_t = mu_0 # Initial state mean (scalar)
        self.P_t = P_0   # Initial state covariance (scalar)
        
        self.mu_history = [mu_0]
        self.P_history = [P_0]
        
        # UKF Parameters
        self.n = 1 # State dimension = 1 for SVSSM
        self.alpha_ukf = 1e-3
        self.beta_ukf = 2.0
        self.kappa = 0.0
        self.lam = self.alpha_ukf**2 * (self.n + self.kappa) - self.n
        
        # Weights
        self.Wm0 = self.lam / (self.n + self.lam)
        self.Wc0 = self.Wm0 + (1 - self.alpha_ukf**2 + self.beta_ukf)
        self.Wi  = 0.5 / (self.n + self.lam)
        
        

    def _generate_sigmas(self, x, P):
        """
        Selecting a set of sample points that precisely capture the mean 
        and covariance of the current state distribution.
        
        Args:
            x : state mean
            P : covariance
            
        Returns:
            Upper and lower bounds on sampling distribution
        """
        
        sigma_dist = tf.sqrt((self.n + self.lam) * P)
        return tf.stack([x, x + sigma_dist, x - sigma_dist])
    
    

    def filter(self, observations):
        """
        Applies the UKF to a sequence of observations.
        
        Args:
            observations (tf.Tensor): A tensor of observations y[0], y[1], ..., y[T] 
                                      of shape (T+1, dim).
        Returns:
            (tf.Tensor, tf.Tensor): Tensors containing the filtered mean mu[t|t] 
                                    and covariance P[t|t] history.
        """
        
        for t in range(tf.shape(observations)[0]):
            sigmas = self._generate_sigmas(self.mu_t, self.P_t)
            sigmas_pred = self.alpha * sigmas
            
            mu_pred = (self.Wm0 * sigmas_pred[0]) + tf.reduce_sum(self.Wi * sigmas_pred[1:])
            
            diff = sigmas_pred - mu_pred
            P_pred = (self.Wc0 * diff[0]**2) + tf.reduce_sum(self.Wi * diff[1:]**2) + self.sigma**2
            
            # Update
            y_obs = observations[t]
            z_meas = y_obs**2
            sigmas_next = self._generate_sigmas(mu_pred, P_pred)
            
            Z_sigmas = (self.beta**2) * tf.math.exp(sigmas_next) # h(x) = beta^2 * exp(x)
            
            z_pred = (self.Wm0 * Z_sigmas[0]) + tf.reduce_sum(self.Wi * Z_sigmas[1:])
            
            # Measure Covariance
            R_t = 2.0 * tf.square((self.beta**2) * tf.math.exp(mu_pred))
            
            diff_z = Z_sigmas - z_pred
            S = (self.Wc0 * diff_z[0]**2) + tf.reduce_sum(self.Wi * diff_z[1:]**2) + R_t
            
            # Cross Covariance
            diff_x = sigmas_next - mu_pred
            Pxz = (self.Wc0 * diff_x[0] * diff_z[0]) + tf.reduce_sum(self.Wi * diff_x[1:] * diff_z[1:])
            
            # Kalman Gain
            K = Pxz / S
            
            # Final Update
            res = z_meas - z_pred
            self.mu_t = mu_pred + (K * res)
            self.P_t = P_pred - (K * S * K)
            
            self.mu_history.append(self.mu_t)
            self.P_history.append(self.P_t)
            
        return tf.stack(self.mu_history), tf.stack(self.P_history)
    
    
    
    
    
    
    
    
    
    ##########################################################################################################
    
    
if __name__ == "__main__":
    print("---- SVSSM Data Generation")
    STEPS = 100
    sv_model = SVSSM(alpha=0.91, sigma=0.5, beta=0.5, seed=42)
    observations, true_states = sv_model.generate_data(STEPS)
    
    print(f"Generated {STEPS+1} observations.")
    print("---- Running Unscented Kalman Filter (UKF)")
    ukf = SV_UKF(alpha=0.91, sigma=0.5, beta=0.5, mu_0=0.0, P_0=1.0)
    ukf_mu, ukf_cov = ukf.filter(observations)
    print(ukf_mu[:5])
    
    
    
    
    
    