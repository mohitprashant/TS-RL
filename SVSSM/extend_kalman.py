# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 19:35:43 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
from svssm import SVSSM

tfd = tfp.distributions


class SV_EKF():
    """
    Extended Kalman Filter for the Stochastic Volatility Model.
    """
    def __init__(self, alpha : float, sigma : float, beta : float, mu_0 : tf.Tensor, P_0 : tf.Tensor):
        self.alpha = tf.constant(alpha, dtype=tf.float32) # Persistence parameter
        self.sigma = tf.constant(sigma, dtype=tf.float32) # State noise std dev
        self.beta = tf.constant(beta, dtype=tf.float32) # Observation scaling factor

        self.mu_t = mu_0 # Initial state mean (scalar)
        self.P_t = P_0   # Initial state covariance (scalar)
        
        self.mu_history = [mu_0]
        self.P_history = [P_0]
        
        
        
    def filter(self, observations):
        """
        Applies the EKF to a sequence of observations.
        
        Args:
            observations (tf.Tensor): A tensor of observations y[0], y[1], ..., y[T] 
                                      of shape (T+1, dim).
        Returns:
            (tf.Tensor, tf.Tensor): Tensors containing the filtered mean mu[t|t] 
                                    and covariance P[t|t] history.
        """
        for t in range(tf.shape(observations)[0]):
            # x = alpha * x
            mu_pred = self.alpha * self.mu_t
            
            # P = alpha^2 * P + sigma^2
            P_pred = (self.alpha**2 * self.P_t) + self.sigma**2
            
            
            # z = y^2 (Squared Observation Strategy)
            y_obs = observations[t]
            z_meas = y_obs**2
            
            # Jacobian Calculation
            # h(x) = E[y^2] = beta^2 * exp(x)
            with tf.GradientTape() as tape:
                tape.watch(mu_pred)
                h_x = (self.beta**2) * tf.math.exp(mu_pred)
            
            H = tape.gradient(h_x, mu_pred) # d(h)/dx
            
            res = z_meas - h_x
            
            # Var(y^2) approx 2 * (beta^2 * exp(x))^2
            R_t = 2.0 * tf.square(h_x)
            
            # Res covar
            S = (H * P_pred * H) + R_t
            
            # Kalman Gain: K = P*H / S
            K = (P_pred * H) / S
            
            # Update
            self.mu_t = mu_pred + (K * res)
            self.P_t = (1.0 - K * H) * P_pred
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
    print("-" * 30)
    print("---- Running Extended Kalman Filter (EKF)")
    ekf = SV_EKF(alpha=0.91, sigma=0.5, beta=0.5, mu_0=0.0, P_0=1.0)
    ekf_mu, ekf_cov = ekf.filter(observations)
    
    print(ekf_mu[:5])
    
    
    
    