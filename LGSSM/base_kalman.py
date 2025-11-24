# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:13:09 2025

@author: 18moh
"""


from lgssm import LGSSM
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np


class KalmanFilter():
    """
    Kalman Filter for a Linear Gaussian State Space Model.
    """
    def __init__(self, A, C, Q, R, mu_0, P_0):
        """
        Initializes the Kalman Filter with LGSSM parameters
        
        Args:
            A (tf.Tensor): Transition matrix (dim, dim).
            C (tf.Tensor): Observation matrix (dim, dim).
            Q (tf.Tensor): Process noise covariance matrix (dim, dim).
            R (tf.Tensor): Observation noise covariance matrix (dim, dim).
            mu_0 (tf.Tensor): Initial state mean estimate mu[0|0] (dim).
            P_0 (tf.Tensor): Initial state covariance estimate P[0|0] (dim, dim).
        """
        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        
        self.mu_t = tf.expand_dims(mu_0, axis=1) # Current mean estimate
        self.P_t = P_0   # Current covariance estimate
        
        self.mu_history = [mu_0] 
        self.P_history = [P_0]
        self.cond_num_history = []
        
    
    def filter(self, observations):
        """
        Applies the Kalman Filter to a sequence of observations y[0], y[1], ...
        
        Args:
            observations (tf.Tensor): A tensor of observations y[0], y[1], ..., y[T] 
                                      of shape (T+1, dim).
        Returns:
            (tf.Tensor, tf.Tensor): Tensors containing the filtered mean mu[t|t] 
                                    and covariance P[t|t] history.
        """
        
        for t in range(1, tf.shape(observations)[0]):
            y_t = tf.expand_dims(observations[t], axis=1) 
            
            # State prediction: mu[t|t-1] = A @ mu[t-1|t-1]
            mu_pred = self.A @ self.mu_t
            # Cov. prediction: P[t|t-1] = A @ P[t-1|t-1] @ A^T + Q
            P_pred = self.A @ self.P_t @ tf.transpose(self.A) + self.Q
            
            
            # Residual: z = y[t] - C @ mu[t|t-1]
            res = y_t - self.C @ mu_pred
            # Res. Covariance: S = C @ P[t|t-1] @ C^T + R
            res_cov = self.C @ P_pred @ tf.transpose(self.C) + self.R
            s, _, _ = tf.linalg.svd(res_cov) 
            sigma_max = s[0]
            sigma_min = s[-1] + 1e-12
            self.cond_num_history.append(sigma_max/sigma_min)
            
            # Kalman Gain: K = P[t|t-1] @ C^T @ S^-1
            K = P_pred @ tf.transpose(self.C) @ tf.linalg.inv(res_cov)
            
            # State Update: mu[t|t] = mu[t|t-1] + K @ residual
            self.mu_t = mu_pred + K @ res
            # Covariance Update: P[t|t] = (I - K @ C) @ P[t|t-1]
            I = tf.eye(self.A.shape[0], dtype=self.A.dtype)
            self.P_t = (I - K @ self.C) @ P_pred
            
            self.mu_history.append(tf.squeeze(self.mu_t)) 
            self.P_history.append(self.P_t)
            
        return tf.stack(self.mu_history, axis=0), tf.stack(self.P_history, axis=0), tf.stack(self.cond_num_history, axis=0)
    
    
    
    def jsc_filter(self, observations):
        """
        Applies the Kalman Filter with Joseph-stabilized covariance update to a 
        sequence of observations y[0], y[1], ...
        
        Args:
            observations (tf.Tensor): A tensor of observations y[0], y[1], ..., y[T] 
                                      of shape (T+1, dim).
        Returns:
            (tf.Tensor, tf.Tensor): Tensors containing the filtered mean mu[t|t] 
                                    and covariance P[t|t] history (stabilized)
        """
        
        for t in range(1, tf.shape(observations)[0]):
            y_t = tf.expand_dims(observations[t], axis=1) 
            
            # State prediction: mu[t|t-1] = A @ mu[t-1|t-1]
            mu_pred = self.A @ self.mu_t
            # Cov. prediction: P[t|t-1] = A @ P[t-1|t-1] @ A^T + Q
            P_pred = self.A @ self.P_t @ tf.transpose(self.A) + self.Q
            
            
            # Residual: z = y[t] - C @ mu[t|t-1]
            res = y_t - self.C @ mu_pred
            # Res. Covariance: S = C @ P[t|t-1] @ C^T + R
            res_cov = self.C @ P_pred @ tf.transpose(self.C) + self.R
            s, _, _ = tf.linalg.svd(res_cov) 
            sigma_max = s[0]
            sigma_min = s[-1] + 1e-12
            self.cond_num_history.append(sigma_max/sigma_min)
            
            # Kalman Gain: K = P[t|t-1] @ C^T @ S^-1
            K = P_pred @ tf.transpose(self.C) @ tf.linalg.inv(res_cov)
            
            # State Update: mu[t|t] = mu[t|t-1] + K @ residual
            self.mu_t = mu_pred + K @ res
            
            # Covariance Update: Joseph Stabilized Form = P[t|t] = (I - K @ C) @ P[t|t-1] @ (I - K @ C)^T + K @ R @ K^T
            I = tf.eye(self.A.shape[0], dtype=self.A.dtype)
            I_minus_KC = I - K @ self.C 
            term1 = I_minus_KC @ P_pred @ tf.transpose(I_minus_KC)
            term2 = K @ self.R @ tf.transpose(K)
            self.P_t = term1 + term2
            
            self.mu_history.append(tf.squeeze(self.mu_t)) 
            self.P_history.append(self.P_t)
            
        return tf.stack(self.mu_history, axis=0), tf.stack(self.P_history, axis=0), tf.stack(self.cond_num_history, axis=0)




##########################################################################################################





if __name__ == "__main__":
    print("---- LGSSM Data Generation")
    
    # LGSSM parameters for a tracking problem
    dt = 1.0 # time step
    DIM = 2
    
    A_mat = tf.constant([[1.0, dt], [0.0, 1.0]]) # Constant velocity model
    C_mat = tf.constant([[1.0, 0.0], [0.0, 1.0]]) # Observe both position and velocity
    B_mat = tf.eye(DIM) * 0.1 # Small process noise
    D_mat = tf.eye(DIM) * 1.0 # Large observation noise (noisy measurements)
    x_init = tf.constant([0.0, 1.0]) # Initial state: position 0, velocity 1
    
    
    lg = LGSSM(
        DIM, 
        A=A_mat, 
        C=C_mat, 
        B=B_mat, 
        D=D_mat,
        x=x_init,
        seed=42
    )
    
    
    # Generate the full sequence of noisy observations
    STEPS = 50
    observations = lg.generate_data(STEPS) 
    print(f"Generated {STEPS+1} observations of shape: {observations.shape}")
    ground_truth_states = tf.stack(lg.state_history, axis=0)
    
    
    # Run Kalman Filter
    print("-" * 30)
    print("---- Running Kalman Filter")
    A, C, Q, R, mu_0, P_0 = lg.get_lgssm_params()
    kf = KalmanFilter(A, C, Q, R, mu_0, P_0)
    filtered_means, filtered_covs, con = kf.jsc_filter(observations)
    # filtered_means, filtered_covs, con = kf.filter(observations)
    
    print(f"Filtered means (mu[t|t]) shape: {filtered_means.shape}")
    print(f"Filtered covariances (P[t|t]) shape: {filtered_covs.shape}")
    
    
    # Evaluate Filter
    # Calculate the Root Mean Square Error (RMSE) of the filter estimate
    position_error = filtered_means[:, 0] - ground_truth_states[:, 0]
    rmse = tf.sqrt(tf.reduce_mean(tf.square(position_error)))
    
    print("-" * 30)
    print(f"Filtered Position RMSE: {rmse.numpy():.4f}")
    
    # Compare with the RMSE of the raw observation
    observation_position_error = observations[:, 0] - ground_truth_states[:, 0]
    obs_rmse = tf.sqrt(tf.reduce_mean(tf.square(observation_position_error)))
    
    print(f"Raw Observation Position RMSE: {obs_rmse.numpy():.4f}")
    
    
    
    
    
    
    
    
    