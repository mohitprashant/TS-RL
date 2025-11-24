# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 16:57:45 2025

@author: 18moh
"""

from lgssm import LGSSM
from base_kalman import KalmanFilter
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


def simple_moving_average_tf(data: tf.Tensor, window: int) -> tf.Tensor:
    """
    Computes the Simple Moving Average (SMA) for the first dimension of the data
    using TensorFlow's 1D convolution.
    
    Args:
        data (tf.Tensor): The input time series data of shape (T, D).
        window (int): The size of the averaging window.
        
    Returns:
        tf.Tensor: The filtered data of shape (T, D).
    """
    
    
    # Input data shape is (T, D). We are smoothing the first dimension (index 0).
    T, D = tf.shape(data)[0], tf.shape(data)[1]
    data_1d = data[:, 0]
    input_tensor = tf.expand_dims(tf.expand_dims(data_1d, axis=0), axis=-1)
    
    # Kernel shape is [window, 1, 1]
    kernel_weights = tf.ones([window, 1, 1], dtype=data.dtype) / tf.cast(window, data.dtype)
    smoothed_output = tf.nn.conv1d(
        input=input_tensor, 
        filters=kernel_weights, 
        stride=1, 
        padding='VALID'
    )
    
    # Squeeze to shape (T - window + 1)
    smoothed_output = tf.squeeze(smoothed_output)
    
    # Pad the output to match the original length T
    padding_size = window - 1
    nan_padding = tf.fill(tf.stack([padding_size]), tf.constant(float('nan'), dtype=data.dtype))
    smoothed_padded = tf.concat([nan_padding, smoothed_output], axis=0)
    
    ma_output = tf.Variable(tf.zeros((T, D), dtype=data.dtype))
    ma_output[:, 0].assign(smoothed_padded)
    if(D > 1):
        ma_output[:, 1:].assign(data[:, 1:])
        
    return ma_output.read_value()



def joseph_stabilized_analysis():
    # seed_range = [x for x in range(42+1)]
    
    # LGSSM parameters for tracking problem
    dt = 1.0
    DIM = 2
    
    A_mat = tf.constant([[1.0, dt], [0.0, 1.0]]) # Constant velocity model
    C_mat = tf.constant([[1.0, 0.0], [0.0, 1.0]]) # Observe both position and velocity
    B_mat = tf.eye(DIM) * 0.1 # Small process noise
    D_mat = tf.eye(DIM) * 1.0 # Large observation noise (noisy measurements)
    x_init = tf.constant([0.0, 1.0]) # Initial state: position 0, velocity 1
    
    
    STEPS = 100
    
    # Generate Data (Ground Truth)
    lg = LGSSM(DIM, A=A_mat, C=C_mat, B=B_mat, D=D_mat, x=x_init, seed=42)
    observations = lg.generate_data(STEPS) 
    ground_truth_states = tf.stack(lg.state_history, axis=0)
    A, C, Q, R, mu_0, P_0 = lg.get_lgssm_params()
    
    print("---- Kalman Filter Stability and Optimality Analysis")
    print(f"Total time steps: {STEPS + 1}")
    print("-" * 30)
    

    # Run Standard Filter
    print("---------- 1. Standard Covariance Update")
    kf_standard = KalmanFilter(A, C, Q, R, mu_0, P_0)
    mu_std, P_std, cond_std = kf_standard.filter(observations)
    
    rmse_std = tf.sqrt(tf.reduce_mean(tf.square(mu_std[:, 0] - ground_truth_states[:, 0])))
    max_cond_std = tf.reduce_max(cond_std)
    
    print(f"-> Filtered Position RMSE: {rmse_std.numpy():.4f}")
    print(f"-> Max Innovation Cov. Cond. Num: {max_cond_std.numpy():.2e}")
    
    # Check for potential non-PSD (Negative Eigen.)
    min_eig_std = tf.reduce_min(tf.linalg.eigvalsh(P_std[-1]))
    print(f"-> Min Eigenvalue of last P[t|t]: {min_eig_std.numpy():.2e}")
    if min_eig_std.numpy() < -1e-8:
        print("WARNING: Standard P[t|t] may have lost positive semi-definiteness.")

    print("-" * 30)
    
    
    # Run Stabilized Filter (Joseph)
    print("---------- 2. Joseph Stabilized Covariance Update")
    kf_joseph = KalmanFilter(A, C, Q, R, mu_0, P_0)
    mu_joseph, P_joseph, cond_joseph = kf_joseph.jsc_filter(observations)

    rmse_joseph = tf.sqrt(tf.reduce_mean(tf.square(mu_joseph[:, 0] - ground_truth_states[:, 0])))
    jsf_max_cond_std = tf.reduce_max(cond_std)
    
    print(f"-> Filtered Position RMSE: {rmse_joseph.numpy():.4f}")
    print(f"-> Max Innovation Cov. Cond. Num: {jsf_max_cond_std.numpy():.2e}")
    
    min_eig_joseph = tf.reduce_min(tf.linalg.eigvalsh(P_joseph[-1]))
    print(f"-> Min Eigenvalue of last P[t|t]: {min_eig_joseph.numpy():.2e}")
    if min_eig_joseph.numpy() < -1e-8:
        print("WARNING: Joseph P[t|t] also lost PSD. This is extremely rare and suggests massive ill-conditioning.")
    
    print("-" * 30)
    

    # Comparative Analysis
    print("---------- 3. Filtering Optimality and Stability Summary")
    
    
    # Optimality Check (Mean Estimates)
    mean_diff = tf.reduce_max(tf.abs(mu_std - mu_joseph))
    print(f"-> Max difference in means (mu_std vs. mu_joseph): {mean_diff.numpy():.2e}")
    
    
    # Covariance Comparison (Stability Check)
    P_diff = tf.reduce_max(tf.abs(P_std - P_joseph))
    print(f"-> Max difference in covariances (P_std vs. P_joseph): {P_diff.numpy():.2e}")
    
    
    # Conditioning Analysis
    avg_cond = tf.reduce_mean(cond_std)
    print(f"-> Average Innovation Covariance Condition Number: {avg_cond.numpy():.2e}")



def mean_filter_analysis():
    # LGSSM parameters for a tracking problem
    dt = 1.0
    DIM = 2
    
    A_mat = tf.constant([[1.0, dt], [0.0, 1.0]]) # Constant velocity model
    C_mat = tf.constant([[1.0, 0.0], [0.0, 1.0]]) # Observe both position and velocity
    B_mat = tf.eye(DIM) * 0.1 # Small process noise
    D_mat = tf.eye(DIM) * 1.0 # Large observation noise (noisy measurements)
    x_init = tf.constant([0.0, 1.0]) # Initial state: position 0, velocity 1
    
    
    
    STEPS = 100 
    WINDOW_SIZE = 10 
    
    # Generate Data
    lg = LGSSM(DIM, A=A_mat, C=C_mat, B=B_mat, D=D_mat, x=x_init, seed=42)
    observations = lg.generate_data(STEPS)
    ground_truth_states = tf.stack(lg.state_history, axis=0)
    A, C, Q, R, mu_0, P_0 = lg.get_lgssm_params()
    
    print("---- Pure TensorFlow Moving Average Comparison")
    print("-" * 30)
    
    # Run Kalman Filter
    kf = KalmanFilter(A, C, Q, R, mu_0, P_0)
    mu_kf, P_kf_tensor, con = kf.filter(observations)
    
    
    # Run Simple Moving Average (TensorFlow only)
    ma_estimates = simple_moving_average_tf(observations, WINDOW_SIZE)
    
    
    # KF RMSE
    rmse_kf = tf.sqrt(tf.reduce_mean(tf.square(mu_kf[:, 0] - ground_truth_states[:, 0])))
    
    # MA RMSE (filter out NaN values)
    start_index = WINDOW_SIZE - 1 
    ma_valid_estimates = ma_estimates[start_index:, 0]
    ground_truth_valid = ground_truth_states[start_index:, 0]
    
    rmse_ma = tf.sqrt(tf.reduce_mean(tf.square(ma_valid_estimates - ground_truth_valid)))

    # Baseline Raw RMSE
    rmse_raw = tf.sqrt(tf.reduce_mean(tf.square(observations[:, 0] - ground_truth_states[:, 0])))
    
    print(f"---- Results Summary")
    print(f"-> Raw Observation Position RMSE (Baseline): {rmse_raw.numpy():.4f}")
    print(f"-> Kalman Filter Position RMSE: {rmse_kf.numpy():.4f}")
    print(f"-> Moving Average (W={WINDOW_SIZE}) Position RMSE: {rmse_ma.numpy():.4f}")

    # Plot the results (Converting Tensors to NumPy for Matplotlib)
    plt.figure(figsize=(12, 6))
    time = np.arange(tf.shape(ground_truth_states)[0].numpy())
    
    # Position (State Dimension 0)
    plt.plot(time, ground_truth_states[:, 0].numpy(), 'k-', label='Ground Truth State (Position)', linewidth=2)
    plt.plot(time, observations[:, 0].numpy(), 'b.', alpha=0.3, label='Noisy Observations')
    plt.plot(time, mu_kf[:, 0].numpy(), 'r-', label='Kalman Filter Estimate', linewidth=2)
    plt.plot(time, ma_estimates[:, 0].numpy(), 'g--', label=f'Moving Average (W={WINDOW_SIZE})', linewidth=2)

    plt.title('Kalman Filter vs. Moving Average Estimate')
    plt.xlabel('Time Step')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    joseph_stabilized_analysis()
    print("\n\n\n\n")
    mean_filter_analysis()
    
    