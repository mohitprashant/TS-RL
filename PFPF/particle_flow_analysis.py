# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 22:35:58 2025

@author: 18moh
"""

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
from particle_flow import ParticleFlowBase
from ledh import LEDHFilter
from edh import EDHFilter
from invert import IPFFilter
from svssm import SVSSM
import matplotlib.pyplot as plt
import time



if __name__ == "__main__":
    # Parameters
    STEPS = 100
    PARTICLES = 200
    FLOW_STEPS = 50
    
    # Generate Data
    sv = SVSSM(alpha=0.91, sigma=0.5, beta=0.5, seed=42)
    y_data, x_data = sv.generate_data(STEPS)
    
    # Define T_steps based on actual data length
    T_LEN = len(y_data)
    t_steps = np.arange(T_LEN)

    print(f"Data Shapes -> Y: {y_data.shape}, X: {x_data.shape}")

    # Run Filters
    print("Running EDH...")
    edh = EDHFilter(0.91, 0.5, 0.5, PARTICLES, FLOW_STEPS)
    est_edh = edh.filter(y_data)
    
    print("Running LEDH...")
    ledh = LEDHFilter(0.91, 0.5, 0.5, PARTICLES, FLOW_STEPS)
    est_ledh = ledh.filter(y_data)
    
    print("Running IPF...")
    ipf = IPFFilter(0.91, 0.5, 0.5, PARTICLES, FLOW_STEPS)
    est_ipf = ipf.filter(y_data)

    # Helper to align and calculate RMSE
    def robust_rmse(est_tensor, true_tensor):
        e = tf.reshape(est_tensor, [-1]).numpy()
        t = tf.reshape(true_tensor, [-1]).numpy()
        
        # Align lengths: Truncate to the minimum common length
        L = min(len(e), len(t))
        e = e[:L]
        t = t[:L]
        
        # Remove NaNs if any slipped through
        valid_mask = ~np.isnan(e)
        if np.sum(valid_mask) == 0: return np.nan
        
        return np.sqrt(np.mean((e[valid_mask] - t[valid_mask])**2))

    # Calculate RMSE
    rmse_edh = robust_rmse(est_edh, x_data)
    rmse_ledh = robust_rmse(est_ledh, x_data)
    rmse_ipf = robust_rmse(est_ipf, x_data)
    
    print(f"\nRMSE Results:")
    print(f"EDH:  {rmse_edh:.4f}")
    print(f"LEDH: {rmse_ledh:.4f}")
    print(f"IPF:  {rmse_ipf:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))
    def plot_robust(ax, time_arr, data_tensor, **kwargs):
        d = tf.reshape(data_tensor, [-1]).numpy()
        L = min(len(time_arr), len(d))
        ax.plot(time_arr[:L], d[:L], **kwargs)

    plot_robust(plt, t_steps, x_data, color='k', label='True State', linewidth=2, alpha=0.6)
    plot_robust(plt, t_steps, est_edh, color='r', linestyle='--', label=f'EDH (RMSE={rmse_edh:.2f})')
    plot_robust(plt, t_steps, est_ledh, color='b', linestyle='-.', label=f'LEDH (RMSE={rmse_ledh:.2f})')
    plot_robust(plt, t_steps, est_ipf, color='g', linestyle=':', label=f'IPF (RMSE={rmse_ipf:.2f})', linewidth=2)
    
    plt.title('Particle Flow Filters on SVSSM (Fixed)')
    plt.ylabel('Log Volatility')
    plt.xlabel('Time Step')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()