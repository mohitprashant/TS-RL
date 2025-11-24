# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 20:12:10 2025

@author: 18moh
"""

from svssm import SVSSM
from unsent_kalman import SV_UKF
from extend_kalman import SV_EKF
from particle_filter import SV_ParticleFilter

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc


def run_benchmark(name, filter_obj, observations, true_states):
    print(f"Running {name}...", end=" ")
    
    tracemalloc.start()
    start_time = time.time()
    result = filter_obj.filter(observations)
    
    if(isinstance(result, tuple)):
        estimates, aux_data = result
    else:
        estimates, aux_data = result, None
        
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    runtime = (end_time - start_time) * 1000 
    peak_mem = peak / 1024 
    est_flat = tf.reshape(estimates, [-1])
    true_flat = tf.reshape(true_states, [-1])
    
    
    if(len(est_flat) > len(true_flat)):
        est_flat = est_flat[1:]
        
    min_len = min(len(est_flat), len(true_flat))
    est_flat = est_flat[:min_len]
    true_flat = true_flat[:min_len]
    
    rmse = tf.sqrt(tf.reduce_mean(tf.square(est_flat - true_flat))).numpy()
    
    # Check for NaN
    is_stable = not np.isnan(rmse)
    
    print(f"Done. RMSE: {rmse:.4f}")
    
    return {
        "name": name,
        "estimates": estimates, 
        "aux": aux_data,
        "rmse": rmse,
        "runtime_ms": runtime,
        "peak_mem_kb": peak_mem,
        "stable": is_stable
    }






if __name__ == "__main__":
    # Configuration
    STEPS = 200
    SEED = 123
    ALPHA, SIGMA, BETA = 0.91, 0.5, 0.5
    PARTICLES = 1000
    
    # Generate Data
    sv = SVSSM(alpha=ALPHA, sigma=SIGMA, beta=BETA, seed=SEED)
    y_data, x_data = sv.generate_data(STEPS)
    T_LEN = len(y_data) 
    t_steps = np.arange(T_LEN)
    
    
    # Instantiate Filters
    ekf = SV_EKF(ALPHA, SIGMA, BETA, 0.0, 1.0)
    ukf = SV_UKF(ALPHA, SIGMA, BETA, 0.0, 1.0)
    pf = SV_ParticleFilter(ALPHA, SIGMA, BETA, PARTICLES, 0.0, 1.0)
    
    
    # Run Benchmarks
    results = []
    results.append(run_benchmark("EKF", ekf, y_data, x_data))
    results.append(run_benchmark("UKF", ukf, y_data, x_data))
    results.append(run_benchmark("PF (N=1000)", pf, y_data, x_data))
    
    
    # Visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    
    
    # Plot 1: State Trajectories
    ax1 = fig.add_subplot(gs[0])
    
    
    # Plot Truth
    ax1.plot(t_steps, x_data.numpy(), 'k-', lw=2, alpha=0.6, label='True State')
    
    colors = {'EKF': 'red', 'UKF': 'blue', 'PF (N=1000)': 'green'}
    styles = {'EKF': '--', 'UKF': '-.', 'PF (N=1000)': ':'}
    
    for res in results:
        est = tf.reshape(res['estimates'], [-1]).numpy()
        if(len(est) > len(t_steps)):
            est = est[:len(t_steps)]
        current_t = t_steps[:len(est)]
        
        ax1.plot(current_t, est, color=colors[res['name']], ls=styles[res['name']], label=res['name'])
        
    ax1.set_title(f'SVSSM Filtering Comparison (Steps={STEPS})')
    ax1.set_ylabel('Log Volatility (X)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    
    # Plot 2: Absolute Error
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for res in results:
        est = tf.reshape(res['estimates'], [-1]).numpy()
        
        limit = min(len(est), len(x_data))
        est_sliced = est[:limit]
        truth_sliced = x_data.numpy()[:limit]
        current_t = t_steps[:limit]
        
        error = np.abs(est_sliced - truth_sliced)
        ax2.plot(current_t, error, color=colors[res['name']], alpha=0.7, label=res['name'])
    
    ax2.set_ylabel('|Error|')
    ax2.set_title('Absolute Estimation Error')
    ax2.grid(True, alpha=0.3)
    
    
    # Plot 3: Particle Degeneracy (N_eff)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    pf_res = results[2]
    if(pf_res['aux'] is not None):
        neff = pf_res['aux'].numpy()
        neff_steps = np.arange(len(neff))
        
        ax3.plot(neff_steps, neff, color='green', label='Effective Sample Size (N_eff)')
        ax3.axhline(PARTICLES/2, color='orange', ls='--', label='Resample Threshold (N/2)')
        ax3.set_ylim(0, PARTICLES * 1.1)
        ax3.set_ylabel('N_eff')
        ax3.set_xlabel('Time Step')
        ax3.set_title('Particle Filter Degeneracy & Resampling')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    # Console Output Table
    print("\n" + "="*80)
    print(f"{'Filter':<15} | {'RMSE':<10} | {'Time (ms)':<10} | {'Mem (KB)':<10} | {'Stable'}")
    print("-" * 80)
    for res in results:
        print(f"{res['name']:<15} | {res['rmse']:.4f}     | {res['runtime_ms']:.2f}      | {res['peak_mem_kb']:.2f}      | {res['stable']}")
    print("="*80)
    
    
    
    
    
    