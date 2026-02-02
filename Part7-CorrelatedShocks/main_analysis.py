import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

from msvssm import MultivariateStochasticVolatilityModel
from improved_sinkhorn import SinkhornParticleFilter, ImprovedSinkhornParticleFilter


tf.random.set_seed(42)
np.random.seed(42)

DTYPE = tf.float32
tfd = tfp.distributions


def build_high_dim_covariance(p, rho_obs, rho_state, rho_leverage):
    """
    Creates a block correlation matrix capturing:
    1. Correlations within observation noise (rho_obs).
    2. Correlations within state transition noise (rho_state).
    3. Cross-correlations between observation and state noise (rho_leverage).

    Args:
        p (int): Dimensionality of the time series.
        rho_obs (float): Correlation coefficient between observation dimensions.
        rho_state (float): Correlation coefficient between state dimensions.
        rho_leverage (float): "Leverage effect" correlation between state x_t 
            and observation y_t noise terms.

    Returns:
        np.ndarray: A symmetric, positive-definite covariance matrix of shape (2p, 2p).
    """
    std_eps = np.random.uniform(0.5, 0.8, p)                    # Random standard deviations
    std_eta = np.random.uniform(0.15, 0.25, p)
    joint_std = np.concatenate([std_eps, std_eta])
    
    corr = np.eye(2 * p)                                            # Initialize Correlation Matrix
    corr[0:p, 0:p] = rho_obs + (1-rho_obs)*np.eye(p)                   # Observation noise correlations
    corr[p:2*p, p:2*p] = rho_state + (1-rho_state)*np.eye(p)          # Block 2: State noise correlations
    
    for i in range(p):                                                      # Off-Diagonal Blocks: Leverage effects
        for j in range(p):
            corr[i, p+j] = rho_leverage
            corr[p+j, i] = rho_leverage
            
    D = np.diag(joint_std)
    Sigma = D @ corr @ D                                    
    return (Sigma + Sigma.T) / 2                                   # Symmetrize



def run_experiment():
    p = 20
    T = 100
    N_particles = 100
    
    phi_val = np.linspace(0.90, 0.99, p).astype(np.float32)
    beta_val = np.full(p, 0.5).astype(np.float32)
    sigma_matrix = build_high_dim_covariance(p, 0.6, 0.4, -0.2)
    gt_model = MultivariateStochasticVolatilityModel(phi_val, sigma_matrix, beta_val)
    true_x, obs = gt_model.simulate(T)

    phi_var = tf.Variable(phi_val, dtype=DTYPE)
    beta_var = tf.Variable(beta_val, dtype=DTYPE)

    print(f"Running Original Sinkhorn (p={p})...")
    tracemalloc.start()
    start_orig = time.time()
    
    with tf.GradientTape() as tape:
        model_orig = MultivariateStochasticVolatilityModel(phi_var, sigma_matrix, beta_var)
        filter_orig = SinkhornParticleFilter(num_particles=N_particles, epsilon=0.1)
        est_orig = filter_orig.run(model_orig, obs)
        rmse_orig = tf.sqrt(tf.reduce_mean(tf.square(true_x - est_orig)))
        
    grads_orig = tape.gradient(rmse_orig, [phi_var, beta_var])
    grad_norm_orig = tf.linalg.global_norm(grads_orig)
    
    time_orig = time.time() - start_orig
    _, mem_orig = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Running Improved Sinkhorn (p={p})...")
    tracemalloc.start()
    start_imp = time.time()
    
    with tf.GradientTape() as tape:
        model_imp = MultivariateStochasticVolatilityModel(phi_var, sigma_matrix, beta_var)
        filter_imp = ImprovedSinkhornParticleFilter(model_imp.precision_eta, num_particles=N_particles, epsilon=0.1)
        est_imp = filter_imp.run(model_imp, obs)
        rmse_imp = tf.sqrt(tf.reduce_mean(tf.square(true_x - est_imp)))
        
    grads_imp = tape.gradient(rmse_imp, [phi_var, beta_var])
    grad_norm_imp = tf.linalg.global_norm(grads_imp)
    
    time_imp = time.time() - start_imp
    _, mem_imp = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("\n" + "="*95)
    print(f"High-Dimensional Sinkhorn Results (p={p}, N={N_particles})")
    print("="*95)
    print(f"{'Method':<22} | {'RMSE':<12} | {'GradNorm':<12} | {'Time(s)':<10} | {'Mem(MB)':<10}")
    print("-" * 95)
    print(f"{'Original':<22} | {rmse_orig.numpy():<12.4f} | {grad_norm_orig.numpy():<12.4f} | {time_orig:<10.4f} | {mem_orig/1024**2:<10.2f}")
    print(f"{'Improved (Geometry)':<22} | {rmse_imp.numpy():<12.4f} | {grad_norm_imp.numpy():<12.4f} | {time_imp:<10.4f} | {mem_imp/1024**2:<10.2f}")
    print("-" * 95)
    print(f"Improvement: {(1 - rmse_imp/rmse_orig)*100:.2f}% (RMSE)")
    print("="*95)

    plot_dims = 3                        # Only plot 3 dim
    time_steps = np.arange(T)
    true_np = true_x.numpy()
    orig_np = est_orig.numpy()
    imp_np = est_imp.numpy()
    
    fig, axes = plt.subplots(plot_dims, 1, figsize=(12, 12), sharex=True)
    
    for i in range(plot_dims):
        ax = axes[i]
        rmse_d_orig = np.sqrt(np.mean((true_np[:,i]-orig_np[:,i])**2))
        rmse_d_imp = np.sqrt(np.mean((true_np[:,i]-imp_np[:,i])**2))
        
        ax.plot(time_steps, true_np[:, i], 'k-', linewidth=2, alpha=0.5, label='True State')
        ax.plot(time_steps, orig_np[:, i], 'r--', label=f'Original (RMSE={rmse_d_orig:.2f})')
        ax.plot(time_steps, imp_np[:, i], 'b--', label=f'Improved (RMSE={rmse_d_imp:.2f})')
        
        ax.set_ylabel(f'Dim {i}\nLog-Vol')
        if i == 0: ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, alpha=0.3)
        
    plt.xlabel('Time Step')
    plt.suptitle(f"Filter Performance (First 3 of {p} Dimensions)\nMetric: Specific Relative Entropy")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

if __name__ == "__main__":
    run_experiment()