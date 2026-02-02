import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc

from svssm import StochasticVolatilityModel
from cnf import CNFParticleFilter
from sinkhorn import SinkhornParticleFilter
from opr import OptimalPlacementParticleFilter
from soft_resample import SoftResamplingParticleFilter

# DTYPE = tf.float64 
DTYPE = tf.float32
tfd = tfp.distributions


def run_filter(filter_obj, observations, true_states):
    """
    Executes the filtering loop for a specific particle filter object and collects metrics.

    Args:
        filter_obj: An instance of a Particle Filter (e.g., SinkhornParticleFilter). 
            Must implement .step() and .initial_dist().
        observations (tf.Tensor): The sequence of observed data points (Y_1:T).
        true_states (tf.Tensor): The ground truth latent states (X_1:T) for RMSE calculation.

    Returns:
        dict: A dictionary containing performance metrics:
            - 'rmse' (tf.Tensor): Root Mean Square Error of the state estimate.
            - 'time' (float): Total execution time in seconds.
            - 'mem' (float): Peak memory usage during the filtering loop in MB.
            - 'ess' (tf.Tensor): Average Effective Sample Size across all timesteps.
            - 'cond' (tf.Tensor): Average condition number of the transport/covariance 
               matrix (if applicable).
            - 'estimates' (np.ndarray): The sequence of weighted mean state estimates.
    """
    T = observations.shape[0]
    N = filter_obj.num_particles
    true_states = tf.convert_to_tensor(true_states, dtype=DTYPE)
    
    particles = filter_obj.initial_dist().sample(N)
    log_weights = tf.fill([N], -tf.math.log(float(N)))
    
    estimates = []
    ess_history = []
    cond_history = []
    
    tracemalloc.start()
    start_time = time.time()
    
    for t in range(T):
        particles, log_weights, P = filter_obj.step(particles, log_weights, observations[t])            # Step
             
        step_lik = tf.reduce_logsumexp(log_weights)                                                     # Metrics
        w_norm = tf.exp(log_weights - step_lik)
        ess = 1.0 / tf.reduce_sum(w_norm**2)
        ess_history.append(ess)
        
        if(P is not None):
            try:
                cond = tf.linalg.cond(P + tf.eye(N)*1e-5)
            except:
                cond = tf.constant(float('nan'))
            cond_history.append(cond)
            
        w_curr = tf.exp(log_weights - tf.reduce_logsumexp(log_weights))                                 # Estimate (Weighted Mean)
        est = tf.reduce_sum(particles * w_curr)
        estimates.append(est)
        
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - start_time
    estimates_tensor = tf.stack(estimates)
    mse = tf.reduce_mean(tf.square(true_states - estimates_tensor))
    rmse = tf.sqrt(mse)
    avg_cond = tf.reduce_mean(cond_history) if cond_history else float('nan')
    
    return {
        'rmse': rmse,
        'time': total_time,
        'mem': peak_mem / 1024**2, # Convert to MB
        'ess': tf.reduce_mean(ess_history),
        'cond': avg_cond,
        'estimates': estimates_tensor.numpy()
    }


def run_experiment_and_plot():
    """
    Sets up the SVSSM experiment, runs multiple filter variants, and plots results.
    """
    alpha_val = 0.91
    sigma_val = 1.0
    beta_val = 0.5
    T = 50
    N = 100
    model = StochasticVolatilityModel(alpha_val, sigma_val, beta_val)
    true_x, obs = model.simulate(T)
    
    methods = [
        ('Sinkhorn', SinkhornParticleFilter, {'epsilon': 0.5}),
        ('Soft', SoftResamplingParticleFilter, {'soft_alpha': 0.5}),
        ('OPR', OptimalPlacementParticleFilter, {}),
        ('CNF-PF', CNFParticleFilter, {}) 
    ]
    
    results = []
    all_estimates = {}
    print(f"{'Method':<10} | {'RMSE':<8} | {'GradNorm':<10} | {'Time(s)':<8} | {'Mem(MB)':<8} | {'ESS':<8}")
    print("-" * 80)
    
    for name, cls, kwargs in methods:
        a_var = tf.Variable(alpha_val, dtype=DTYPE)
        s_var = tf.Variable(sigma_val, dtype=DTYPE)
        b_var = tf.Variable(beta_val, dtype=DTYPE)
        pf = cls(a_var, s_var, b_var, num_particles=N, **kwargs)

        with tf.GradientTape() as tape:
            out = run_filter(pf, obs, true_x)
            loss = out['rmse']
        grads = tape.gradient(loss, [a_var, s_var, b_var])
        
        if all(g is not None for g in grads):
            grad_norm = tf.linalg.global_norm(grads).numpy()
        else:
            grad_norm = 0.0
            
        print(f"{name:<10} | {out['rmse']:<8.4f} | {grad_norm:<10.4f} | {out['time']:<8.4f} | {out['mem']:<8.4f} | {out['ess']:<8.1f}")
        results.append({'method': name, 'rmse': out['rmse'],'mem': out['mem']})
        all_estimates[name] = out['estimates']

    plt.figure(figsize=(12, 6))
    time_steps = np.arange(T)
    plt.plot(time_steps, true_x, 'k-', linewidth=2, label='True State', alpha=0.8)

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for i, (name, est) in enumerate(all_estimates.items()):
        plt.plot(time_steps, est, linestyle='--', marker='o', markersize=4, 
                 label=f'{name} Est', color=colors[i], alpha=0.7)
        
    plt.title("State Estimation Comparison: True vs Filter Estimates")
    plt.xlabel("Time Step")
    plt.ylabel("Latent State X")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment_and_plot()