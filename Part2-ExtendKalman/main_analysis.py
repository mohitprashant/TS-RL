import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
import time
import tracemalloc
from svssm import StochasticVolatilityModel
from ukf import UnscentedKalmanFilter
from ekf import ExtendedKalmanFilter
from particle_filter import ParticleFilter

tfd = tfp.distributions


def generate_simulation_data(alpha, sigma, beta, T):
    """
    Generates synthetic ground truth states and observations.

    Args:
        alpha (float): Autoregression coefficient.
        sigma (float): State noise std dev.
        beta (float): Observation scale.
        T (int): Number of time steps.

    Returns:
        tuple: (true_states, observations)
    """
    print(f"Generating Synthetic Data (T={T})...")
    sv = StochasticVolatilityModel(alpha, sigma, beta)
    return sv.simulate(T)


def run_kalman_baselines(alpha, sigma, beta, observations, true_states):
    """
    Runs the deterministic/linearized baselines (EKF and UKF).

    Returns:
        list: A list of result dictionaries for EKF and UKF.
    """
    print("Running EKF & UKF Baselines...")
    results = []
    
    ekf = ExtendedKalmanFilter(alpha, sigma, beta)
    results.append(ekf.run_filter(observations, true_states))

    ukf = UnscentedKalmanFilter(alpha, sigma, beta)
    results.append(ukf.run_filter(observations, true_states))
    return results


def run_particle_filter_experiments(alpha, sigma, beta, obs, true_x, 
                                    particle_counts, thresholds):
    """
    Runs a grid search over Particle Filter hyperparameters.

    Args:
        particle_counts (list): List of integers for N.
        thresholds (list): List of floats for resampling threshold ratios.

    Returns:
        list: A list of result dictionaries for all PF variations.
    """
    print("Running Particle Filter Parameter Sweep...")
    results = []
    
    for N in particle_counts:
        for th in thresholds:
            print(f"  -> Executing PF: N={N}, Threshold Ratio={th}")
            pf = ParticleFilter(alpha, sigma, beta, num_particles=N, resample_threshold_ratio=th)
            res = pf.run_filter(obs, true_x)
            results.append(res)
            
    return results


def display_performance_table(results):
    """
    Prints a formatted ASCII table of the performance metrics.
    """
    print("\n" + "="*85)
    print(f"{'Performance Comparison Summary':^85}")
    print("="*85)
    print(f"{'Method':<30} | {'RMSE':<10} | {'Time(s)':<10} | {'Mem(KB)':<10} | {'Avg ESS':<10}")
    print("-" * 85)
    
    for r in results:
        ess_str = f"{r['ess_avg']:.1f}" if 'ess_avg' in r else "---"
        print(f"{r['label']:<30} | {r['rmse']:<10.4f} | {r['time']:<10.4f} | {r['mem']/1024:<10.1f} | {ess_str:<10}")
    print("="*85)


def plot_comparative_analysis(results, true_x, T):
    """
    Generates the comparative plot.
    """
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    for res in results:
        is_pf = 'PF' in res['label']
        marker = 'o' if is_pf else 'D'
        size = res['particles']/10 if is_pf else 100

        if 'EKF' in res['label']: color = 'red'
        elif 'UKF' in res['label']: color = 'orange'
        else: color = 'blue'
        ax1.scatter(res['time'], res['rmse'], s=size, c=color, alpha=0.6, edgecolors='k')
    ax1.set_xlabel('Runtime (s)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Trade-off: Accuracy vs Computational Cost')
    ax1.grid(True, alpha=0.3)
    

    ax2 = fig.add_subplot(2, 2, 2)
    t_steps = np.arange(T)
    ax2.plot(t_steps, true_x, 'k-', lw=1, alpha=0.5, label='True State')
    ukf_res = next(r for r in results if 'UKF' in r['label'])
    ekf_res = next(r for r in results if 'EKF' in r['label'])
    pf_results = [r for r in results if 'PF' in r['label']]             # Take best PF results
    best_pf = min(pf_results, key=lambda x: x['rmse'])

    ax2.plot(t_steps, best_pf['estimates'], 'blue', lw=1, label=f"Best PF (RMSE={best_pf['rmse']:.2f})")
    ax2.plot(t_steps, ukf_res['estimates'], 'orange', lw=1, label=f"UKF (RMSE={ukf_res['rmse']:.2f})")
    ax2.plot(t_steps, ekf_res['estimates'], 'red', lw=1, label=f"EKF (RMSE={ekf_res['rmse']:.2f})")
    ax2.set_title(f"Trajectory Tracking: UKF vs EKF vs Best PF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    

    ax3 = fig.add_subplot(2, 2, 3)
    thresholds = sorted(list(set(r['threshold_ratio'] for r in pf_results)))
    for th in thresholds:
        subset = [r for r in pf_results if r['threshold_ratio'] == th]
        subset.sort(key=lambda x: x['particles'])
        x_vals = [r['particles'] for r in subset]
        y_vals = [r['rmse'] for r in subset]
        ax3.plot(x_vals, y_vals, marker='o', label=f'Resample Threshold={th}')
    ax3.set_xlabel('Number of Particles (N)')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Convergence: Error vs Particle Count')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()




##################################################################################################

def main():
    """
    Orchestrates the full analysis pipeline.
    """
    alpha, sigma, beta = 0.91, 1.0, 0.5
    T = 50

    true_x, observations = generate_simulation_data(alpha, sigma, beta, T)
    baseline_results = run_kalman_baselines(alpha, sigma, beta, observations, true_x)
    pf_results = run_particle_filter_experiments(
        alpha, sigma, beta, observations, true_x,
        particle_counts=[50, 200, 1000],
        thresholds=[0.2, 0.5, 0.9]
    )
    
    all_results = baseline_results + pf_results
    display_performance_table(all_results)
    plot_comparative_analysis(all_results, true_x, T)

if __name__ == "__main__":
    main()