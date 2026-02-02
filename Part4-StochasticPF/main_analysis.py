import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import tracemalloc

from opt_inv_pf import StochPFPF
from ledh_inv_pf import LEDH_PFPF
from lorenz96 import Lorenz96Model

tfd = tfp.distributions



def compute_omat(particles, weights, true_state):
    """
    Computes the Optimal Mass Transport (OMAT) metric approximation.

    Args:
        particles (tf.Tensor): Particle states. Shape (N, K).
        weights (tf.Tensor): Normalized particle weights. Shape (N,).
        true_state (tf.Tensor): Ground truth state vector. Shape (K,).

    Returns:
        float: The weighted average Euclidean distance.
    """
    diff = tf.norm(particles - true_state, axis=-1)
    return tf.reduce_sum(diff * weights).numpy()


def run_experiment():
    """
    Executes an experiment between LEDH-PFPF and Stoch-PFPF on the Lorenz96 model.
    """
    K = 10                                                                         # Sim configs
    T = 50
    N = 100
    print(f"Running PFPF Comparison: Lorenz 96 (d={K}), N={N}, Steps={T}")
    l96 = Lorenz96Model(K=K)
    # tf.random.set_seed(123)
    x_true = tf.fill([K], l96.F)
    for _ in range(200): x_true = l96.rk4_step(x_true)
    
    true_states_list = []                                                          # Data Gen
    obs_list = []
    for _ in range(T):
        x_true = l96.transition(x_true)
        y = l96.observation(x_true)
        true_states_list.append(x_true)
        obs_list.append(y)
    
    true_states = tf.stack(true_states_list)
    observations = tf.stack(obs_list)
    
    filters = {                                                                     # Filters
        'LEDH_PFPF': LEDH_PFPF(l96, N, num_steps=20),
        'Stoch_PFPF': StochPFPF(l96, N, num_steps=20, mu=0.1)
    }
    
    results = {}                                                                    # sim loop
    for name, flt in filters.items():
        print(f"Running {name}...")
        tf.random.set_seed(42)
        np.random.seed(42)
        
        particles = tfd.MultivariateNormalDiag(loc=tf.fill([K], l96.F), scale_diag=tf.ones(K)*2.0).sample(N)
        weights = tf.fill([N], 1.0/float(N))
        m_ekf = tf.fill([K], l96.F)
        P_ekf = tf.eye(K)
        
        mse_sum = 0.0
        omat_sum = 0.0
        cond_sum = 0.0
        ess_sum = 0.0
        
        tracemalloc.start()
        start_time = time.time()
        
        for t in range(T):
            particles, weights, m_ekf, P_ekf, est, ess, cond = flt.run_step(
                particles, weights, m_ekf, P_ekf, observations[t])
            
            err = true_states[t] - est
            mse_val = tf.reduce_mean(err**2).numpy()
            
            if np.isnan(mse_val) or mse_val > 1e6:
                print(f"  [ERROR] Instability at step {t} for {name}")
                break
                
            mse_sum += mse_val
            omat_sum += compute_omat(particles, weights, true_states[t])
            cond_sum += cond.numpy()
            ess_sum += ess.numpy()
            
            if t % 10 == 0:
                print(f"  Step {t}/{T} - RMSE: {np.sqrt(mse_val):.2f}, Cond: {cond:.1f}, ESS: {ess:.1f}")

        exec_time = time.time() - start_time
        curr_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name] = {
            'RMSE': np.sqrt(mse_sum / T),
            'OMAT': omat_sum / T,
            'Avg Cond': cond_sum / T,
            'Avg ESS': ess_sum / T,
            'Time': exec_time,
            'Mem': peak_mem / 1e6
        }

    print("\n" + "="*100)
    print(f"{'Method':<15} | {'RMSE':<10} | {'OMAT':<10} | {'Avg Cond':<10} | {'Avg ESS':<10} | {'Time (s)':<10} | {'Mem (MB)':<10}")
    print("-" * 100)
    for k, v in results.items():
        print(f"{k:<15} | {v['RMSE']:<10.4f} | {v['OMAT']:<10.4f} | {v['Avg Cond']:<10.2f} | {v['Avg ESS']:<10.1f} | {v['Time']:<10.4f} | {v['Mem']:<10.2f}")
    print("="*100)

if __name__ == "__main__":
    run_experiment()