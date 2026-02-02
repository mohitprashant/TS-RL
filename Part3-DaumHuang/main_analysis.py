import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import tracemalloc
from flow_base import AuxiliaryEKF
from ledh import LEDHFilter
from edh import EDHFilter
from ledh_inv_pf import PFPF_LEDHFilter
from edh_inv_pf import PFPF_EDHFilter

from lorenz96 import Lorenz96Model

tfd = tfp.distributions



def compute_omat(particles, weights, true_state):
    """
    Computes OMAT metric (Weighted Euclidean Error).
    
    Args:
        particles (tf.Tensor): Particle cloud (N, K).
        weights (tf.Tensor): Weights (N,).
        true_state (tf.Tensor): Ground truth (K,).
        
    Returns:
        float: OMAT value.
    """
    diff = tf.norm(particles - true_state, axis=-1)
    return tf.reduce_sum(diff * weights).numpy()


def run_comparison(particle_num):
    """
    Runs the comparative experiment across different state dimensions (K=40, K=80).
    Tracks RMSE, OMAT, ESS, Execution Time, and Memory Usage.
    """
    Dimensions = [10, 20, 50]
    T = 100
    N = particle_num
    
    for K in Dimensions:
        print(f"\n{'#'*60}")
        print(f" EXPERIMENT: Lorenz 96 with Dimension K={K}, Particle Num={N}")
        print(f"{'#'*60}")
        
        l96 = Lorenz96Model(K=K)

        print("Generating Ground Truth...")
        x_true = tf.fill([K], l96.F)
        for _ in range(200): x_true = l96.rk4_step(x_true)
        
        true_states = []
        observations = []
        for _ in range(T):
            x_true = l96.transition(x_true)
            true_states.append(x_true)
            observations.append(l96.observation(x_true))
        
        true_states = tf.stack(true_states)
        observations = tf.stack(observations)

        filters = {
            'EDH': EDHFilter(l96, N),
            'LEDH': LEDHFilter(l96, N),
            'PFPF_EDH': PFPF_EDHFilter(l96, N),
            'PFPF_LEDH': PFPF_LEDHFilter(l96, N)
        }
        
        results = {}
        for name, flt in filters.items():
            print(f"Running {name}...")

            particles = tfd.MultivariateNormalDiag(loc=tf.fill([K], l96.F), scale_diag=tf.ones(K)).sample(N)
            weights = tf.fill([N], 1.0/N)
            ekf = AuxiliaryEKF(l96)
            m_ekf = tf.fill([K], l96.F)
            P_ekf = tf.eye(K)
            
            mse_sum = 0.0
            omat_sum = 0.0
            ess_list = []
            cond_list = []
            
            tracemalloc.start()
            start_time = time.time()

            for t in range(T):
                if 'PFPF' in name:
                    particles, weights, m_ekf, P_ekf, est, ess, cond = flt.run_step(
                        particles, weights, m_ekf, P_ekf, observations[t])
                else:
                    particles, m_ekf, P_ekf, est, ess, cond = flt.run_step(
                        particles, m_ekf, P_ekf, observations[t])
                    weights = tf.fill([N], 1.0/N)
                    
                err = true_states[t] - est
                mse_sum += tf.reduce_mean(err**2).numpy()
                omat_sum += compute_omat(particles, weights, true_states[t])
                ess_list.append(ess)
                cond_list.append(cond)
                
            exec_time = time.time() - start_time
            curr_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            results[name] = {
                'RMSE': np.sqrt(mse_sum / T),
                'OMAT': omat_sum / T,
                'Min ESS' : np.min(ess_list),
                'Avg ESS': np.mean(ess_list),
                'Time (s)': exec_time,
                'Mem (MB)': peak_mem / 1024 / 1024,
                'Max Cond' : np.max(cond_list),
                'Avg Cond': np.mean(cond_list)
            }

        print(f"\nResults for K={K}, N={N}:")
        print("="*95)
        print(f"{'Method':<15} | {'RMSE':<10} | {'OMAT':<10} | {'Avg ESS':<10} | {'Min ESS':<10} |{'Time (s)':<10} | {'Mem (MB)':<10} | {'Avg Cond':<10} | {'Max Cond':<10}")
        print("-" * 95)
        for k, v in results.items():
            print(f"{k:<15} | {v['RMSE']:<10.4f} | {v['OMAT']:<10.4f} | {v['Avg ESS']:<10.1f} | {v['Min ESS']:<10.1f} | {v['Time (s)']:<10.4f} | {v['Mem (MB)']:<10.2f} | {v['Avg Cond']:<10.2f} | {v['Max Cond']:<10.2f}")
        print("="*95)


if __name__ == "__main__":
    run_comparison(10)
    run_comparison(20)
    run_comparison(50)
    run_comparison(100)