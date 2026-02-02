import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from lgssm import LinearGaussianSSM
from kalman import KalmanFilter

tfd = tfp.distributions


class MeanFilter:
    """
    Moving average filter.
    """
    def __init__(self, window_size=5):
        """
        Initialize a Mean Filter with the corresponding window size
        """
        self.window_size = window_size

    def run_filter(self, observations):
        """
        Args:
            observations: (T, ny)
        Returns:
            means: (T, ny) - Note: assumes state space is same dimension as observation for comparison,
                           or effectively estimates position from position observations.
        """
        T = observations.shape[0]
        obs_np = observations.numpy()
        estimates = []
        buffer = []
        
        for t in range(T):
            buffer.append(obs_np[t])
            if(len(buffer) > self.window_size):
                buffer.pop(0)
            est = np.mean(buffer, axis=0)
            estimates.append(est)
            
        return np.array(estimates)



#####################################################################################################

def compare_filters():
    dt = 0.1
    T = 50
    A = [[1.0, dt], [0.0, 1.0]]                            # LGSSM Configs
    B = [[0.5 * dt**2, 0.0], [dt, 0.0]]
    C = [[1.0, 0.0]]
    D = [[0.5]]
    Sigma_init = [[5.0, 0.0], [0.0, 5.0]]

    ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)         # Run filter in Joseph form and regular
    true_states, observations = ssm.simulate(T)
    
    kf = KalmanFilter(A, B, C, D, Sigma_init)         
    mu_joseph, P_joseph, cond_S, cond_P_joseph = kf.run_filter(observations, joseph_form=True)
    mu_regular, P_regular, _, cond_P_regular = kf.run_filter(observations, joseph_form=False)
    
    mf = MeanFilter(window_size=10)                        # Run Mean Filter
    mu_mean = mf.run_filter(observations)

    # Calculate Errors (RMSE) for Position (State dimension 0)
    pos_true = true_states.numpy()[:, 0]
    rmse_joseph = np.sqrt(np.mean((pos_true - mu_joseph.numpy()[:, 0])**2))
    rmse_regular = np.sqrt(np.mean((pos_true - mu_regular.numpy()[:, 0])**2))
    rmse_mean = np.sqrt(np.mean((pos_true - mu_mean[:, 0])**2))
    
    print(f"{'Filter Type':<20} | {'RMSE (Position)':<15}")
    print("-" * 40)
    print(f"{'KF (Joseph)':<20} | {rmse_joseph:.4f}")
    print(f"{'KF (Regular)':<20} | {rmse_regular:.4f}")
    print(f"{'Mean Filter':<20} | {rmse_mean:.4f}")
    
    # Difference between covariance matrices (Frobenius norm)
    cov_diff = tf.norm(P_joseph - P_regular, ord='fro', axis=[-2, -1])          
    print("\nMax Covariance Difference (Joseph - Regular): {:.2e}".format(np.max(cov_diff)))

    # Plot Graphs
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax = axes[0]
    ax.plot(pos_true, 'k-', lw=1, label='True Position')
    ax.scatter(np.arange(T), observations.numpy()[:, 0], c='r', s=5, alpha=0.3, label='Observations')
    ax.plot(mu_joseph.numpy()[:, 0], 'b--', lw=1.5, label=f'KF (RMSE={rmse_joseph:.2f})')
    ax.plot(mu_mean[:, 0], 'g:', lw=2, label=f'Mean Filter (RMSE={rmse_mean:.2f})')
    ax.set_ylabel('Position')
    ax.set_title('Tracking Performance: KF vs Mean Filter')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(cov_diff, 'm-')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title('Covariance Discrepancy: || P_joseph - P_regular ||')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(cond_P_joseph, 'b-', label='Cond(P) Joseph')
    ax.plot(cond_P_regular, 'r--', label='Cond(P) Regular')
    ax.set_ylabel('Condition Number')
    ax.set_xlabel('Time Step')
    ax.set_title('Condition Number of State Covariance Matrix')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_filters()