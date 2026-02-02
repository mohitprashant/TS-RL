import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from stochastic_pf import StochasticParticleFlow

DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')



class ParticleFlowSimulation:
    """
    Handles the execution of Monte Carlo simulations using a StochasticParticleFlow model.
    """
    def __init__(self, model):
        """
        Initializes the simulation manager.

        Args:
            model (StochasticParticleFlow): An instance of the model class containing
                                            problem constants, Jacobians, and the optimized schedule.
        """
        self.model = model

    def run_single_simulation(self, method, num_particles, dt):
        """
        Runs a single particle flow simulation.
        
        Args:
            method (str): 'linear' or 'optimal' schedule.
            num_particles (int): Number of particles.
            dt (float): Time step.
            
        Returns:
            tuple: (final_mse, final_trace_P)
        """
        steps = int(1.0 / dt)
        z_std = tf.random.normal((num_particles, 2), mean=0.0, stddev=1.0, dtype=DTYPE)                  # Init particles
        particles = self.model.prior_mu + tf.matmul(z_std, tf.transpose(self.model.L_prior))
        
        dw_seq = tf.random.normal((steps, num_particles, 2), dtype=DTYPE) * tf.sqrt(tf.constant(dt, dtype=DTYPE))
        curr = particles
        
        for k in range(steps):
            t = k * dt

            if(method == 'linear'):
                beta = tf.constant(t, dtype=DTYPE)
                u = tf.constant(1.0, dtype=DTYPE)
            else:
                if(self.model.beta_grid is None):
                    raise ValueError("Model schedule not solved. Call solve_optimal_schedule() first.")
                    
                beta = tfp.math.interp_regular_1d_grid(x=tf.constant(t, dtype=DTYPE), 
                    x_ref_min=0.0, x_ref_max=1.0, y_ref=self.model.beta_grid)
                u = tfp.math.interp_regular_1d_grid(x=tf.constant(t, dtype=DTYPE), 
                    x_ref_min=0.0, x_ref_max=1.0, y_ref=self.model.u_grid)
            
            # Flow Update
            mu_curr = tf.reduce_mean(curr, axis=0)
            H_local = self.model.get_H_jacobian(mu_curr)
            Ah_local = -tf.matmul(tf.transpose(H_local), tf.matmul(self.model.inv_R, H_local))
            
            M = self.model.M0_mat + beta * (-Ah_local)
            inv_M = tf.linalg.inv(M)
            Ah_paper = Ah_local

            K2 = u * inv_M                                                        # K2 = u * M^-1
            term_k1 = tf.matmul(tf.matmul(inv_M, Ah_paper), inv_M)
            K1_raw = 0.5 * self.model.Q_diff + 0.5 * u * term_k1
            
            k1_diag = tf.linalg.diag_part(K1_raw)                                 # Non-negative diagonal
            k1_diag_safe = tf.maximum(k1_diag, 1e-8)
            K1 = tf.linalg.set_diag(K1_raw, k1_diag_safe)

            diff = curr - self.model.prior_mu                                     # Gradients
            grad_log_p0 = -tf.matmul(diff, self.model.inv_P_prior) 
            
            h_val = self.model.h_meas(curr)
            resid = self.model.z_meas - h_val
            
            Ht_invR = tf.matmul(tf.transpose(H_local), self.model.inv_R)         # Get drift and diffusion
            grad_log_h = tf.matmul(resid, tf.transpose(Ht_invR))
            grad_log_p = (1.0 - beta) * grad_log_p0 + beta * grad_log_h
            drift = tf.matmul(grad_log_p, tf.transpose(K1)) + tf.matmul(grad_log_h, tf.transpose(K2))
            diffusion = dw_seq[k] * self.model.sqrt_Q
            curr = curr + drift * dt + diffusion

        mean_est = tf.reduce_mean(curr, axis=0)                                  # Get final measures
        cov_est = tfp.stats.covariance(curr)
        final_mse = tf.reduce_sum(tf.square(mean_est - self.model.target_true))
        final_trace = tf.linalg.trace(cov_est)
        
        return final_mse, final_trace


    def run_monte_carlo_batch(self, method='optimal', num_runs=20, num_particles=50, dt=0.01):
        """
        Executes a batch of Monte Carlo simulations to evaluate filter consistency.

        Args:
            method (str): The scheduling method ('linear' or 'optimal').
            num_runs (int): The number of independent Monte Carlo trials.
            num_particles (int): Number of particles per trial.
            dt (float): Integration step size.

        Returns:
            tuple: A tuple (mse_tensor, trace_tensor) containing results for all runs.
        """
        print(f"Running Monte Carlo Batch: {method}...")
        mse_list = []
        tr_P_list = []
        # tf.random.set_seed(42)
        
        for _ in range(num_runs):
            mse, trace = self.run_single_simulation(method, num_particles, dt)
            mse_list.append(mse)
            tr_P_list.append(trace)
            
        return tf.stack(mse_list), tf.stack(tr_P_list)


##############################################################################################################################


def plot_results(model, mse_lin, mse_opt, tr_lin, tr_opt):
    """
    Visualization function.
    """
    lam = model.lambda_grid.numpy()
    beta = model.beta_grid.numpy()
    u = model.u_grid.numpy()
    
    r_stiff_opt = []
    r_stiff_lin = []
    M0 = model.M0_mat.numpy()
    Mh = model.Mh_nominal.numpy()
    Q = model.Q_diff.numpy()
    
    for i, t in enumerate(lam):
        M = M0 + beta[i] * Mh
        inv_M = np.linalg.inv(M)
        F_opt = -0.5 * Q @ M + 0.5 * u[i] * inv_M @ Mh
        eigs = np.linalg.eigvals(F_opt)
        r_stiff_opt.append(np.max(np.abs(np.real(eigs))) / np.min(np.abs(np.real(eigs))))
        
        M_l = M0 + t * Mh
        inv_M_l = np.linalg.inv(M_l)
        F_l = -0.5 * Q @ M_l + 0.5 * 1.0 * inv_M_l @ Mh
        eigs_l = np.linalg.eigvals(F_l)
        r_stiff_lin.append(np.max(np.abs(np.real(eigs_l))) / np.min(np.abs(np.real(eigs_l))))

    fig2, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Replication of Dai (2021) Figure 2', fontsize=14)

    axs[0, 0].plot(lam, beta, 'r-', label='Optimal')
    axs[0, 0].plot(lam, lam, 'b--', label='Linear')
    axs[0, 0].set_title(r'(a) $\beta(\lambda)$'); axs[0, 0].legend(); axs[0, 0].grid(True)

    axs[0, 1].plot(lam, beta - lam, 'k-')
    axs[0, 1].set_title(r'(b) Error'); axs[0, 1].grid(True)

    axs[1, 0].plot(lam, u, 'b-')
    axs[1, 0].set_title(r'(c) $u^*(\lambda)$'); axs[1, 0].grid(True)

    axs[1, 1].semilogy(lam, r_stiff_opt, 'r-', label='Optimal')
    axs[1, 1].semilogy(lam, r_stiff_lin, 'b--', label='Linear')
    axs[1, 1].set_title(r'(d) $R_{stiff}$'); axs[1, 1].legend(); axs[1, 1].grid(True)
    plt.tight_layout(); plt.show()

    fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle('Performance Comparison', fontsize=14)
    axs3[0].boxplot([mse_lin.numpy(), mse_opt.numpy()], labels=['Linear', 'Optimal'])
    axs3[0].set_title('Mean Squared Error (MSE)'); axs3[0].grid(True)
    axs3[1].boxplot([tr_lin.numpy(), tr_opt.numpy()], labels=['Linear', 'Optimal'])
    axs3[1].set_title('Trace of Covariance (Tr(P))'); axs3[1].grid(True)
    plt.show()


def main():
    spf = StochasticParticleFlow()
    spf.solve_optimal_schedule()
    sim_manager = ParticleFlowSimulation(spf)
    mse_lin, tr_lin = sim_manager.run_monte_carlo_batch('linear')
    mse_opt, tr_opt = sim_manager.run_monte_carlo_batch('optimal')
    
    print("\n" + "="*65)
    print(f"{'Run':<5} {'MSE_lin':<12} {'MSE_opt':<12} {'Tr(P)_lin':<12} {'Tr(P)_opt':<12}")
    print("-" * 65)
    m_lin, t_lin = mse_lin.numpy(), tr_lin.numpy()
    m_opt, t_opt = mse_opt.numpy(), tr_opt.numpy()
    for i in range(len(m_lin)):
        print(f"{i+1:<5} {m_lin[i]:<12.4f} {m_opt[i]:<12.4f} {t_lin[i]:<12.2f} {t_opt[i]:<12.2f}")
    print("-" * 65)
    print(f"{'Avg':<5} {np.mean(m_lin):<12.4f} {np.mean(m_opt):<12.4f} {np.mean(t_lin):<12.2f} {np.mean(t_opt):<12.2f}")
    print("="*65)

    plot_results(spf, mse_lin, mse_opt, tr_lin, tr_opt)



if __name__ == "__main__":
    main()