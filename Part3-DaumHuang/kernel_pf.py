import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from lorenz96 import Lorenz96Model

tfd = tfp.distributions



class ParticleFlowKernel:
    """
    Computes the kernelized drift term for the Particle Flow Filter.
    """
    def __init__(self, alpha, kernel_type):
        """
        Initializes the kernel parameters.

        Args:
            alpha (float): Scaling factor for the kernel bandwidth. Controls the 
                           interaction radius between particles.
            kernel_type (str): The type of kernel to use. Options are:
                - 'matrix': Uses an element-wise bandwidth scaled by the local variance (B_diag).
                            Effective for high dimensions as it prevents flow stagnation.
                - 'scalar': Uses a single scalar bandwidth derived from the median pairwise 
                            distance heuristic.
        """
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        self.type = kernel_type

    @tf.function
    def compute_drift(self, particles, grad_log_p, B_diag):
        """
        Computes the drift vector f_s(x) using the kernel trick.

        The drift is approximated as:
        f_s(x) = (1/N) * sum_j [ k(x, x_j) * grad_log_p(x_j) + div_x(k(x, x_j)) ]

        Args:
            particles (tf.Tensor): Current particle states. Shape (dim, N).
            grad_log_p (tf.Tensor): Gradient of the log-homotopy (log-likelihood + log-prior).
                                    Shape (dim, N).
            B_diag (tf.Tensor): Diagonal of the particle covariance matrix. Used for 
                                bandwidth scaling. Shape (dim,).

        Returns:
            tf.Tensor: The computed drift vector for each particle. Shape (dim, N).
        """
        N = tf.cast(tf.shape(particles)[1], tf.float32)
        dim = tf.shape(particles)[0]
        
        diff = tf.expand_dims(particles, 2) - tf.expand_dims(particles, 1)
        diff_sq = tf.square(diff)

        if self.type == 'matrix':
            scale = 2.0 * self.alpha * (B_diag + 1e-5) 
            scale = tf.reshape(scale, [dim, 1, 1])
            K_vals = tf.exp(-diff_sq / scale) 
            
            term1 = tf.reduce_sum(K_vals * tf.expand_dims(grad_log_p, 1), axis=2)
            div_K = K_vals * (diff / (scale * 0.5))
            term2 = tf.reduce_sum(div_K, axis=2)
            return (term1 + term2) / N
        else:
            # avg_var = tf.reduce_mean(B_diag)
            # scale_scalar = 2.0 * self.alpha * avg_var * tf.cast(dim, tf.float32) 
            # dist_sq_scalar = tf.reduce_sum(diff_sq, axis=0) 
            # K_scalar = tf.exp(-dist_sq_scalar / scale_scalar)

            dists = tf.reduce_sum(diff_sq, axis=0) 
            mask = tf.eye(N) * 1e30
            dists_masked = dists + mask
            median_dist = tfp.stats.percentile(dists_masked, 50.0)
            scale_scalar = median_dist / (2.0 * np.log(2.0))
            
            K_scalar = tf.exp(-dists / scale_scalar)
            
            grad_exp = tf.expand_dims(grad_log_p, 1) 
            K_broadcast = tf.expand_dims(K_scalar, 0) 
            term1 = tf.reduce_sum(K_broadcast * grad_exp, axis=2)
            
            factor = diff / (scale_scalar * 0.5) 
            div_K = K_broadcast * factor
            term2 = tf.reduce_sum(div_K, axis=2)
            return (term1 + term2) / N
        


class PFF:
    """
    Particle Flow Filter (PFF) implementation.
    """
    def __init__(self, n_particles, dim, kernel_type, model):
        """
        Initializes PFF.

        Args:
            n_particles (int): Number of particles (ensemble members).
            dim (int): Dimensionality of the state space (K).
            kernel_type (str): 'scalar' or 'matrix' kernel for the flow approximation.
            model (Lorenz96Model): The dynamic system model.
        """
        self.kernel = ParticleFlowKernel(alpha=0.05, kernel_type=kernel_type)
        self.dim = dim
        self.Np = n_particles
        self.model = model


    @tf.function
    def update(self, particles, y, R_val):
        """
        Performs the measurement update step using Particle Flow.

        Integrates the particles from pseudo-time s=0 (Prior) to s=1 (Posterior)
        using an Euler integration scheme.

        Args:
            particles (tf.Tensor): Prior particle states. Shape (dim, N).
            y (tf.Tensor): Current observation vector. Shape (dim,).
            R_val (tf.Tensor): Observation noise variance (scalar).

        Returns:
            tf.Tensor: Posterior particle states. Shape (dim, N).
        """
        x = particles
        dt = tf.constant(0.05, dtype=tf.float32)
        n_steps = 20
        
        cov_diag = tfp.stats.variance(x, sample_axis=1) + 1e-3
        D = tf.expand_dims(cov_diag, 1)

        for _ in range(n_steps):
            x = self._flow_step(x, y, R_val, cov_diag, D, dt)
        return x


    def _flow_step(self, x, y, R_val, B_diag, D, dt):
        """
        Executes a single Euler integration step in pseudo-time.

        Args:
            x (tf.Tensor): Current particle positions in flow.
            y (tf.Tensor): Observation.
            R_val (tf.Tensor): Observation noise variance.
            B_diag (tf.Tensor): Diagonal of particle covariance (for kernel width).
            D (tf.Tensor): Diffusion matrix diagonal (precomputed).
            dt (float): Integration step size.

        Returns:
            tf.Tensor: Updated particle positions.
        """
        innov = tf.expand_dims(y, 1) - x
        grad_lik = innov / R_val
        
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        grad_pri = -(x - mean) / tf.expand_dims(B_diag, 1)
        
        grad_log_p = grad_lik + grad_pri
        drift = self.kernel.compute_drift(x, grad_log_p, B_diag)
        
        return x + dt * (D * drift)


################################################################################################################################


def main():
    K, T, N_particles = 40, 60, 30
    obs_std = 1.0
    R_val = tf.constant(obs_std**2, dtype=tf.float32)

    l96 = Lorenz96Model(K=K, obs_std=obs_std)
    states_true, obs_all = l96.simulate(T, burn_in=200)
    particles_init = l96.initial_dist().sample(N_particles)
    particles_s = tf.transpose(particles_init)
    particles_m = tf.transpose(particles_init)

    pff_scalar = PFF(N_particles, K, 'scalar', l96)
    pff_matrix = PFF(N_particles, K, 'matrix', l96)

    @tf.function
    def predict_step(p):
        return tf.transpose(tf.vectorized_map(l96.rk4_step, tf.transpose(p)))
    
    history = {
        's_rmse': [], 'm_rmse': [], 
        's_spread': [], 'm_spread': [],
        's_omat': [], 'm_omat': []
    }

    print(f"Running Optimized Filter (K={K})...")
    for t in range(T):
        particles_s = pff_scalar.update(particles_s, obs_all[t], R_val)
        particles_m = pff_matrix.update(particles_m, obs_all[t], R_val)

        def get_stats(p, true):
            err = tf.reduce_mean(p, axis=1) - true
            rmse = tf.sqrt(tf.reduce_mean(tf.square(err)))
            dist = tf.norm(p - tf.expand_dims(true, 1), axis=0)
            omat = tf.reduce_mean(dist)
            spread = tf.sqrt(tf.reduce_mean(tfp.stats.variance(p, sample_axis=1)))
            return rmse.numpy(), spread.numpy(), omat.numpy()

        rs, ss, os = get_stats(particles_s, states_true[t])
        rm, sm, om = get_stats(particles_m, states_true[t])
        
        history['s_rmse'].append(rs); history['s_spread'].append(ss); history['s_omat'].append(os)
        history['m_rmse'].append(rm); history['m_spread'].append(sm); history['m_omat'].append(om)

        if(t < T - 1):
            particles_s = predict_step(particles_s)
            particles_m = predict_step(particles_m)

    print(f"\n{'='*60}")
    print(f"{'METRIC SUMMARY (Avg over ' + str(T) + ' steps)':^60}")
    print(f"{'='*60}")
    print(f"{'Metric':<15} | {'Scalar Kernel':<20} | {'Matrix Kernel':<20}")
    print(f"{'-'*60}")
    print(f"{'RMSE':<15} | {np.mean(history['s_rmse']):<20.4f} | {np.mean(history['m_rmse']):<20.4f}")
    print(f"{'OMAT':<15} | {np.mean(history['s_omat']):<20.4f} | {np.mean(history['m_omat']):<20.4f}")
    print(f"{'Spread':<15} | {np.mean(history['s_spread']):<20.4f} | {np.mean(history['m_spread']):<20.4f}")
    print(f"{'='*60}\n")

    plot_full_analysis(history, states_true, particles_s, particles_m)

def plot_full_analysis(h, states_true, p_s, p_m):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].plot(h['s_rmse'], 'r-', alpha=0.5, label='Scalar RMSE')
    ax[0].plot(h['s_spread'], 'r--', label='Scalar Spread (Collapse)')
    ax[0].plot(h['m_rmse'], 'b-', alpha=0.5, label='Matrix RMSE')
    ax[0].plot(h['m_spread'], 'b--', label='Matrix Spread (Stable)')
    ax[0].set_title("Filtering Stability")
    ax[0].set_xlabel("Time Step")
    ax[0].set_ylabel("Magnitude")
    ax[0].legend()

    i1, i2 = 19, 20
    ax[1].scatter(p_s.numpy()[i1, :], p_s.numpy()[i2, :], c='red', marker='x', alpha=0.5, label='Scalar')
    ax[1].scatter(p_m.numpy()[i1, :], p_m.numpy()[i2, :], edgecolors='blue', facecolors='none', label='Matrix')
    ax[1].scatter(states_true[-1].numpy()[i1], states_true[-1].numpy()[i2], c='black', marker='*', s=200, label='Truth')
    ax[1].set_title(f"Marginal Distribution ($X_{{{i1}}}$ vs $X_{{{i2}}}$)")
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()