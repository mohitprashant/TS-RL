import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
from svssm import StochasticVolatilityModel



# DTYPE = tf.float64
DTYPE = tf.float32
tfd = tfp.distributions



class SinkhornParticleFilter:
    """
    Differentiable Particle Filter using Sinkhorn (Entropy-Regularized OT) Resampling.
    """
    def __init__(self, alpha, sigma, beta, num_particles=100, epsilon=0.5, n_iter=20):
        """
        Initializes the Sinkhorn Particle Filter.

        Args:
            alpha (tf.Tensor/float): Autoregressive coefficient for state transition.
            sigma (tf.Tensor/float): Standard deviation of process noise.
            beta (tf.Tensor/float): Scaling factor for observation volatility.
            num_particles (int): Number of particles (N) to use.
            epsilon (tf.Tensor/float): Entropy regularization strength. Higher 
                values lead to smoother, more diffused transport.
            n_iter (int): Number of Sinkhorn iterations for potential convergence.
        """
        self.alpha = tf.convert_to_tensor(alpha, dtype=DTYPE)
        self.sigma = tf.convert_to_tensor(sigma, dtype=DTYPE)
        self.beta = tf.convert_to_tensor(beta, dtype=DTYPE)
        self.num_particles = num_particles
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=DTYPE)
        self.n_iter = n_iter
        

    def initial_dist(self):
        """
        Initial state distribution X_1.
        
        Assumes the process is at a stationary distribution: N(0, sigma^2 / (1 - alpha^2)).

        Returns:
            tfd.Distribution: A TensorFlow Probability Normal distribution.
        """
        variance = self.sigma**2 / (1.0 - self.alpha**2)
        return tfd.Normal(loc=0.0, scale=tf.sqrt(variance))


    def transition(self, particles):
        """
        State transition model: X_t ~ N(alpha * X_{t-1}, sigma).

        Args:
            particles (tf.Tensor): Current particle states.

        Returns:
            tf.Tensor: Sampled next-state particles.
        """
        return tfd.Normal(loc=self.alpha * particles, scale=self.sigma).sample()


    def log_likelihood(self, y_obs, particles):
        """
        Observation log-likelihood: log p(y_t | x_t) for a Stochastic Volatility Model.

        Args:
            y_obs (tf.Tensor): The scalar observation at time t.
            particles (tf.Tensor): Current particle states.

        Returns:
            tf.Tensor: Log-probabilities of the observation given each particle.
        """
        safe_particles = tf.clip_by_value(particles, -20.0, 20.0)     # Clamp particles to prevent exp() explosion
        scale = self.beta * tf.exp(safe_particles / 2.0)
        return tfd.Normal(loc=0.0, scale=scale).log_prob(y_obs)


    def sinkhorn_potentials(self, log_a, log_b, C):
        """
        Computes Sinkhorn potentials (f, g) using stabilized log-domain updates.
        
        Solves dual problem of entropy-regularized optimal transport to find 
        scaling factors that transform the cost matrix into a doubly stochastic 
        transport plan.

        Args:
            log_a (tf.Tensor): Log-source weights (normalized particle weights).
            log_b (tf.Tensor): Log-target weights (usually uniform 1/N).
            C (tf.Tensor): Cost matrix (squared Euclidean distance between particles).

        Returns:
            tuple: (f, g) Sinkhorn dual potentials.
        """
        N = self.num_particles
        f = tf.zeros((N,), dtype=DTYPE)
        g = tf.zeros((N,), dtype=DTYPE)
        eps = self.epsilon
        
        for _ in range(self.n_iter):                                          # Sinkhorn iterations
            tmp_f = log_b[None, :] + (g[None, :] - C) / eps                   # Update f
            f_update = -eps * tf.reduce_logsumexp(tmp_f, axis=1)
            f = 0.5 * (f + f_update)
            
            tmp_g = log_a[:, None] + (f[:, None] - C) / eps                   # Update g
            g_update = -eps * tf.reduce_logsumexp(tmp_g, axis=0)
            g = 0.5 * (g + g_update)
        return f, g


    def sinkhorn_resample(self, particles, log_weights):
        """
        Performs Soft Resampling via Optimal Transport.
        
        Maps the current weighted particle set to a new set of particles with 
        uniform weights by applying the barycentric projection derived from the 
        optimal transport plan P.

        Args:
            particles (tf.Tensor): Current particle states.
            log_weights (tf.Tensor): Current unnormalized log-weights.

        Returns:
            tuple: (new_particles, new_log_weights, P) where P is the transport matrix.
        """
        N = self.num_particles
        log_weights = tf.maximum(log_weights, -1e9) 
        log_w_norm = log_weights - tf.reduce_logsumexp(log_weights)            # Normalize weights
        
        log_b = tf.fill([N], -tf.math.log(float(N)))                           # Target weights
        diff = particles[:, None] - particles[None, :]                         # Cost Matrix 
        C = diff**2

        f, g = self.sinkhorn_potentials(log_w_norm, log_b, C)
        
        log_P = (f[:, None] + g[None, :] - C) / self.epsilon + log_w_norm[:, None] + log_b[None, :]      # Transport Matrix in log domain
        P = tf.exp(log_P) 
        P = tf.debugging.check_numerics(P, "Transport Matrix P contains NaNs")                           # Check nan and then transport
        new_particles = float(N) * tf.linalg.matvec(tf.transpose(P), particles)
        
        new_log_weights = log_b
        return new_particles, new_log_weights, P


    def run_filter(self, observations, true_states=None):
        """
        Runs the Sinkhorn Particle Filter over a sequence of observations.

        Args:
            observations (tf.Tensor): Time series of observed data.
            true_states (tf.Tensor, optional): Ground truth latent states for RMSE.

        Returns:
            dict: Performance and state estimation metrics:
                - 'rmse': Root Mean Square Error.
                - 'time': Total runtime.
                - 'mem': Peak memory usage in bytes.
                - 'ess_avg': Average Effective Sample Size.
                - 'cond_avg': Average condition number of the transport matrix.
                - 'log_likelihood': Total estimated log-marginal likelihood.
                - 'estimates': Tensor of mean state estimates at each time step.
        """
        T = observations.shape[0]
        N = self.num_particles
        
        particles = self.initial_dist().sample(N)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        
        estimates = []
        ess_history = []
        cond_history = []
        log_likelihood_est = 0.0
        
        tracemalloc.start()
        start_time = time.time()

        for t in range(T):
            particles = self.transition(particles)
            log_lik = self.log_likelihood(observations[t], particles)                  # Weight particles
            log_weights += log_lik
            
            step_log_lik = tf.reduce_logsumexp(log_weights)
            log_likelihood_est += step_log_lik 

            log_w_norm = log_weights - step_log_lik                                    # ESS calc
            w_norm = tf.exp(log_w_norm)
            ess = 1.0 / tf.reduce_sum(w_norm**2)
            ess_history.append(ess)
            
            particles, log_weights, P = self.sinkhorn_resample(particles, log_weights)   # Sinkhorn Resampling step
            
            try:                                                                         # Hnandle nan exceptions
                P_stable = P + tf.eye(N) * 1e-5
                cond_num = tf.linalg.cond(P_stable)
            except Exception:
                cond_num = tf.constant(float('nan'))
            cond_history.append(cond_num)

            est = tf.reduce_mean(particles)                                               # Estimate
            estimates.append(est)

        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_time = time.time() - start_time
        
        estimates_tensor = tf.stack(estimates)
        rmse = np.sqrt(np.mean((true_states.numpy() - estimates_tensor.numpy())**2)) if true_states is not None else 0.0
        
        avg_cond = tf.reduce_mean(cond_history)
        if(tf.math.is_nan(avg_cond)): 
            avg_cond = tf.constant(0.0)

        return {
            'rmse': rmse,
            'time': total_time,
            'mem': peak_mem,
            'ess_avg': tf.reduce_mean(ess_history),
            'cond_avg': avg_cond,
            'log_likelihood': log_likelihood_est,
            'estimates': estimates_tensor
        }
    

##################################################################################################################################

def run_experiment_grid():
    alpha_val = 0.91
    sigma_val = 1.0
    beta_val = 0.5
    T = 50
    N = 100 
    model = StochasticVolatilityModel(alpha_val, sigma_val, beta_val)
    true_x, observations = model.simulate(T)

    epsilons = [0.1, 0.2, 0.5, 0.7, 1.0, 1.5]
    n_iters_list = [10, 20, 30, 40, 50, 70, 100]
    results = []
    print(f"{'Epsilon':<8} | {'Iters':<6} | {'RMSE':<10} | {'GradNorm':<12} | {'Time(s)':<10}")
    print("-" * 75)

    for n_iter in n_iters_list:
        for eps in epsilons:
            alpha_var = tf.Variable(alpha_val, dtype=DTYPE)
            sigma_var = tf.Variable(sigma_val, dtype=DTYPE)
            beta_var = tf.Variable(beta_val, dtype=DTYPE)
            
            with tf.GradientTape() as tape:
                dpf = SinkhornParticleFilter(alpha_var, sigma_var, beta_var, num_particles=N, epsilon=eps, n_iter=n_iter)
                out = dpf.run_filter(observations, true_x)
                loss = -out['log_likelihood']
                
            grads = tape.gradient(loss, [alpha_var, sigma_var, beta_var])                    # Compute grad
            
            if(all(g is not None for g in grads)):
                grad_norm = tf.linalg.global_norm(grads).numpy()
            else:
                grad_norm = 0.0

            res_entry = {
                'epsilon': eps,
                'n_iter': n_iter,
                'rmse': out['rmse'],
                'cond': out['cond_avg'].numpy(),
                'diff_norm': grad_norm,
                'time': out['time'],
                'mem': out['mem']
            }
            results.append(res_entry)
            print(f"{eps:<8.1f} | {n_iter:<6} | {out['rmse']:<10.4f} | {grad_norm:<12.4f} | {out['time']:<10.4f}")
    return results


def plot_joint_results(results):
    epsilons = sorted(list(set(r['epsilon'] for r in results)))
    n_iters = sorted(list(set(r['n_iter'] for r in results)))
    cm = plt.get_cmap('viridis')
    colors = [cm(x) for x in np.linspace(0, 1, len(n_iters))]
    
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(1, 3, 1)
    for i, n_iter in enumerate(n_iters):
        subset = [r for r in results if r['n_iter'] == n_iter]
        subset.sort(key=lambda x: x['epsilon'])
        x = [r['epsilon'] for r in subset]
        y = [r['rmse'] for r in subset]
        ax1.plot(x, y, marker='o', markersize=4, label=f'Iter={n_iter}', color=colors[i])
        
    ax1.set_xlabel('Epsilon')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Regularization')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(1, 3, 2)
    for i, n_iter in enumerate(n_iters):
        subset = [r for r in results if r['n_iter'] == n_iter]
        subset.sort(key=lambda x: x['epsilon'])
        x = [r['epsilon'] for r in subset]
        y = [r['cond'] for r in subset]
        ax2.plot(x, y, marker='x', markersize=4, label=f'Iter={n_iter}', color=colors[i])
    ax2.set_xlabel('Epsilon')
    ax2.set_ylabel('Avg Condition Number (log scale)')
    ax2.set_yscale('log')
    ax2.set_title('Matrix Stability vs Regularization')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(1, 3, 3)
    rep_eps = [epsilons[0], epsilons[len(epsilons)//2], epsilons[-1]]
    for eps in rep_eps:
        subset = [r for r in results if r['epsilon'] == eps]
        subset.sort(key=lambda x: x['n_iter'])
        x = [r['n_iter'] for r in subset]
        y = [r['time'] for r in subset]
        ax3.plot(x, y, marker='s', label=f'Eps={eps}')
    ax3.set_xlabel('Sinkhorn Iterations')
    ax3.set_ylabel('Runtime (s)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_experiment_grid()
    plot_joint_results(results)