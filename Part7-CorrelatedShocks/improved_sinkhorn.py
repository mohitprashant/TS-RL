import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float32
tfd = tfp.distributions


class SinkhornParticleFilter:
    """
    Standard Differentiable Particle Filter using Sinkhorn Resampling (Euclidean Cost).
    """
    def __init__(self, num_particles=100, epsilon=0.5, n_iter=15):
        """
        Initializes the Sinkhorn Filter parameters.

        Args:
            num_particles (int): The number of particles (N) to use.
            epsilon (float): The regularization strength. Controls the smoothness 
                of the transport plan.
            n_iter (int): Number of iterations for the Sinkhorn-Knopp algorithm.
        """
        self.num_particles = num_particles
        self.epsilon = tf.convert_to_tensor(epsilon, dtype=DTYPE)
        self.n_iter = n_iter

    def compute_cost(self, particles):
        """
        Computes the pairwise squared Euclidean distance matrix.

        Cost C_{ij} = ||x_i - x_j||^2

        Args:
            particles (tf.Tensor): A tensor of shape (N, p) representing N particles 
                in p dimensions.

        Returns:
            tf.Tensor: A Cost matrix of shape (N, N).
        """
        diff = particles[:, None, :] - particles[None, :, :]
        return tf.reduce_sum(tf.square(diff), axis=-1)


    def run(self, model, observations):
        """
        Executes the filtering loop over the observation sequence.

        Args:
            model: An object (e.g., MultivariateStochasticVolatilityModel) providing 
                .initial_dist(), .transition_dist(), and .observation_dist().
            observations (tf.Tensor): Sequence of observations (T, p).

        Returns:
            tf.Tensor: The sequence of estimated posterior means (T, p).
        """
        T = observations.shape[0]
        N = self.num_particles
        particles = model.initial_dist().sample(N)
        log_weights = tf.fill([N], -tf.math.log(float(N)))
        estimates = []
        
        for t in range(T):
            particles = model.transition_dist(particles).sample()
            log_lik = model.observation_dist(particles).log_prob(observations[t])
            log_lik = tf.clip_by_value(log_lik, -100.0, 100.0)
            log_weights += log_lik
            
            particles = self.sinkhorn_resample(particles, log_weights)
            log_weights = tf.fill([N], -tf.math.log(float(N)))
            estimates.append(tf.reduce_mean(particles, axis=0))
            
        return tf.stack(estimates)


    def sinkhorn_resample(self, particles, log_weights):
        """
        Performs the Soft Resampling step using the Sinkhorn algorithm.

        Args:
            particles (tf.Tensor): Current particle states (N, p).
            log_weights (tf.Tensor): Unnormalized log-weights (N,).

        Returns:
            tf.Tensor: The transported particles (N, p).
        """
        N = self.num_particles
        log_a = log_weights - tf.reduce_logsumexp(log_weights)
        log_b = tf.fill([N], -tf.math.log(float(N)))
        
        C = self.compute_cost(particles)
        
        scale = tf.reduce_max(tf.stop_gradient(C))
        eps = self.epsilon * tf.maximum(scale, 1.0)
        f = tf.zeros(N); g = tf.zeros(N)
        
        for _ in range(self.n_iter):
            f = 0.5 * (f - eps * tf.reduce_logsumexp(log_b[None,:] + (g[None,:] - C)/eps, axis=1))
            g = 0.5 * (g - eps * tf.reduce_logsumexp(log_a[:,None] + (f[:,None] - C)/eps, axis=0))
            
        log_P = (f[:,None] + g[None,:] - C)/eps + log_a[:,None] + log_b[None,:]
        P = tf.exp(log_P)
        return float(N) * tf.matmul(P, particles, transpose_a=True)


class ImprovedSinkhornParticleFilter(SinkhornParticleFilter):
    """
    Improved Sinkhorn Filter using Mahalanobis Distance.
    This class uses the Mahalanobis distance weighted by the inverse covariance 
    matrix of the state noise.

    Cost C(x, y) = (x - y)^T * Omega * (x - y)
    """
    def __init__(self, precision_matrix, num_particles=100, epsilon=0.5, n_iter=15):
        """
        Initializes the improved filter.

        Args:
            precision_matrix (tf.Tensor): The inverse of the state covariance matrix (p, p).
            num_particles (int): Number of particles.
            epsilon (float): Regularization strength.
            n_iter (int): Number of Sinkhorn iterations.
        """
        super().__init__(num_particles, epsilon, n_iter)
        self.Omega = precision_matrix 

    def compute_cost(self, particles):
        """
        Computes pairwise Mahalanobis distance.

        Args:
            particles (tf.Tensor): Shape (N, p).

        Returns:
            tf.Tensor: Cost matrix of shape (N, N).
        """
        diff = particles[:, None, :] - particles[None, :, :]                        # Compute difference vectors
        N = self.num_particles
        diff_flat = tf.reshape(diff, (N*N, -1))
        projected = tf.matmul(diff_flat, self.Omega)                               # Project differences
        cost_flat = tf.reduce_sum(diff_flat * projected, axis=1)                    # Compute dot product
        return tf.reshape(cost_flat, (N, N))
