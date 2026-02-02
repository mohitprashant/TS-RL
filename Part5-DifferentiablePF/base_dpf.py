import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class BaseParticleFilter:
    def __init__(self, alpha, sigma, beta, num_particles):
        """
        Initializes the model parameters.

        Args:
            alpha (tf.Tensor): Persistence parameter of the latent state (|alpha| < 1).
            sigma (tf.Tensor): Standard deviation of the state transition noise.
            beta (tf.Tensor): Global scaling factor for observation volatility.
            num_particles (int): Number of particles used in the filtering approximation.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.beta = beta
        self.num_particles = num_particles


    def initial_dist(self):
        """
        Defines the initial distribution of the latent state at t=0.

        Calculates the stationary distribution of the AR(1) process, 
        ensuring the filter starts in a steady state.

        Returns:
            tfd.Normal: A Normal distribution with mean 0 and stationary variance.
        """
        variance = self.sigma**2 / (1.0 - self.alpha**2)
        return tfd.Normal(loc=0.0, scale=tf.sqrt(variance))


    def transition(self, particles):
        """
        Propagates particles through the state transition model (Prediction).

        X_t = alpha * X_{t-1} + N(0, sigma^2).

        Args:
            particles (tf.Tensor): Tensor of particle states from the previous timestep.

        Returns:
            tf.Tensor: Newly sampled particle states.
        """
        return tfd.Normal(loc=self.alpha * particles, scale=self.sigma).sample()


    def log_likelihood(self, y_obs, particles):
        """
        Computes the log-likelihood of an observation given the particles.

        y_t ~ N(0, (beta * exp(x_t / 2))^2)

        Args:
            y_obs (tf.Tensor): The observed value at the current timestep.
            particles (tf.Tensor): The current set of particles representing x_t.

        Returns:
            tf.Tensor: The log-probability density for each particle.
        """
        safe_particles = tf.clip_by_value(particles, -20.0, 20.0)      # clip for numerical stability during exp
        scale = self.beta * tf.exp(safe_particles / 2.0)
        return tfd.Normal(loc=0.0, scale=scale).log_prob(y_obs)