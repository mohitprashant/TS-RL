import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

class Lorenz96Model:
    """
    Lorenz 96 State Space Model.
    
    A continuous-time dynamical system often used as a proxy for atmospheric 
    turbulence and chaotic systems.
    
    Dynamics (ODE): 
        dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F
    
    Observation: 
        Y_k = X_k + v_k  where v_k ~ N(0, R)
    """
    def __init__(self, K=40, F=8.0, dt=0.05, process_std=0.1, obs_std=1.0):
        """
        Initializes the Lorenz 96 model parameters.

        Args:
            K (int): State dimension (number of variables). Default is 40.
            F (float): Forcing constant. F=8.0 typically induces chaotic behavior.
            dt (float): Integration time step. Default is 0.05.
            process_std (float): Standard deviation of the additive process noise.
            obs_std (float): Standard deviation of the additive observation noise.
        """
        self.dtype = tf.float32
        self.K = K
        self.F = tf.constant(F, dtype=self.dtype)
        self.dt = tf.constant(dt, dtype=self.dtype)
        
        # Noise Covariances
        self.Q_diag = tf.fill([K], tf.constant(process_std, dtype=self.dtype)**2)
        self.R_diag = tf.fill([K], tf.constant(obs_std, dtype=self.dtype)**2)
        self.R = tf.linalg.diag(self.R_diag)
        self.Q = tf.linalg.diag(self.Q_diag)


    def rk4_step(self, x):
        """
        Performs a single Runge-Kutta 4 (RK4) integration step for the deterministic dynamics.
        
        Args:
            x (tf.Tensor): Current state tensor of shape (..., K).
            
        Returns:
            tf.Tensor: Next state tensor after time dt, shape (..., K).
        """
        def ode(state):
            x_p1 = tf.roll(state, shift=-1, axis=-1)
            x_m1 = tf.roll(state, shift=1, axis=-1)
            x_m2 = tf.roll(state, shift=2, axis=-1)
            return (x_p1 - x_m2) * x_m1 - state + self.F

        k1 = ode(x)
        k2 = ode(x + 0.5 * self.dt * k1)
        k3 = ode(x + 0.5 * self.dt * k2)
        k4 = ode(x + self.dt * k3)
        return x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def transition(self, x):
        """
        Applies the stochastic state transition: x_k = f(x_{k-1}) + q_k.
        
        Args:
            x (tf.Tensor): State at time k-1.
            
        Returns:
            tf.Tensor: State at time k with process noise added.
        """
        mu = self.rk4_step(x)
        return mu + tf.random.normal(tf.shape(mu), stddev=tf.sqrt(self.Q_diag[0]))


    def observation(self, x):
        """
        Generates a noisy observation from the state: y_k = x_k + r_k.
        
        Args:
            x (tf.Tensor): Current state vector.
            
        Returns:
            tf.Tensor: Noisy observation vector.
        """
        return x + tf.random.normal(tf.shape(x), stddev=tf.sqrt(self.R_diag[0]))


    def get_jacobian(self, x):
        """
        Computes the Jacobian F = df/dx using Automatic Differentiation.
        
        Args:
            x (tf.Tensor): Input state tensor.
            
        Returns:
            tf.Tensor: Jacobian matrix of the RK4 transition function at x.
        """
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.rk4_step(x)
        return tape.batch_jacobian(y, x)
    

    def simulate(self, T, burn_in=1000):
        """
        Simulates the Lorenz 96 system for T time steps.
        
        Includes a burn-in period to allow the system to settle onto its chaotic attractor
        before recording begins.

        Args:
            T (int): Number of time steps to record.
            burn_in (int): Number of steps to discard initially. Default 1000.

        Returns:
            tuple: (states, observations)
                - states (tf.Tensor): Recorded states of shape (T, K).
                - observations (tf.Tensor): Recorded observations of shape (T, K).
        """
        current_state = self.initial_dist().sample()

        print(f"Burning in for {burn_in} steps...")
        for _ in range(burn_in):
            current_state = self.rk4_step(current_state)

        print(f"Simulating for {T} steps...")
        x_history = []
        y_history = []
        
        for _ in range(T):
            dist = self.transition_dist(current_state)
            current_state = dist.sample()
            
            obs_dist = self.observation_dist(current_state)
            current_obs = obs_dist.sample()
            
            x_history.append(current_state)
            y_history.append(current_obs)
            
        return tf.stack(x_history), tf.stack(y_history)
    

    def initial_dist(self):
        """
        Returns the initial distribution for the state X_0.
        
        Initialized near the equilibrium forcing F with small perturbations to 
        ensure divergence into chaotic behavior.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Initial distribution.
        """
        loc = tf.fill([self.K], self.F)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.ones(self.K)*0.1)        


    def transition_dist(self, x_prev):
        """
        Returns the probabilistic state transition: p(x_t | x_{t-1}).
        
        Modeled as: x_t = RK4(x_{t-1}) + Gaussian Noise

        Args:
            x_prev (tf.Tensor): State at time t-1.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Transition distribution.
        """
        loc = self.rk4_step(x_prev)
        return tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.fill([self.K], self.Q_diag[0]))
    
    
    def observation_dist(self, x_curr):
        """
        Returns the observation distribution: p(y_t | x_t).
        
        Modeled as fully observed state with Gaussian noise: y_t = x_t + Noise.

        Args:
            x_curr (tf.Tensor): State at time t.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Observation distribution.
        """
        return tfd.MultivariateNormalDiag(loc=x_curr, scale_diag=tf.fill([self.K], self.R_diag[0]))



####################################################################################################


def main():
    K = 40          # Standard L96 dimension
    F = 8.0         # Standard chaotic forcing
    dt = 0.05       # Integration step
    T = 500         # Steps to record

    l96 = Lorenz96Model(K=K, F=F, dt=dt, process_std=0.01, obs_std=1.0)
    states, observations = l96.simulate(T, burn_in=500)
    states_np = states.numpy()
    obs_np = observations.numpy()


    plt.figure(figsize=(12, 6))
    plt.imshow(states_np.T, aspect='auto', cmap='RdBu_r', origin='lower',
               extent=[0, T, 0, K], vmin=0, vmax=12)
    plt.colorbar(label='State Value ($X_k$)')
    plt.title(f'Lorenz 96 Hovmoller Diagram (F={F}, K={K})')
    plt.xlabel('Time Step')
    plt.ylabel('State Index (k)')
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(states_np[:, 0], states_np[:, 1], states_np[:, 2], 
            lw=0.8, color='teal', label='True Trajectory')
    ax.scatter(obs_np[:, 0], obs_np[:, 1], obs_np[:, 2], 
               color='red', s=10, alpha=0.4, label='Observations', marker='o')

    ax.set_title("Phase Space Projection (X0, X1, X2)\nTrajectory vs Noisy Observations")
    ax.set_xlabel("X0")
    ax.set_ylabel("X1")
    ax.set_zlabel("X2")
    ax.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()