import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np

tfd = tfp.distributions

class LinearGaussianSSM:
    """
    Implements the Linear Gaussian Model described in Example 2.
    
    State Space: X in R^{n_x}
    Observation Space: Y in R^{n_y}
    
    Equations:
      X_1 ~ N(0, Sigma_init)
      X_n = A * X_{n-1} + B * V_n,  where V_n ~ N(0, I)
      Y_n = C * X_n + D * W_n,      where W_n ~ N(0, I)
    """
    def __init__(self, A, B, C, D, Sigma_init):
        """
        Initialize matrices as TensorFlow tensors.
        
        Args:
            A: State transition matrix (n_x, n_x)
            B: State noise mapping matrix (n_x, n_v)
            C: Observation matrix (n_y, n_x)
            D: Observation noise mapping matrix (n_y, n_w)
            Sigma_init: Initial state covariance matrix (n_x, n_x)
        """
        self.dtype = tf.float32
        self.A = tf.convert_to_tensor(A, dtype=self.dtype)
        self.B = tf.convert_to_tensor(B, dtype=self.dtype)
        self.C = tf.convert_to_tensor(C, dtype=self.dtype)
        self.D = tf.convert_to_tensor(D, dtype=self.dtype)
        self.Sigma_init = tf.convert_to_tensor(Sigma_init, dtype=self.dtype)

        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        
        # Pre-compute Cholesky factors for efficiency and stability in TFP
        # Covariance for transition: Q = B * B^T
        # Covariance for observation: R = D * D^T
        self.Q_scale = tf.linalg.cholesky(tf.matmul(self.B, self.B, transpose_b=True))
        self.R_scale = tf.linalg.cholesky(tf.matmul(self.D, self.D, transpose_b=True))
        self.Init_scale = tf.linalg.cholesky(self.Sigma_init)

    def initial_dist(self):
        """Returns the distribution for X_1 ~ N(0, Sigma) """
        return tfd.MultivariateNormalTriL(
            loc=tf.zeros(self.nx, dtype=self.dtype),
            scale_tril=self.Init_scale
        )

    def transition_dist(self, x_prev):
        """
        Returns p(x_n | x_{n-1}) = N(A x_{n-1}, B B^T) 
        """
        loc = tf.linalg.matvec(self.A, x_prev)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.Q_scale)

    def observation_dist(self, x_curr):
        """
        Returns p(y_n | x_n) = N(C x_n, D D^T) 
        """
        loc = tf.linalg.matvec(self.C, x_curr)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=self.R_scale)

    def simulate(self, T):
        """
        Simulates the system for T time steps.
        """
        x_history = []
        y_history = []

        # Sample X_1
        x_curr = self.initial_dist().sample()
        y_curr = self.observation_dist(x_curr).sample()
        
        x_history.append(x_curr)
        y_history.append(y_curr)

        # Iterate for T-1 steps
        for _ in range(1, T):
            # Sample X_n | X_{n-1}
            x_curr = self.transition_dist(x_curr).sample()
            
            # Sample Y_n | X_n
            y_curr = self.observation_dist(x_curr).sample()
            
            x_history.append(x_curr)
            y_history.append(y_curr)

        return tf.stack(x_history), tf.stack(y_history)


##############################################################################################

def main():
    dt = 0.1
    T = 100
    A = [[1.0, dt],                  # Transition Matrix A: Constant velocity model
         [0.0, 1.0]]
    
    B = [[0.5 * dt**2, 0.0],         # Noise Mapping B: Random acceleration
         [dt, 0.0]]  
    
    C = [[1.0, 0.0]]                 # Observation Matrix C: We only observe position
    D = [[0.5]]                      # Observation Noise Mapping D: Scalar noise
    Sigma_init = [[1.0, 0.0],        # Initial Covariance
                  [0.0, 1.0]]

    ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
    

    print(f"Simulating Linear Gaussian Model for T={T} steps...")
    states, observations = ssm.simulate(T)
    states_np = states.numpy()
    obs_np = observations.numpy()
    time_steps = np.arange(T)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(time_steps, states_np[:, 0], label='True State (Position)', linewidth=2)
    plt.scatter(time_steps, obs_np[:, 0], color='red', s=15, alpha=0.6, label='Observations')
    plt.title('State X[0] vs Observations Y')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(time_steps, states_np[:, 1], color='orange', label='True State (Velocity)')
    plt.title('Latent State X[1] (Velocity)')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    print("Plot generated.")
    plt.show()


if __name__ == "__main__":
    main()