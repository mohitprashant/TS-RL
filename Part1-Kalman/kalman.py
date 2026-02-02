import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from lgssm import LinearGaussianSSM

tfd = tfp.distributions


class KalmanFilter:
    """
    Kalman Filter.
    
    The filter models a system with the following dynamics:
    x_{t} = A * x_{t-1} + w_t,  w_t ~ N(0, B*B^T)
    y_{t} = C * x_t + v_t,      v_t ~ N(0, D*D^T)
    """

    def __init__(self, A, B, C, D, Sigma_init):
        """
        Initializes the Kalman Filter parameters.

        Args:
            A (array-like): State transition matrix.
            B (array-like): Process noise shaping matrix (Q = B @ B.T).
            C (array-like): Observation matrix.
            D (array-like): Observation noise shaping matrix (R = D @ D.T).
            Sigma_init (array-like): Initial state covariance matrix.
        """
        self.dtype = tf.float32
        self.A = tf.convert_to_tensor(A, dtype=self.dtype)
        self.C = tf.convert_to_tensor(C, dtype=self.dtype)
        self.Q = tf.matmul(tf.convert_to_tensor(B, dtype=self.dtype), 
                           tf.convert_to_tensor(B, dtype=self.dtype), transpose_b=True)
        self.R = tf.matmul(tf.convert_to_tensor(D, dtype=self.dtype), 
                           tf.convert_to_tensor(D, dtype=self.dtype), transpose_b=True)
        
        self.P_init = tf.convert_to_tensor(Sigma_init, dtype=self.dtype)
        
        self.nx = self.A.shape[0]
        self.ny = self.C.shape[0]
        self.eye_nx = tf.eye(self.nx, dtype=self.dtype)
        

    def get_condition_number(self, matrix):
        """
        Manually computes condition number using SVD.
        Cond(A) = sigma_max / sigma_min
        """
        s = tf.linalg.svd(matrix, compute_uv=False)       # compute_uv=False because we only need singular values (s)
        return tf.reduce_max(s) / tf.reduce_min(s)

    def predict(self, x_curr, P_curr):
        """
        Performs the Kalman Filter prediction step (Time Update).

        Args:
            x_curr (tf.Tensor): Current state estimate.
            P_curr (tf.Tensor): Current state covariance.

        Returns:
            tuple: (x_pred, P_pred) predicted state and covariance for the next step.
        """
        x_pred = tf.linalg.matvec(self.A, x_curr)
        P_pred = tf.matmul(self.A, tf.matmul(P_curr, self.A, transpose_b=True)) + self.Q
        return x_pred, P_pred


    def update(self, x_pred, P_pred, y_obs, joseph_form=True):
        """
        Performs the Kalman Filter update step (Measurement Update).

        Args:
            x_pred (tf.Tensor): Predicted state from the predict() step.
            P_pred (tf.Tensor): Predicted covariance from the predict() step.
            y_obs (tf.Tensor): Actual measurement observed at this timestep.
            joseph_form (bool): If True, uses the Joseph Form for covariance update 
                to ensure the resulting matrix remains symmetric and positive-definite.

        Returns:
            tuple: (x_new, P_new, cond_S, cond_P) updated state, covariance, 
                and condition numbers for S and P.
        """
        
        y_pred = tf.linalg.matvec(self.C, x_pred)
        y_res = y_obs - y_pred                                                            # Innovation
        
        S = tf.matmul(self.C, tf.matmul(P_pred, self.C, transpose_b=True)) + self.R       # Innovation Covariance S
        cond_S = self.get_condition_number(S)                                             # Stability Check: Condition Number of S

        CP_T = tf.matmul(self.C, P_pred, transpose_b=True)                                # Kalman Gain
        K_transpose = tf.linalg.solve(S, CP_T) 
        K = tf.transpose(K_transpose)
        
        x_new = x_pred + tf.linalg.matvec(K, y_res)                                       # State Update
        
        KC = tf.matmul(K, self.C)                                                         # Covariance Update
        I_KC = self.eye_nx - KC
        
        if(joseph_form):
            P_term1 = tf.matmul(I_KC, tf.matmul(P_pred, I_KC, transpose_b=True))
            P_term2 = tf.matmul(K, tf.matmul(self.R, K, transpose_b=True))
            P_new = P_term1 + P_term2
        else:
            P_new = tf.matmul(I_KC, P_pred)

        cond_P = self.get_condition_number(P_new)                                        # Stability Check: Condition Number of P_new
        return x_new, P_new, cond_S, cond_P


    def run_filter(self, observations, joseph_form=True):
        """
        Runs the filter over a sequence of observations.

        Args:
            observations (tf.Tensor): A sequence of measurement vectors.
            joseph_form (bool): Whether to use the Joseph Form update.

        Returns:
            tuple: (means, covariances, cond_S_history, cond_P_history) 
                tensors containing the filter's history.
        """
        T = observations.shape[0]
        x_curr = tf.zeros(self.nx, dtype=self.dtype)
        P_curr = self.P_init
        
        means = []
        covariances = []
        cond_S_history = []
        cond_P_history = []
        
        # Initial step
        x_curr, P_curr, cond_S, cond_P = self.update(x_curr, P_curr, observations[0], joseph_form)
        means.append(x_curr)
        covariances.append(P_curr)
        cond_S_history.append(cond_S)
        cond_P_history.append(cond_P)
        
        # Primary Control Loop
        for t in range(1, T):
            x_pred, P_pred = self.predict(x_curr, P_curr)
            x_curr, P_curr, cond_S, cond_P = self.update(x_pred, P_pred, observations[t], joseph_form)
            
            means.append(x_curr)
            covariances.append(P_curr)
            cond_S_history.append(cond_S)
            cond_P_history.append(cond_P)
            
        return (tf.stack(means), tf.stack(covariances), 
                tf.stack(cond_S_history), tf.stack(cond_P_history))



###################################################################################################

def main():
    dt = 0.1
    T = 100
    A = [[1.0, dt], [0.0, 1.0]]
    B = [[0.5 * dt**2, 0.0], [dt, 0.0]]
    C = [[1.0, 0.0]]
    D = [[0.5]]
    Sigma_init = [[1.0, 0.0], [0.0, 1.0]]

    ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
    states, observations = ssm.simulate(T)
    kf = KalmanFilter(A, B, C, D, Sigma_init)
    means, covs, cond_S, cond_P = kf.run_filter(observations, joseph_form=True)


    time_steps = np.arange(T)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(time_steps, states.numpy()[:, 0], 'k-', label='True')
    plt.plot(time_steps, means.numpy()[:, 0], 'b--', label='Estimate')
    plt.title("Position Estimate")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(time_steps, cond_P.numpy(), 'purple')
    plt.title("Condition Number of State Covariance (P)")
    plt.ylabel("Condition Number")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel("Time Step")
    
    plt.subplot(2, 2, 3)
    plt.plot(time_steps, cond_S.numpy(), 'green')
    plt.title("Condition Number of Innovation Covariance (S)")
    plt.ylabel("Condition Number")
    plt.xlabel("Time Step")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print(f"Max Condition Number (P): {np.max(cond_P.numpy()):.2f}")
    print(f"Max Condition Number (S): {np.max(cond_S.numpy()):.2f}")

if __name__ == "__main__":
    main()