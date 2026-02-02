import tensorflow as tf
import tensorflow_probability as tfp

DTYPE = tf.float64
tf.keras.backend.set_floatx('float64')

class StochasticParticleFlow:
    """
    Stiffness Mitigation in Stochastic Particle Flow Filters.

    Attributes:
        target_true (tf.Tensor): The ground truth target location [x, y].
        sensor1 (tf.Tensor): Location of the first sensor.
        sensor2 (tf.Tensor): Location of the second sensor.
        prior_mu (tf.Tensor): Mean of the prior Gaussian distribution.
        P_prior (tf.Tensor): Covariance of the prior Gaussian.
        R (tf.Tensor): Measurement noise covariance matrix.
        Q_diff (tf.Tensor): Diffusion process noise covariance.
        mu_weight (tf.Tensor): The penalty weight for the stiffness term in the 
                               cost function. Higher values prioritize reducing stiffness.
        M0_mat (tf.Tensor): The inverse of the prior covariance (Information matrix).
        Mh_nominal (tf.Tensor): The nominal measurement information matrix evaluated 
                                at the ground truth, used for the ODE approximation.
    """
    def __init__(self):
        self.target_true = tf.constant([4.0, 4.0], dtype=DTYPE)
        self.sensor1 = tf.constant([3.5, 0.0], dtype=DTYPE)
        self.sensor2 = tf.constant([-3.5, 0.0], dtype=DTYPE)
        
        self.prior_mu = tf.constant([3.0, 0.0], dtype=DTYPE)                                            # Prior (Gaussian)
        self.P_prior = tf.linalg.diag(tf.constant([1000.0, 2.0], dtype=DTYPE))
        self.inv_P_prior = tf.linalg.inv(self.P_prior)
        self.L_prior = tf.linalg.cholesky(self.P_prior)

        self.R = tf.linalg.diag(tf.constant([0.04, 0.04], dtype=DTYPE))                                # Measurement Noise and Q
        self.inv_R = tf.linalg.inv(self.R)
        self.Q_diff = tf.linalg.diag(tf.constant([4.0, 0.4], dtype=DTYPE))
        self.sqrt_Q = tf.math.sqrt(tf.linalg.diag_part(self.Q_diff))
        
        self.mu_weight = tf.constant(0.2, dtype=DTYPE)                                                 # Optimization Weight
        self.z_meas = tf.constant([0.4754, 1.1868], dtype=DTYPE)                                       # Measurement Z

        self.M0_mat = self.inv_P_prior                                                                 # Matrices for BVP Optimization
        
        H_nominal = self.get_H_jacobian(self.target_true)                                              # Nominal Hessian
        Ah_nominal = -tf.matmul(tf.transpose(H_nominal), tf.matmul(self.inv_R, H_nominal))
        self.Mh_nominal = -Ah_nominal

        self.lambda_grid = None                                                                        # Storage for schedule
        self.beta_grid = None
        self.u_grid = None


    def h_meas(self, x):
        """
        Computes the nonlinear measurement function h(x).

        Calculates the bearings (angles) from two fixed sensors to the target state x.

        Args:
            x (tf.Tensor): State vector(s). Shape (2,) or (Batch, 2).

        Returns:
            tf.Tensor: Measurement vector z = [angle1, angle2]. 
                       Shape matches input batch dimension.
        """
        if(len(x.shape) == 1):
            x = tf.expand_dims(x, 0)
        
        dx1 = x[:, 0] - self.sensor1[0]
        dy1 = x[:, 1] - self.sensor1[1]
        z1 = tf.math.atan2(dy1, dx1)

        dx2 = x[:, 0] - self.sensor2[0]
        dy2 = x[:, 1] - self.sensor2[1]
        z2 = tf.math.atan2(dy2, dx2)
        return tf.stack([z1, z2], axis=1)


    def get_H_jacobian(self, x_lin):
        """
        Computes the analytical Jacobian H = dh/dx at a linearization point.

        Args:
            x_lin (tf.Tensor): The state vector at which to linearize. Shape (2,).

        Returns:
            tf.Tensor: The Jacobian matrix of shape (2, 2).
                       Element [i, j] is d(z_i) / d(x_j).
        """
        dx1 = x_lin[0] - self.sensor1[0]
        dy1 = x_lin[1] - self.sensor1[1]
        r2_1 = dx1**2 + dy1**2 + 1e-9
        
        dx2 = x_lin[0] - self.sensor2[0]
        dy2 = x_lin[1] - self.sensor2[1]
        r2_2 = dx2**2 + dy2**2 + 1e-9

        row1 = tf.stack([-dy1 / r2_1, dx1 / r2_1])
        row2 = tf.stack([-dy2 / r2_2, dx2 / r2_2])
        return tf.stack([row1, row2])


    @tf.function
    def stiffness_ode_fn(self, t, y):
        """
        Defines the Euler-Lagrange Ordinary Differential Equations (ODEs) for the 
        optimal control problem.

        Args:
            t (float): Current pseudo-time lambda (0.0 to 1.0).
            y (tf.Tensor): State vector containing [beta, u].

        Returns:
            tf.Tensor: Derivatives [d(beta)/dt, d(u)/dt].
        """
        beta = y[0]
        u = y[1]
        M = self.M0_mat + beta * self.Mh_nominal
        inv_M = tf.linalg.inv(M)
        Ah_paper = -self.Mh_nominal 
        
        term1 = tf.linalg.trace(Ah_paper) * tf.linalg.trace(inv_M)
        term2 = tf.linalg.trace(M) * tf.linalg.trace(tf.matmul(tf.matmul(inv_M, inv_M), Ah_paper))
        
        d_kappa = -(term1 + term2)
        d2_beta = self.mu_weight * d_kappa
        return tf.stack([u, d2_beta])


    def solve_optimal_schedule(self):
        """
        Solves the Boundary Value Problem (BVP) to find the optimal schedule beta(lambda).

        Returns:
            tuple:
                lambda_grid (tf.Tensor): The grid of pseudo-time steps (0 to 1).
                beta_grid (tf.Tensor): The optimized schedule values.
                u_grid (tf.Tensor): The rate of change of the schedule.
        """
        solver = tfp.math.ode.DormandPrince()
        def evaluate_boundary(u_initial):
            y0 = tf.stack([tf.constant(0.0, dtype=DTYPE), u_initial])
            results = solver.solve(self.stiffness_ode_fn, 
                                   initial_time=0.0, 
                                   initial_state=y0, 
                                   solution_times=[1.0])
            return results.states[-1, 0]

        u_min = tf.constant(0.1, dtype=DTYPE)
        u_max = tf.constant(5.0, dtype=DTYPE)
        
        # Bisection method, cause we only have IVP :(
        u_best = u_min
        for _ in range(25):
            u_mid = (u_min + u_max) / 2.0
            beta_val = evaluate_boundary(u_mid)
            if tf.abs(beta_val - 1.0) < 1e-4:
                u_best = u_mid
                break
            if(beta_val > 1.0):
                u_max = u_mid
            else:
                u_min = u_mid
            u_best = u_mid
            
        print(f"BVP Converged: u(0)={u_best.numpy():.4f}")
        
        t_eval = tf.linspace(0.0, 1.0, 50)
        y0 = tf.stack([tf.constant(0.0, dtype=DTYPE), u_best])
        results = solver.solve(self.stiffness_ode_fn, 
                               initial_time=0.0, 
                               initial_state=y0, 
                               solution_times=t_eval)
        
        self.lambda_grid = t_eval
        self.beta_grid = tf.clip_by_value(results.states[:, 0], 0.0, 1.0)
        self.u_grid = results.states[:, 1]
        return self.lambda_grid, self.beta_grid, self.u_grid

