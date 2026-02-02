import tensorflow as tf
import tensorflow_probability as tfp
from ledh_inv_pf import AuxiliaryEKF
from ledh_inv_pf import robust_svd_eig
from ledh_inv_pf import robust_pinv

tfd = tfp.distributions



class HomotopySolver:
    """
    Solves for the optimal stiffness-mitigating schedule beta(lambda).
    """
    def __init__(self, model, mu=0.1, steps=20):
        """
        Initializes the solver.

        Args:
            model: The system model containing noise matrices (R, Q).
            mu (float): The weighting factor for the penalty term in the cost function.
                        Higher mu penalizes high velocity (u) more, leading to smoother 
                        but potentially stiffer schedules.
            steps (int): Number of integration steps for the flow.
        """
        self.model = model
        self.mu = tf.constant(mu, dtype=tf.float64)
        self.steps = steps


    @tf.function
    def compute_kappa_grad(self, beta, H0, dH):
        """
        Computes the gradient of the condition number with respect to beta.
        d(cond(M))/d(beta), where M = H0 + beta * dH.

        Args:
            beta (tf.Tensor): Current schedule value.
            H0 (tf.Tensor): Prior information matrix.
            dH (tf.Tensor): Difference between Measurement and Prior information matrices.

        Returns:
            tf.Tensor: The gradient scalar, clipped for stability.
        """
        beta_c = tf.clip_by_value(beta, 0.0, 1.0)
        M = H0 + beta_c * dH
        is_bad = tf.reduce_any(tf.math.is_nan(M))                           # Check for Nan
        
        def safe_grad():
            e_vals, e_vecs = robust_svd_eig(M) 
            lam_max = e_vals[0]
            lam_min = e_vals[-1]
            v_max = e_vecs[:, 0]
            v_min = e_vecs[:, -1]
            
            term_min = tf.tensordot(v_min, tf.linalg.matvec(dH, v_min), axes=1)
            term_max = tf.tensordot(v_max, tf.linalg.matvec(dH, v_max), axes=1)
            
            safe_min = tf.maximum(lam_min, 1e-4)
            grad = (safe_min * term_max - lam_max * term_min) / (tf.square(safe_min) + 1e-12)
            return tf.clip_by_value(grad, -100.0, 100.0)

        return tf.cond(is_bad, lambda: 0.0, safe_grad)


    @tf.function
    def integrate_ode(self, v0, H0, dH):
        """
        Integrates the Euler-Lagrange ODE for the schedule.
        The ODE is: d^2(beta)/dt^2 = mu * grad_beta(condition_number).

        Args:
            v0 (float): Initial velocity (slope) of the schedule, u(0).
            H0 (tf.Tensor): Prior information matrix.
            dH (tf.Tensor): Information difference matrix.

        Returns:
            tuple:
                - final_beta (tf.Tensor): The value of beta at t=1.
                - beta_arr (tf.Tensor): The full trajectory of beta values.
                - vel_arr (tf.Tensor): The full trajectory of velocity values (u).
        """
        dt = 1.0 / float(self.steps)
        state = tf.stack([0.0, v0])
        
        beta_arr = tf.TensorArray(tf.float64, size=self.steps+1).write(0, 0.0)
        vel_arr = tf.TensorArray(tf.float64, size=self.steps+1).write(0, v0)
        
        for i in tf.range(self.steps):
            def ode_func(y):
                accel = self.mu * self.compute_kappa_grad(y[0], H0, dH)
                return tf.stack([y[1], accel])
            
            k1 = ode_func(state)
            k2 = ode_func(state + 0.5*dt*k1)
            k3 = ode_func(state + 0.5*dt*k2)
            k4 = ode_func(state + dt*k3)
            state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
            beta_arr = beta_arr.write(i+1, state[0])
            vel_arr = vel_arr.write(i+1, state[1])
            
        return state[0], beta_arr.stack(), vel_arr.stack()


    @tf.function
    def solve(self, P_pred):
        """
        Solves the Boundary Value Problem (BVP) for the optimal schedule.
        Uses a Shooting Method (Secant method) to find the initial velocity v0 
        such that beta(1) = 1.0, given beta(0) = 0.0.

        Args:
            P_pred (tf.Tensor): The predicted covariance matrix from the EKF.

        Returns:
            tuple: (betas, velocities)
                - betas (tf.Tensor): The optimized schedule values. Shape (steps+1,).
                - velocities (tf.Tensor): The rate of change of the schedule. Shape (steps+1,).
        """
        H0 = robust_pinv(P_pred)
        Hh = self.model.R_inv
        dH = Hh - H0

        v_a, v_b = 0.5, 1.5
        b_a, _, _ = self.integrate_ode(v_a, H0, dH)
        b_b, _, _ = self.integrate_ode(v_b, H0, dH)

        if(tf.math.is_nan(b_a) or tf.math.is_nan(b_b)):                              # Check nans
             return tf.linspace(0.0, 1.0, self.steps+1), tf.ones(self.steps+1)

        err_a = b_a - 1.0
        err_b = b_b - 1.0
        
        init = [0, v_a, v_b, err_a, err_b]
        
        def body(i, v_p, v_c, e_p, e_c):
            denom = e_c - e_p
            safe_denom = tf.where(tf.abs(denom) < 1e-6, 1e-6*tf.sign(denom), denom)
            v_new = v_c - e_c * (v_c - v_p) / safe_denom
            v_new = tf.clip_by_value(v_new, 0.1, 10.0)
            
            b_end, _, _ = self.integrate_ode(v_new, H0, dH)
            return i+1, v_c, v_new, e_c, b_end - 1.0

        _, _, v_opt, _, _ = tf.while_loop(lambda i, vp, vc, ep, ec: i < 8, body, init)
        _, betas, vels = self.integrate_ode(v_opt, H0, dH)
        
        valid = tf.math.logical_and(                                               # Validity check
            tf.math.is_finite(betas[-1]),
            tf.abs(betas[-1] - 1.0) < 0.2
        )
        
        return tf.cond(valid,
            lambda: (tf.clip_by_value(betas, 0.0, 1.0), vels),
            lambda: (tf.linspace(0.0, 1.0, self.steps+1), tf.ones(self.steps+1))
        )



class StochPFPF:
    """
    Optimal Homotopy Particle Flow Particle Filter (Stoch-PFPF).
    """
    def __init__(self, model, num_particles=100, num_steps=20, mu=0.1):
        """
        Initializes the Stoch-PFPF filter.

        Args:
            model: The dynamic system model.
            num_particles (int): Number of particles.
            num_steps (int): Number of flow integration steps.
            mu (float): Weighting factor for the schedule optimization.
        """
        self.model = model
        self.N = num_particles
        self.steps = num_steps
        self.homotopy_solver = HomotopySolver(model, mu, num_steps)


    @tf.function
    def run_step(self, particles, weights, m_ekf, P_ekf, y_curr):
        """
        Executes one time step of the filter.

        Args:
            particles (tf.Tensor): Current particles (N, K).
            weights (tf.Tensor): Current weights (N,).
            m_ekf (tf.Tensor): Auxiliary EKF mean.
            P_ekf (tf.Tensor): Auxiliary EKF covariance.
            y_curr (tf.Tensor): Current observation.

        Returns:
            tuple: Updated (particles, weights, m_ekf, P_ekf, estimate, ess, average_condition_num).
        """
        ekf = AuxiliaryEKF(self.model)                                            # EKF Pref
        m_pred, P_pred = ekf.predict(m_ekf, P_ekf)

        betas, beta_dots = self.homotopy_solver.solve(P_pred)                      # Solve for optimal homotopy
        dt = 1.0 / float(self.steps)
        
        particles_prop = self.model.transition(particles)
        H0 = robust_pinv(P_pred)                                                   # Matrices for Flow
        Hh = self.model.R_inv
        dH = Hh - H0
        
        Hh_y = tf.linalg.matvec(Hh, y_curr)
        I = tf.eye(self.model.K)
        log_det_J = tf.zeros(self.N)
        total_cond = 0.0
        
        for k in tf.range(self.steps):                                             # Migrate particles
            beta = betas[k]
            beta_dot = beta_dots[k]
            
            M = H0 + beta * dH + 1e-6 * I
            M_inv = robust_pinv(M)
            A = -0.5 * beta_dot * M_inv @ Hh
            b_vec = -1.0 * (A @ tf.expand_dims(y_curr, -1))
            b_vec = tf.squeeze(b_vec)

            drift = tf.transpose(tf.matmul(A, particles_prop, transpose_b=True)) + b_vec
            drift = tf.clip_by_value(drift, -100.0, 100.0)                           # Clip to prevent explosion
            
            particles_prop = particles_prop + dt * drift

            with tf.device("/CPU:0"):                                                # Get metrics
                mat_step = I + dt * A
                log_det_J += tf.math.log(tf.abs(tf.linalg.det(mat_step)) + 1e-12)
                s = tf.linalg.svd(M, compute_uv=False)
                total_cond += tf.reduce_max(s) / (tf.reduce_min(s) + 1e-9)

        obs_dist = tfd.MultivariateNormalDiag(loc=particles_prop, scale_diag=tf.sqrt(self.model.R_diag))       # Update weights
        log_lik = obs_dist.log_prob(y_curr)
        log_w = tf.math.log(weights + 1e-12) + log_lik + log_det_J
        log_w = tf.where(tf.math.is_finite(log_w), log_w, -1e9 * tf.ones_like(log_w))
        
        w_unnorm = tf.exp(log_w - tf.reduce_max(log_w))
        weights = w_unnorm / (tf.reduce_sum(w_unnorm) + 1e-12)
        ess = 1.0 / (tf.reduce_sum(weights**2) + 1e-9)
        

        def resample():                                                                                         # Resample if low ESS :(
            idxs = tf.random.categorical(tf.reshape(log_w, [1, -1]), self.N)[0]
            return tf.gather(particles_prop, idxs), tf.fill([self.N], 1.0/float(self.N))

        particles_final, weights_final = tf.cond(ess < self.N/2.0, resample, lambda: (particles_prop, weights))
        m_upd, P_upd = ekf.update(m_pred, P_pred, y_curr)
        est = tf.reduce_sum(particles_final * weights_final[:, None], axis=0)
        
        return particles_final, weights_final, m_upd, P_upd, est, ess, total_cond / float(self.steps)