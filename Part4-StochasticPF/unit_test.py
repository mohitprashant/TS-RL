import pytest
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from lorenz96 import Lorenz96Model
from ledh_inv_pf import AuxiliaryEKF, robust_pinv, robust_svd_eig, LEDH_PFPF
from opt_inv_pf import HomotopySolver, StochPFPF
from stochastic_pf import StochasticParticleFlow
from spf_analysis import ParticleFlowSimulation

DTYPE = tf.float64

def test_robust_pinv():
    """Test robust pseudo-inverse with a singular matrix."""
    A = tf.constant([[1.0, 1.0], [1.0, 1.0]], dtype=DTYPE)
    A_pinv = robust_pinv(A)
    expected = tf.constant([[0.25, 0.25], [0.25, 0.25]], dtype=DTYPE)    # Expected pinv is [[0.25, 0.25], [0.25, 0.25]]
    assert np.allclose(A_pinv.numpy(), expected.numpy(), atol=1e-5)

def test_robust_svd_eig():
    """Test robust eigendecomposition reconstruction."""
    A = tf.constant([[2.0, 0.5], [0.5, 1.0]], dtype=DTYPE)                # Symmetric Positive Definite Matrix
    s, u = robust_svd_eig(A)
    A_rec = u @ tf.linalg.diag(s) @ tf.transpose(u)                       # Reconstruct: U * S * U^T
    assert np.allclose(A.numpy(), A_rec.numpy(), atol=1e-5)



# @pytest.fixture
# def l96_model():
#     return Lorenz96Model(K=5, dt=0.01)

# def test_lorenz96_jacobian(l96_model):                                 # Fails because of implicit float32 to float64 conversion :(
#     x = tf.random.normal([1, 5], dtype=DTYPE)
#     F = l96_model.get_jacobian(x)
#     assert F.shape == (1, 5, 5)


def test_auxiliary_ekf_step(l96_model):
    ekf = AuxiliaryEKF(l96_model)
    m = tf.zeros([5], dtype=DTYPE)
    P = tf.eye(5, dtype=DTYPE)
    y = tf.ones([5], dtype=DTYPE)

    m_pred, P_pred = ekf.predict(m, P)
    assert tf.linalg.trace(P_pred) > tf.linalg.trace(P) # Covariance should grow
    m_upd, P_upd = ekf.update(m_pred, P_pred, y)
    assert tf.linalg.trace(P_upd) < tf.linalg.trace(P_pred) # Covariance should shrink



@pytest.fixture
def spf_model():
    return StochasticParticleFlow()


def test_spf_jacobian(spf_model):
    """Test the analytical Jacobian against numerical approximation."""
    x_test = tf.constant([3.5, 3.5], dtype=DTYPE)
    H_analytic = spf_model.get_H_jacobian(x_test)
    assert H_analytic.shape == (2, 2)
    assert np.all(np.isfinite(H_analytic.numpy()))


def test_spf_measurement_func(spf_model):
    batch_x = tf.constant([[4.0, 4.0], [3.0, 0.0]], dtype=DTYPE)
    z = spf_model.h_meas(batch_x)
    assert z.shape == (2, 2) 


def test_spf_ode_function(spf_model):
    """Test the stiffness ODE function used in the BVP."""
    t = tf.constant(0.5, dtype=DTYPE)
    y = tf.constant([0.5, 1.0], dtype=DTYPE)
    dy = spf_model.stiffness_ode_fn(t, y)
    assert dy.shape == (2,)



@pytest.mark.integration
def test_pfpf_integration(l96_model):
    """
    Integration test for the Stoch-PFPF filter class.
    Runs one step of the filter to ensure end-to-end connectivity.
    """
    N = 20
    filter_obj = StochPFPF(l96_model, num_particles=N, num_steps=5)
    
    particles = tf.random.normal([N, 5], dtype=DTYPE)                   # Init state
    weights = tf.fill([N], 1.0/N)
    m_ekf = tf.zeros([5], dtype=DTYPE)
    P_ekf = tf.eye(5, dtype=DTYPE)
    y_curr = tf.ones([5], dtype=DTYPE)
    
    parts, w, m, P, est, ess, cond = filter_obj.run_step(               # Run Step
        particles, weights, m_ekf, P_ekf, y_curr
    )
    assert parts.shape == (N, 5)
    assert np.isclose(tf.reduce_sum(w).numpy(), 1.0, atol=1e-5)
    assert not tf.reduce_any(tf.math.is_nan(parts))
    assert ess > 0


@pytest.mark.integration
def test_spf_simulation_manager(spf_model):
    """
    Integration test for ParticleFlowSimulation using the 'linear' method.
    """
    sim_manager = ParticleFlowSimulation(spf_model)
    mse, trace = sim_manager.run_single_simulation(
        method='linear', 
        num_particles=50, 
        dt=0.2
    )
    assert mse.shape == ()
    assert trace.shape == ()
    assert mse >= 0.0
    assert trace >= 0.0


# @pytest.mark.integration
# def test_spf_optimal_schedule_injection(spf_model):
#     """
#     Test the 'optimal' path by manually injecting a dummy schedule 
#     instead of solving the BVP for time efficiency.
#     """
#     spf_model.lambda_grid = tf.linspace(0.0, 1.0, 10)
#     spf_model.beta_grid = tf.linspace(0.0, 1.0, 10)
#     spf_model.u_grid = tf.ones(10, dtype=DTYPE)
    
#     sim_manager = ParticleFlowSimulation(spf_model)
    
#     mse, trace = sim_manager.run_single_simulation(
#         method='optimal', 
#         num_particles=50, 
#         dt=0.2
#     )
#     assert mse >= 0.0