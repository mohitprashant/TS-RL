import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from lorenz96 import Lorenz96Model
from flow_base import AuxiliaryEKF
from edh import EDHFilter
from ledh import LEDHFilter
from edh_inv_pf import PFPF_EDHFilter
from ledh_inv_pf import PFPF_LEDHFilter
from kernel_pf import PFF, ParticleFlowKernel


@pytest.fixture
def model_params():
    return {
        'K': 10,
        'F': 8.0,
        'dt': 0.05,
        'process_std': 0.1,
        'obs_std': 1.0
    }

@pytest.fixture
def lorenz_model(model_params):
    return Lorenz96Model(**model_params)

@pytest.fixture
def initial_state(model_params):
    K = model_params['K']
    return tf.random.normal([K])

@pytest.fixture
def particle_cloud(model_params):
    N = 20
    K = model_params['K']
    return tf.random.normal([N, K])


def test_lorenz96_rk4_integration(lorenz_model, initial_state):
    """Test that RK4 integration changes the state."""
    next_state = lorenz_model.rk4_step(initial_state)
    assert next_state.shape == initial_state.shape
    assert not np.allclose(next_state.numpy(), initial_state.numpy()), "State should evolve over time"


def test_lorenz96_jacobian_shape(lorenz_model, initial_state):
    """Test Jacobian calculation shape."""
    x_batch = tf.expand_dims(initial_state, 0) # (1, K)
    J = lorenz_model.get_jacobian(x_batch)
    K = lorenz_model.K
    assert J.shape == (1, K, K)


def test_lorenz96_transition_stochasticity(lorenz_model, initial_state):
    """Test that transition adds noise."""
    tf.random.set_seed(42)
    s1 = lorenz_model.transition(initial_state)
    tf.random.set_seed(42)
    s2 = lorenz_model.transition(initial_state)
    
    assert np.allclose(s1.numpy(), s2.numpy())                             # Same seed should produce same result

    s3 = lorenz_model.transition(initial_state)                            # Different call should likely be different (without seed reset)
    assert not np.allclose(s1.numpy(), s3.numpy())



def test_auxiliary_ekf_update(lorenz_model, initial_state):
    ekf = AuxiliaryEKF(lorenz_model)
    K = lorenz_model.K
    m = initial_state
    P = tf.eye(K)
    y = initial_state + 0.1
    m_upd, P_upd = ekf.update(m, P, y)
    
    assert m_upd.shape == (K,)
    assert P_upd.shape == (K, K)
    assert tf.linalg.trace(P_upd) < tf.linalg.trace(P)                    # Covariance should shrink after update


@pytest.mark.parametrize("kernel_type", ['scalar', 'matrix'])
def test_kernel_pff_update(kernel_type, lorenz_model, particle_cloud):
    """Test the Kernel PFF implementation."""
    N, K = particle_cloud.shape
    particles_t = tf.transpose(particle_cloud)
    pff = PFF(n_particles=N, dim=K, kernel_type=kernel_type, model=lorenz_model)
    y = tf.reduce_mean(particles_t, axis=1) 
    R_val = tf.constant(1.0)
    updated_particles = pff.update(particles_t, y, R_val)
    
    assert updated_particles.shape == (K, N)
    assert not np.allclose(updated_particles.numpy(), particles_t.numpy())


def test_kernel_drift_computation(lorenz_model):
    """Unit test specifically for the kernel drift function."""
    dim = 5
    N = 10
    particles = tf.random.normal([dim, N])
    grad_log_p = tf.random.normal([dim, N])
    B_diag = tf.ones([dim])
    
    kernel_s = ParticleFlowKernel(alpha=0.1, kernel_type='scalar')                     # Test Scalar Kernel
    drift_s = kernel_s.compute_drift(particles, grad_log_p, B_diag)
    assert drift_s.shape == (dim, N)
    
    kernel_m = ParticleFlowKernel(alpha=0.1, kernel_type='matrix')                     # Test Matrix Kernel
    drift_m = kernel_m.compute_drift(particles, grad_log_p, B_diag)
    assert drift_m.shape == (dim, N)