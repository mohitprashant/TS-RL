import pytest
import tensorflow as tf
import numpy as np
from kalman import KalmanFilter
from lgssm import LinearGaussianSSM



@pytest.fixture
def ssm_params():
    """Provides standard parameters for a 1D constant velocity model."""
    dt = 0.1
    A = [[1.0, dt], [0.0, 1.0]]
    B = [[0.5 * dt**2, 0.0], [dt, 0.0]]
    C = [[1.0, 0.0]]
    D = [[0.5]]
    Sigma_init = [[1.0, 0.0], [0.0, 1.0]]
    return A, B, C, D, Sigma_init


def test_lgssm_simulation_shapes(ssm_params):
    """Verify that the LGSSM simulation returns correct tensor shapes."""
    A, B, C, D, Sigma_init = ssm_params
    T = 20
    ssm = LinearGaussianSSM(A, B, C, D, Sigma_init)
    states, observations = ssm.simulate(T)
    
    assert states.shape == (T, 2)
    assert observations.shape == (T, 1)


def test_kalman_filter_dimensions(ssm_params):
    """Check if Kalman Filter initialization correctly identifies dimensions."""
    A, B, C, D, Sigma_init = ssm_params
    kf = KalmanFilter(A, B, C, D, Sigma_init)
    
    assert kf.nx == 2
    assert kf.ny == 1
    assert kf.Q.shape == (2, 2)
    assert kf.R.shape == (1, 1)


def test_kalman_filter_stability_metrics(ssm_params):
    """Ensure condition numbers are positive and logical."""
    A, B, C, D, Sigma_init = ssm_params
    kf = KalmanFilter(A, B, C, D, Sigma_init)
    
    obs = tf.constant([[1.0], [1.1]], dtype=tf.float32)
    means, covs, cond_S, cond_P = kf.run_filter(obs)
    
    assert tf.reduce_all(cond_S >= 1.0)
    assert tf.reduce_all(cond_P >= 1.0)


def test_prediction_step_logic(ssm_params):
    """Manually check one prediction step for a simple case."""
    A, B, C, D, Sigma_init = ssm_params
    kf = KalmanFilter(A, B, C, D, Sigma_init)
    
    x_init = tf.constant([10.0, 1.0], dtype=tf.float32) # pos=10, vel=1
    P_init = tf.eye(2)
    
    x_pred, P_pred = kf.predict(x_init, P_init)
    
    # Expected position: 10 + (0.1 * 1) = 10.1
    assert x_pred.numpy()[0] == pytest.approx(10.1)
    # Covariance should increase after prediction due to process noise Q
    assert tf.linalg.trace(P_pred) > tf.linalg.trace(P_init)


def test_update_step_reduces_uncertainty(ssm_params):
    """The state covariance P should generally decrease after a measurement update."""
    A, B, C, D, Sigma_init = ssm_params
    kf = KalmanFilter(A, B, C, D, Sigma_init)
    
    x_pred = tf.constant([0.0, 0.0], dtype=tf.float32)
    P_pred = tf.eye(2) * 2.0
    y_obs = tf.constant([0.5], dtype=tf.float32)
    
    x_new, P_new, _, _ = kf.update(x_pred, P_pred, y_obs)
    
    # Measurement should reduce uncertainty (trace of covariance)
    assert tf.linalg.trace(P_new) < tf.linalg.trace(P_pred)