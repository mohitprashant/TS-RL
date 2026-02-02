import pytest
import tensorflow as tf
import numpy as np
from svssm import StochasticVolatilityModel
from ekf import ExtendedKalmanFilter
from ukf import UnscentedKalmanFilter
from particle_filter import ParticleFilter


@pytest.fixture
def model_params():
    """Standard parameters for the Stochastic Volatility Model."""
    return {
        "alpha": 0.91,
        "sigma": 1.0,
        "beta": 0.5,
        "T": 30
    }


def test_svssm_simulation_output(model_params):
    """Verify the Stochastic Volatility Model produces correct tensor shapes."""
    sv = StochasticVolatilityModel(model_params["alpha"], model_params["sigma"], model_params["beta"])
    states, observations = sv.simulate(model_params["T"])
    
    assert states.shape == (model_params["T"],)
    assert observations.shape == (model_params["T"],)
    assert not tf.reduce_any(tf.math.is_nan(states))


def test_ekf_performance(model_params):
    """Verify EKF runs and returns expected dictionary keys."""
    sv = StochasticVolatilityModel(**{k: v for k, v in model_params.items() if k != 'T'})
    true_x, obs = sv.simulate(model_params["T"])
    ekf = ExtendedKalmanFilter(model_params["alpha"], model_params["sigma"], model_params["beta"])
    results = ekf.run_filter(obs, true_x)
    
    assert results['label'] == 'EKF'
    assert results['rmse'] >= 0.0
    assert len(results['estimates']) == model_params["T"]
    assert results['time'] > 0


def test_ukf_sigma_points(model_params):
    """Ensure UKF generates the correct number of sigma points (2n+1)."""
    ukf = UnscentedKalmanFilter(model_params["alpha"], model_params["sigma"], model_params["beta"])
    x = tf.constant(0.5, dtype=tf.float32)
    P = tf.constant(1.0, dtype=tf.float32)
    
    sig_pts = ukf.generate_sigma_points(x, P)
    # n=1 for this model, so 2(1)+1 = 3 sigma points
    assert sig_pts.shape == (3,)


def test_particle_filter_resampling(model_params):
    """Verify Particle Filter calculates ESS and respects thresholds."""
    N = 100
    pf = ParticleFilter(model_params["alpha"], model_params["sigma"], model_params["beta"], 
                        num_particles=N, resample_threshold_ratio=0.5)
    
    sv = StochasticVolatilityModel(**{k: v for k, v in model_params.items() if k != 'T'})
    true_x, obs = sv.simulate(10)
    results = pf.run_filter(obs, true_x)
    
    assert 'ess_avg' in results
    assert results['ess_avg'] <= N
    assert results['particles'] == N


@pytest.mark.parametrize("filter_class", [ExtendedKalmanFilter, UnscentedKalmanFilter])
def test_deterministic_filters_consistency(filter_class, model_params):
    """Ensure filters handle zero-observation edge cases gracefully."""
    f = filter_class(model_params["alpha"], model_params["sigma"], model_params["beta"])
    obs = tf.zeros((5,), dtype=tf.float32)
    
    # Running without true_states should set RMSE to 0.0 instead of crashing
    results = f.run_filter(obs)
    assert results['rmse'] == 0.0

