import pytest
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from svssm import StochasticVolatilityModel
from sinkhorn import SinkhornParticleFilter
from soft_resample import SoftResamplingParticleFilter
from opr import OptimalPlacementParticleFilter
from cnf import CNFParticleFilter

tfd = tfp.distributions

@pytest.fixture
def model_params():
    """Default parameters for the SVSSM model."""
    return {
        "alpha": 0.91,
        "sigma": 1.0,
        "beta": 0.5,
        "T": 20, 
        "N": 50 
    }

@pytest.fixture
def synthetic_data(model_params):
    """Generates a small batch of synthetic data for testing."""
    model = StochasticVolatilityModel(
        model_params["alpha"], 
        model_params["sigma"], 
        model_params["beta"]
    )
    states, obs = model.simulate(model_params["T"])
    return states, obs

@pytest.fixture
def initial_vars(model_params):
    """Returns differentiable variables for filter initialization."""
    a = tf.Variable(model_params["alpha"], dtype=tf.float32)
    s = tf.Variable(model_params["sigma"], dtype=tf.float32)
    b = tf.Variable(model_params["beta"], dtype=tf.float32)
    return a, s, b



FILTER_CLASSES = [
    (SinkhornParticleFilter, {'epsilon': 0.5, 'n_iter': 5}),
    (SoftResamplingParticleFilter, {'soft_alpha': 0.5}),
    (OptimalPlacementParticleFilter, {}),
    (CNFParticleFilter, {})
]


@pytest.mark.parametrize("filter_cls, kwargs", FILTER_CLASSES)
def test_filter_initialization(initial_vars, model_params, filter_cls, kwargs):
    """Test that filters initialize correctly."""
    a, s, b = initial_vars
    pf = filter_cls(a, s, b, num_particles=model_params['N'], **kwargs)
    
    assert pf.num_particles == model_params['N']
    # Check if initial distribution is created
    d_init = pf.initial_dist()
    assert isinstance(d_init, tfp.distributions.Normal)


@pytest.mark.parametrize("filter_cls, kwargs", FILTER_CLASSES)
def test_filter_single_step(initial_vars, model_params, filter_cls, kwargs):
    """Test a single .step() call for tensor shape correctness."""
    a, s, b = initial_vars
    N = model_params['N']
    pf = filter_cls(a, s, b, num_particles=N, **kwargs)
    particles = tf.zeros((N,), dtype=tf.float32)
    log_weights = tf.fill((N,), -tf.math.log(float(N)))
    y_obs = tf.constant(0.1, dtype=tf.float32)
    
    # Run step
    new_particles, new_log_weights, extra = pf.step(particles, log_weights, y_obs)
    
    assert new_particles.shape == (N,)
    assert new_log_weights.shape == (N,)
    assert tf.reduce_any(tf.math.is_finite(new_log_weights))




# --- Integration Tests: Differentiability & Full Loop ---

@pytest.mark.parametrize("filter_cls, kwargs", FILTER_CLASSES)
def test_full_filter_loop_and_metrics(initial_vars, synthetic_data, model_params, filter_cls, kwargs):
    """
    Integration Test:
    Run the filter for T steps and verify RMSE/ESS metrics are calculated.
    """
    a, s, b = initial_vars
    true_x, obs = synthetic_data
    pf = filter_cls(a, s, b, num_particles=model_params['N'], **kwargs)
    estimates = []
    particles = pf.initial_dist().sample(model_params['N'])
    log_weights = tf.fill([model_params['N']], -tf.math.log(float(model_params['N'])))
    
    for t in range(model_params['T']):
        particles, log_weights, _ = pf.step(particles, log_weights, obs[t])
        w = tf.exp(log_weights - tf.reduce_logsumexp(log_weights))
        est = tf.reduce_sum(particles * w)
        estimates.append(est)
        
    estimates = tf.stack(estimates)
    rmse = tf.sqrt(tf.reduce_mean((estimates - true_x)**2))
    
    assert estimates.shape == (model_params['T'],)
    assert rmse >= 0.0
    assert not tf.math.is_nan(rmse)



@pytest.mark.parametrize("filter_cls, kwargs", FILTER_CLASSES)
def test_differentiability(initial_vars, synthetic_data, model_params, filter_cls, kwargs):
    """
    Integration Test:
    CRITICAL: Verify that gradients flow through the filter step back to parameters.
    """
    a, s, b = initial_vars
    true_x, obs = synthetic_data
    pf = filter_cls(a, s, b, num_particles=model_params['N'], **kwargs)
    
    with tf.GradientTape() as tape:
        T_short = 5
        estimates = []
        particles = pf.initial_dist().sample(model_params['N'])
        log_weights = tf.fill([model_params['N']], -tf.math.log(float(model_params['N'])))
        
        for t in range(T_short):
            particles, log_weights, _ = pf.step(particles, log_weights, obs[t])
            w = tf.exp(log_weights - tf.reduce_logsumexp(log_weights))
            est = tf.reduce_sum(particles * w)
            estimates.append(est)
            
        est_tensor = tf.stack(estimates)
        loss = tf.reduce_mean((est_tensor - true_x[:T_short])**2)
        
    grads = tape.gradient(loss, [a, s, b])
    assert len(grads) == 3
    for g, name in zip(grads, ['alpha', 'sigma', 'beta']):
        assert g is not None, f"Gradient for {name} was None (graph broken)"
        assert not tf.reduce_any(tf.math.is_nan(g)), f"Gradient for {name} contained NaNs"