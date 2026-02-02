import pytest
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from msvssm import MultivariateStochasticVolatilityModel
from improved_sinkhorn import SinkhornParticleFilter, ImprovedSinkhornParticleFilter

tf.random.set_seed(42)
np.random.seed(42)



@pytest.fixture
def msv_params():
    """Parameters for a small 2D multivariate model."""
    p = 2
    phi = tf.constant([0.9, 0.8], dtype=tf.float32)
    beta = tf.constant([0.5, 0.5], dtype=tf.float32)
    sigma_eps = tf.eye(p) * 0.5
    sigma_eta = tf.eye(p) * 0.1             # valid covariance matrix (positive definite)
    top = tf.concat([sigma_eps, tf.zeros((p, p))], axis=1)
    bot = tf.concat([tf.zeros((p, p)), sigma_eta], axis=1)
    sigma_full = tf.concat([top, bot], axis=0)
    return phi, sigma_full, beta, p

@pytest.fixture
def model_and_data(msv_params):
    """Returns an initialized model and synthetic data."""
    phi, sigma, beta, p = msv_params
    model = MultivariateStochasticVolatilityModel(phi, sigma, beta)
    states, obs = model.simulate(T=20)
    return model, states, obs



def test_msv_initialization(msv_params):
    """Verify model splits the covariance matrix correctly."""
    phi, sigma, beta, p = msv_params
    model = MultivariateStochasticVolatilityModel(phi, sigma, beta)
    
    assert model.p == p
    assert model.sigma_eps.shape == (p, p)
    assert model.sigma_eta.shape == (p, p)
    assert model.chol_eps.shape == (p, p)
    assert model.chol_eta.shape == (p, p)


def test_msv_simulation_shapes(model_and_data):
    """Verify simulation output dimensions."""
    model, states, obs = model_and_data
    T = 20
    p = model.p
    assert states.shape == (T, p)
    assert obs.shape == (T, p)
    assert not tf.reduce_any(tf.math.is_nan(states))


def test_msv_distributions(model_and_data):
    """Verify distribution methods return valid TFP objects."""
    model, _, _ = model_and_data
    p = model.p
    d0 = model.initial_dist()
    assert isinstance(d0, tfp.distributions.MultivariateNormalDiag)
    sample0 = d0.sample()
    assert sample0.shape == (p,)
    
    x_prev = tf.zeros((10, p))
    d_trans = model.transition_dist(x_prev)
    assert d_trans.event_shape == (p,)
    assert d_trans.batch_shape == (10,)

    x_curr = tf.zeros((10, p))
    d_obs = model.observation_dist(x_curr)
    assert d_obs.event_shape == (p,)


def test_euclidean_cost():
    """Verify squared Euclidean distance calculation."""
    pf = SinkhornParticleFilter(num_particles=2)
    # Particles: [0, 0] and [3, 4]
    particles = tf.constant([[0.0, 0.0], [3.0, 4.0]], dtype=tf.float32)
    
    cost = pf.compute_cost(particles)
    expected = np.array([[0.0, 25.0], [25.0, 0.0]])
    np.testing.assert_allclose(cost.numpy(), expected, atol=1e-5)


def test_mahalanobis_cost():
    """Verify Mahalanobis cost scales dimensions correctly."""
    precision = tf.constant([[2.0, 0.0], [0.0, 1.0]], dtype=tf.float32)
    pf = ImprovedSinkhornParticleFilter(precision_matrix=precision, num_particles=2)
    particles = tf.constant([[0.0, 0.0], [1.0, 1.0]], dtype=tf.float32)
    cost = pf.compute_cost(particles)
    expected_val = 3.0
    assert cost[0, 1].numpy() == pytest.approx(expected_val)
    assert cost[1, 0].numpy() == pytest.approx(expected_val)


def test_sinkhorn_resample_output_shape(msv_params):
    """Check that resampling returns correct particle tensor shape."""
    phi, sigma, beta, p = msv_params
    N = 10
    pf = SinkhornParticleFilter(num_particles=N, epsilon=0.1, n_iter=5)
    particles = tf.random.normal((N, p))
    log_weights = tf.zeros((N,))
    new_particles = pf.sinkhorn_resample(particles, log_weights)
    assert new_particles.shape == (N, p)
    assert not tf.reduce_any(tf.math.is_nan(new_particles))


@pytest.mark.integration
def test_filter_runs_end_to_end(model_and_data):
    """
    Run both Standard and Improved filters on the model.
    Verify they complete and return trajectories of correct shape.
    """
    model, _, obs = model_and_data
    T = obs.shape[0]
    p = model.p
    N = 20
    
    pf_std = SinkhornParticleFilter(num_particles=N, epsilon=0.5, n_iter=5)
    est_std = pf_std.run(model, obs)
    assert est_std.shape == (T, p)

    pf_imp = ImprovedSinkhornParticleFilter(model.precision_eta, num_particles=N, epsilon=0.5, n_iter=5)
    est_imp = pf_imp.run(model, obs)
    assert est_imp.shape == (T, p)


@pytest.mark.integration
def test_differentiability_check(msv_params, model_and_data):
    """
    Ensure we can backprop through the Improved Sinkhorn Filter.
    This is critical for the 'Differentiable' aspect.
    """
    phi_val, sigma, beta_val, _ = msv_params
    _, _, obs = model_and_data
    
    phi_var = tf.Variable(phi_val)
    beta_var = tf.Variable(beta_val)
    
    with tf.GradientTape() as tape:
        model = MultivariateStochasticVolatilityModel(phi_var, sigma, beta_var)
        pf = ImprovedSinkhornParticleFilter(model.precision_eta, num_particles=10, n_iter=5)
        
        est = pf.run(model, obs)
        loss = tf.reduce_mean(tf.square(est))
        
    grads = tape.gradient(loss, [phi_var, beta_var])
    
    assert grads[0] is not None
    assert grads[1] is not None
    assert not tf.reduce_any(tf.math.is_nan(grads[0]))