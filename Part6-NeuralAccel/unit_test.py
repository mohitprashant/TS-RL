import pytest
import tensorflow as tf
import numpy as np

from svssm import StochasticVolatilityModel
from gradnetot_analysis import ConditionalGNM, CondGradNet, GradNetParticleFilter
from deeponet_analysis import GNM_Module, DeepONetGradNet, NeuralParticleFilter


@pytest.fixture
def svssm_params():
    """Standard parameters for the Stochastic Volatility Model."""
    return {
        "alpha": 0.91,
        "sigma": 1.0,
        "beta": 0.5,
        "T": 20,    
        "N": 50        
    }

@pytest.fixture
def synthetic_data(svssm_params):
    """Generates synthetic data (True States and Observations)."""
    model = StochasticVolatilityModel(
        svssm_params["alpha"], 
        svssm_params["sigma"], 
        svssm_params["beta"]
    )
    states, obs = model.simulate(svssm_params["T"])
    return model, states, obs


def test_conditional_gnm_layer_shapes():
    """Verify Conditional GNM layer handles input/context shapes correctly."""
    batch_size = 10
    embed_dim = 32
    x = tf.random.normal((batch_size, 1))
    y = tf.random.normal((batch_size, 1))

    layer = ConditionalGNM(embed_dim=embed_dim)
    output = layer(x, y)
    assert output.shape == (batch_size, 1)
    assert len(layer.trainable_variables) > 0


def test_cond_gradnet_forward_pass():
    """Verify the full CondGradNet model forward pass."""
    batch_size = 10
    num_modules = 2
    x = tf.random.normal((batch_size, 1))
    y = tf.random.normal((batch_size, 1))
    model = CondGradNet(num_modules=num_modules)
    output = model([x, y])
    assert output.shape == (batch_size, 1)


def test_gradnet_pf_pretraining_logic(synthetic_data):
    """Ensure GradNetParticleFilter pretraining runs without error."""
    model, _, _ = synthetic_data
    pf = GradNetParticleFilter(num_particles=20, lr=0.01)
    try:
        pf.pretrain(model, steps=2, batch_size=16)
    except Exception as e:
        pytest.fail(f"GradNet pretraining failed with error: {e}")


def test_gradnet_pf_execution(synthetic_data):
    """Verify GradNetParticleFilter runs on data and returns estimates."""
    model, _, obs = synthetic_data
    pf = GradNetParticleFilter(num_particles=20)
    pf.net([tf.zeros((1,1)), tf.zeros((1,1))])
    est = pf.run(model, obs)
    # assert est.shape == (obs.shape[0],)
    assert not tf.reduce_any(tf.math.is_nan(est))


def test_deeponet_trunk_basis_module():
    """Test the GNM_Module (Trunk Basis) shapes."""
    batch_size = 10
    x = tf.random.normal((batch_size, 1))
    module = GNM_Module(embed_dim=16)
    output = module(x)
    assert output.shape == (batch_size, 1)


def test_deeponet_full_architecture():
    """Test the DeepONet merging Trunk (x) and Branch (y) outputs."""
    batch_size = 10
    num_basis = 5
    x = tf.random.normal((batch_size, 1))
    y = tf.random.normal((batch_size, 1))
    net = DeepONetGradNet(num_basis=num_basis, embed_dim=16)
    output = net([x, y])
    assert output.shape == (batch_size, 1)



def test_deeponet_variable_particle_counts(synthetic_data):
    """
    Verify DeepONet can run with different N than trained.
    This tests the 'Operator Generalization' capability.
    """
    model, _, obs = synthetic_data
    pf = NeuralParticleFilter(net_type='deeponet', num_particles=10)
    pf.pretrain(model, steps=1)
    
    est_10 = pf.run(model, obs, num_particles=10)
    assert est_10.shape == (obs.shape[0],)
    
    est_50 = pf.run(model, obs, num_particles=50)
    assert est_50.shape == (obs.shape[0],)
    assert not np.allclose(est_10.numpy(), est_50.numpy())

