import jax.numpy as jnp
import haiku as hk
from functools import partial

from .q_network import QNetwork, nn
from .noisy import noisy_network
from .lstm import lstm, value
from .cnn import cnn

def get_network(hidden_size: int, output_size: int, x: jnp.ndarray):
    q = QNetwork(hidden_size, output_size)
    return q(x)


def build_network(hidden_size: int, output_size: int,
                  model_str: str = 'nn'):
    """
    TODO: Do we want a bias unit with our LFA?
    Currently yes. But if we do some stuff with TC then we need to change this.
    """
    # q_network_fn = QNetwork(hidden_size, output_size)
    # network = hk.without_apply_rng(hk.transform(q_network_fn))
    # return network
    layers = [hidden_size]

    if model_str == 'linear':
        layers = []

    if model_str == "noisy":
        network_fn = partial(noisy_network, layers, output_size)
        network = hk.transform(network_fn)
    elif model_str == 'nn' or model_str == 'linear':
        network_fn = partial(nn, layers, output_size)
        network = hk.without_apply_rng(hk.transform(network_fn))
    elif model_str == 'lstm':
        network_fn = partial(lstm, hidden_size, output_size)
        network = hk.without_apply_rng(hk.transform(network_fn))
    elif model_str == 'seq_value':
        network_fn = partial(value, hidden_size, output_size)
        network = hk.without_apply_rng(hk.transform(network_fn))
    elif model_str == 'cnn':
        network_fn = partial(cnn, hidden_size, output_size)
        network = hk.without_apply_rng(hk.transform(network_fn))
    else:
        raise NotImplementedError

    return network


