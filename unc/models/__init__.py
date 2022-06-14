import jax.numpy as jnp
import haiku as hk
from functools import partial

from .q_network import QNetwork, nn
from .gvfn import gvfn
from .noisy import noisy_network
from .lstm import lstm, value
from .cnn import cnn
from .cnn_lstm import cnn_lstm

def get_network(hidden_size: int, output_size: int, x: jnp.ndarray):
    q = QNetwork(hidden_size, output_size)
    return q(x)


def build_network(hidden_size: int, output_size: int,
                  model_str: str = 'nn', with_bias: bool = True,
                  init: str = 'fan_avg', n_actions_gvfs: int = None,
                  layers: int = 1):
    """
    with_bias: only set to false if we're in the linear LFA w/ g.t. states. (TABULAR)
    gvf_feature_idxes: indices of gvf features.
    """
    # q_network_fn = QNetwork(hidden_size, output_size)
    # network = hk.without_apply_rng(hk.transform(q_network_fn))
    # return network
    hidden_layers = []

    if model_str != 'linear':
        for _ in range(layers):
            hidden_layers.append(hidden_size)

    if init == 'fan_avg':
        init = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')
    elif init == 'zero':
        init = hk.initializers.Constant(0)

    if model_str == "noisy":
        network_fn = partial(noisy_network, hidden_layers, output_size)
        network = hk.transform(network_fn)
    elif (model_str == 'nn' or model_str == 'linear') and n_actions_gvfs is not None:
        network_fn = partial(gvfn, hidden_layers, n_actions_gvfs, output_size,
                             with_bias=with_bias, init=init)
        network = hk.without_apply_rng(hk.transform(network_fn))
    elif model_str == 'nn' or model_str == 'linear':
        network_fn = partial(nn, hidden_layers, output_size,
                             with_bias=with_bias, init=init)
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
    elif model_str == 'cnn_lstm':
        network_fn = partial(cnn_lstm, hidden_size, output_size)
        network = hk.without_apply_rng(hk.transform(network_fn))
    else:
        raise NotImplementedError

    return network


