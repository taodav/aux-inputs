import jax.numpy as jnp
import haiku as hk
from functools import partial

from .q_network import QNetwork, nn
from .gvfn import mult_action_gvfn
from .noisy import noisy_network
from .lstm import lstm, value
from .cnn import cnn
from .cnn_lstm import cnn_lstm
from .actor_critic import actor_nn, critic_nn, actor_cnn, critic_cnn

def get_network(hidden_size: int, output_size: int, x: jnp.ndarray):
    q = QNetwork(hidden_size, output_size)
    return q(x)


def build_network(hidden_size: int, output_size: int,
                  model_str: str = 'nn', with_bias: bool = True,
                  init: str = 'fan_avg', n_predictions: int = 0,
                  action_cond: str = None,
                  layers: int = 1,
                  tile_code_gvfs: bool = False):
    """
    with_bias: only set to false if we're in the linear LFA w/ g.t. states. (TABULAR)
    gvf_feature_idxes: indices of gvf features.
    """
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
    elif (model_str == 'nn' or model_str == 'linear') and n_predictions > 0:
        if action_cond == 'mult':
            # We have a sigmoid output if we DON'T tile code our inputs.
            network_fn = partial(mult_action_gvfn, hidden_layers, output_size, n_predictions,
                                 sigmoid_output=not tile_code_gvfs, with_bias=with_bias, init=init)
        else:
            raise NotImplementedError
            # network_fn = partial(gvfn, hidden_layers, n_actions_gvfs, output_size,
            #                      with_bias=with_bias, init=init)
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
    elif model_str == 'actor_critic':
        actor_fn = partial(actor_nn, hidden_layers, output_size)
        actor_network = hk.without_apply_rng(hk.transform(actor_fn))

        critic_fn = partial(critic_nn, hidden_layers)
        critic_network = hk.without_apply_rng(hk.transform(critic_fn))
        return (actor_network, critic_network)
    elif model_str == 'cnn_actor_critic':
        actor_fn = partial(actor_cnn, hidden_size, output_size)
        actor_network = hk.without_apply_rng(hk.transform(actor_fn))

        critic_fn = partial(critic_cnn, hidden_size)
        critic_network = hk.without_apply_rng(hk.transform(critic_fn))
        return (actor_network, critic_network)

    else:
        raise NotImplementedError

    return network


