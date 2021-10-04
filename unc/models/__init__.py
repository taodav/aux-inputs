import jax.numpy as jnp
import haiku as hk
from functools import partial

from .q_network import QNetwork, nn
from .noisy import NoisyQNetwork


def get_network(hidden_size: int, output_size: int, x: jnp.ndarray):
    q = QNetwork(hidden_size, output_size)
    return q(x)


def build_network(hidden_size: int, output_size: int):
    # hfunc = partial(get_network, hidden_size, output_size)
    # network = hk.without_apply_rng(hk.transform(hfunc))
    # return network
    layers = [hidden_size]
    network = partial(nn, layers, output_size)
    network = hk.without_apply_rng(hk.transform(network))

    return network


