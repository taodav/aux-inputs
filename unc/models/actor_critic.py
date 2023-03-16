import jax.numpy as jnp
import numpy as np
import haiku as hk
import jax
from typing import List
from haiku import Conv2D


def actor_nn(layers: List[int], actions: int, x: np.ndarray,
       init: hk.initializers.Initializer = hk.initializers.RandomUniform(-3e-3, 3e-3)):

    hidden = []
    for layer in layers:
        hidden.append(hk.Linear(layer))
        hidden.append(jax.nn.relu)
    hidden.append(hk.Linear(actions, w_init=init))
    hidden.append(jax.nn.softmax)

    pi_func = hk.Sequential(hidden)

    return pi_func(x)


def critic_nn(layers: List[int], x: np.ndarray,
             init: hk.initializers.Initializer = hk.initializers.RandomUniform(-3e-3, 3e-3)):

    hidden = []
    for layer in layers:
        hidden.append(hk.Linear(layer))
        hidden.append(jax.nn.relu)
    hidden.append(hk.Linear(1, w_init=init))

    v_func = hk.Sequential(hidden)

    return v_func(x)

def actor_cnn(hidden_size: int, actions: int, x: np.ndarray):
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    # TODO: we hardcode these sizes in first. There MAY be a use to making this nicer.
    kernel_sizes = [5, 4, 2]
    if list(x.shape[1:-1]) == [25, 25]:
        kernel_sizes = [11, 9, 7]
    elif list(x.shape[1:-1]) == [21, 21]:
        kernel_sizes = [10, 7, 1]
    elif list(x.shape[1:-1]) == [17, 17]:
        kernel_sizes = [8, 6, 5]
    elif list(x.shape[1:-1]) == [5, 5]:
        kernel_sizes = [3, 2, 2]

    convs = hk.Sequential([
        Conv2D(output_channels=32, kernel_shape=kernel_sizes[0], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu,
        Conv2D(output_channels=hidden_size, kernel_shape=kernel_sizes[1], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu,
        Conv2D(output_channels=hidden_size, kernel_shape=kernel_sizes[2], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu
    ])
    policy = hk.Sequential([
        # hk.Linear(hidden_size, w_init=init, b_init=b_init),
        # jax.nn.relu,
        hk.Linear(actions, w_init=init, b_init=b_init),
        jax.nn.softmax
    ])
    flatten = hk.Flatten()

    hidden = convs(x)
    flat_hidden = flatten(hidden)
    return policy(flat_hidden)

def critic_cnn(hidden_size: int, x: np.ndarray):
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    # TODO: we hardcode these sizes in first. There MAY be a use to making this nicer.
    kernel_sizes = [5, 4, 2]
    if list(x.shape[1:-1]) == [25, 25]:
        kernel_sizes = [11, 9, 7]
    elif list(x.shape[1:-1]) == [21, 21]:
        kernel_sizes = [10, 7, 1]
    elif list(x.shape[1:-1]) == [17, 17]:
        kernel_sizes = [8, 6, 5]
    elif list(x.shape[1:-1]) == [5, 5]:
        kernel_sizes = [3, 2, 2]

    convs = hk.Sequential([
        Conv2D(output_channels=32, kernel_shape=kernel_sizes[0], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu,
        Conv2D(output_channels=hidden_size, kernel_shape=kernel_sizes[1], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu,
        Conv2D(output_channels=hidden_size, kernel_shape=kernel_sizes[2], stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu
    ])
    values = hk.Sequential([
        # hk.Linear(hidden_size, w_init=init, b_init=b_init),
        # jax.nn.relu,
        hk.Linear(1, w_init=init, b_init=b_init),
    ])
    flatten = hk.Flatten()

    hidden = convs(x)
    flat_hidden = flatten(hidden)
    return values(flat_hidden)
