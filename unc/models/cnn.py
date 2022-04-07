import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from haiku import Conv2D
from typing import Optional, Tuple


def cnn(hidden_size: int, actions: int, x: np.ndarray):
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    convs = hk.Sequential([
        Conv2D(output_channels=32, kernel_shape=4, stride=2, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu,
        Conv2D(output_channels=hidden_size, kernel_shape=3, stride=1, padding="VALID",
               w_init=init, b_init=b_init),
        jax.nn.relu
    ])
    values = hk.Sequential([
        # hk.Linear(hidden_size, w_init=init, b_init=b_init),
        # jax.nn.relu,
        hk.Linear(actions, w_init=init, b_init=b_init),
    ])
    flatten = hk.Flatten()

    hidden = convs(x)
    flat_hidden = flatten(hidden)
    return values(flat_hidden)

