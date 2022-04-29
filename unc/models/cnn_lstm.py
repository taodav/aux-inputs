import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from haiku import Conv2D

from .lstm import LSTM


def cnn_lstm(hidden_size: int, actions: int, x: np.ndarray, h: np.ndarray):
    """
    Apply a cnn + lstm forward pass over the sequence x.
    :param hidden_size: hidden size of GRU.
    :param actions: number of actions of our action-value approximator.
    :param x: sequence of inputs, of size (batch_size x seq_len x channels x h x w).
    :param h: initial hidden states.
    :return: outputs of size
    """
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    # TODO: we hardcode these sizes in first. There MAY be a use to making this nicer.
    kernel_sizes = [5, 4, 2]
    if list(x.shape[1:]) == [17, 17, 6]:
        kernel_sizes = [8, 6, 5]
    elif list(x.shape[1:]) == [5, 5, 6]:
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
        jax.nn.relu,
        hk.Linear(hidden_size, w_init=init, b_init=b_init),
        hk.Flatten()
    ])

    lstm = LSTM(hidden_size)

    linear = hk.Linear(actions, w_init=init, b_init=b_init)

    hidden = hk.BatchApply(convs)(x)

    hc, final_hidden = hk.dynamic_unroll(lstm, jnp.transpose(hidden, (1, 0, 2)), h)
    outs = hk.BatchApply(linear)(jnp.transpose(hc[:, 0], (1, 0, 2)))

    return outs, hc, final_hidden

