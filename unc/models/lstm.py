import numpy as np
import jax.numpy as jnp
import haiku as hk


def lstm(hidden_size: int, actions: int, x: np.ndarray, h: np.ndarray):
    """
    Apply a lstm forward pass over the sequence x.
    :param hidden_size: hidden size of GRU.
    :param actions: number of actions of our action-value approximator.
    :param x: sequence of inputs, of size (batch_size x seq_len x *inp_size).
    :param h: initial hidden states.
    :return: outputs of size
    """
    init = hk.initializers.VarianceScaling(np.sqrt(2), 'fan_avg', 'uniform')
    b_init = hk.initializers.Constant(0)

    recurrent_func = hk.LSTM(hidden_size)

    hiddens, final_hidden = hk.dynamic_unroll(recurrent_func, jnp.transpose(x, (1, 0, 2)), h)
    linear = hk.Linear(actions, w_init=init, b_init=b_init)
    outs = hk.BatchApply(linear)(jnp.transpose(hiddens, (1, 0, 2)))

    return outs, final_hidden




