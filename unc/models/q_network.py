import jax.numpy as jnp
import numpy as np
import haiku as hk
import jax
from typing import List


def nn(layers: List[int], actions: int, x: np.ndarray, with_bias: bool = True,
       init: hk.initializers.Initializer = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')):
    b_init = hk.initializers.Constant(0)

    hidden = []
    for layer in layers:
        hidden.append(hk.Linear(layer, w_init=init, b_init=b_init))
        hidden.append(jax.nn.relu)

    hidden = hk.Sequential(hidden)

    values = hk.Sequential([
        hk.Linear(actions, w_init=init, b_init=b_init, with_bias=with_bias)
    ])

    h = hidden(x)
    return values(h)


class QNetwork(hk.Module):
    def __init__(self, n_hidden: int, n_actions: int, name=None):
        super(QNetwork, self).__init__(name=name)
        self.n_hidden = n_hidden
        self.n_actions = n_actions

    def __call__(self, x: jnp.ndarray):
        input_size = x.shape[-1]
        w_init = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')
        b_init = hk.initializers.Constant(0)
        w1 = hk.get_parameter("w1", shape=[input_size, self.n_hidden], dtype=x.dtype, init=w_init)
        b1 = hk.get_parameter("b1", shape=[self.n_hidden], dtype=x.dtype, init=b_init)

        w2 = hk.get_parameter("w2", shape=[self.n_hidden, self.n_actions], dtype=x.dtype, init=w_init)
        b2 = hk.get_parameter("b2", shape=[self.n_actions], dtype=x.dtype, init=b_init)

        o1 = jnp.dot(x, w1) + b1
        a1 = jax.nn.relu(o1)

        o2 = jnp.dot(a1, w2) + b2
        return o2
