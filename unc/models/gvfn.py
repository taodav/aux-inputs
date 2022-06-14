import jax.numpy as jnp
import haiku as hk
import jax
from typing import List


def gvfn(layers: List[int],
         n_actions: int,
         actions: int, x: jnp.ndarray, with_bias: bool = True,
         init: hk.initializers.Initializer = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')):
    """
    gvf_features: index of features we want to pass through a sigmoid.
    """

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
    outputs = values(h)

    q_vals = outputs[:, :n_actions]
    predictions = outputs[:, n_actions:]
    predictions = jax.nn.sigmoid(predictions)

    return jnp.concatenate((q_vals, predictions), axis=-1)
