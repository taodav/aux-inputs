import jax.numpy as jnp
import numpy as np
import haiku as hk
import jax
from typing import List


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
