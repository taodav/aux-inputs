import jax.numpy as jnp
import haiku as hk
import jax
from typing import List


def gvfn(layers: List[int],
         n_actions: int,
         n_predictions: int,
         x: jnp.ndarray,
         with_bias: bool = True,
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
        hk.Linear(n_actions + n_predictions, w_init=init, b_init=b_init, with_bias=with_bias)
    ])

    h = hidden(x)
    outputs = values(h)

    q_vals = outputs[:, :n_actions]
    predictions = outputs[:, n_actions:]
    predictions = jax.nn.sigmoid(predictions)

    return jnp.concatenate((q_vals, predictions), axis=-1)


def mult_action_gvfn(layers: List[int],
                     n_actions: int,
                     n_predictions: int,
                     x: jnp.ndarray,
                     one_hot_actions: jnp.ndarray,
                     with_bias: bool = True,
                     sigmoid_output: bool = True,
                     init: hk.initializers.Initializer = hk.initializers.VarianceScaling(jnp.sqrt(2), 'fan_avg', 'uniform')):
    """
    Multiplicative actions GVFN.
    x: batch_size x features_size
    one_hot_actions: batch_size x n_actions
    """
    b_init = hk.initializers.Constant(0)

    w1 = hk.get_parameter('w1', shape=(n_actions, x.shape[-1], layers[0]), init=init)
    selected_action_w1 = jnp.dot(one_hot_actions, w1)
    current_output = jnp.dot(x, selected_action_w1)

    if with_bias:
        b1 = hk.get_parameter('b1', shape=(n_actions, layers[0]), init=b_init)
        selected_action_b1 = jnp.dot(one_hot_actions, b1)
        current_output += selected_action_b1

    current_output = jax.nn.relu(current_output)

    i = 0
    while i < (len(layers) - 1):
        wi = hk.get_parameter(f'w{i}', shape=(n_actions, layers[i], layers[i + 1]), init=init)
        selected_action_wi = jnp.dot(one_hot_actions, wi)
        current_output = jnp.dot(current_output, selected_action_wi)

        if with_bias:
            bi = hk.get_parameter(f'b{i}', shape=(n_actions, layers[i + 1]), init=b_init)
            selected_action_bi = jnp.dot(one_hot_actions, bi)
            current_output += selected_action_bi

        current_output = jax.nn.relu(current_output)

    value_w = hk.get_parameter('wv', shape=(n_actions, layers[-1], n_predictions), init=init)
    selected_action_wv = jnp.dot(one_hot_actions, value_w)
    value = jnp.dot(current_output, selected_action_wv)

    if with_bias:
        value_b = hk.get_parameter('bv', shape=(n_actions, n_predictions), init=b_init)
        value += jnp.dot(one_hot_actions, value_b)

    if sigmoid_output:
        value = jax.nn.sigmoid(value)

    return value




