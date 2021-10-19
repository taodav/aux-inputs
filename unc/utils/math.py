import jax.lax
import jax.numpy as jnp
from jax.ops import index_add


def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets) ** 2
    return jnp.mean(squared_diff)


def seq_sarsa_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: jnp.ndarray):
    """
    sequential version of sarsa loss
    First axis of all tensors are the sequence length.
    :return:
    """
    target = r + g * q1[jnp.arange(next_a.shape[0]), next_a]
    target = jax.lax.stop_gradient(target)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - target


def sarsa_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: int):
    target = r + g * q1[next_a]
    target = jax.lax.stop_gradient(target)
    return q[a] - target


def expected_sarsa_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: int,
                         eps: float = 0.1):
    next_greedy_action = q1.argmax()
    pi = jnp.ones_like(q1) * (eps / q1.shape[-1])
    pi = index_add(pi, next_greedy_action, (1 - eps))
    e_q1 = (pi * q1).sum(axis=-1)
    target = r + g * e_q1
    target = jax.lax.stop_gradient(target)
    return q[a] - target


def qlearning_error(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, *args):
    target = r + g * q1.max()
    target = jax.lax.stop_gradient(target)
    return q[a] - target
