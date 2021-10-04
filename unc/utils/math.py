import jax.lax
import jax.numpy as jnp


def relu(x: jnp.ndarray):
    return jnp.maximum(0, x)


def mse(predictions: jnp.ndarray, targets: jnp.ndarray = None):
    if targets is None:
        targets = jnp.zeros_like(predictions)
    squared_diff = 0.5 * (predictions - targets) ** 2
    return jnp.mean(squared_diff)


def sarsa_loss(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: int):
    target = r + g * q1[next_a]
    target = jax.lax.stop_gradient(target)
    return q[a] - target


def expected_sarsa_loss(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, *args,
                        eps: float = 0.1):
    next_greedy_action = q1.argmax()
    pi = jnp.ones_like(q1) * (eps / q1.shape[-1])
    pi[next_greedy_action] += (1 - eps)
    e_q1 = (pi * q1).sum(axis=-1)
    target = r + g * e_q1
    target = jax.lax.stop_gradient(target)
    return q[a] - target


def qlearning_loss(q: jnp.ndarray, a: int, r: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, *args):
    target = r + g * q1.max()
    target = jax.lax.stop_gradient(target)
    return q[a] - target
