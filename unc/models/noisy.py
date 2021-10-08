import jax.numpy as jnp
import jax
import numpy as np

import haiku as hk

from typing import List


# Factorised NoisyLinear layer with bias
# from https://github.com/deepmind/dqn_zoo/blob/master/dqn_zoo/networks.py
def noisy_linear(num_outputs: int,
                 weight_init_stddev: float,
                 with_bias: bool = True):
  """Linear layer with weight randomization http://arxiv.org/abs/1706.10295."""

  def make_noise_sqrt(rng, shape):
    noise = jax.random.truncated_normal(rng, lower=-2., upper=2., shape=shape)
    return jax.lax.stop_gradient(jnp.sign(noise) * jnp.sqrt(jnp.abs(noise)))

  def net_fn(inputs):
    """Function representing a linear layer with learned noise distribution."""
    num_inputs = inputs.shape[-1]
    max_val = np.sqrt(1 / num_inputs)
    mu_initializer = hk.initializers.RandomUniform(-max_val, max_val)
    mu_layer = hk.Linear(
        num_outputs,
        name='mu',
        with_bias=with_bias,
        w_init=mu_initializer,
        b_init=mu_initializer)
    sigma_initializer = hk.initializers.Constant(  #
        weight_init_stddev / jnp.sqrt(num_inputs))
    sigma_layer = hk.Linear(
        num_outputs,
        name='sigma',
        with_bias=True,
        w_init=sigma_initializer,
        b_init=sigma_initializer)

    # Broadcast noise over batch dimension.
    input_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_inputs])
    output_noise_sqrt = make_noise_sqrt(hk.next_rng_key(), [1, num_outputs])

    # Factorized Gaussian noise.
    mu = mu_layer(inputs)
    noisy_inputs = input_noise_sqrt * inputs
    sigma = sigma_layer(noisy_inputs) * output_noise_sqrt
    return mu + sigma

  return net_fn


def noisy_network(layers: List[int], actions: int, x: np.ndarray, noisy_std_init: float = 0.5):
    hidden = []
    for layer in layers:
        hidden.append(noisy_linear(layer, noisy_std_init))
        hidden.append(jax.nn.relu)

    hidden = hk.Sequential(hidden)

    values = hk.Sequential([
        noisy_linear(actions, noisy_std_init)
    ])
    h = hidden(x)
    return values(h)
