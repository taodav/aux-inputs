import jax.numpy as jnp
import jax
import numpy as np

import haiku as hk

from .q_network import QNetwork


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


class NoisyQNetwork(QNetwork):
    def __init__(self, n_features: int, n_hidden: int, n_actions: int, noisy_std: float = 0.1):
        super(NoisyQNetwork, self).__init__(n_features, n_hidden, n_actions)
        self.noisy_std = noisy_std

        self.l1 = NoisyLinear(self.n_features, self.n_hidden, std_init=self.noisy_std)
        self.l2 = NoisyLinear(self.n_hidden, self.n_actions, std_init=self.noisy_std)

    def reset_noise(self):
        self.l1.reset_noise()
        self.l2.reset_noise()
