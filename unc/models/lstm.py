import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
from haiku import RNNCore, LSTMState
from typing import Optional, Tuple


# The below was taken from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
# and modified.
def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_map(broadcast, nest)


class LSTM(RNNCore):
  r"""Long short-term memory (LSTM) RNN core.
  The implementation is based on :cite:`zaremba2014recurrent`. Given
  :math:`x_t` and the previous state :math:`(h_{t-1}, c_{t-1})` the core
  computes
  .. math::
     \begin{array}{ll}
     i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i) \\
     f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f) \\
     g_t = \tanh(W_{ig} x_t + W_{hg} h_{t-1} + b_g) \\
     o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o) \\
     c_t = f_t c_{t-1} + i_t g_t \\
     h_t = o_t \tanh(c_t)
     \end{array}
  where :math:`i_t`, :math:`f_t`, :math:`o_t` are input, forget and
  output gate activations, and :math:`g_t` is a vector of cell updates.
  The output is equal to the new hidden, :math:`h_t`.
  Notes:
    Forget gate initialization:
      Following :cite:`jozefowicz2015empirical` we add 1.0 to :math:`b_f`
      after initialization in order to reduce the scale of forgetting in
      the beginning of the training.
  """

  def __init__(self, hidden_size: int, name: Optional[str] = None):
    """Constructs an LSTM.
    Args:
      hidden_size: Hidden layer size.
      name: Name of the module.
    """
    super().__init__(name=name)
    self.hidden_size = hidden_size

  def __call__(
      self,
      inputs: jnp.ndarray,
      prev_state: LSTMState,
  ) -> Tuple[jnp.ndarray, LSTMState]:
    if len(inputs.shape) > 2 or not inputs.shape:
      raise ValueError("LSTM input must be rank-1 or rank-2.")
    x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
    gated = hk.Linear(4 * self.hidden_size)(x_and_h)
    # TODO(slebedev): Consider aligning the order of gates with Sonnet.
    # i = input, g = cell_gate, f = forget_gate, o = output_gate
    i, g, f, o = jnp.split(gated, indices_or_sections=4, axis=-1)
    f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
    c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g)
    h = jax.nn.sigmoid(o) * jnp.tanh(c)
    return jnp.stack([h, c]), LSTMState(h, c)

  def initial_state(self, batch_size: Optional[int]) -> LSTMState:
    state = LSTMState(hidden=jnp.zeros([self.hidden_size]),
                      cell=jnp.zeros([self.hidden_size]))
    if batch_size is not None:
      state = add_batch(state, batch_size)
    return state


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

    recurrent_func = LSTM(hidden_size)

    hc, final_hidden = hk.dynamic_unroll(recurrent_func, jnp.transpose(x, (1, 0, 2)), h)
    linear = hk.Linear(actions, w_init=init, b_init=b_init)
    outs = hk.BatchApply(linear)(jnp.transpose(hc[:, 0], (1, 0, 2)))

    return outs, hc, final_hidden




