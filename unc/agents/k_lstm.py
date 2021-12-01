import numpy as np
import haiku as hk
from optax import GradientTransformation
import optax
from jax import random, jit, vmap
import jax
import jax.numpy as jnp
from jax.ops import index_add
from functools import partial
from typing import Tuple, List

from unc.args import Args
from unc.utils import Batch
from unc.utils.math import mse, seq_sarsa_error
from .lstm import LSTMAgent


class kLSTMAgent(LSTMAgent):
    def __init__(self, rnn_network: hk.Transformed,
                 value_network: hk.Transformed,
                 rnn_optimizer: GradientTransformation,
                 value_optimizer: GradientTransformation,
                 n_features: int,
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        """
        LSTM agent that produces k outputs.
        Uses stats of k outputs as input to value network
        :param rnn_network:
        :param value_network:
        :param rnn_optimizer:
        :param value_optimizer:
        :param n_features:
        :param n_actions:
        :param rand_key:
        :param args:
        """
        self.n_features = n_features
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        # k RNN params
        self.k = args.k_rnn_hs
        self.same_params = args.same_k_rnn_params

        self.trunc = args.trunc
        self.init_hidden_var = args.init_hidden_var
        self.hidden_state = None

        num_param_sets = self.k if not self.same_params else 1
        network_rand_keys = random.split(rand_key, num=num_param_sets + 2)
        self._rand_key = network_rand_keys[0]
        value_init_rand_key = network_rand_keys[1]
        rnn_init_rand_keys = network_rand_keys[2:]

        self.rnn_network = rnn_network
        self.rnn_network_apply_act = vmap(self.rnn_network.apply, in_axes=(0, None, 0))

        self.value_network = value_network
        self.reset()

        k_inputs = jnp.zeros((self.k, 1, self.trunc, self.n_features))
        # Here we instantiate k RNNs... or do we?
        rnn_init_fn = vmap(self.rnn_network.init)
        self.rnn_network_params = rnn_init_fn(rnn_init_rand_keys, k_inputs, self.hidden_state)

        self.value_network_params = self.value_network.init(rng=value_init_rand_key, x=jnp.zeros((1, 2 * self.n_hidden)))

        self.rnn_optimizer = rnn_optimizer
        rnn_optimizer_init_fn = vmap(self.rnn_optimizer.init)
        self.rnn_optimizer_states = rnn_optimizer_init_fn(self.rnn_network_params)

        self.value_optimizer = value_optimizer
        self.value_optimizer_state = self.value_optimizer.init(self.value_network_params)

        # This is for the super call to functional_update
        self.network = rnn_network
        self.optimizer = rnn_optimizer
        # essentially we update our RNNs through the functional_update call.
        self.vmjit_functional_rnn_update = jit(vmap(super(kLSTMAgent, self).functional_update, in_axes=0))

        self.eps = args.epsilon
        self.device = args.device
        self.args = args
        self.curr_q = None

        self.er_hidden_update = args.er_hidden_update
        self.reset()

        if self.er_hidden_update == "grad":
            raise NotImplementedError
            # self.hidden_optimizer = optax.sgd(args.step_size)
            # # This line.... it does nothing.
            # self.hidden_optimizer_state = self.hidden_optimizer.init(self.hidden_state)

        self.error_fn = None
        if args.algo == 'sarsa':
            self.error_fn = seq_sarsa_error
        else:
            raise NotImplementedError

    @property
    def state_shape(self) -> Tuple[int, int, int]:
        return self.k, 2, self.n_hidden

    @property
    def state(self):
        return jnp.concatenate([self.hidden_state.hidden, self.hidden_state.cell], axis=1)

    def reset(self):
        """
        Reset LSTM hidden states.
        :return:
        """
        hs = jnp.zeros([self.k, 1, self.n_hidden])
        cs = jnp.zeros([self.k, 1, self.n_hidden])
        if self.init_hidden_var > 0.:
            self._rand_key, keys = random.split(self._rand_key, num=3)
            hs = random.normal(keys[0], shape=[self.k, 1, self.n_hidden]) * self.init_hidden_var
            cs = random.normal(keys[1], shape=[self.k, 1, self.n_hidden]) * self.init_hidden_var
        self.hidden_state = hk.LSTMState(hidden=hs, cell=cs)

    def act(self, obs: np.ndarray):
        obs = jnp.expand_dims(obs, 1)
        action, self._rand_key, self.hidden_state, self.curr_q = \
            self.functional_act(obs, self.hidden_state, self.rnn_network_params, self.value_network_params, self._rand_key)
        return action

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       hidden_state: jnp.ndarray,
                       rnn_network_params: hk.Params,
                       value_network_params: hk.Params,
                       rand_key: random.PRNGKey) -> Tuple[np.ndarray, random.PRNGKey, hk.LSTMState, np.ndarray]:
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, new_hidden_state, qs = self.greedy_act(state, hidden_state, rnn_network_params, value_network_params)
        probs = index_add(probs, greedy_idx, 1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key, new_hidden_state, qs

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, hidden_state: np.ndarray,
                   rnn_network_params: hk.Params, value_network_params: hk.Params) -> jnp.ndarray:
        qs, _, new_hidden_state = self.Qs(state, hidden_state, rnn_network_params, value_network_params)
        return jnp.argmax(qs, axis=1), new_hidden_state, qs

    def Qs(self, state: np.ndarray, hidden_state: np.ndarray,
           rnn_network_params: hk.Params, value_network_params: hk.Params, *args) \
            -> Tuple[jnp.ndarray, hk.LSTMState, jnp.ndarray]:
        k_q_vals, all_hiddens, new_lstm_states = self.rnn_network_apply_act(rnn_network_params, state, hidden_state)
        hidden_states = new_lstm_states.hidden[:, 0]
        hidden_means = hidden_states.mean(axis=0)
        hidden_vars = ((hidden_states - hidden_means) ** 2).mean(axis=0)
        value_inputs = jnp.concatenate((hidden_means, hidden_vars))[None, :]
        q_vals = self.value_network.apply(value_network_params, value_inputs)
        return q_vals, all_hiddens, new_lstm_states

    def _rewrap_hidden(self, lstm_states: jnp.ndarray):
        return hk.LSTMState(hidden=lstm_states[:, :, 0], cell=lstm_states[:, :, 1])

    def update(self, lb: Batch) -> Tuple[float, dict]:

        # we only need first state.
        lstm_state = self._rewrap_hidden(lb.state[:, :, 0])
        lstm_next_state = self._rewrap_hidden(lb.next_state[:, :, 0])

        loss, self.rnn_network_params, self.optimizer_state, hidden_states, next_hidden_states = \
            self.vmjit_functional_rnn_update(self.rnn_network_params, self.rnn_optimizer_states,
                                             lstm_state, lb.obs, lb.action, lstm_next_state,
                                             lb.next_obs, lb.gamma, lb.reward, lb.next_action, lb.zero_mask)

        """
        TODO:
        1. Refactor sample_k so that we don't lose all hidden states and next hidden states. Use these (flattened) for TD update.
        2. find mean/var of this, concatenate to form features for both state and next state
        3. do the TD update with value_network.
        
        STUFF TO THINK ABOUT:
        STALENESS: LSTM hidden states in buffer are very likely to be stale. Could we instead do online learning for the
        value head?
        """
        print("holy shiet i worked")
        # Next we update our value estimator