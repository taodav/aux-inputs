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
from unc.utils.math import mse, seq_sarsa_error, sarsa_error
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
        if num_param_sets == 1:
            rnn_init_rand_keys = rnn_init_rand_keys[0]

        self.rnn_network = rnn_network
        axes = (0, None, 1) if not self.same_params else (None, None, 1)
        self.rnn_network_apply_act = vmap(self.rnn_network.apply, in_axes=axes)

        self.value_network = value_network
        self.reset()

        # Here we instantiate k RNNs...
        # Unless we share params. Then we let the apply fn do all the batching for us.
        if self.same_params:
            x = jnp.zeros((1, self.trunc, self.n_features))
            rnn_init_fn = self.rnn_network.init
            init_hs = jax.tree_map(lambda x: x[:, 0], self.hidden_state)
        else:
            x = jnp.zeros((self.k, 1, self.trunc, self.n_features))
            rnn_init_fn = vmap(self.rnn_network.init)
            init_hs = self.hidden_state
        self.rnn_network_params = rnn_init_fn(rnn_init_rand_keys, x, init_hs)

        # if we include cell state
        # self.value_network_params = self.value_network.init(rng=value_init_rand_key, x=jnp.zeros((1, 4 * self.n_hidden)))

        # if we use only hidden state
        self.value_network_params = self.value_network.init(rng=value_init_rand_key, x=jnp.zeros((1, self.trunc, 2 * self.n_hidden)))

        self.rnn_optimizer = rnn_optimizer
        if self.same_params:
            rnn_optimizer_init_fn = self.rnn_optimizer.init
        else:
            rnn_optimizer_init_fn = vmap(self.rnn_optimizer.init)
        self.rnn_optimizer_states = rnn_optimizer_init_fn(self.rnn_network_params)

        self.value_optimizer = value_optimizer
        self.value_optimizer_state = self.value_optimizer.init(self.value_network_params)

        # This is for the super call to functional_update
        self.network = rnn_network
        self.optimizer = rnn_optimizer

        # essentially we update our RNNs through the functional_update call.
        params_axes = (0, 0)
        if self.same_params:
            params_axes = (None, None)
        axes = params_axes + (1, None, None, 1) + tuple(None for _ in range(5))
        self.vmjit_functional_rnn_update = jit(vmap(super(kLSTMAgent, self).functional_update, in_axes=axes))

        self.eps = args.epsilon
        self.device = args.device
        self.args = args

        # FOR DEBUGGING
        self.curr_q = None
        self.rnn_curr_q = None

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
        return 2, self.k, self.n_hidden

    def reset(self):
        """
        Reset LSTM hidden states.
        :return:
        """
        hs = jnp.zeros([self.k, self.n_hidden])
        cs = jnp.zeros([self.k, self.n_hidden])
        if self.init_hidden_var > 0.:
            keys = random.split(self._rand_key, num=3)
            self._rand_key = keys[0]
            hs = random.normal(keys[1], shape=[self.k, self.n_hidden]) * self.init_hidden_var
            cs = random.normal(keys[2], shape=[self.k, self.n_hidden]) * self.init_hidden_var
        lstm_state = hk.LSTMState(hidden=hs, cell=cs)
        broadcast = lambda x: jnp.broadcast_to(x, (1,) + x.shape)
        batch_hs = jax.tree_map(broadcast, lstm_state)
        self.hidden_state = batch_hs

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
        probs = probs.at[greedy_idx].add(1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key, new_hidden_state, qs

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, hidden_state: np.ndarray,
                   rnn_network_params: hk.Params, value_network_params: hk.Params) -> jnp.ndarray:
        qs, _, new_hidden_state = self.Qs(state, hidden_state, rnn_network_params, value_network_params)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    @staticmethod
    def transpose_vmap_output(*args):
        res = []
        for a in args:
            if isinstance(a, jnp.ndarray):
                res.append(jnp.swapaxes(a, 0, 1))
            else:
                res.append(jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), a))
        return tuple(res)

    def Qs(self, state: np.ndarray, hidden_state: np.ndarray,
           rnn_network_params: hk.Params, value_network_params: hk.Params, *args) \
            -> Tuple[jnp.ndarray, hk.LSTMState, jnp.ndarray]:
        """
        To get Q values, we need to calculate our hidden state stats for both our current state and next state.
        Then we use these as features.
        """
        k_q_vals, all_hiddens, new_lstm_states = self.rnn_network_apply_act(rnn_network_params, state, hidden_state)

        all_hiddens = jnp.transpose(all_hiddens, (3, 1, 0, 2, 4))

        k_q_vals, new_lstm_states = self.transpose_vmap_output(k_q_vals, new_lstm_states)

        # Only use hidden states, not cell state
        hidden_state = all_hiddens[:, :, :, 0]

        hidden_means = hidden_state.mean(axis=2)
        hidden_vars = hidden_state.var(axis=2)
        value_inputs = jnp.concatenate((hidden_means, hidden_vars), axis=-1)

        q_vals = self.value_network.apply(value_network_params, value_inputs)
        return q_vals, all_hiddens, new_lstm_states

    def _value_loss(self, network_params: hk.Params,
                    state: np.ndarray,
                    action: np.ndarray,
                    next_state: np.ndarray,
                    gamma: np.ndarray,
                    reward: np.ndarray,
                    zero_mask: np.ndarray,
                    next_action: np.ndarray = None):
        q = self.value_network.apply(network_params, state)
        q1 = self.value_network.apply(network_params, next_state)

        batch_loss = vmap(self.error_fn)
        td_err = batch_loss(q, action, reward, gamma, q1, next_action)
        td_err *= zero_mask
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def functional_value_update(self,
                                network_params: hk.Params,
                                optimizer_state: hk.State,
                                hidden_state: np.ndarray,
                                action: np.ndarray,
                                hidden_next_state: np.ndarray,
                                gamma: np.ndarray,
                                reward: np.ndarray,
                                next_action: np.ndarray,
                                zero_mask: np.ndarray
                          ) -> Tuple[float, hk.Params, hk.State]:
        # Get mean an variances of our hidden states as inputs to our value head
        hses, next_hses = hidden_state[:, :, :, 0], hidden_next_state[:, :, :, 0]
        mean_hs, mean_next_hs = hses.mean(axis=2), next_hses.mean(axis=2)
        var_hs, var_next_hs = hses.var(axis=2), next_hses.var(axis=2)
        state, next_state = jnp.concatenate((mean_hs, var_hs), axis=-1), jnp.concatenate((mean_next_hs, var_next_hs), axis=-1)

        # Finally, we update
        loss, grad = jax.value_and_grad(self._value_loss)(network_params, state, action, next_state, gamma, reward, zero_mask, next_action)
        updates, optimizer_state = self.value_optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    @staticmethod
    @jit
    def flatten_params(params: hk.Params):
        """
        Flatten k params to one by averaging
        """
        return jax.tree_map(lambda p: jnp.mean(p, axis=0), params)

    def update(self, lb: Batch) -> Tuple[float, dict]:
        # we only need first state.
        lstm_state = self._rewrap_hidden(lb.state[:, 0])
        lstm_next_state = self._rewrap_hidden(lb.next_state[:, 0])

        # we simply use jit + vmap to turn our previous functional_update function
        # to an update for all our RNNs.
        # TODO: Different kinds of RNN updates and diff inputs to value head.
        rnn_loss, self.rnn_network_params, self.optimizer_state, hidden_states, next_hidden_states = \
            self.vmjit_functional_rnn_update(self.rnn_network_params, self.rnn_optimizer_states,
                                             lstm_state, lb.obs, lb.action, lstm_next_state,
                                             lb.next_obs, lb.gamma, lb.reward, lb.next_action, lb.zero_mask)

        if self.same_params:
            self.rnn_network_params = self.flatten_params(self.rnn_network_params)
            self.optimizer_state = self.flatten_params(self.optimizer_state)
            hidden_states, next_hidden_states = jnp.transpose(hidden_states, (1, 2, 0, 3, 4)), jnp.transpose(next_hidden_states, (1, 2, 0, 3, 4))
            rnn_loss = jnp.mean(rnn_loss)

        # Next we update our value estimator
        # take all hidden states and update
        value_loss, self.value_network_params, self.value_optimizer_state = \
            self.functional_value_update(self.value_network_params, self.value_optimizer_state,
                                         hidden_states, lb.action, next_hidden_states,
                                         lb.gamma, lb.reward, lb.next_action,
                                         lb.zero_mask)

        return value_loss, {"value_loss": value_loss, "rnn_loss": rnn_loss}
