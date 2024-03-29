import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp
from optax import GradientTransformation
from jax import random, jit, vmap
from functools import partial
from typing import Tuple, Iterable

from unc.args import Args
from unc.utils import Batch
from unc.utils.math import mse, seq_sarsa_error
from .dqn import DQNAgent


class LSTMAgent(DQNAgent):
    def __init__(self, network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        """
        LSTM agent.
        """
        self.features_shape = features_shape
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self.trunc = args.trunc
        self.init_hidden_var = args.init_hidden_var
        self.hidden_state = None

        self._rand_key, network_rand_key = random.split(rand_key)
        self.network = network
        self.reset()
        self.network_params = self.network.init(rng=network_rand_key, x=jnp.zeros((1, self.trunc, *self.features_shape)),
                                                h=self.hidden_state)
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon
        self.args = args
        self.curr_q = None

        self.er_hidden_update = args.er_hidden_update
        self.reset()
        if self.er_hidden_update == "grad":
            self.hidden_optimizer = optax.sgd(args.step_size)
            # This line.... it does nothing.
            self.hidden_optimizer_state = self.hidden_optimizer.init(self.hidden_state)

        self.error_fn = None
        if args.algo == 'sarsa':
            self.error_fn = seq_sarsa_error
        else:
            raise NotImplementedError
        # elif args.algo == 'esarsa':
        #     self.error_fn = expected_sarsa_error
        # elif args.algo == 'qlearning':
        #     self.error_fn = qlearning_error

    @property
    def state_shape(self) -> Tuple[int, int]:
        return 2, self.n_hidden

    @property
    def state(self):
        return jnp.concatenate([self.hidden_state.hidden, self.hidden_state.cell], axis=0)

    def reset(self):
        """
        Reset LSTM hidden states.
        :return:
        """
        broadcast = lambda x: jnp.broadcast_to(x, (1,) + x.shape)
        hs = jnp.zeros([self.n_hidden])
        cs = jnp.zeros([self.n_hidden])
        if self.init_hidden_var > 0.:
            self._rand_key, keys = random.split(self._rand_key, num=3)
            hs = random.normal(keys[0], shape=[self.n_hidden]) * self.init_hidden_var
            cs = random.normal(keys[1], shape=[self.n_hidden]) * self.init_hidden_var
        lstm_state = hk.LSTMState(hidden=hs, cell=cs)
        batch_hs = jax.tree_map(broadcast, lstm_state)
        self.hidden_state = batch_hs

    def act(self, obs: np.ndarray):
        obs = jnp.expand_dims(obs, 1)
        action, self._rand_key, self.hidden_state, self.curr_q = self.functional_act(obs, self.hidden_state, self.network_params, self._rand_key)
        return action

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       hidden_state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey) -> Tuple[np.ndarray, random.PRNGKey, hk.LSTMState, np.ndarray]:
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, new_hidden_state, qs = self.greedy_act(state, hidden_state, network_params)
        probs = probs.at[greedy_idx].add(1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key, new_hidden_state, qs

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get greedy actions given a state
        :param state: (b x timesteps x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        qs, _, new_hidden_state = self.Qs(state, hidden_state, network_params=network_params)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    def Qs(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params, *args) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.network.apply(network_params, state, hidden_state)

    def _loss(self, network_params: hk.Params,
              hidden_state: np.ndarray,
              state: np.ndarray,
              action: np.ndarray,
              next_hidden_state: np.ndarray,
              next_state: np.ndarray,
              gamma: np.ndarray,
              reward: np.ndarray,
              next_action: np.ndarray,
              zero_mask: np.ndarray):
        q, hiddens, _ = self.network.apply(network_params, state, hidden_state)
        q1, target_hiddens, _ = self.network.apply(network_params, next_state, next_hidden_state)

        batch_loss = vmap(self.error_fn)
        td_err = batch_loss(q, action, reward, gamma, q1, next_action)  # Should be batch x seq_len

        # Don't learn from the values past dones.
        td_err *= zero_mask
        return mse(td_err), (hiddens, target_hiddens)

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          hidden_state: hk.LSTMState,
                          state: np.ndarray,
                          action: np.ndarray,
                          next_hidden_state: hk.LSTMState,
                          next_state: np.ndarray,
                          gamma: np.ndarray,
                          reward: np.ndarray,
                          next_action: np.ndarray,
                          zero_mask: np.ndarray
                          ) -> Tuple[float, hk.Params, hk.State, jnp.ndarray, jnp.ndarray]:
        """
        :return: loss, network parameters, optimizer state and all hidden states (bs x timesteps x 2 x n_hidden)
        """
        outs, grad = jax.value_and_grad(self._loss, has_aux=True)(network_params, hidden_state, state, action, next_hidden_state,
                                                    next_state, gamma, reward, next_action, zero_mask)
        loss, (all_hidden_states, all_target_hidden_states) = outs
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state, \
               jnp.transpose(all_hidden_states, axes=(2, 0, 1, 3)), \
               jnp.transpose(all_target_hidden_states, axes=(2, 0, 1, 3))

    @partial(jit, static_argnums=0)
    def functional_update_hidden(self,
                                 network_params: hk.Params,
                                 optimizer_state: hk.State,
                                 hs_optimizer_state: hk.State,
                                 hidden_state: hk.LSTMState,
                                 state: np.ndarray,
                                 action: np.ndarray,
                                 next_hidden_state: hk.LSTMState,
                                 next_state: np.ndarray,
                                 gamma: np.ndarray,
                                 reward: np.ndarray,
                                 next_action: np.ndarray,
                                 zero_mask: np.ndarray
                                 ) -> Tuple[float, hk.Params, hk.State, hk.LSTMState, hk.State]:
        """
        functional update, but we also update (and return) the updated hidden state
        :return:
        """
        outs, grad = jax.value_and_grad(self._loss, argnums=(0, 1), has_aux=True)(network_params, hidden_state, state, action, next_hidden_state,
                                                    next_state, gamma, reward, next_action, zero_mask)
        loss, _, _ = outs
        network_grads, hs_grads = grad
        updates, optimizer_state = self.optimizer.update(network_grads, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        # now we update our hidden state
        hs_updates, hs_optimizer_state = self.hidden_optimizer.update(hs_grads, hs_optimizer_state, hidden_state)
        hidden_state = optax.apply_updates(hidden_state, hs_updates)

        return loss, network_params, optimizer_state, hidden_state, hs_optimizer_state

    def _rewrap_hidden(self, lstm_state: jnp.ndarray):
        return hk.LSTMState(hidden=lstm_state[:, 0], cell=lstm_state[:, 1])

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """
        # We only need the first timestep hidden state for lstm state.
        lstm_state = self._rewrap_hidden(b.state[:, 0])
        lstm_next_state = self._rewrap_hidden(b.next_state[:, 0])

        other_info = {}
        if self.er_hidden_update == 'grad':
            loss, self.network_params, self.optimizer_state, lstm_state, self.hidden_optimizer_state = \
                self.functional_update_hidden(self.network_params,
                                              self.optimizer_state,
                                              self.hidden_optimizer_state,
                                              lstm_state, b.obs, b.action,
                                              lstm_next_state, b.next_obs, b.gamma, b.reward,
                                              b.next_action, b.zero_mask)
            other_info['first_hidden_state'] = lstm_state
        elif self.er_hidden_update == 'update':
            loss, self.network_params, self.optimizer_state, hidden_states, next_hidden_states = \
                self.functional_update(self.network_params,
                                       self.optimizer_state,
                                       lstm_state, b.obs, b.action,
                                       lstm_next_state, b.next_obs, b.gamma, b.reward,
                                       b.next_action, b.zero_mask)
            other_info['next_hidden_states'] = next_hidden_states

        else:
            loss, self.network_params, self.optimizer_state, _, _ = \
                self.functional_update(self.network_params,
                                       self.optimizer_state,
                                       lstm_state, b.obs, b.action,
                                       lstm_next_state, b.next_obs, b.gamma, b.reward,
                                       b.next_action, b.zero_mask)

        return loss, other_info

