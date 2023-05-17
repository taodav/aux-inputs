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
from unc.agents.lstm import LSTMAgent


# def seq_sarsa_mc_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray, g: jnp.ndarray):
#     # here, q is MC q, but same same
#     # This could be simpler if we assume gamma_terminal is on, but let's not
#     shunted_discount = jnp.concatenate([jnp.ones_like(g[0:1]), g[:-1]])
#     discount = jnp.cumprod(shunted_discount)
#
#     discounted_r = r * discount
#     cumulative_discounted_r = jnp.cumsum(discounted_r[::-1])[::-1]
#
#     # If discount is 0, then cumulative_discounted_r is 0 as well, so we're safe from blowups
#     # After shunting, it should never be that anyways though to be fair
#     corrected_cumulative_r = cumulative_discounted_r / jnp.maximum(discount, 1e-5)
#     target = jax.lax.stop_gradient(corrected_cumulative_r)
#
#     q_vals = q[jnp.arange(a.shape[0]), a]
#     return q_vals - target

def seq_sarsa_mc_error(q: jnp.ndarray, a: jnp.ndarray, ret: jnp.ndarray):
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - ret

def seq_sarsa_lambda_error(qtd: jnp.ndarray, qmc: jnp.ndarray, a: jnp.ndarray):
    q_vals_td = qtd[jnp.arange(a.shape[0]), a]
    q_vals_mc = qmc[jnp.arange(a.shape[0]), a]
    q_vals_mc = jax.lax.stop_gradient(q_vals_mc)

    return q_vals_td - q_vals_mc


def seq_sarsa_lambda_returns_error(q: jnp.ndarray, a: jnp.ndarray, r: jnp.ndarray,
                                   g: jnp.ndarray, q1: jnp.ndarray, next_a: jnp.ndarray,
                                   lambda_: float):
    # If scalar make into vector.
    lambda_ = jnp.ones_like(g) * lambda_
    q1_vals = q1[jnp.arange(next_a.shape[0]), next_a]

    # Work backwards to compute `G_{T-1}`, ..., `G_0`.
    def _body(acc, xs):
        returns, discounts, values, lambda_ = xs
        acc = returns + discounts * ((1 - lambda_) * values + lambda_ * acc)
        return acc, acc

    _, returns = jax.lax.scan(
        _body, q1_vals[-1], (r, g, q1_vals, lambda_), reverse=True)
    # _body, v_t[-1], (r, g, v_t, lambda_), reverse=True)

    lambda_returns = jax.lax.stop_gradient(returns)
    q_vals = q[jnp.arange(a.shape[0]), a]
    return q_vals - lambda_returns


class MultiHeadLSTMAgent(LSTMAgent):
    def __init__(self, network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        """
        Two-headed LSTM agent, where one head learns w/ Sarsa,
        and the other head learns with MC
        """
        
        super().__init__(network, optimizer, features_shape, n_actions, rand_key, args)
        self.action_mode = args.action_selector_mode
        self.update_mode = args.update_mode
        self.lambda_1 = args.lambda_1
        self.gamma_terminal = args.gamma_terminal
        self.reward_scale = args.reward_scale
        self.lambda_coefficient = args.lambda_coefficient
        self.gamma = args.discounting

        self.td_error_fn = seq_sarsa_error
        self.batch_td_error_fn = vmap(self.td_error_fn)
        self.batch_mc_error_fn = vmap(seq_sarsa_mc_error)
        assert self.lambda_1 == 1., "TD(lambda) not implemented for lambda < 1."
        # if self.lambda_1 < 1.:
        #     self.batch_mc_error_fn = vmap(seq_sarsa_lambda_returns_error,
        #                                   in_axes=(0, 0, 0, 0, 0, 0, None))
        self.batch_lambda_error_fn = vmap(seq_sarsa_lambda_error)

    def both_Qs(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params, *args):
        return self.network.apply(network_params, state, hidden_state)

    def act(self, obs: np.ndarray, head: str = None):
        head = self.action_mode if head is None else head
        obs = jnp.expand_dims(obs, 1)
        action, self._rand_key, self.hidden_state, self.curr_q = \
            self.functional_act(obs, self.hidden_state,
                                self.network_params,
                                self._rand_key,
                                head=head)
        return action

    @partial(jit, static_argnums=[0, 5])
    def functional_act(self, state: jnp.ndarray,
                       hidden_state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey,
                       head: str = None) -> Tuple[np.ndarray, random.PRNGKey, hk.LSTMState, np.ndarray]:
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, new_hidden_state, qs = self.greedy_act(state, hidden_state, network_params, head=head)
        probs = probs.at[greedy_idx].add(1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key, new_hidden_state, qs

    @partial(jit, static_argnums=[0, 4])
    def greedy_act(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params,
                   head: str = None) -> jnp.ndarray:
        """
        Get greedy actions given a state
        :param state: (b x timesteps x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        head = self.action_mode if head is None else head

        qs, new_hidden_state = self.Qs(state, hidden_state, network_params=network_params, head=head)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    def Qs(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params, *args,
           head: str = None) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        head = self.action_mode if head is None else head
        q_td, q_mc, new_hidden = self.both_Qs(state, hidden_state, network_params, *args)
        if head == 'mc':
            return q_mc, new_hidden
        elif head == 'td':
            return q_td, new_hidden
        else:
            raise NotImplementedError

    def _loss(self, network_params: hk.Params,
              hidden_state: np.ndarray,
              state: np.ndarray,
              action: np.ndarray,
              next_hidden_state: np.ndarray,
              next_state: np.ndarray,
              gamma: np.ndarray,
              effective_reward: np.ndarray,
              effective_return: np.ndarray,
              next_action: np.ndarray,
              zero_mask: np.ndarray,
              mode: str = 'lambda'):
        #(B x T x A)
        td0_q_all, td_lambda_q_all, new_hidden = self.both_Qs(state, hidden_state, network_params)

        td0_q_s0 = td0_q_all[:, :-1, :]

        td_lambda_q_s0 = td_lambda_q_all[:, :-1, :]
        td_lambda_q_s1 = td_lambda_q_all[:, 1:, :]

        # td0_err

        td0_err = self.batch_td_error_fn(td0_q_s0, action, effective_reward,
                                         gamma, td_lambda_q_s1, next_action)
        # if self.lambda_1 < 1.:
        #     td_lambda_err = self.batch_mc_error_fn(td_lambda_q_s0, action, effective_reward,
        #                                            gamma, td_lambda_q_s1, next_action, self.lambda_1)
        # else:
        td_lambda_err = self.batch_mc_error_fn(td_lambda_q_s0, action, effective_return)

        lambda_err = self.batch_lambda_error_fn(td0_q_s0, td_lambda_q_s0, action)
        td0_err, td_lambda_err, lambda_err = mse(td0_err), mse(td_lambda_err), mse(lambda_err)
        if mode == 'td0':
            main_loss = td0_err
        elif mode == 'td_lambda':
            main_loss = td_lambda_err
        elif mode == 'both':
            main_loss = td0_err + td_lambda_err
        elif mode == 'lambda':
            main_loss = td0_err + td_lambda_err + (self.lambda_coefficient * lambda_err)
        else:
            assert NotImplementedError

        return main_loss, {
            'td0_loss': td0_err,
            'td_lambda_loss': td_lambda_err,
            'lambda_loss': lambda_err
        }

    @partial(jit, static_argnums=(0, 1))
    def functional_update(self,
                          mode: str, # td0, both, lambda, td_lambda
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          hidden_state: hk.LSTMState,
                          state: np.ndarray,
                          action: np.ndarray,
                          next_hidden_state: hk.LSTMState,
                          next_state: np.ndarray,
                          gamma: np.ndarray,
                          reward: np.ndarray,
                          returns: np.ndarray,
                          next_action: np.ndarray,
                          zero_mask: np.ndarray
                          ) -> Tuple[float, hk.Params, hk.State]:
        (loss, aux_loss), grad = jax.value_and_grad(self._loss, has_aux=True)(network_params, hidden_state, state, action, next_hidden_state,
                                                    next_state, gamma, reward, returns, next_action, zero_mask, mode=mode)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, aux_loss, network_params, optimizer_state

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """
        # We only need the first timestep hidden state for lstm state.
        lstm_state = self._rewrap_hidden(b.state[:, 0])
        lstm_next_state = self._rewrap_hidden(b.next_state[:, 0])

        loss, aux_loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.update_mode,
                                   self.network_params,
                                   self.optimizer_state,
                                   lstm_state, b.obs, b.action,
                                   lstm_next_state, b.next_obs, b.gamma, b.reward,
                                   b.returns,
                                   b.next_action, b.zero_mask)
        return loss, aux_loss
