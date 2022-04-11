import numpy as np
import haiku as hk
import jax
import optax
import jax.numpy as jnp
from jax import vmap, random, jit
from functools import partial
from typing import Tuple

from unc.utils.math import mse
from unc.utils.data import Batch
from .dqn import DQNAgent


class NoisyNetAgent(DQNAgent):

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey) \
            -> Tuple[np.ndarray, random.PRNGKey]:
        """
        Get action. for noisy nets we act greedily.
        :param state: (*state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: epsilon-greedy action
        """
        key, subkey = random.split(rand_key)
        greedy_idx, _ = self.greedy_act(state, network_params, subkey)

        return greedy_idx, key

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, network_params: hk.Params,
                   rand_key: random.PRNGKey) -> Tuple[jnp.ndarray, random.PRNGKey]:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        qs, rand_key = self.Qs(state, network_params, rand_key)
        return jnp.argmax(qs, axis=1), rand_key

    def Qs(self, state: np.ndarray, network_params: hk.Params, rand_key: random.PRNGKey) -> \
            Tuple[jnp.ndarray, random.PRNGKey]:
        """
        Get all Q-values given a state. Needs RNG inputs
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        rand_key, sub_key = random.split(rand_key)
        return self.network.apply(network_params, sub_key, state), rand_key

    def _loss(self, network_params: hk.Params,
              state: np.ndarray,
              action: np.ndarray,
              next_state: np.ndarray,
              gamma: np.ndarray,
              reward: np.ndarray,
              rand_key: random.PRNGKey,
              next_action: np.ndarray = None):
        rand_key, *rand_keys = random.split(self._rand_key, 3)

        q = self.network.apply(network_params, rand_keys[0], state)
        q1 = self.network.apply(network_params, rand_keys[1],  next_state)

        batch_loss = vmap(self.error_fn)
        td_err = batch_loss(q, action, reward, gamma, q1, next_action)
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          state: np.ndarray,
                          action: np.ndarray,
                          next_state: np.ndarray,
                          gamma: np.ndarray,
                          reward: np.ndarray,
                          next_action: np.ndarray,
                          rand_key: random.PRNGKey,
                          ) -> Tuple[float, hk.Params, hk.State, random.PRNGKey]:
        loss, grad = jax.value_and_grad(self._loss)(network_params, state, action, next_state, gamma, reward, rand_key,
                                                    next_action=next_action)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state, rand_key

    def update(self, b: Batch) -> float:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """

        loss, self.network_params, self.optimizer_state, self._rand_key = \
            self.functional_update(self.network_params, self.optimizer_state, b.state, b.action, b.next_state, b.gamma,
                                   b.reward, b.next_action, self._rand_key)
        return loss

    def act(self, state: np.ndarray) -> np.ndarray:
        action, self._rand_key = self.functional_act(state, self.network_params, self._rand_key)
        return action
