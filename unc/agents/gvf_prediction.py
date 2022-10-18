import haiku as hk
import numpy as np
import jax
import optax
import jax.numpy as jnp
from jax import random, jit, vmap
from optax import GradientTransformation
from typing import Iterable, Tuple
from functools import partial
from PyFixedReps import TileCoder

from unc.args import Args
from unc.utils.data import Batch
from unc.utils.math import mse
from .dqn import DQNAgent


class GVFPredictionAgent(DQNAgent):
    """
    This is ONLY a prediction agent.
    Act does nothing except return a uniformly sampled action.
    """
    def __init__(self,
                 network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 n_gvfs: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        self.n_gvfs = n_gvfs
        self.tile_code_gvfs = args.tile_code_gvfs
        if self.tile_code_gvfs:
            self.tc = TileCoder({
                'tiles': args.gvf_tiles,
                'tilings': args.gvf_tilings,
                'dims': self.n_gvfs,

                'input_ranges': [(0, 1) for _ in range(self.n_gvfs)],
                'scale_output': False
            })
            features_shape = (features_shape[0] + self.tc.features(),)
        else:
            features_shape = (features_shape[0] + self.n_gvfs,)

        self.features_shape = features_shape
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self._rand_key, network_rand_key = random.split(rand_key)
        self.network = network
        one_hot_action_example = jnp.zeros((1, self.n_actions))
        one_hot_action_example = one_hot_action_example.at[0].set(1)
        self.network_params = self.network.init(rng=network_rand_key,
                                                x=jnp.zeros((1, *self.features_shape)),
                                                one_hot_actions=one_hot_action_example)
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon
        self.args = args
        self.curr_q = None

    def reset(self):
        """
        Returns first initialized GVF predictions
        """
        return jnp.zeros((1, self.n_gvfs))
        # if '2' in self.args.env:
        #     # Traverse probability here. This init is b/c we know we initialize with
        #     # rewards present in both cases
        #     # TODO: refactor for any init.
        #     self.current_gvf_predictions += self.args.discounting * 0.6

    def policy(self, state: jnp.ndarray, *args, **kwargs) -> jnp.ndarray:
        return np.ones((state.shape[0], self.n_actions)) / self.n_actions

    def one_hot(self, idxes: jnp.ndarray):
        one_hot_action = jnp.zeros((idxes.shape[0], self.n_actions))
        one_hot_action = one_hot_action.at[jnp.arange(idxes.shape[0]), idxes].set(1)
        return one_hot_action

    @partial(jit, static_argnums=0)
    def Vs(self, state: jnp.ndarray, action: jnp.ndarray,
           obs: jnp.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        obs_and_state = jnp.concatenate((obs, state), axis=-1)
        one_hot_action = self.one_hot(action)
        Vs = self.network.apply(network_params, obs_and_state, one_hot_action)
        return Vs

    def act(self, state: jnp.ndarray) -> jnp.ndarray:
        action_key, self._rand_key = random.split(self._rand_key, 2)
        action = random.choice(action_key, self.n_actions, shape=(state.shape[0],))
        return action

    def predictions(self, prev_predictions: jnp.ndarray, prev_actions: jnp.ndarray,
                    obs: jnp.ndarray) -> jnp.ndarray:
        """
        update_predictions takes as input:
        predictions: s_{t - 1}, batch_size x n_predictions
        actions: a_{t - 1}, batch_size x n_actions (not one-hot)
        observations: o_t, batch_size x *features_shape
        """
        if self.tile_code_gvfs:
            prev_predictions = self.batch_encode(prev_predictions)

        return self.Vs(prev_predictions, prev_actions, obs, self.network_params)

    @staticmethod
    def gvf_off_policy_td_error(v: jnp.ndarray, c: jnp.ndarray,
                                g: jnp.ndarray, v1: jnp.ndarray,
                                is_ratio: jnp.ndarray):
        target = c + g * v1
        target = jax.lax.stop_gradient(target)

        return is_ratio * (v - target)

    def _loss(self, network_params: hk.Params,
              state: jnp.ndarray,
              prev_action: jnp.ndarray,
              next_state: jnp.ndarray,
              action: jnp.ndarray,
              cumulant_termination: jnp.ndarray,
              cumulants: jnp.ndarray,
              is_ratio: jnp.ndarray,
              ):
        """
        TODO
        Assume first elements of cumulants and cumulant_termination are reward and discount for reward.
        The rest are our GVFs.
        """
        outputs = self.network.apply(network_params, state, prev_action)
        next_outputs = self.network.apply(network_params, next_state, action)

        batch_loss = vmap(self.gvf_off_policy_td_error)
        td_err = batch_loss(outputs, cumulants, cumulant_termination, next_outputs, is_ratio)
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def functional_mult_action_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          obs: jnp.ndarray,
                          prev_prediction: jnp.ndarray,
                          prev_action: jnp.ndarray,
                          next_obs: jnp.ndarray,
                          prediction: jnp.ndarray,
                          action: jnp.ndarray,
                          gamma: jnp.ndarray,
                          cumulants: jnp.ndarray,
                          cumulant_termination: jnp.ndarray,
                          is_ratio: jnp.ndarray,
                          ) -> Tuple[float, hk.Params, hk.State]:
        obs_and_prev_prediction = jnp.concatenate((obs, prev_prediction), axis=-1)
        next_obs_and_prediction = jnp.concatenate((next_obs, prediction), axis=-1)
        one_hot_prev_action = self.one_hot(prev_action)
        one_hot_action = self.one_hot(action)
        cumulant_termination = gamma * (1 - cumulant_termination)

        loss, grad = jax.value_and_grad(self._loss)(network_params, obs_and_prev_prediction, one_hot_prev_action,
                                                    next_obs_and_prediction, one_hot_action, cumulant_termination,
                                                    cumulants, is_ratio)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def batch_encode(self, arr: np.ndarray):
        return jnp.stack([self.tc.encode(s) for s in arr])

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data.
        :param batch: Batch of data
        :return: loss
        """
        prev_predictions, predictions = b.prev_predictions, b.predictions
        if self.tile_code_gvfs:
            prev_predictions = self.batch_encode(prev_predictions)
            predictions = self.batch_encode(predictions)

        loss, self.network_params, self.optimizer_state = \
            self.functional_mult_action_update(self.network_params, self.optimizer_state,
                                   b.obs, prev_predictions, b.prev_action, b.next_obs,
                                   predictions, b.action,
                                   b.gamma, b.cumulants, b.cumulant_terminations,
                                   b.impt_sampling_ratio)
        return loss, {}

