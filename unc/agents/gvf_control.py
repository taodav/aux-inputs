import haiku as hk
import jax
import optax
import jax.numpy as jnp
from jax import random, jit, vmap
from optax import GradientTransformation
from typing import Iterable, Tuple
from functools import partial

from .dqn import DQNAgent
from unc.args import Args
from unc.utils.data import Batch
from unc.utils.math import mse


class GVFControlAgent(DQNAgent):
    def __init__(self,
                 gvf_idxes: jnp.ndarray,
                 network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args
                 ):

        super(GVFControlAgent, self).__init__(network, optimizer, features_shape, n_actions, rand_key, args)
        self.current_gvf_predictions = None
        self.gvf_idxes = gvf_idxes
        self.use_tc = 't' in self.args.env
        self.denom = self.n_actions if self.n_actions > 0 else 0.0001

    def reset(self):
        self.current_gvf_predictions = jnp.zeros((1, len(self.gvf_idxes)))
        # if '2' in self.args.env:
        #     # Traverse probability here. This init is b/c we know we initialize with
        #     # rewards present in both cases
        #     # TODO: refactor for any init.
        #     self.current_gvf_predictions += self.args.discounting * 0.6

    def Qs(self, state: jnp.ndarray, network_params: hk.Params, *args) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.Qs_and_cumulant_Vs(state, network_params, *args)[0]

    @partial(jit, static_argnums=0)
    def Qs_and_cumulant_Vs(self, state: jnp.ndarray, network_params: hk.Params, *args) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get all reward Q-values and the Cumulant Vs.
        """
        Qs_and_Vs = self.network.apply(network_params, state)
        Qs = Qs_and_Vs[:, :self.n_actions]
        cumulant_Vs = Qs_and_Vs[:, self.n_actions:]
        return Qs, cumulant_Vs

    def act(self, state: jnp.ndarray) -> jnp.ndarray:
        if self.n_actions > 0:
            action, self._rand_key, self.curr_q, self.current_gvf_predictions =\
                self.functional_act(state, self.network_params, self._rand_key)
            return action
        else:
            self.update_predictions(state)
            return jnp.array([])

    def update_predictions(self, state: jnp.ndarray) -> jnp.ndarray:
        _, self.current_gvf_predictions = self.Qs_and_cumulant_Vs(state, self.network_params)

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey) \
            -> Tuple[jnp.ndarray, random.PRNGKey, jnp.ndarray, jnp.ndarray]:
        """
        Get epsilon-greedy actions given a state
        :param state: (*state.shape) State to find actions
        :param network_params: Potentially use another model to find action-values.
        :return: epsilon-greedy action
        """
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx, qs, cumulant_vs = self.greedy_act(state, network_params)
        probs = probs.at[greedy_idx].add(1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, jnp.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key, qs, cumulant_vs

    def gvf_td_sarsa_error(self, q: jnp.ndarray, a: int, c: jnp.ndarray, g: jnp.ndarray, q1: jnp.ndarray, next_a: int,
                           is_ratio: jnp.ndarray):
        one_and_is_ratio = jnp.concatenate((jnp.array([1]), is_ratio))
        next_idxes = jnp.concatenate([jnp.array([next_a]), self.gvf_idxes]).astype(int)
        target = c + g * q1[next_idxes]
        target = jax.lax.stop_gradient(target)

        idxes = jnp.concatenate([jnp.array([a]), self.gvf_idxes]).astype(int)
        return one_and_is_ratio * (q[idxes] - target)

    def _loss(self, network_params: hk.Params,
              state: jnp.ndarray,
              action: jnp.ndarray,
              next_state: jnp.ndarray,
              cumulant_termination: jnp.ndarray,
              cumulants: jnp.ndarray,
              next_action: jnp.ndarray,
              is_ratio: jnp.ndarray,
              ):
        """
        TODO
        Assume first elements of cumulants and cumulant_termination are reward and discount for reward.
        The rest are our GVFs.
        """
        outputs = self.network.apply(network_params, state)
        next_outputs = self.network.apply(network_params, next_state)

        batch_loss = vmap(self.gvf_td_sarsa_error)
        td_err = batch_loss(outputs, action, cumulants, cumulant_termination, next_outputs, next_action, is_ratio)
        # td_err = self.gvf_td_sarsa_error(outputs[0], action[0], cumulants[0], cumulant_termination[0], next_outputs[0], next_action[0], is_ratio[0])
        return mse(td_err)

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: jnp.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        qs, cumulant_vs = self.Qs_and_cumulant_Vs(state, network_params=network_params)
        return jnp.argmax(qs, axis=1), qs, cumulant_vs

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          network_params: hk.Params,
                          optimizer_state: hk.State,
                          state: jnp.ndarray,
                          action: jnp.ndarray,
                          next_state: jnp.ndarray,
                          gamma: jnp.ndarray,
                          reward: jnp.ndarray,
                          next_action: jnp.ndarray,
                          cumulants: jnp.ndarray,
                          cumulant_termination: jnp.ndarray,
                          is_ratio: jnp.ndarray,
                          ) -> Tuple[float, hk.Params, hk.State]:
        cumulants = jnp.concatenate([jnp.expand_dims(reward, -1), cumulants], axis=-1)
        cumulant_termination = jnp.concatenate([jnp.expand_dims(gamma, -1), cumulant_termination], axis=-1)

        loss, grad = jax.value_and_grad(self._loss)(network_params, state, action,
                                                    next_state, cumulant_termination,
                                                    cumulants, next_action, is_ratio)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data.
        :param batch: Batch of data
        :return: loss
        """

        loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.network_params, self.optimizer_state,
                                   b.obs, b.action,
                                   b.next_obs, b.gamma, b.reward, b.next_action,
                                   b.cumulants, b.cumulant_terminations, b.impt_sampling_ratio)
        return loss, {}

