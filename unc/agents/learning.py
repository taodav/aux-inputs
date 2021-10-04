import jax
import jax.numpy as jnp
import dill
import numpy as np
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from optax import GradientTransformation
from pathlib import Path
from typing import Tuple

from unc.args import Args
from unc.models import QNetwork
from unc.utils.math import sarsa_loss, expected_sarsa_loss, qlearning_loss, mse
from unc.utils.data import Batch

from .base import Agent


class LearningAgent(Agent):
    def __init__(self, network: hk.Transformed,
                 optimizer: GradientTransformation,
                 n_features: int,
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        self.n_features = n_features
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self.network = network
        self.network_params = self.network.init(rng=rand_key, x=jnp.zeros((1, self.n_features)))
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon
        self.device = args.device
        self.args = args

        self.loss_fn = None
        if args.algo == 'sarsa':
            self.loss_fn = sarsa_loss
        elif args.algo == 'esarsa':
            self.loss_fn = expected_sarsa_loss
        elif args.algo == 'qlearning':
            self.loss_fn = qlearning_loss

        self._rand_key = rand_key

    def set_eps(self, eps: float):
        self.eps = eps

    # @partial(jit, static_argnums=0)
    def act(self, state: jnp.ndarray) -> np.ndarray:
        """
        Get epsilon-greedy actions given a state
        :param state: (*state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: epsilon-greedy action
        """
        self._rand_key, subkey = random.split(self._rand_key)
        if random.uniform(subkey) > self.eps:
            return self.greedy_act(state, self.network_params)

        self._rand_key, subkey = random.split(self._rand_key)
        return random.choice(subkey, np.arange(self.n_actions), shape=(state.shape[0],))

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        qs = self.Qs(state, network_params=network_params)
        return jnp.argmax(qs, axis=1)

    def Qs(self, state: np.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.network.apply(network_params, state)

    def Q(self, state: np.ndarray, action: np.ndarray, network_params: hk.Params = None) -> jnp.ndarray:
        """
        Get action-values given a state and action
        :param state: (b x *state.shape) State to find action-values
        :param action: (b) Actions for action-values
        :param model: Optional. Potenially use another model
        :return: (b) Action-values
        """
        qs = self.Qs(state, network_params=network_params)
        return qs[jnp.arange(action.shape[0]), action]

    def _loss(self, network_params: hk.Params,
              state: np.ndarray,
              action: np.ndarray,
              next_state: np.ndarray,
              gamma: np.ndarray,
              reward: np.ndarray,
              next_action: np.ndarray = None):
        q = self.network.apply(network_params, state)
        q1 = self.network.apply(network_params, next_state)

        batch_loss = vmap(self.loss_fn)
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
                          ) -> Tuple[float, hk.Params, hk.State]:
        loss, grad = jax.value_and_grad(self._loss)(network_params, state, action, next_state, gamma, reward, next_action)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def update(self, b: Batch) -> float:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """

        loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.network_params, self.optimizer_state, b.state, b.action, b.next_state, b.gamma, b.reward, b.next_state)
        return loss

    def save(self, path: Path):
        """
        Saves the agent's parameters in a given path.
        :param path: path to save to (including file name).
        """
        to_save = {
            'network_params': self.network_params,
            'n_features': self.n_features,
            'n_hidden': self.n_hidden,
            'n_actions': self.n_actions,
            'optimizer_state': self.optimizer_state,
            'args': self.args.as_dict(),
            'rand_key': self._rand_key
        }
        dill.dump(to_save, path)

    @staticmethod
    def load(path: Path) -> Agent:
        loaded = dill.load(path)
        args = Args()
        args.from_dict(loaded['args'])

        network = QNetwork(loaded['n_hidden'], loaded['n_actions'])
        optimizer = optax.adam(args.step_size)

        agent = LearningAgent(network, optimizer, loaded['n_features'], loaded['n_actions'], loaded['rng'], args)
        agent.network_params = loaded['network_params']
        agent.optimizer_state = loaded['optimizer_state']

        return agent

