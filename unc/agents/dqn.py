import jax
import jax.numpy as jnp
import dill
import numpy as np
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from jax.ops import index_add
from optax import GradientTransformation
from pathlib import Path
from typing import Tuple, Callable, Iterable

from unc.args import Args
from unc.models import build_network
from unc.utils.math import sarsa_error, expected_sarsa_error, qlearning_error, mse
from unc.utils.data import Batch

from .base import Agent


class DQNAgent(Agent):
    def __init__(self, network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        self.features_shape = features_shape
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self._rand_key, network_rand_key = random.split(rand_key)
        self.network = network
        self.network_params = self.network.init(rng=network_rand_key, x=jnp.zeros((1, *self.features_shape)))
        self.optimizer = optimizer
        self.optimizer_state = self.optimizer.init(self.network_params)
        self.eps = args.epsilon
        self.device = args.device
        self.args = args

        self.error_fn = None
        if args.algo == 'sarsa':
            self.error_fn = sarsa_error
        elif args.algo == 'esarsa':
            self.error_fn = expected_sarsa_error
        elif args.algo == 'qlearning':
            self.error_fn = qlearning_error

    def set_eps(self, eps: float):
        self.eps = eps

    def act(self, state: np.ndarray) -> np.ndarray:
        action, self._rand_key = self.functional_act(state, self.network_params, self._rand_key)
        return action

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey) \
            -> Tuple[np.ndarray, random.PRNGKey]:
        """
        Get epsilon-greedy actions given a state
        :param state: (*state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: epsilon-greedy action
        """
        probs = jnp.zeros(self.n_actions) + self.eps / self.n_actions
        greedy_idx = self.greedy_act(state, network_params)
        probs = index_add(probs, greedy_idx, 1 - self.eps)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=probs, shape=(state.shape[0],)), key

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

    def Qs(self, state: np.ndarray, network_params: hk.Params, *args) -> jnp.ndarray:
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
                          ) -> Tuple[float, hk.Params, hk.State]:
        loss, grad = jax.value_and_grad(self._loss)(network_params, state, action, next_state, gamma, reward, next_action)
        updates, optimizer_state = self.optimizer.update(grad, optimizer_state, network_params)
        network_params = optax.apply_updates(network_params, updates)

        return loss, network_params, optimizer_state

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """

        loss, self.network_params, self.optimizer_state = \
            self.functional_update(self.network_params, self.optimizer_state, b.obs, b.action, b.next_obs, b.gamma, b.reward, b.next_action)
        return loss, {}

    def save(self, path: Path):
        """
        Saves the agent's parameters in a given path.
        :param path: path to save to (including file name).
        """
        to_save = {
            'network_params': self.network_params,
            'features_shape': self.features_shape,
            'n_hidden': self.n_hidden,
            'n_actions': self.n_actions,
            'optimizer_state': self.optimizer_state,
            'args': self.args.as_dict(),
            'rand_key': self._rand_key
        }
        with open(path, "wb") as f:
            dill.dump(to_save, f)

    @staticmethod
    def load(path: Path, agent_class: Callable) -> Agent:
        with open(path, "rb") as f:
            loaded = dill.load(f)
        args = Args()
        args.from_dict(loaded['args'])

        model_str = args.arch
        if model_str == 'nn' and args.exploration == 'noisy':
            model_str = args.exploration
        network = build_network(args.n_hidden, loaded['n_actions'], model_str=model_str)
        optimizer = optax.adam(args.step_size)

        agent = agent_class(network, optimizer, loaded['features_shape'], loaded['n_actions'], loaded['rand_key'], args)
        agent.network_params = loaded['network_params']
        agent.optimizer_state = loaded['optimizer_state']

        return agent

