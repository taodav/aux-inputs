import jax
import jax.numpy as jnp
import distrax
import dill
import numpy as np
import haiku as hk
import optax
from functools import partial
from jax import random, jit, vmap
from optax import GradientTransformation
from pathlib import Path
from typing import Tuple, Callable, Iterable

from unc.args import Args
from unc.models import build_network
from unc.utils.data import Batch

from .base import Agent

def ppo_loss(v_obs: np.ndarray, v_target: np.ndarray,
             pi: np.ndarray, action: int, old_log_prob: np.ndarray,
             advantages: np.ndarray, eps: float):
    ## critic loss
    critic_loss = 0.5 * ((v_obs - v_target) ** 2)

    ## policy losses
    dist = distrax.Categorical(probs=pi)

    # entropy
    entropy_loss = -dist.entropy()
    # policy gradient
    log_prob = dist.log_prob(action)

    ratio = np.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = np.clip(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -np.fmin(p_loss1, p_loss2)

    loss = policy_loss + 0.001 * entropy_loss + critic_loss

    return loss.sum()

class PPOAgent(Agent):
    def __init__(self, actor_network: hk.Transformed,
                 critic_network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args,
                 ppo_eps: float = 0.2):
        self.features_shape = features_shape
        self.n_hidden = args.n_hidden
        self.n_actions = n_actions

        self._rand_key, actor_rand_key, critic_rand_key = random.split(rand_key, 3)
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.actor_network_params = self.actor_network.init(rng=actor_rand_key, x=jnp.zeros((1, *self.features_shape)))
        self.critic_network_params = self.critic_network.init(rng=critic_rand_key, x=jnp.zeros((1, *self.features_shape)))

        self.optimizer = optimizer
        self.actor_optimizer_state = self.optimizer.init(self.actor_network_params)
        self.critic_optimizer_state = self.optimizer.init(self.critic_network_params)
        self.eps = args.epsilon
        self.ppo_eps = ppo_eps
        self.args = args
        self.curr_q = None

        self.batch_error_fn = vmap(ppo_loss)

    def set_eps(self, eps: float):
        self.eps = eps

    def get_eps(self) -> float:
        return self.eps

    def act(self, state: np.ndarray) -> np.ndarray:
        action, self._rand_key, _ = self.functional_act(state, self.actor_network_params, self._rand_key)
        return action

    def policy(self, state: jnp.ndarray, actor_network_params: hk.Params) -> Tuple[jnp.ndarray, None]:
        probs = self.actor_network.apply(actor_network_params, state)
        return probs, None

    @partial(jit, static_argnums=0)
    def functional_act(self, state: jnp.ndarray,
                       network_params: hk.Params,
                       rand_key: random.PRNGKey) \
            -> Tuple[np.ndarray, random.PRNGKey, jnp.ndarray]:
        """
        Get epsilon-greedy actions given a state
        :param state: (*state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: epsilon-greedy action
        """
        policy, _ = self.policy(state, network_params)

        key, subkey = random.split(rand_key)

        return random.choice(subkey, np.arange(self.n_actions), p=policy, shape=(state.shape[0],)), key, None

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, network_params: hk.Params) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        pi = self.actor_network.apply(state, network_params=network_params)
        return jnp.argmax(pi, axis=1), pi

    def Vs(self, state: np.ndarray, critic_network_params: hk.Params, *args) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.critic_network.apply(critic_network_params, state)

    def _loss(self, pi_params: hk.Params,
              v_params: hk.Params,
              state: np.ndarray,
              action: np.ndarray,
              old_log_prob: np.ndarray,
              v_target: np.ndarray,
              advantages: np.ndarray):
        pi = self.actor_network.apply(pi_params, state)
        v_obs = self.critic_network.apply(v_params, state)

        err = self.batch_error_fn(v_obs, v_target, pi, action, old_log_prob, advantages, self.ppo_eps)
        return err

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          pi_params: hk.Params,
                          v_params: hk.Params,
                          pi_optimizer_state: hk.State,
                          v_optimizer_state: hk.State,
                          state: np.ndarray,
                          action: np.ndarray,
                          old_log_prob: np.ndarray,
                          v_target: np.ndarray,
                          advantages: np.ndarray
                          ):
        loss, (pi_grad, v_grad) = jax.value_and_grad(self._loss, argnums=[0, 1])(pi_params, v_params, state, action, old_log_prob, v_target, advantages)
        # actor update
        updates, optimizer_state = self.optimizer.update(pi_grad, pi_optimizer_state, pi_params)
        pi_params = optax.apply_updates(pi_params, updates)

        # critic update
        updates, optimizer_state = self.optimizer.update(v_grad, v_optimizer_state, v_params)
        v_params = optax.apply_updates(v_params, updates)

        return loss, (pi_params, v_params), optimizer_state

    def update(self, b: Batch) -> Tuple[float, dict]:
        """
        Update given a batch of data
        :param batch: Batch of data
        :return: loss
        """

        loss, network_params, self.optimizer_state = \
            self.functional_update(self.actor_network_params, self.critic_network_params,
                                   self.actor_optimizer_state, self.critic_optimizer_state,
                                   b.obs, b.action, b.old_log_prob, b.v_target, b.advantages)
        self.actor_network_params, self.critic_network_params = network_params
        return loss, {}

    def save(self, path: Path):
        """
        Saves the agent's parameters in a given path.
        :param path: path to save to (including file name).
        """
        to_save = {
            'actor_network_params': self.actor_network_params,
            'critic_network_params': self.critic_network_params,
            'features_shape': self.features_shape,
            'n_hidden': self.n_hidden,
            'n_actions': self.n_actions,
            'ppo_eps': self.ppo_eps,
            'actor_optimizer_state': self.actor_optimizer_state,
            'critic_optimizer_state': self.critic_optimizer_state,
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
        agent.actor_network_params = loaded['actor_network_params']
        agent.critic_network_params = loaded['critic_network_params']
        agent.actor_optimizer_state = loaded['actor_optimizer_state']
        agent.critic_optimizer_state = loaded['critic_optimizer_state']

        return agent

