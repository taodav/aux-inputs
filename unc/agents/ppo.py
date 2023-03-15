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
from unc.utils.math import mse

from .base import Agent

@partial(jit, static_argnames='gae_lambda')
def process_sampled_batch(b: Batch, gae_lambda: float = .95):
    """
    This function process a sampled sequence batch of (obs, action, gamma, reward, log_prob, value)
    into a batch for updating a PPO agent.
    It explicitly calculates an advantage.
    """

    advantages = []
    gae = np.zeros(b.reward.shape[0])
    gae_lambda_arr = np.ones_like(gae) * gae_lambda
    for t in reversed(range(b.reward.shape[-1])):
        value_diff = b.gamma[:, t] * b.value[:, t + 1] - b.value[:, t]
        delta = b.reward[:, t] + value_diff
        gae = delta + b.gamma[:, t] * gae_lambda_arr * gae
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = jnp.array(advantages).T
    v_target = advantages + b.value[:, :-1]
    return Batch(obs=b.obs, action=b.action, gamma=b.gamma, reward=b.reward, log_prob=b.log_prob,
                 value=v_target, advantages=advantages)

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

    ratio = jnp.exp(log_prob - old_log_prob)
    p_loss1 = ratio * advantages
    p_loss2 = jnp.clip(ratio, 1 - eps, 1 + eps) * advantages
    policy_loss = -jnp.fmin(p_loss1, p_loss2)

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
                 ppo_eps: float = 0.2,
                 ppo_lambda: float = 0.95):
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
        self.ppo_lambda = ppo_lambda
        self.args = args
        self.curr_q = None
        self.curr_pi = None

        self.batch_error_fn = vmap(ppo_loss)

    def reset(self):
        super().reset()
        self.curr_pi = None

    def set_eps(self, eps: float):
        self.eps = eps

    def get_eps(self) -> float:
        return self.eps

    def act(self, state: np.ndarray) -> np.ndarray:
        assert state.shape[0] <= 1
        action, self._rand_key, batch_curr_pi = self.functional_act(state, self.actor_network_params, self._rand_key)
        self.curr_pi = batch_curr_pi[0]
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
        assert state.shape[0] <= 1
        policy, _ = self.policy(state, network_params)

        key, subkey = random.split(rand_key)

        # return random.choice(subkey, np.expand_dims(np.arange(self.n_actions), 0).repeat(state.shape[0], 0),
        #                      p=policy, shape=(state.shape[0],)), key, policy
        return random.choice(subkey, np.arange(self.n_actions),
                             p=policy[0], shape=(state.shape[0],)), key, policy

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

    @jit
    def V(self, state: np.ndarray, critic_network_params: hk.Params, *args) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        return self.critic_network.apply(critic_network_params, state)

    def _loss(self, pi_params: hk.Params,
              v_params: hk.Params,
              b: Batch):
        curr_obs = b.obs[:, :-1]
        flat_obs = curr_obs.reshape((-1, *curr_obs.shape[2:]))
        flat_pi = self.actor_network.apply(pi_params, flat_obs)
        flat_v = self.critic_network.apply(v_params, flat_obs)[:, 0]

        pi = flat_pi.reshape(curr_obs.shape[0], curr_obs.shape[1], *flat_pi.shape[1:])
        v = flat_v.reshape(curr_obs.shape[0], curr_obs.shape[1], *flat_v.shape[1:])

        err = self.batch_error_fn(v, b.value, pi, b.action, b.log_prob, b.advantages, np.ones(b.obs.shape[0])*self.ppo_eps)

        return err.mean()

    @partial(jit, static_argnums=0)
    def functional_update(self,
                          pi_params: hk.Params,
                          v_params: hk.Params,
                          pi_optimizer_state: hk.State,
                          v_optimizer_state: hk.State,
                          b: Batch
                          ):
        processed_b = process_sampled_batch(b, gae_lambda=self.ppo_lambda)
        loss, (pi_grad, v_grad) = jax.value_and_grad(self._loss, argnums=[0, 1])(pi_params, v_params, processed_b)
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
                                   b)
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

