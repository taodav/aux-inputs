import numpy as np
import haiku as hk
import optax
import jax
import jax.numpy as jnp
from optax import GradientTransformation
from functools import partial
from jax import random, jit, nn, vmap
from typing import Tuple

from unc.args import Args
from .lstm import LSTMAgent


class DistributionalLSTMAgent(LSTMAgent):
    def __init__(self, network: hk.Transformed,
                 optimizer: GradientTransformation,
                 n_features: int,
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):
        super(DistributionalLSTMAgent, self).__init__(
            network, optimizer, n_features, n_actions, rand_key, args
        )

        self.atoms = args.atoms
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.support = jnp.linspace(self.v_min, self.v_max, self.atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.error_fn = self.seq_dist_sarsa_err

    @partial(jit, static_argnums=0)
    def greedy_act(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params) -> jnp.ndarray:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param network_params: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        TODO: add option to add value head
        """
        qs, _, new_hidden_state = self.Qs(state, hidden_state, network_params=network_params)
        return jnp.argmax(qs[:, 0], axis=1), new_hidden_state, qs

    def Qs(self, state: np.ndarray, hidden_state: np.ndarray, network_params: hk.Params, *args) -> jnp.ndarray:
        """
        Get all Q-values given a state.
        :param state: (b x timesteps x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x timesteps x actions) torch.tensor full of action-values.
        """
        dist_flattened_qs, all_hidden_state, new_hidden_state = self.network.apply(network_params, state, hidden_state)
        tsteps = dist_flattened_qs.shape[1]
        out_qs = jnp.reshape(dist_flattened_qs, (-1, tsteps, self.n_actions, self.atoms))
        dist_qs = nn.softmax(out_qs, axis=-1)
        qs = (dist_qs * self.support).sum(-1)
        return qs, all_hidden_state, new_hidden_state

    # TODO: update _loss to reflect value head

    @partial(jit, static_argnums=0)
    def seq_dist_sarsa_err(self,
                           q: jnp.ndarray,
                           a: jnp.ndarray,
                           r: jnp.ndarray,
                           g: jnp.ndarray,
                           q1: jnp.ndarray,
                           next_a: jnp.ndarray):
        """
        Distributional RL loss. Largely adapted from
        https://github.com/Kaixhin/Rainbow/blob/master/agent.py
        TODO: SOMETHING IN HERE IS SLOW AS FUCK
        """
        timesteps = q.shape[0]

        # current action distribution
        out_qs = jnp.reshape(q, (timesteps, self.n_actions, self.atoms))
        qs_a = out_qs[jnp.arange(timesteps), a]
        log_ps_a = nn.log_softmax(qs_a, axis=-1)

        # target
        out_q1s = jnp.reshape(q1, (timesteps, self.n_actions, self.atoms))
        qns_a = out_q1s[jnp.arange(timesteps), next_a]
        pns_a = nn.softmax(qns_a, axis=-1)

        # now compute Tz
        family = jnp.repeat(jnp.expand_dims(self.support, 0), timesteps, axis=0)
        t_z = jnp.expand_dims(r, -1) + jnp.expand_dims(g, -1) * family
        # have to clip to be within v_min and v_max
        t_z = jnp.clip(t_z, self.v_min, self.v_max)

        # project onto fixed support
        b = (t_z - self.v_min) / self.delta_z

        # fix disappearing prob. mass
        l, u = jnp.floor(b).astype(jnp.int32), jnp.ceil(b).astype(jnp.int32)
        lower_offset = ((u > 0) * (l == u)).astype(jnp.int32)
        l -= lower_offset

        upper_offset = ((l < (self.atoms - 1)) * (l == u)).astype(jnp.int32)
        u += upper_offset

        # distribute the prob of Tz
        m = jnp.zeros((timesteps * self.atoms))
        offset = np.repeat(jnp.linspace(0, ((timesteps - 1) * self.atoms), timesteps).astype(jnp.int32)[:, np.newaxis], self.atoms, 1)

        m = m.at[jnp.reshape(l + offset, -1)].add(jnp.reshape(pns_a * (u.astype(np.float32) - b), -1))
        m = m.at[jnp.reshape(u + offset, -1)].add(jnp.reshape(pns_a * (b - l.astype(np.float32)), -1))
        m = jnp.reshape(m, (timesteps, self.atoms))
        m = jax.lax.stop_gradient(m)

        loss = -jnp.sum(m * log_ps_a, axis=-1)
        return loss

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
        return jnp.mean(td_err), (hiddens, target_hiddens)



