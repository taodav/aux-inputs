from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.tiger import Tiger


class TigerWrapper(gym.Wrapper):
    priority = 1

    def __init__(self, env: Union[Tiger, TigerWrapper]):
        super(TigerWrapper, self).__init__(env)
        self.rng = env.rng

        if isinstance(self.env, TigerWrapper):
            assert self.env.priority <= self.priority

    @property
    def weights(self):
        return self.env.weights

    @weights.setter
    def weights(self, weights):
        self.env.weights = weights

    @property
    def state_max(self):
        return self.env.state_max

    @property
    def state_min(self):
        return self.env.state_min

    @property
    def particles(self):
        return self.env.particles

    @property
    def state_space(self):
        return self.env.state_space

    @particles.setter
    def particles(self, particles):
        self.env.particles = particles

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def size(self) -> int:
        return self.env.size

    @state.setter
    def state(self, state) -> None:
        self.env.state = state

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        return self.env.batch_get_obs(states)

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.env.get_obs(state, *args, **kwargs)

    def get_reward(self, *args, **kwargs) -> float:
        return self.env.get_reward(*args, **kwargs)

    def get_terminal(self) -> bool:
        return self.env.get_terminal()

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.env.batch_transition(states, actions)

    def transition(self, state: np.ndarray, action: int, *args, **kwargs) -> np.ndarray:
        return self.env.transition(state, action, *args, **kwargs)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        return self.env.emit_prob(state, obs)

    def sample_position(self, n: int = 10) -> np.ndarray:
        return self.env.sample_position(n)

    def sample_all_states(self) -> np.ndarray:
        return self.env.sample_all_states()

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode=mode, **kwargs)
