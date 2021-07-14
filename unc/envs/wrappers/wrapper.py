from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.compass import CompassWorld


class CompassWorldWrapper(gym.Wrapper):
    priority = 1

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper]):
        super(CompassWorldWrapper, self).__init__(env)
        self.random_start = env.random_start
        self.rng = env.rng

        if isinstance(self.env, CompassWorldWrapper):
            assert self.env.priority <= self.priority

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @state.setter
    def state(self, state) -> None:
        self.env.state = state

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.env.get_obs(state)

    def get_reward(self) -> float:
        return self.env.get_reward()

    def get_terminal(self) -> bool:
        return self.env.get_terminal()

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        return self.env.transition(state, action)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        return self.env.emit_prob(state, obs)

    def sample_states(self, n: int = 10) -> np.ndarray:
        return self.env.sample_states(n)

    def sample_all_states(self) -> np.ndarray:
        return self.env.sample_all_states()
