import numpy as np
import gym
from typing import Any

from unc.envs import Environment


class SimpleChain(Environment):
    def __init__(self, n: int = 10):
        """
        Simple func. approx single chain. Always returns an observation of 1.
        :param n: length of chain
        """
        self.n = n
        self._state = np.zeros(self.n)
        self.current_idx = 0
        self.observation_space = gym.spaces.MultiBinary(1)
        self.action_space = gym.spaces.Discrete(1)

    @property
    def state(self):
        return self._state

    def reset(self):
        self._state = np.zeros(self.n)
        self.current_idx = 0
        self._state[self.current_idx] = 1
        return self.get_obs(self._state)

    def get_reward(self, prev_state: np.ndarray = None, action: int = None) -> int:
        if self.state[-1] == 1:
            return 1
        return 0

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return np.ndarray([1])

    def get_terminal(self) -> bool:
        return self._state[-1] == 1

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        # Just go right
        idx = state.nonzero()[0].item()
        state[min(state.shape[0] - 1, idx + 1)] = 1
        state[idx] = 0

        return state

    def emit_prob(self, state: Any, obs: np.ndarray) -> float:
        return 1

    def step(self, action: int):
        self._state = self.transition(self._state.copy(), action)
        return self.get_obs(self._state), self.get_reward(), self.get_terminal(), {}

