import numpy as np
import gym
from typing import Any

from unc.envs import Environment

class SimpleChain(Environment):
    def __init__(self, n: int = 10, reward_in_obs: bool = False):
        """
        Simple func. approx single chain. Always returns an observation of 1.
        :param n: length of chain
        """
        self.n = n
        self.reward_in_obs = reward_in_obs
        self._state = np.zeros(self.n)
        self.current_idx = 0
        if self.reward_in_obs:
            self.observation_space = gym.spaces.MultiBinary(2)
        else:
            self.observation_space = gym.spaces.MultiBinary(1)
        self.action_space = gym.spaces.Discrete(1)

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        self._state = state

    def reset(self):
        self.state = np.zeros(self.n)
        self.current_idx = 0
        self.state[self.current_idx] = 1
        return self.get_obs(self.state)

    def get_reward(self, state: np.ndarray = None, prev_state: np.ndarray = None, action: int = None) -> int:
        if state is None:
            state = self.state
        if state[-1] == 1:
            return 1
        return 0

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        if self.reward_in_obs:
            return np.array([1, self.get_reward(state=state)])
        return np.array([1])

    def get_terminal(self) -> bool:
        return self.state[-1] == 1

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        # Just go right
        idx = state.nonzero()[0].item()
        state[min(state.shape[0] - 1, idx + 1)] = 1
        state[idx] = 0

        return state

    def emit_prob(self, state: Any, obs: np.ndarray) -> float:
        return 1

    def step(self, action: int):
        self.state = self.transition(self.state.copy(), action)
        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}

class FullyObservableSimpleChain(SimpleChain):
    def __init__(self, n: int = 10, reward_in_obs: bool = False):
        super().__init__(n=n, reward_in_obs=reward_in_obs)
        self.observation_space = gym.spaces.MultiBinary(n)

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.state.copy()
