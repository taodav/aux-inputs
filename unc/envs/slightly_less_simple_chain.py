import numpy as np
import gym

from unc.envs import Environment


class SlightlyLessSimpleChain(Environment):
    def __init__(self, n: int = 10):
        """
        (Slightly less) simple chain environment, where we have two actions
        (left, right) and our observations are a one-hot encoding of the current
        state.
        """
        self.n = n
        self._state = np.zeros(self.n)
        self.current_idx = 0
        self.observation_space = gym.spaces.MultiBinary(self.n)
        self.action_space = gym.spaces.Discrete(2)

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
        return state.copy()

    def get_terminal(self) -> bool:
        return self.state[-1] == 1

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        idx = state.nonzero()[0].item()
        new_idx = -1
        if action == 0:
            # Go left
            new_idx = max(0, idx - 1)
            state[new_idx] = 1
        elif action == 1:
            # Go right
            new_idx = min(state.shape[0] - 1, idx + 1)
            state[new_idx] = 1
        else:
            raise NotImplementedError

        if new_idx != idx:
            state[idx] = 0

        return state

    def step(self, action: int):
        self.state = self.transition(self.state.copy(), action)
        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}
