import gym
import numpy as np
from typing import Tuple

from .base import Environment


class Tiger(Environment):
    def __init__(self,
                 rng: np.random.RandomState = np.random.RandomState(),
                 noise: float = 0.15,
                 random_non_listen_obs: bool = False
                 ):
        super(Tiger, self).__init__()
        """
        Actions are defined as such:
        0 - Open door 0
        1 - Open door 1
        2 - Listen
        """
        self.action_space = gym.spaces.Discrete(3)
        self.rng = rng

        # State is a 2 dimensional array, with elements:
        # [tiger_position, last_action]
        self._state = None
        self.noise = noise

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        self._state = state

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(2)

        tiger_position = state[0]
        prev_action = state[1]

        if prev_action == 2:
            if self.rng.random() > self.noise:
                obs[tiger_position] = 1
            else:
                obs[1 - tiger_position] = 1
        elif prev_action == -1:
            # Here is the special case of initial observation. We show a random tiger position
            rand_tiger_pos = self.rng.choice([0, 1])
            obs[rand_tiger_pos] = 1
        else:
            obs[tiger_position] = 1

        return obs

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        new_state[1] = action

        return new_state

    def get_reward(self, prev_state: np.ndarray = None, action: int = None) -> float:
        if self.state[1] > 1:
            return -0.1
        elif self.state[1] == self.state[0]:
            return 1
        else:
            return -1

    def get_terminal(self) -> bool:
        return self.state[1] < 2

    def reset(self) -> np.ndarray:
        tiger_position = self.rng.choice([0, 1])
        self.state = np.array([tiger_position, -1])

        return self.get_obs(self.state)

    def emit_prob(self, states: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Get emittance probabilities.
        states: shape is batch_size x 2
        obs: shape is batch_size x 2
        """
        return obs[states[:, 0]] * (1 - self.noise)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.state = self.transition(self.state, action)
        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}

