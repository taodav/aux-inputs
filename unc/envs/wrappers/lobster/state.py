import gym
import numpy as np
from typing import Tuple, Union

from .wrapper import LobsterFishing, LobsterFishingWrapper


class GroundTruthStateWrapper(LobsterFishingWrapper):
    """
    Lobster environment with the ground-truth state as observation
    """
    priority = 2

    state_mapping = [
        [0, 3, 6, 9],
        [1, 4, 7, 10],
        [2, 5, 8, 11]
    ]

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(GroundTruthStateWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(12), high=np.ones(12)
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        state = state.astype(int)

        state_vector = np.zeros(12)
        possible_states = self.state_mapping[state[0]]

        r1 = state[1]
        r2 = state[2]

        if r1 and r2:
            rew_idx = 0
        elif not r1 and r2:
            rew_idx = 1
        elif r1 and not r2:
            rew_idx = 2
        else:
            rew_idx = 3

        state_idx = possible_states[rew_idx]

        state_vector[state_idx] = 1

        return state_vector

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
