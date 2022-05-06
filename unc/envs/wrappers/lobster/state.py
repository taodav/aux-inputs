import gym
import numpy as np
from typing import Tuple, Union

from .wrapper import LobsterFishing, LobsterFishingWrapper


class GroundTruthStateWrapper(LobsterFishingWrapper):
    """
    Lobster environment with the ground-truth state as observation
    """
    priority = 2

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(GroundTruthStateWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(7), high=np.ones(7)
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        state = state.astype(int)
        position = np.zeros(3)
        position[state[0]] = 1
        r1_obs = np.zeros(2)
        r2_obs = np.zeros(2)

        r1_obs[state[1]] = 1
        r2_obs[state[2]] = 1

        return np.concatenate((position, r1_obs, r2_obs))

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
