import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing


class GVFWrapper(LobsterFishingWrapper):
    """
    In this wrapper, we create observations that just tile code the
    predictions, and append these predictions onto the observation.
    """
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(GVFWrapper, self).__init__(env)

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9 + 2),
            high=np.ones(9 + 2),
        )

        self._predictions = None

    @property
    def predictions(self):
        return self._predictions

    @predictions.setter
    def predictions(self, predictions: np.ndarray):
        self._predictions = predictions

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        unwrapped_obs = self.unwrapped.get_obs(state)

        return np.concatenate((unwrapped_obs, self.predictions), axis=-1)

    @property
    def gvf_idxes(self):
        return np.arange(9, 9 + 2)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
