from PyFixedReps import TileCoder
import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing


class GVFTileCodingWrapper(LobsterFishingWrapper):
    """
    In this wrapper, we create observations that just tile code the
    predictions, and append these predictions onto the observation.
    """
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(GVFTileCodingWrapper, self).__init__(env)
        self.gvf_features = 2
        self.tc1 = TileCoder({
            # 'tiles': 8,
            'tiles': 16,
            # 'tilings': 1,
            'tilings': 4,
            'dims': 1,

            'input_ranges': [(0, 1)],
            'scale_output': False
        })
        self.tc2 = TileCoder({
            # 'tiles': 8,
            'tiles': 16,
            # 'tilings': 1,
            'tilings': 4,
            'dims': 1,

            'input_ranges': [(0, 1)],
            'scale_output': False
        })

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9 + self.tc1.features() + self.tc2.features()),
            high=np.ones(9 + self.tc1.features() + self.tc2.features()),
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

        tc1_obs = self.tc1.encode(self._predictions[0:1])
        tc2_obs = self.tc1.encode(self._predictions[1:2])

        return np.concatenate((unwrapped_obs, tc1_obs, tc2_obs), axis=0)

    @property
    def gvf_idxes(self):
        return np.arange(9, 9 + self.gvf_features)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
