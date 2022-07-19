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
        self.gvf_features = 1
        # self.gvf_features = 2
        self.tcs = [TileCoder({
            # 'tiles': 8,
            'tiles': 16,
            # 'tilings': 1,
            'tilings': 4,
            'dims': 1,

            'input_ranges': [(0, 1)],
            'scale_output': False
        }) for _ in range(self.gvf_features)]

        tc_features = sum(tc.features() for tc in self.tcs)

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9 + tc_features),
            high=np.ones(9 + tc_features),
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

        tc_obses = []
        for i, tc in enumerate(self.tcs):
            tc_obses.append(tc.encode(self._predictions[i : i + 1]))
        # tc1_obs = self.tc1.encode(self._predictions[0:1])
        # tc2_obs = self.tc2.encode(self._predictions[1:2])

        return np.concatenate((unwrapped_obs, *tc_obses), axis=0)

    @property
    def gvf_idxes(self):
        return np.arange(9, 9 + self.gvf_features)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
