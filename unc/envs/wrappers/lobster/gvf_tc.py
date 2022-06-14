from PyFixedReps import TileCoder
import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing, all_lobster_states


class GVFTileCodingWrapper(LobsterFishingWrapper):
    """
    In this wrapper, we create observations that just tile code the
    predictions, and append these predictions onto the observation.
    """
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper],
                 max_episode_steps: int = 200):
        super(GVFTileCodingWrapper, self).__init__(env)
        self.gvf_features = 2
        self.tc = TileCoder({
            'tiles': 4,
            'tilings': 16,
            'dim': self.gvf_features,

            'input_ranges': [(0, max_episode_steps), (0, max_episode_steps)],
            'scale_input': False
        })

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9 + self.tc.features()),
            high=np.ones(9 + self.tc.features()),
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

        tc_obs = self.tc.encode(self._predictions)

        return np.concatenate((unwrapped_obs, tc_obs), axis=0)

    @property
    def gvf_idxes(self):
        return np.arange(9, 9 + self.gvf_features)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
