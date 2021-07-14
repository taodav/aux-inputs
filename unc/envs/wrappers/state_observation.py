import gym
import numpy as np
from typing import Union

from unc.envs.wrappers import CompassWorldWrapper
from unc.envs import CompassWorld


class StateObservationWrapper(CompassWorldWrapper):
    """
    Compass World with underlying state in the observation.

    Observations are structured like so:
    [y, x, dir, *obs]
    """
    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args, **kwargs):
        super(StateObservationWrapper, self).__init__(env, *args, **kwargs)

        self.observation_space = gym.spaces.Box(low=np.array([1, 1, 0, 0, 0, 0, 0, 0]), high=np.array([6, 6, 3, 1, 1, 1, 1, 1]), dtype=np.int16)

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = super(StateObservationWrapper, self).get_obs(state)
        state = self.state.copy()
        
        return np.concatenate((state, obs), axis=0)


