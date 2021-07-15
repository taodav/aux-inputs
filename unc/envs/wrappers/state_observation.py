import gym
import numpy as np
from typing import Union, Tuple

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
        obs = self.env.get_obs(state)
        state = state.copy()
        
        return np.concatenate((state, obs), axis=0)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info




