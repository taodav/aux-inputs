import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs.compass import CompassWorld
from unc.particle_filter import state_stats


class LocalStateObservationWrapper(CompassWorldWrapper):
    priority = 3

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args,
                 mean_only: bool = False, vars_only: bool = False, **kwargs):
        """
        Observations are structured like so:
        [mean_y, var_y, mean_x, var_x, mean_dir, var_dir, *obs]

        if mean_only is True:
        [mean_y, mean_x, mean_dir, *obs]

        similar for if vars_only is True.
        Locally encode observation wrapper. Local in this case meaning
        with summary statistics of the particles
        :param env:
        """
        super(LocalStateObservationWrapper, self).__init__(env, *args, **kwargs)
        assert not (mean_only and vars_only), "Can't have both mean and vars only"

        assert hasattr(self, 'particles') and hasattr(self, 'weights')
        self.mean_only = mean_only
        self.vars_only = vars_only

        if self.mean_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([1, 1, 0, 0, 0, 0, 0, 0]),
                high=np.array([6, 6, 4, 1, 1, 1, 1, 1]))
        elif self.vars_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, 0, 0, 0, 0]),
                high=np.array([36, 36, 16, 1, 1, 1, 1, 1])
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
                high=np.array([6, 36, 6, 36, 4, 16, 1, 1, 1, 1, 1]))

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Observation with mean/variance of pf prepended to the original observation
        :param state:
        :return:
        """
        mean, variance = state_stats(self.particles, self.weights)
        if self.mean_only:
            pf_state = np.array(mean)
        elif self.vars_only:
            pf_state = np.array(variance)
        else:
            pf_state = np.array(list(zip(mean, variance))).flatten()

        original_obs = self.env.get_obs(state)
        return np.concatenate((pf_state, original_obs), axis=0)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
