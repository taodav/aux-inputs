import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs.compass import CompassWorld
from unc.particle_filter import state_stats


class ObservationStatsWrapper(CompassWorldWrapper):
    priority = 3

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args,
                 mean_only: bool = False, vars_only: bool = False, **kwargs):
        """
        We encode our next observation given an action.
        Which action do we choose? Currently we use next action given the current policy.
        So, our observation is of shape
        obs_shape + num_actions + next_obs_shape
        :param env:
        """
        super(ObservationStatsWrapper, self).__init__(env, *args, **kwargs)
        assert not (mean_only and vars_only), "Can't have both mean and vars only"

        assert hasattr(self, 'particles') and hasattr(self, 'weights')

        self.mean_only = mean_only
        self.vars_only = vars_only

        if self.mean_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1,
                               1, 1, 1, 1, 1]))
        elif self.vars_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1,
                               1, 1, 1, 1, 1]))
        else:
            self.observation_space = gym.spaces.Box(

                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1]))

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Observation with mean/variance of pf prepended to the original observation
        :param state:
        :return:
        """

        # Update our particles and weights after doing a transition
        all_particle_obs = self.batch_get_obs(self.particles)

        # mean, variance = state_stats(all_particle_obs, np.ones(self.particles.shape[0]) / self.particles.shape[0])
        mean, variance = state_stats(all_particle_obs, self.weights)
        if self.mean_only:
            pf_state = np.array(mean)
        elif self.vars_only:
            pf_state = np.array(variance)
        else:
            pf_state = np.array(list(zip(mean, variance))).flatten()

        original_obs = self.env.get_obs(state)
        return np.concatenate((original_obs, pf_state), axis=0)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state, action), reward, done, info
