import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs.compass import CompassWorld
from unc.particle_filter import batch_step
from unc.particle_filter import state_stats


class NextObservationStatsWrapper(CompassWorldWrapper):
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
        super(NextObservationStatsWrapper, self).__init__(env, *args, **kwargs)
        assert not (mean_only and vars_only), "Can't have both mean and vars only"

        assert hasattr(self, 'particles') and hasattr(self, 'weights')

        self.mean_only = mean_only
        self.vars_only = vars_only

        if self.mean_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1,
                               1, 1, 1, 1, 1]))
        elif self.vars_only:
            self.observation_space = gym.spaces.Box(
                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1,
                               1, 1, 1, 1, 1]))
        else:
            self.observation_space = gym.spaces.Box(

                low=np.array([0, 0, 0, 0, 0,
                              0, 0, 0,
                              0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0]),
                high=np.array([1, 1, 1, 1, 1,
                               1, 1, 1,
                               1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1]))

    def get_obs(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Observation with mean/variance of pf prepended to the original observation
        :param state:
        :return:
        """

        # Update our particles and weights after doing a transition
        batch_actions = np.zeros(self.particles.shape[0], dtype=int) + action
        next_particles = self.batch_transition(self.particles, batch_actions)
        mean, variance = state_stats(next_particles, np.ones(self.particles.shape[0]) / self.particles.shape[0])

        # mean, variance = state_stats(next_particles, self.weights)
        if self.mean_only:
            pf_state = np.array(mean)
        elif self.vars_only:
            pf_state = np.array(variance)
        else:
            pf_state = np.array(list(zip(mean, variance))).flatten()

        original_obs = self.env.get_obs(state)
        return np.concatenate((pf_state, original_obs), axis=0)
