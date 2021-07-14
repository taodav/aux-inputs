import gym
import numpy as np
from typing import Union

from unc.envs import CompassWorld
from unc.envs.wrappers import CompassWorldWrapper
from unc.particle_filter import step, state_stats


class ParticleFilterWrapper(CompassWorldWrapper):
    """
    Particle filter observations.

    Observations are structured like so:
    [mean_y, var_y, mean_x, var_x, mean_dir, var_dir, *obs]
    """

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args, **kwargs):
        super(ParticleFilterWrapper, self).__init__(env, *args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            high=np.array([6, 36, 6, 36, 4, 16, 1, 1, 1, 1, 1]))

        self.particles = None
        self.weights = None

    @property
    def priority(self) -> int:
        return 2

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Observation with mean/variance of pf prepended to the original observation
        :param state:
        :return:
        """
        mean, variance = state_stats(self.particles, self.weights)
        pf_state = np.array(list(zip(mean, variance)))
        
        original_obs = super(ParticleFilterWrapper, self).get_obs(state)
        return np.concatenate((pf_state, original_obs), axis=0)

    def reset(self, **kwargs) -> np.ndarray:
        super(ParticleFilterWrapper, self).reset(**kwargs)
        # Instantiate particles and weights
        self.particles = self.sample_all_states()
        self.weights = np.ones(self.particles.shape[0]) / self.particles.shape[0]

        # Update them based on the first observation
        original_obs = super(ParticleFilterWrapper, self).get_obs(self.state)
        self.weights, self.particles = step(self.weights, self.particles, original_obs,
                                            self.transition, self.emit_prob)

        return self.get_obs(self.state)

    def step(self, action: int):
        original_obs, reward, done, info = super(ParticleFilterWrapper, self).step(action)

        # Update our particles and weights after doing a transition
        self.weights, self.particles = step(self.weights, self.particles, original_obs,
                                            self.transition, self.emit_prob)

        return self.get_obs(self.state), reward, done, info

