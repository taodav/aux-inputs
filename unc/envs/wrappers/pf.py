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

    if mean_only is True:
    [mean_y, mean_x, mean_dir, *obs]
    """
    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args,
                 update_weight_interval: int = 1, mean_only: bool = False,
                 vars_only: bool = False, **kwargs):
        super(ParticleFilterWrapper, self).__init__(env, *args, **kwargs)
        assert not (mean_only and vars_only), "Can't have both mean and vars only"
        self.particles = None
        self.weights = None
        self.env_step = 0
        self.update_weight_interval = update_weight_interval
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
        # Instantiate particles and weights
        self.particles = self.sample_all_states()
        self.weights = np.ones(self.particles.shape[0]) / self.particles.shape[0]

        # Update them based on the first observation
        original_obs = self.env.get_obs(self.state)
        self.weights, self.particles = step(self.weights, self.particles, original_obs,
                                            self.transition, self.emit_prob)

        return self.get_obs(self.state)

    def step(self, action: int):
        original_obs, reward, done, info = self.env.step(action)
        self.env_step += 1

        # Update our particles and weights after doing a transition
        self.weights, self.particles = step(self.weights, self.particles, original_obs,
                                            self.transition, self.emit_prob, action=action,
                                            update_weights=self.env_step % self.update_weight_interval == 0)

        return self.get_obs(self.state), reward, done, info

