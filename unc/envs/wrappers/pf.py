import gym
import numpy as np
from typing import Union

from unc.envs import CompassWorld
from unc.envs.wrappers import CompassWorldWrapper
from unc.particle_filter import step, resample, state_stats


class ParticleFilterWrapper(CompassWorldWrapper):
    """
    Particle filter observations.

    Observations are structured like so:
    [mean_y, var_y, mean_x, var_x, mean_dir, var_dir, *obs]

    if mean_only is True:
    [mean_y, mean_x, mean_dir, *obs]

    similar for if vars_only is True.

    n_particles describes the number of particles to start with and maintain.
    if the value is set to -1, we start with |S| - 1 number of particles (number of states
    less terminal state).
    """
    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args,
                 update_weight_interval: int = 1, mean_only: bool = False,
                 vars_only: bool = False, resample_interval: int = None,
                 n_particles: int = -1,
                 **kwargs):
        super(ParticleFilterWrapper, self).__init__(env, *args, **kwargs)
        assert not (mean_only and vars_only), "Can't have both mean and vars only"
        self.particles = None
        self.weights = None
        self.env_step = 0
        self.update_weight_interval = update_weight_interval
        self.mean_only = mean_only
        self.vars_only = vars_only
        self.n_particles = n_particles
        self.resample_interval = resample_interval if resample_interval is not None else float('inf')

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
        if self.n_particles == -1:
            self.particles = self.sample_all_states()
        else:
            self.particles = self.sample_states(n=self.n_particles)
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

        if self.env_step % self.resample_interval == 0:
            self.weights, self.particles = resample(self.weights, self.particles, rng=self.rng)

        return self.get_obs(self.state), reward, done, info

