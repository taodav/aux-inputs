import numpy as np
from typing import Union

from unc.envs.compass import CompassWorld
from unc.particle_filter import batch_step, resample
from .wrapper import CompassWorldWrapper


class CompassParticleFilterWrapper(CompassWorldWrapper):
    """
    Particle filter (not incl observations, see local_state wrapper).


    n_particles describes the number of particles to start with and maintain.
    if the value is set to -1, we start with |S| - 1 number of particles (number of states
    less terminal state).
    """
    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], *args,
                 update_weight_interval: int = 1,  resample_interval: int = None,
                 n_particles: int = -1,
                 **kwargs):
        super(CompassParticleFilterWrapper, self).__init__(env, *args, **kwargs)
        self.particles = None
        self.weights = None
        self.env_step = 0
        self.update_weight_interval = update_weight_interval
        self.n_particles = n_particles
        self.resample_interval = resample_interval if resample_interval is not None else float('inf')

    def reset(self, **kwargs) -> np.ndarray:
        color_obs = self.env.reset(**kwargs)
        # Instantiate particles and weights
        if self.n_particles == -1:
            self.particles = self.sample_all_states()
        else:
            self.particles = self.sample_states(n=self.n_particles)
        self.weights = np.ones(self.particles.shape[0]) / self.particles.shape[0]

        # Update them based on the first observation
        self.weights, self.particles = batch_step(self.weights, self.particles, color_obs,
                                            self.batch_transition, self.emit_prob)

        return color_obs

    def step(self, action: int):
        color_obs, reward, done, info = self.env.step(action)
        self.env_step += 1

        # Update our particles and weights after doing a transition
        self.weights, self.particles = batch_step(self.weights, self.particles, color_obs,
                                            self.batch_transition, self.emit_prob, action=action,
                                            update_weights=self.env_step % self.update_weight_interval == 0)

        if self.weights is None:
            # If all our weights are 0, we get None for weights and have to
            # re-initialize particles
            if self.n_particles == -1:
                self.particles = self.sample_all_states()
            else:
                self.particles = self.sample_states(n=self.n_particles)

            self.weights = np.ones(self.particles.shape[0]) / self.particles.shape[0]

        if self.env_step % self.resample_interval == 0:
            self.weights, self.particles = resample(self.weights, self.particles, rng=self.rng)

        return color_obs, reward, done, info

