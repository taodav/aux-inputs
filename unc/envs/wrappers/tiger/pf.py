import numpy as np
from typing import Union, Tuple

from unc.envs.tiger import Tiger
from unc.particle_filter import batch_step, resample

from .wrapper import TigerWrapper


class TigerParticleFilterWrapper(TigerWrapper):
    priority = 2

    def __init__(self, env: Union[Tiger, TigerWrapper], *args,
                 update_weight_interval: int = 1,  resample_interval: int = None,
                 n_particles: int = 10,
                 **kwargs):
        super(TigerParticleFilterWrapper, self).__init__(env, *args, **kwargs)
        self.particles = None
        self.weights = None
        self.env_step = 0
        self.update_weight_interval = update_weight_interval
        self.n_particles = n_particles
        self.resample_interval = resample_interval if resample_interval is not None else float('inf')

    def reset(self, **kwargs) -> np.ndarray:
        """
        Here emittance probs are all 1 anyways. We don't need to take an empty step.
        """
        obs = self.env.reset()
        self.particles = self.rng.choice([0, 1], self.n_particles)

        self.weights = np.ones(self.n_particles) / self.n_particles

        return obs

    def step(self, action: int):
        prev_state = self.state
        self.state, self.particles, self.weights = self.transition(self.state, action, self.particles, self.weights)
        self.env_step += 1

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {}

    def transition(self, state: np.ndarray, action: int, particles: np.ndarray = None,
                   weights: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        new_state = self.env.transition(state, action)

        if particles is None or weights is None:
            return new_state

        obs = self.get_obs(new_state)

        # Update our particles and weights after doing a transition
        new_weights, new_particles = batch_step(weights, particles, obs,
                                                      self.batch_transition, self.emit_prob, action=action,
                                                      update_weights=True)

        if weights is None:
            # TODO: In case we need particle reinvigoration
            raise NotImplementedError()

        if self.env_step % self.resample_interval == 0:
            new_weights, new_particles = resample(new_weights, new_particles, rng=self.rng)

        return new_state, new_particles, new_weights
