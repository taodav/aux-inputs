import numpy as np
from typing import Union, Tuple

from unc.envs.rocksample import RockSample
from unc.particle_filter import batch_step, resample
from .wrapper import RockSampleWrapper


class RocksParticleFilterWrapper(RockSampleWrapper):
    """
    Particle filter for only the rock observations.

    Code: p

    Observations are structured like so:
    [pos_x, pos_y, rock_1_weights, rock_2_weights, ..., rock_k_weights]

    """
    priority = 2

    def __init__(self, env: Union[RockSample, RockSampleWrapper], *args,
                 update_weight_interval: int = 1,  resample_interval: int = None,
                 n_particles: int = 100,
                 **kwargs):
        super(RocksParticleFilterWrapper, self).__init__(env, *args, **kwargs)
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
        # self.particles = np.zeros((self.n_particles, self.rocks))
        self.particles = self.rng.choice([0, 1], (self.n_particles, self.rocks))

        self.weights = np.ones(self.particles.shape[0]) / self.particles.shape[0]

        return obs

    def obs2state(self, obses: np.ndarray) -> np.ndarray:
        """
        Assume that first dimension of obses is n_particles.
        :param obses: (n_particles, obs_shape)
        :return: (n_particles, state_shape)
        """
        ap = self.agent_position.copy()
        rm = self.rock_morality.copy()
        sr = self.sampled_rocks.copy()
        partial_state = np.concatenate([ap, rm, sr])[None, :]
        repeated_partial_state = partial_state.repeat(obses.shape[0], axis=0)
        particle_states = np.concatenate([repeated_partial_state, obses], axis=1)
        return particle_states

    def step(self, action: int):
        prev_state = self.state
        self.state, self.particles, self.weights = self.transition(self.state, action, self.particles, self.weights)
        self.env_step += 1

        rock_idx = action - 5
        if rock_idx >= 0:
            self.checked_rocks[rock_idx] = True

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {}

    def transition(self, state: np.ndarray, action: int, particles: np.ndarray = None,
                   weights: np.ndarray = None) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        new_state = self.env.transition(state, action)

        if particles is None or weights is None:
            return new_state

        obs = self.get_obs(new_state)

        # Update our particles and weights after doing a transition
        particle_states = self.obs2state(particles)
        new_weights, new_particle_states = batch_step(weights, particle_states, obs,
                                                      self.batch_transition, self.emit_prob, action=action,
                                                      update_weights=True)
        new_particles = new_particle_states[:, -self.rocks:]

        if weights is None:
            # TODO: In case we need particle reinvigoration
            raise NotImplementedError()

        if self.env_step % self.resample_interval == 0:
            new_weights, new_particles = resample(new_weights, new_particles, rng=self.rng)

        return new_state, new_particles, new_weights
