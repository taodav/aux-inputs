import numpy as np
from functools import partial
from jax import jit, random, vmap
from jax.ops import index_update
from .wrapper import RockSampleWrapper


class PerfectSensorWrapper(RockSampleWrapper):

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        action = actions[0]
        bs = states.shape[0]
        positions = states[:, :2]
        rock_moralities = states[:, 2:2 + self.k]
        sampled_rockses = states[:, 2 + self.k:2 + 2 * self.k]
        current_rocks_obses = states[:, 2 + 2 * self.k:]

        rock_positionses = np.repeat(np.expand_dims(self.rock_positions, 0), bs, axis=0)
        direction_mappings = np.repeat(np.expand_dims(self.direction_mapping, 0), bs, axis=0)
        position_maxes = np.repeat(np.expand_dims(self.position_max, 0), bs, axis=0)
        position_mins = np.repeat(np.expand_dims(self.position_min, 0), bs, axis=0)
        rand_keys = random.split(self.env.rand_key, num=bs + 1)
        self.env.rand_key, rand_keys = rand_keys[0], rand_keys[1:]

        if action > 4:
            # CHECK
            new_rocks_obses, rand_keys = vmap(self._check_transition)(current_rocks_obses, positions, rock_positionses,
                                                                      rock_moralities, rand_keys, actions)
            current_rocks_obses = new_rocks_obses
        elif action == 4:
            # SAMPLING
            new_sampled_rockses, new_rocks_obses, new_rock_moralities = vmap(self.env._sample_transition)(
                positions, sampled_rockses, current_rocks_obses, rock_moralities, rock_positionses
            )
            sampled_rockses, current_rocks_obses, rock_moralities = new_sampled_rockses, new_rocks_obses, new_rock_moralities

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            positions = vmap(self.env._move_transition)(positions, direction_mappings,
                                                    position_maxes, position_mins,
                                                    actions)
        return np.concatenate([positions, rock_moralities, sampled_rockses, current_rocks_obses], axis=1)

    @partial(jit, static_argnums=0)
    def _check_transition(self, current_rocks_obs: np.ndarray,
                          position: np.ndarray,
                          rock_positions: np.ndarray,
                          rock_morality: np.ndarray,
                          rand_key: random.PRNGKey,
                          action: int):
        rock_idx = action - 5
        # w.p. prob we return correct rock observation.
        rock_obs = rock_morality[rock_idx]

        new_rocks_obs = index_update(current_rocks_obs, rock_idx, rock_obs)
        return new_rocks_obs, rand_key

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        position, rock_morality, sampled_rocks, current_rocks_obs = self.unpack_state(state)

        if action > 4:
            # CHECK
            new_rocks_obs, self.env.rand_key = self._check_transition(current_rocks_obs.copy(), position, self.rock_positions,
                                                                  rock_morality, self.rand_key, action)
            current_rocks_obs = new_rocks_obs
        elif action == 4:
            # SAMPLING
            new_sampled_rocks, new_rocks_obs, new_rock_morality = self.env._sample_transition(
                position, sampled_rocks, current_rocks_obs, rock_morality, self.rock_positions
            )
            sampled_rocks, current_rocks_obs, rock_morality = new_sampled_rocks, new_rocks_obs, new_rock_morality

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            position = self.env._move_transition(position, self.direction_mapping,
                                             self.position_max, self.position_min,
                                             action)

        return self.pack_state(position, rock_morality, sampled_rocks, current_rocks_obs)


    def step(self, action: int):
        prev_state = self.state
        self.state = self.transition(self.state, action)

        rock_idx = action - 5
        if rock_idx >= 0:
            self.env.checked_rocks[rock_idx] = True

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {}
