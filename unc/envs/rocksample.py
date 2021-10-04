import json
import gym
import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
from jax.ops import index_update
from typing import Tuple
from itertools import product
from pathlib import Path

from unc.envs.base import Environment
from unc.utils.data import euclidian_dist, half_dist_prob


class RockSample(Environment):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, config_file: Path, rng: np.random.RandomState, rand_key: jax.random.PRNGKey):
        """
        RockSample environment.
        Observations: position (2) and rock goodness/badness
        Actions: k + 5 actions. Actions are as follows:
        0: North, 1: East, 2: South, 3: West,
        4: Sample,
        5: Check rock 1, ..., k + 5: Check rock k
        :param config_file: config file for RockSample (located in unc/envs/configs)
        :param seed: random seed for the environment.
        """
        with open(config_file) as f:
            config = json.load(f)
        self.size = config['size']
        self.k = config['rocks']
        self.half_efficiency_distance = config['half_efficiency_distance']
        self.bad_rock_reward = config['bad_rock_reward']
        self.good_rock_reward = config['good_rock_reward']
        self.exit_reward = config['exit_reward']

        self.observation_space = gym.spaces.MultiBinary(self.k + 2)
        self.state_space = gym.spaces.MultiBinary(2 + 3 * self.k)
        self.action_space = gym.spaces.Discrete(self.k + 5)
        self.rng = rng
        self.rand_key = rand_key
        self.position_max = np.array([self.size - 1, self.size - 1])
        self.position_min = np.array([0, 0])

        self.rock_positions = None
        self.rock_morality = None
        self.agent_position = None
        self.sampled_rocks = None
        self.checked_rocks = None
        self.current_rocks_obs = np.zeros(self.k, dtype=np.int16)

        # Given a random seed, generate the map
        self.generate_map()

    @property
    def state(self):
        """
        return the underlying state of the environment.
        State consists of 3k + 2 features
        2 are the position
        k are for the underlying rock morality
        k are for which rocks were already sampled
        k are for what the current rock observation is
        IN THIS ORDER
        :return: underlying environment state
        """
        ap = self.agent_position.copy()
        rm = self.rock_morality.copy()
        sr = self.sampled_rocks.copy()
        cro = self.current_rocks_obs.copy()
        return np.concatenate([ap, rm, sr, cro])

    @state.setter
    def state(self, state: np.ndarray):
        ap, rm, sr, cro = self.unpack_state(state)
        self.agent_position = ap
        self.rock_morality = rm
        self.sampled_rocks = sr
        self.current_rocks_obs = cro

    def generate_map(self):
        rows_range = np.arange(0, self.size)
        cols_range = rows_range[:-1]
        possible_rock_positions = np.array(list(product(rows_range, cols_range)), dtype=np.int16)
        all_positions_idx = self.rng.choice(possible_rock_positions.shape[0], self.k, replace=False)
        all_positions = possible_rock_positions[all_positions_idx]
        self.rock_positions = all_positions

    def sample_positions(self, n: int = 1):
        rows_range = np.arange(0, self.size)
        cols_range = rows_range[:-1]
        possible_positions = np.array(list(product(rows_range, cols_range)), dtype=np.int16)
        sample_idx = self.rng.choice(possible_positions.shape[0], n, replace=True)
        sample_position = np.array(possible_positions[sample_idx], dtype=np.int16)
        return sample_position

    def sample_all_states(self, rock_morality: np.ndarray = None) -> np.ndarray:
        """
        Return observations from all possible non-terminal agent positions,
        with a given rock morality.
        :param rock_morality:
        :return:
        """
        if rock_morality is None:
            rock_morality = self.rock_morality

        states = []

        for y in range(self.size):
            for x in range(self.size - 1):
                pos = np.array([y, x])
                state = np.concatenate([pos, rock_morality, self.sampled_rocks, self.current_rocks_obs])
                states.append(state)

        return np.stack(states)

    def sample_morality(self) -> np.ndarray:
        assert self.rock_positions is not None
        rand_int = self.rng.random_integers(0, (1 << self.k) - 1)
        return ((rand_int & (1 << np.arange(self.k))) > 0).astype(int)

    def get_terminal(self) -> bool:
        return self.agent_position[1] == (self.size - 1)

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Observation is dependent on action here.
        Observation is a k + 2 vector:
        first two features are the position.
        last k features are the current_rock_obs.
        :param state:
        :param action:
        :return:
        """
        position, _, _, rocks_obs = self.unpack_state(state)

        return np.concatenate([position, rocks_obs])

    def get_reward(self, prev_state: np.ndarray, action: int) -> int:
        """
        NOTE: this should be called AFTER transition happens
        :param action: action that you're planning on taking
        :return: reward
        """
        rew = 0

        if action == 4:
            # If we're SAMPLING
            ele = (self.rock_positions == self.agent_position)
            idx = np.nonzero(ele[:, 0] & ele[:, 1])[0]
            if idx.shape[0] > 0:
                idx = idx[0]
                # If we're on a rock, we get rewards by sampling accordingly
                _, prev_rock_morality, _, _ = self.unpack_state(prev_state)
                rew = self.good_rock_reward if prev_rock_morality[idx] > 0 else self.bad_rock_reward

        elif action < 4:
            # If we're MOVING
            if self.get_terminal():
                rew = self.exit_reward

        return rew

    def reset(self):
        self.sampled_rocks = np.zeros(len(self.rock_positions)).astype(bool)
        self.checked_rocks = np.zeros_like(self.sampled_rocks).astype(bool)
        self.rock_morality = self.sample_morality()
        self.agent_position = self.sample_positions(1)[0]
        self.current_rocks_obs = np.zeros_like(self.current_rocks_obs)

        return self.get_obs(self.state)

    def unpack_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position = state[:2]
        rock_morality = state[2:2 + self.k]
        sampled_rocks = state[2 + self.k:2 + 2 * self.k]
        current_rocks_obs = state[2 + 2 * self.k:]
        return position, rock_morality, sampled_rocks, current_rocks_obs

    @staticmethod
    def pack_state(position: np.ndarray, rock_morality: np.ndarray, sampled_rocks: np.ndarray,
                   current_rocks_obs: np.ndarray):
        return np.concatenate([position, rock_morality, sampled_rocks, current_rocks_obs])

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @partial(jit, static_argnums=0)
    def _check_transition(self, current_rocks_obs: np.ndarray,
                          position: np.ndarray,
                          rock_positions: np.ndarray,
                          rock_morality: np.ndarray,
                          rand_key: random.PRNGKey,
                          action: int):
        rock_idx = action - 5
        dist = euclidian_dist(position, rock_positions[rock_idx])
        prob = half_dist_prob(dist, self.half_efficiency_distance)

        # w.p. prob we return correct rock observation.
        rock_obs = rock_morality[rock_idx]
        choices = jnp.array([rock_obs, 1 - rock_obs])
        probs = jnp.array([prob, 1 - prob])
        key, subkey = random.split(rand_key)
        rock_obs = random.choice(subkey, choices, (1, ), p=probs)[0]

        new_rocks_obs = index_update(current_rocks_obs, rock_idx, rock_obs)
        return new_rocks_obs, key

    @partial(jit, static_argnums=0)
    def _move_transition(self, position: np.ndarray,
                         direction_mapping: np.ndarray,
                         position_max: np.ndarray,
                         position_min: np.ndarray,
                         action: int):
        new_pos = position + direction_mapping[action]
        position = jnp.maximum(jnp.minimum(new_pos, position_max), position_min)
        return position

    @partial(jit, static_argnums=0)
    def _sample_transition(self, position: np.ndarray,
                           sampled_rocks: np.ndarray,
                           current_rocks_obs: np.ndarray,
                           rock_mortality: np.ndarray,
                           rock_positions: np.ndarray):
        ele = (rock_positions == position)
        bool_pos = (ele[:, 0] & ele[:, 1]).astype(int)

        zero_arr = jnp.zeros_like(current_rocks_obs)
        new_sampled_rocks = jnp.minimum(sampled_rocks + bool_pos, zero_arr + 1)
        new_rocks_obs = jnp.maximum(current_rocks_obs - bool_pos, zero_arr)
        new_rock_morality = jnp.maximum(rock_mortality - bool_pos, zero_arr)
        return new_sampled_rocks, new_rocks_obs, new_rock_morality

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        position, rock_morality, sampled_rocks, current_rocks_obs = self.unpack_state(state)

        if action > 4:
            # CHECK
            new_rocks_obs, self.rand_key = self._check_transition(current_rocks_obs.copy(), position, self.rock_positions,
                                                                      rock_morality, self.rand_key, action)
            current_rocks_obs = new_rocks_obs
        elif action == 4:
            # SAMPLING
            new_sampled_rocks, new_rocks_obs, new_rock_morality = self._sample_transition(
                position, sampled_rocks, current_rocks_obs, rock_morality, self.rock_positions
            )
            sampled_rocks, current_rocks_obs, rock_morality = new_sampled_rocks, new_rocks_obs, new_rock_morality

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            position = self._move_transition(position, self.direction_mapping,
                                             self.position_max, self.position_min,
                                             action)

        return self.pack_state(position, rock_morality, sampled_rocks, current_rocks_obs)

    def emit_prob(self, states: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        With rock sample, we use an UNWEIGHTED particle filter
        (like in the POMCP paper). This function will simply return an array
        of shape (len(states),) of 1's
        :param states: size (batch_size, *state_shape)
        :param obs: size (batch_size, *obs_shape)
        :return: array of 1's of shape (batch_size, )
        """
        if len(states.shape) > 1:
            ones = np.ones(states.shape[0])
        else:
            ones = np.ones(1)
        return ones

    def step(self, action: int):
        prev_state = self.state
        self.state = self.transition(self.state, action)

        rock_idx = action - 5
        if rock_idx >= 0:
            self.checked_rocks[rock_idx] = True

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {}

    def generate_array(self) -> np.ndarray:
        """
        Generate numpy array representing state.
        Mappings are as follows:
        0 = white space
        1 = agent
        2 = rock
        3 = agent + rock
        4 = goal
        :return:
        """
        viz_array = np.zeros((self.size, self.size))

        viz_array[self.rock_positions[:, 0], self.rock_positions[:, 1]] = 2

        viz_array[self.agent_position[0], self.agent_position[1]] += 1

        viz_array[:, self.size - 1] = 4
        return viz_array



