import json
import gym
import numpy as np
from typing import Any, Tuple
from itertools import product
from pathlib import Path

from unc.envs.base import Environment
from unc.utils.data import euclidian_dist, manhattan_dist, half_dist_prob

class RockSample(Environment):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, config_file: Path, seed: int):
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
        self.n = config['size']
        self.k = config['rocks']
        self.half_efficiency_distance = config['half_efficiency_distance']
        self.bad_rock_reward = config['bad_rock_reward']
        self.good_rock_reward = config['good_rock_reward']
        self.exit_reward = config['exit_reward']
        self.seed = seed

        self.observation_space = gym.spaces.Discrete(self.k + 2)
        self.action_space = gym.spaces.Discrete(self.k + 5)
        self.rng = np.random.RandomState(seed)
        self.position_max = [self.n - 1, self.n - 1]
        self.position_min = [0, 0]

        self.rock_positions = None
        self.rock_morality = None
        self.agent_position = None
        self.agent_init_position = None
        self.sampled_rocks = None
        self.current_rocks_obs = np.zeros(self.k)

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
        rows_range = np.arange(0, self.n)
        cols_range = rows_range[:-1]
        possible_rock_positions = np.array(list(product(rows_range, cols_range)))
        all_positions_idx = self.rng.choice(possible_rock_positions.shape[0], self.k + 1, replace=False)
        all_positions = possible_rock_positions[all_positions_idx]
        self.agent_init_position = all_positions[0]
        self.rock_positions = all_positions[1:]

    def sample_morality(self) -> np.ndarray:
        assert self.rock_positions is not None
        rand_int = self.rng.random_integers(0, (1 << self.k) - 1)
        return ((rand_int & (1 << np.arange(self.k))) > 0).astype(int)

    def get_terminal(self) -> bool:
        return self.agent_position[1] == (self.n - 1)

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
        pposition, prock_morality, psampled_rocks, pcurrent_rocks_obs = self.unpack_state(prev_state)

        if action == 4:
            # If we're SAMPLING
            ele = (self.rock_positions == self.agent_position)
            idx = np.nonzero(ele[:, 0] & ele[:, 1])[0]
            if idx.shape[0] > 0:
                # If we're on a rock, we get rewards by sampling accordingly
                rew = self.good_rock_reward if self.rock_morality[idx] > 0 else self.bad_rock_reward

        elif action < 4:
            # If we're MOVING
            if self.get_terminal():
                rew = self.exit_reward

        return rew

    def reset(self):
        self.sampled_rocks = np.zeros(len(self.rock_positions)).astype(bool)
        self.rock_morality = self.sample_morality()
        self.agent_position = self.agent_init_position.copy()

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

    def transition(self, state: Any, action: int) -> np.ndarray:
        position, rock_morality, sampled_rocks, current_rocks_obs = self.unpack_state(state)

        if action > 4:
            # CHECK
            new_rocks_obs = current_rocks_obs.copy()
            rock_idx = action - 5
            dist = euclidian_dist(position, self.rock_positions[rock_idx])
            prob = half_dist_prob(dist, self.half_efficiency_distance)

            # w.p. prob we return correct rock observation.
            rock_obs = rock_morality[rock_idx]
            if self.rng.random() > prob:
                rock_obs = 1 - rock_obs

            new_rocks_obs[rock_idx] = rock_obs
            current_rocks_obs = new_rocks_obs
        elif action == 4:
            # SAMPLING
            ele = (self.rock_positions == position)
            idx = np.nonzero(ele[:, 0] & ele[:, 1])[0]

            if idx.shape[0] > 0:
                # If we're on a rock
                idx = idx[0]
                new_sampled_rocks = sampled_rocks.copy()
                new_rocks_obs = current_rocks_obs.copy()

                new_sampled_rocks[idx] = 1

                # If this rock was actually good, we sampled it now it turns bad.
                # Elif this rock is bad, we sample a bad rock and return 0
                new_rocks_obs[idx] = 0

            # If we sample a space with no rocks, nothing happens for transition.
        else:
            # MOVING
            new_pos = position + self.direction_mapping[action]
            position = np.maximum(np.minimum(new_pos, self.position_max), self.position_min)

        return self.pack_state(position, rock_morality, sampled_rocks, current_rocks_obs)

    def step(self, action: int):
        next_state = self.transition(self.state, action)
        prev_state = self.state.copy()
        self.state = next_state
        rew = self.get_reward(prev_state, action)







