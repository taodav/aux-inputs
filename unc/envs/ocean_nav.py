import gym
import numpy as np
from typing import Tuple


from unc.envs.base import Environment
from unc.utils.data import ind_to_one_hot


def pos_to_map(pos: np.ndarray, size: int):
    pos_map = np.zeros((size, size), dtype=np.int16)
    pos_map[pos[0], pos[1]] = 1
    return pos_map


class OceanNav(Environment):
    """
    Ocean navigation environment. We have quite a few more state variables to deal with here,
    So we keep this "epistemic state" environment in group_info.

    In currents, for each group of currents, we have the following key-value mapping:
    mapping: For this group, where are the currents located?
    refresh_rate: On average, how many timesteps until the current flips?
    directions: Directions in which we could sample from for this group.

    Note: this is currently the fully observable version of this environment.
    For partial observability, use the pertaining wrapper
    """
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, rng: np.random.RandomState, config: dict):
        super(OceanNav, self).__init__()
        self.config = config
        self.size = self.config['size']
        self.rng = rng
        self.current_bump_reward = self.config['current_bump_reward']

        self.position = None
        self.reward = None

        self.position_max = np.array([self.size - 1, self.size - 1], dtype=int)
        self.position_min = np.array([0, 0], dtype=int)

        # state space is flattened current map + position + reward position
        # obstacle_map is considered as part of epistemic state.
        self.state_space = gym.spaces.MultiDiscrete(self.size * self.size + 2 + 2)
        self.action_space = gym.spaces.Discrete(4)

        # low = np.zeros((7, self.size, self.size))
        # high = np.ones((7, self.size, self.size))
        low = np.zeros((self.size, self.size, 7))
        high = np.ones((self.size, self.size, 7))
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

        # Check configs for what this dict looks like
        self.currents = self.config['currents']

        # we get the per-timestep prob of changing
        self.current_inverse_rates = np.array([g['change_rate'] for g in self.currents])
        self.lambs = 1 / self.current_inverse_rates
        self.pmfs_1 = self.lambs * np.exp(-self.lambs)

        # Map of all obstacles
        self.obstacle_map = np.array(self.config['obstacle_map'], dtype=np.int16)

        # Map of all current directions
        self.current_map = np.zeros((self.size, self.size), dtype=np.int16)

        # Start positions to sample from
        self.start_positions = np.array(self.config['starts'], dtype=np.int16)

        # Reward positions to sample from. Note: we can only have 1 active reward in an env currently.
        # TODO (maybe): Extend this to multiple rewards
        self.possible_reward_positions = np.array(self.config['rewards'], dtype=np.int16)

    def reset_currents(self):
        self.current_map = np.zeros((self.size, self.size), dtype=np.int16)
        for curr_info in self.currents:
            sampled_curr_direction = self.rng.choice(curr_info['directions'])
            for y, x in curr_info['mapping']:
                self.current_map[y, x] = sampled_curr_direction + 1

    def reset(self) -> np.ndarray:
        self.reset_currents()
        self.position = self.start_positions[self.rng.choice(range(len(self.start_positions)))]
        self.reward = self.possible_reward_positions[self.rng.choice(range(len(self.possible_reward_positions)))]
        return self.get_obs(self.state)

    @property
    def state(self):
        flattened_current_map = self.current_map.flatten()
        return np.concatenate([flattened_current_map, self.position, self.reward])

    @state.setter
    def state(self, state: np.ndarray):
        self.current_map, self.position, self.reward = self.unpack_state(state)

    def unpack_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_size = self.size * self.size
        flattened_current_map = state[:map_size]
        position = state[map_size:map_size + 2]
        reward = state[-2:]
        return np.reshape(flattened_current_map, (self.size, self.size)), position, reward

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        """
        Observation in this case is 3D array:
        1st dimension is channels (we have 7 currently)
        2nd and 3rd is width x height (size x size).
        Channels are:

        obstacle map
        (5x) current maps
        position map
        reward map
        """
        current_map, position, reward_pos = self.unpack_state(state)
        obstacle_map = np.expand_dims(self.obstacle_map.copy(), -1)

        # One-hot vector, where we delete 0th channel b/c it's implied by the rest
        current_map_one_hot = ind_to_one_hot(current_map, max_val=4)
        current_map_one_hot = current_map_one_hot[:, :, 1:]

        pos_map = np.expand_dims(pos_to_map(position, self.size), -1)
        reward_map = np.expand_dims(pos_to_map(reward_pos, self.size), -1)
        return np.concatenate((obstacle_map, current_map_one_hot, pos_map, reward_map), axis=-1, dtype=float)

    def get_terminal(self) -> bool:
        return np.all(self.position == self.reward)

    def get_reward(self, prev_state: np.ndarray, action: int) -> float:
        if np.all(self.position == self.reward):
            return 1.

        # Here we see if a current has pushed us into a wall
        current_map, position, reward_pos = self.unpack_state(self.state)
        current_direction = current_map[position[0], position[1]]

        # if we're in a current
        if current_direction > 0:
            prev_current_map, prev_pos, reward_pos = self.unpack_state(prev_state)
            post_move_position = self.move(prev_pos, action)
            # if the current didn't move us, that means the current bumped us into a wall.
            if np.all(post_move_position == position):
                return self.current_bump_reward

        return 0.

    def move(self, pos: np.ndarray, action: int):
        new_pos = pos.copy()

        new_pos += self.direction_mapping[action]
        new_pos = np.maximum(np.minimum(new_pos, self.position_max), self.position_min)

        # if we bump into a wall inside the grid
        if self.obstacle_map[new_pos[0], new_pos[1]] > 0:
            new_pos = pos.copy()

        return new_pos

    def tick_currents(self, current_map: np.ndarray):
        """
        See if we change currents or not. If we do,
        update current map
        """
        # sample a bool for each current group
        change_mask = self.rng.binomial(1, p=self.pmfs_1).astype(bool)

        for i, change_bool in enumerate(change_mask):
            if change_bool:

                group_info = self.currents[i]

                # get all current directions we could sample
                all_current_options = group_info['directions'][:]

                # get our the current direction
                sample_position = group_info['mapping'][0]
                prev_current = current_map[sample_position[0], sample_position[1]] - 1
                current_options = [c for c in all_current_options if c != prev_current]

                # sample a new direction
                new_current = self.rng.choice(current_options)

                # set our new direction
                for pos in group_info['mapping']:
                    current_map[pos[0], pos[1]] = new_current + 1

        return current_map

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        current_map, position, reward_pos = self.unpack_state(state)
        new_current_map = current_map.copy()

        # We first move according to our action
        new_pos = self.move(position, action)

        # now we move again according to any currents
        current_direction = current_map[new_pos[0], new_pos[1]]
        if current_direction > 0:
            new_pos = self.move(new_pos, current_direction - 1)

        # now we tick currents
        new_current_map = self.tick_currents(new_current_map)
        flattened_new_current_map = new_current_map.flatten()

        return np.concatenate((flattened_new_current_map, new_pos, reward_pos))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_state = self.state
        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {}

