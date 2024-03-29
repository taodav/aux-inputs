import gym
import numpy as np
from typing import Tuple


from unc.envs.base import Environment
from unc.utils.data import ind_to_one_hot


def pos_to_map(pos: np.ndarray, size: int):
    pos_map = np.zeros((size, size), dtype=np.int16)
    if len(pos.shape) == 2:
        for p in pos:
            if np.all(p > -1):
                pos_map[p[0], p[1]] = 1
    elif len(pos.shape) == 1:
        if np.all(pos > -1):
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

    def __init__(self, rng: np.random.RandomState,
                 config: dict,
                 slip_prob: float = 0.):
        super(OceanNav, self).__init__()
        self.config = config
        self.size = self.config['size']
        self.rng = rng
        self.current_bump_reward = self.config['current_bump_reward']
        self.kelp_prob = 0.

        if "kelp_prob" in self.config:
            self.kelp_prob = self.config["kelp_prob"]

        self.slip_prob = slip_prob
        self.position = None
        self.rewards = None

        self.position_max = np.array([self.size - 1, self.size - 1], dtype=int)
        self.position_min = np.array([0, 0], dtype=int)

        # state space is flattened current map + position + reward position
        # obstacle_map is considered as part of epistemic state.
        self.state_space = gym.spaces.MultiDiscrete(self.size * self.size + 2 + 2)
        self.action_space = gym.spaces.Discrete(4)

        low = np.zeros((self.size, self.size, 7), dtype=int)
        high = np.ones((self.size, self.size, 7), dtype=int)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

        # Check configs for what this dict looks like
        self.currents = self.config['currents']

        # we get the per-timestep prob of changing
        self.current_inverse_rates = np.array([g['change_rate'] for g in self.currents if g['change_rate'] > 0])

        # this mask gives you all the currents that we tick each step.
        # this is an optimization!
        self.tick_mask = np.array([g['change_rate'] > 0 for g in self.currents], dtype=bool)
        self.lambs = 1 / self.current_inverse_rates
        self.pmfs_1 = self.lambs * np.exp(-self.lambs)

        # Map of all obstacles
        self.obstacle_map = np.array(self.config['obstacle_map'], dtype=np.int16)

        self.glass_map = np.array(self.config['glass_map'], dtype=np.int16)

        self.kelp_map = np.zeros_like(self.glass_map)
        if "kelp_map" in self.config:
            self.kelp_map = np.array(self.config['kelp_map'], dtype=np.int16)

        # Map of all current directions
        self.current_map = np.zeros((self.size, self.size), dtype=np.int16)

        # Start positions to sample from
        self.start_positions = np.array(self.config['starts'], dtype=np.int16)

        # Reward positions to sample from. Note: we can only have 1 active reward in an env currently.
        rew_pos_list = [rew_dict['position'] for rew_dict in self.config['rewards']]
        self.possible_reward_positions = np.array(rew_pos_list, dtype=np.int16)

    def reset_currents(self):
        self.current_map = np.zeros((self.size, self.size), dtype=np.int16)
        for curr_info in self.currents:
            if "start_probs" in curr_info:
                sampled_curr_direction = self.rng.choice(curr_info['directions'], p=curr_info["start_probs"])
            else:
                sampled_curr_direction = self.rng.choice(curr_info['directions'])

            for y, x in curr_info['mapping']:
                self.current_map[y, x] = sampled_curr_direction + 1

    def reset(self) -> np.ndarray:
        self.reset_currents()
        self.position = self.start_positions[self.rng.choice(range(len(self.start_positions)))]
        self.rewards = self.possible_reward_positions[self.rng.choice(range(len(self.possible_reward_positions)))][None, :]
        return self.get_obs(self.state)

    @property
    def state(self):
        flattened_current_map = self.current_map.flatten()
        flattened_rewards = self.rewards.flatten()
        return np.concatenate([flattened_current_map, self.position, flattened_rewards])

    @state.setter
    def state(self, state: np.ndarray):
        self.current_map, self.position, self.rewards = self.unpack_state(state)

    def unpack_state(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        map_size = self.size * self.size
        flattened_current_map = state[:map_size]
        position = state[map_size:map_size + 2]
        reward = state[map_size + 2:]
        return np.reshape(flattened_current_map, (self.size, self.size)), position, reward.reshape(-1, 2)

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
        current_map, position, reward_poses = self.unpack_state(state)
        obstacle_map = np.expand_dims(self.obstacle_map.copy(), -1)

        # One-hot vector, where we delete 0th channel b/c it's implied by the rest
        current_map_one_hot = ind_to_one_hot(current_map, max_val=4)
        current_map_one_hot = current_map_one_hot[:, :, 1:]

        pos_map = np.expand_dims(pos_to_map(position, self.size), -1)
        reward_map = np.expand_dims(pos_to_map(reward_poses, self.size), -1)
        return np.concatenate((obstacle_map, current_map_one_hot, pos_map, reward_map), axis=-1, dtype=float)

    def get_terminal(self) -> bool:
        matching_pos = np.all((self.position == self.rewards), axis=-1)
        return np.any(matching_pos)

    def get_current_reward(self, state: np.ndarray, prev_state: np.ndarray, action: int) -> float:
        # Here we see if a current has pushed us into a wall
        current_map, position, reward_pos = self.unpack_state(state)
        current_direction = current_map[position[0], position[1]]

        # if we're in a current
        if current_direction > 0:
            prev_current_map, prev_pos, reward_pos = self.unpack_state(prev_state)
            post_move_position = self.move(prev_pos, action, slippage=False)
            # if the current didn't move us, that means the current bumped us into a wall.
            if np.all(post_move_position == position):
                return self.current_bump_reward
        return 0.

    def get_reward(self, prev_state: np.ndarray, action: int) -> float:
        matching_pos = np.all((self.position == self.rewards), axis=-1)
        if np.any(matching_pos):
            return 1.

        return self.get_current_reward(self.state, prev_state, action)

    def move(self, pos: np.ndarray, action: int, slippage: bool = True):
        new_pos = pos.copy()
        not_kelp_slip = self.kelp_prob == 0. or self.kelp_map[pos[0], pos[1]] == 0 or self.rng.rand() > self.kelp_prob
        not_normal_slip = self.slip_prob == 0. or self.rng.rand() > self.slip_prob

        if not slippage or (not_kelp_slip and not_normal_slip):
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
        change_mask = np.zeros_like(self.tick_mask, dtype=bool)
        change_mask[self.tick_mask] = self.rng.binomial(1, p=self.pmfs_1).astype(bool)

        for i, group_info in enumerate(self.currents):
            if group_info['change_rate'] > 0 and change_mask[i]:

                # get all current directions we could sample
                all_current_options = group_info['directions'][:]

                # get our the current direction
                sample_position = group_info['mapping'][0]
                prev_current = current_map[sample_position[0], sample_position[1]] - 1
                current_options = [c for c in all_current_options if c != prev_current]

                # sample a new direction
                new_current = prev_current
                if len(current_options) > 1:
                    new_current = self.rng.choice(current_options)
                elif current_options:
                    new_current = current_options[0]

                # set our new direction
                for pos in group_info['mapping']:
                    current_map[pos[0], pos[1]] = new_current + 1

        return current_map

    @staticmethod
    def opposite_directions(dir_1: int, dir_2: int):
        return (dir_1 == 0 and dir_2 == 2) or (dir_1 == 2 and dir_2 == 0) or (dir_1 == 1 and dir_2 == 3) or (dir_1 == 3 and dir_2 == 1)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        current_map, position, reward_pos = self.unpack_state(state)
        new_current_map = current_map.copy()

        # We first move according to our action
        # have it so that you can't move against the current direction if it exists
        # you're currently in (swim orthogonal to the rip!)
        old_current_direction = current_map[position[0], position[1]]
        if old_current_direction > 0 and self.opposite_directions(old_current_direction - 1, action):
            new_pos = position.copy()
        else:
            new_pos = self.move(position, action)

        # now we move again according to any currents
        current_direction = current_map[new_pos[0], new_pos[1]]
        if current_direction > 0:
            new_pos = self.move(new_pos, current_direction - 1, slippage=False)

        # now we tick currents
        new_current_map = self.tick_currents(new_current_map)
        flattened_new_current_map = new_current_map.flatten()
        flattened_reward_pos = reward_pos.flatten()

        return np.concatenate((flattened_new_current_map, new_pos, flattened_reward_pos))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_state = self.state
        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {'position': self.position.copy()}


