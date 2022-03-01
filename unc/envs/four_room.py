import gym.spaces
import numpy as np
from typing import Tuple

from .base import Environment


class FourRoom(Environment):
    """
    Our partially observable 4Room environment to demonstrate uncertainty over time.

    Each room has a reward in it, placed at a fixed position.

    After the reward is taken, each reward reappears after a random interval (dist is fixed for each reward).

    Observations are:
    obs[0] -> y-position
    obs[1] -> x-position
    obs[2-5] -> reward of the current room you're in.

    w.r.t. the doorways, doorways are states where you can't see either side.
    """
    reward_positions = np.array([[4, 9], [10, 10], [9, 0]])
    reward_inverse_rates = np.array([80, 80, 80])
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self,
                 rng: np.random.RandomState,
                 random_start: bool = True,
                 reward_val: float = 1.0):
        super(FourRoom, self).__init__()
        self.rng = rng
        self.size = 11
        self.halfway = self.size // 2
        door_from_edge = self.halfway // 2

        low = np.zeros(2 + self.n_rewards)
        high = np.ones_like(low)
        high[0:2] = self.size - 1

        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )
        self.action_space = gym.spaces.Discrete(4)

        self.random_start = random_start
        self.reward_val = reward_val

        self.lambs = 1 / self.reward_inverse_rates
        self.pmfs_1 = self.lambs * np.exp(-self.lambs)

        self.position_max = np.array([self.size - 1, self.size - 1], dtype=int)
        self.position_min = np.array([0, 0], dtype=int)

        self.grid = np.zeros((self.size, self.size))
        self.grid[self.halfway, :] = 1
        self.grid[:, self.halfway] = 1

        # we start in a random position in the first room
        self.init_position_support = []
        if self.random_start:
            for i in range(self.halfway):
                for j in range(self.halfway):
                    pos = [i, j]
                    self.init_position_support.append(pos)
        else:
            self.init_position_support.append([0, 0])
        self.init_position_support = np.array(self.init_position_support, dtype=int)

        self.grid[self.halfway, [door_from_edge, -door_from_edge - 1]] = 0.
        self.grid[[door_from_edge, -door_from_edge - 1], self.halfway] = 0.

        self.time_since_takens = np.zeros(self.n_rewards, dtype=int)
        self.position = np.zeros(2, dtype=int)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.position, self.time_since_takens))

    @state.setter
    def state(self, state: np.ndarray) -> np.ndarray:
        pos, tst = self.unpack_state(state)
        self.position = pos
        self.time_since_takens = tst

    @property
    def n_rewards(self) -> int:
        return self.reward_positions.shape[0]

    def unpack_state(self, state: np.array) -> Tuple[np.array, np.array]:
        return state[:2], state[2:]

    def get_room_idx(self, position: np.ndarray) -> int:
        above = position[0] < self.halfway
        below = position[0] > self.halfway
        left = position[1] < self.halfway
        right = position[1] > self.halfway

        if above:
            if left:
                return 0
            elif right:
                return 1
        elif below:
            if right:
                return 2
            elif left:
                return 3

        return -1

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        pos, tst = self.unpack_state(state)
        room_idx = self.get_room_idx(pos)

        reward_obs = np.zeros(self.n_rewards)

        # if we're not in a doorway or the first room and the reward has not been taken
        # we can observe the reward
        if room_idx > 0 and tst[room_idx - 1] == 0:
            reward_obs[room_idx - 1] = 1

        return np.concatenate((pos, reward_obs))

    def get_reward(self, prev_state: np.ndarray, action: int = None) -> int:
        _, prev_tst = self.unpack_state(prev_state)
        _, curr_tst = self.unpack_state(self.state)
        prev_avail = (prev_tst == 0).astype(int)
        curr_just_unavail = (curr_tst == 1).astype(int)
        return (prev_avail * curr_just_unavail).sum().item()

    def get_terminal(self) -> bool:
        return False

    def reset(self) -> np.ndarray:
        self.time_since_takens = np.zeros(self.n_rewards, dtype=int)
        self.position = self.init_position_support[self.rng.choice(np.arange(len(self.init_position_support)))].copy()
        return self.get_obs(self.state)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        actions: cardinal directions, from north clockwise (4)
        """
        pos, tst = self.unpack_state(state)
        new_pos, new_tst = pos.copy(), tst.copy()
        new_pos += self.direction_mapping[action]

        # if we bump into one of the outer boundaries
        new_pos = np.maximum(np.minimum(new_pos, self.position_max), self.position_min)

        # if we bump into a wall inside the grid
        if self.grid[new_pos[0], new_pos[1]] > 0:
            new_pos = pos

        # now we check if we are at any of the rewards
        matches = new_pos == self.reward_positions
        match_mask = matches[:, 0] * matches[:, 1]

        # see which rewards are currently available
        tst_mask = tst == 0

        # we get a reward if we're at a reward and that reward is available.
        collected_reward = match_mask * tst_mask

        # now we "tick" (progress one step) our tst array
        # first we update our tst
        ticked = (new_tst != 0).astype(int)
        new_tst += ticked

        # we see who gets reset
        reset_mask = self.rng.binomial(1, p=self.pmfs_1)
        new_tst *= (1 - reset_mask)

        # now we remove rewards that have just been collected
        new_tst += collected_reward

        return np.concatenate((new_pos, new_tst))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_state = self.state
        self.state = self.transition(self.state, action)

        # TODO: do we have terminal = True when timesteps are up? Don't think it matters.

        return self.get_obs(self.state), self.get_reward(prev_state), self.get_terminal(), {}

    def generate_array(self) -> np.ndarray:
        """
        Generate a numpy array representing state.
        Mappings for indices are as follows:
        0 = white space
        1 = wall
        2 = agent
        3 = reward
        4 = empty reward
        :return:
        """
        viz_array = self.grid.copy().astype(np.uint8)

        pos, tsts = self.unpack_state(self.state)

        # we show rewards if they're there
        for rew_pos, tst in zip(self.reward_positions, tsts):
            to_set = 4
            if tst == 0:
                to_set = 3

            viz_array[rew_pos[0], rew_pos[1]] = to_set

        # agent position
        viz_array[pos[0], pos[1]] = 2

        return viz_array


