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
    reward_positions = np.array([[1, 1], [4, 9], [10, 10], [9, 0]])
    reward_inverse_rates = np.array([30, 35, 40, 45])
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)


    def __init__(self,
                 rng: np.random.RandomState,
                 random_start: bool = True,
                 reward_val: float = 1.0):
        super(FourRoom, self).__init__()
        self.observation_space = gym.spaces.MultiDiscrete(6)
        self.action_space = gym.spaces.Discrete(4)
        self.rng = rng
        self.size = 11
        self.halfway = self.size // 2
        door_from_edge = self.halfway // 2

        self.random_start = random_start
        self.reward_val = reward_val

        self.lambs = 1 / self.reward_inverse_rates
        self.pmfs_1 = self.lambs * np.exp(-self.lambs)

        self.position_max = np.array([self.size - 1, self.size - 1])
        self.position_min = np.array([0, 0])

        self.grid = np.zeros((self.size, self.size))
        self.grid[self.halfway, :] = 1
        self.grid[:, self.halfway] = 1

        self.grid[self.halfway, [door_from_edge, -door_from_edge - 1]] = 0.
        self.grid[[door_from_edge, -door_from_edge - 1], self.halfway] = 0.

        self.time_since_takens = np.zeros(4)
        self.position = np.zeros(2)

    @property
    def state(self):
        return np.concatenate((self.position, self.time_since_takens))

    def _unpack_state(self, state: np.array) -> Tuple[np.array, np.array]:
        return state[:2], state[2:]

    def get_room_idx(self, position: np.ndarray) -> int:
        above = position[0] > self.halfway
        below = position[0] < self.halfway
        right = position[1] > self.halfway
        left = position[1] > self.halfway

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
        pos, tst = self._unpack_state(state)
        room_idx = self.get_room_idx(pos)

        reward_obs = np.zeros(4)

        # if we're not in a doorway and the reward has not been taken
        # we can observe the reward
        if room_idx > 0 and tst[room_idx] == 0:
            reward_obs[room_idx] = 1

        return np.concatenate((pos, reward_obs))

    def get_reward(self, prev_state: np.ndarray, action: int = None) -> int:
        _, prev_tst = self._unpack_state(prev_state)
        _, curr_tst = self._unpack_state(self.state)
        # TODO

    def _tick_time_since_taken(self, tst: np.ndarray, collected_reward: np.ndarray) -> np.ndarray:
        """
        tst: time since taken array, size 4
        collected_reward: at this current tick, were any of the available
        rewards taken? mask of size 4
        """
        return tst


    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        actions: cardinal directions, from north clockwise (4)
        """
        pos, tst = self._unpack_state(state)
        new_pos, new_tst = pos.copy(), tst.copy()
        new_pos += self.direction_mapping[action]

        # if we bump into one of the outer boundaries
        new_pos = np.maximum(np.minimum(new_pos, self.position_max), self.position_min)

        # if we bump into a wall inside the grid
        if self.grid[new_pos[0], new_pos[1]] > 0:
            new_pos = pos

        # now we check if we are at any of the rewards
        matches = new_pos == self.reward_positions
        match_mask = matches[:, 0] and matches[:, 1]

        # see which rewards are currently available
        tst_mask = tst == 0

        # we get a reward if we're at a reward and that reward is available.
        collected_reward = match_mask * tst_mask

        # now we "tick" (progress one step) our tst array
        # first we update our tst
        new_tst += (new_tst != 0).astype(int)

        # now we remove rewards that have just been collected
        new_tst += collected_reward

        return np.concatenate((new_pos, new_tst))


