import numpy as np
import gym
import matplotlib.pyplot as plt

from typing import Tuple
from .base import Environment


class CompassWorld(Environment):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, size: int = 8,
                 seed: int = 2021,
                 random_start: bool = True):
        super(CompassWorld, self).__init__()
        self.observation_space = gym.spaces.MultiBinary(5)
        self.action_space = gym.spaces.Discrete(3)
        self.size = size
        self.random_start = random_start
        self.rng = np.random.RandomState(seed)

        self._state_max = [self.size - 2, self.size - 2, 3]
        self._state_min = [1, 1, 0]

        self._state = None

        # This is for particle filter states.
        # We need this def so that we can pass particles and weights between wrappers.
        self.particles = None
        self.weights = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        self._state = state

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(5)
        if state[2] == 0:
            # Facing NORTH
            if state[0] == 1:
                obs[0] = 1
        elif state[2] == 1:
            # Facing EAST
            if state[1] == self.size - 2:
                obs[1] = 1
        elif state[2] == 2:
            # Facing SOUTH
            if state[0] == self.size - 2:
                obs[2] = 1
        elif state[2] == 3:
            # Facing WEST
            if state[1] == 1:
                # On the border
                if state[0] == 1:
                    obs[4] = 1
                else:
                    obs[3] = 1
        else:
            raise NotImplementedError()

        return obs

    def get_reward(self) -> int:
        if (self.state == np.array([1, 1, 3])).all():
            return 1
        return 0

    def get_terminal(self) -> bool:
        if (self.state == np.array([1, 1, 3])).all():
            return True
        return False

    def reset(self) -> np.ndarray:
        """
        State description:
        state[0] -> y-position
        state[1] -> x-position
        state[2] -> direction, with mapping below:
        Mapping: 0 = North, 1 = East, 2 = South, 3 = West

        Observation description:
        Binary vector of size 5, each indicating whether you're directly in front of a certain color or not.
        :return:
        """
        if self.random_start:
            all_states = self.sample_all_states()
            eligible_state_indices = np.arange(0, all_states.shape[0])

            # Make sure to remove goal state from start states
            remove_idx = None
            for i in eligible_state_indices:
                if (all_states[i] == np.array([1, 1, 3])).all():
                    remove_idx = i
            delete_mask = np.ones_like(eligible_state_indices, dtype=np.bool)
            delete_mask[remove_idx] = False
            eligible_state_indices = eligible_state_indices[delete_mask, ...]
            start_state_idx = self.rng.choice(eligible_state_indices)

            self.state = all_states[start_state_idx]
        else:
            self.state = np.array([3, 3, self.rng.choice(np.arange(0, 4))], dtype=np.int16)

        return self.get_obs(self.state)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        if action == 0:
            new_state[:-1] += self.direction_mapping[state[-1]]
        elif action == 1:
            new_state[-1] = (state[-1] + 1) % 4
        elif action == 2:
            new_state[-1] = (state[-1] - 1) % 4

        # Wall interactions
        new_state = np.maximum(np.minimum(new_state, self._state_max), self._state_min)

        return new_state

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        """
        Get the probability of emitting a given observation given a state
        :param state: underlying state
        :param obs: observation
        :return: probability of emitting (either 0 or 1 for this deterministic environment).
        """
        ground_truth_obs = self.get_obs(state)
        return float((ground_truth_obs == obs).all())

    def sample_states(self, n: int = 10) -> np.ndarray:
        """
        Sample n random states (for particle filtering)
        :param n: number of states to sample
        :return: sampled states
        """
        poss = self.rng.choice(np.arange(1, self.size - 1, dtype=np.int16), size=(n, 2))
        dirs = self.rng.choice(np.arange(0, len(self.direction_mapping), dtype=np.int16), size=(n, 1))
        return np.concatenate([poss, dirs], axis=-1)

    def sample_all_states(self) -> np.ndarray:
        """
        Sample all available states
        :return:
        """
        states = np.zeros(((self.size - 2) ** 2 * 4, 3), dtype=np.int16)
        c = 0
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                for k in range(0, 4):
                    states[c] = np.array([i, j, k])
                    c += 1

        assert c == states.shape[0]

        return states

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Action description:
        0 = Forward, 1 = Turn Right, 2 = Turn Left
        :param action: Action to take
        :return:
        """
        assert action in list(range(0, 4)), f"Invalid action: {action}"

        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}

    def generate_array(self) -> np.ndarray:
        """
        Generate a numpy array representing agent state.
        Mappings for indices are as follows:
        0 = white space
        1 = orange wall
        2 = yellow wall
        3 = red wall
        4 = blue wall
        5 = green wall
        6 = agent facing NORTH
        7 = agent facing EAST
        8 = agent facing SOUTH
        9 = agent facing WEST
        :return:
        """
        viz_array = np.zeros((self.size, self.size), dtype=np.uint8)

        # WEST wall
        viz_array[:, 0] = 4
        viz_array[1, 0] = 5

        # EAST wall
        viz_array[:, self.size - 1] = 2

        # NORTH wall
        viz_array[0, :] = 1

        # SOUTH wall
        viz_array[self.size - 1, :] = 3

        viz_array[self.state[0], self.state[1]] = self.state[-1] + 6
        return viz_array



if __name__ == "__main__":
    env = CompassWorld()
    env.reset()
    plt.imshow(env.render())
    plt.show()

