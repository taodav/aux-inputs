import numpy as np
from .compass import CompassWorld


class FixedCompassWorld(CompassWorld):
    """
    "Fixed" version of compass world, where the green terminal rewarding tile is
    in the middle of the west wall as opposed to the top left wall.

    This makes it so that the agent can't exploit using the north wall as a "guide"
    to the green wall.
    """
    def reset(self) -> np.ndarray:

        if self.random_start:
            all_states = self.sample_all_states()
            eligible_state_indices = np.arange(0, all_states.shape[0])

            # Make sure to remove goal state from start states
            remove_idx = None
            for i in eligible_state_indices:
                if (all_states[i] == np.array([(self.size - 1) // 2, 1, 3])).all():
                    remove_idx = i
            delete_mask = np.ones_like(eligible_state_indices, dtype=np.bool)
            delete_mask[remove_idx] = False
            eligible_state_indices = eligible_state_indices[delete_mask, ...]
            start_state_idx = self.rng.choice(eligible_state_indices)

            self.state = all_states[start_state_idx]
        else:
            self.state = np.array([3, 3, self.rng.choice(np.arange(0, 4))], dtype=np.int16)

        return self.get_obs(self.state)

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
                if state[0] == (self.size - 1) // 2:
                    obs[4] = 1
                else:
                    obs[3] = 1
        else:
            raise NotImplementedError()

        return obs

    def get_reward(self) -> int:
        if (self.state == np.array([(self.size - 1) // 2, 1, 3])).all():
            return 1
        return 0

    def get_terminal(self) -> bool:
        if (self.state == np.array([(self.size - 1) // 2, 1, 3])).all():
            return True
        return False

    def generate_array(self) -> np.ndarray:
        viz_array = np.zeros((self.size, self.size), dtype=np.uint8)

        # WEST wall
        viz_array[:, 0] = 4
        viz_array[(self.size - 1) // 2, 0] = 5

        # EAST wall
        viz_array[:, self.size - 1] = 2

        # NORTH wall
        viz_array[0, :] = 1

        # SOUTH wall
        viz_array[self.size - 1, :] = 3

        viz_array[self.state[0], self.state[1]] = self.state[-1] + 6
        return viz_array
