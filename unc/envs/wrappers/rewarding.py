import numpy as np

from .wrapper import CompassWorldWrapper


class RewardingWrapper(CompassWorldWrapper):
    """
    Compass World where you get rewarded if you're directly facing the green wall.
    """
    priority = 1

    def get_reward(self) -> int:
        if self.state == np.array([1, 1, 3]):
            return 1
        return 0

    def get_terminal(self) -> bool:
        if self.state == np.array([1, 1, 3]):
            return True
        return False

    def reset(self) -> np.ndarray:
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
