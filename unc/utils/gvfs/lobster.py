import numpy as np

from .base import GeneralValueFunction


class LobsterGVFs(GeneralValueFunction):
    """
    Two GVFs with policies that always do left and right respectively.
    Everything is done with batches!
    """
    def __init__(self,
                 n_actions: int,
                 gamma: float):
        super(LobsterGVFs, self).__init__(n_actions)
        self.action_idxes = np.array([0, 1], dtype=int)
        self.gamma = gamma

    @property
    def n(self):
        return self.action_idxes.shape[0]

    def cumulant(self, obs: np.ndarray) -> np.ndarray:
        """
        Our cumulants for this GVF is whether or not the rewards in
        node 1 or 2 are visible and present.
        obs: batch_size x obs_size
        """
        return obs[:, [4, 7]]

    def termination(self, obs: np.ndarray) -> np.ndarray:
        # return np.zeros((obs.shape[0], 2)) + self.gamma
        return obs[:, [5, 8]] * self.gamma

    def policy(self, state: np.ndarray) -> np.ndarray:
        pis = np.zeros((state.shape[0], 2, self.n_actions))
        pis[:, np.arange(pis.shape[1]), [0, 1]] = 1
        return pis

    def impt_sampling_ratio(self, state: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Return importance sampling ratios for actions left and right.
        """
        is_ratios = super(LobsterGVFs, self).impt_sampling_ratio(state, b)
        # rejection_sampling_mask = is_ratios < 2
        # is_ratios *= rejection_sampling_mask
        return is_ratios[:, np.arange(is_ratios.shape[1]), [0, 1]]
