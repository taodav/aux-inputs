import numpy as np

from .base import GeneralValueFunction


class SimpleChainGVF(GeneralValueFunction):
    """
    One GVF with a policy that always goes right.
    """
    def __init__(self,
                 n_actions: int,
                 gamma: float):
        super(SimpleChainGVF, self).__init__(n_actions)
        self.gamma = gamma

    def cumulant(self, obs: np.ndarray) -> np.ndarray:
        """
        Our cumulants for this GVF is whether or not the rewards in
        node 1 or 2 are visible and present.
        obs: batch_size x obs_size
        """
        return obs[:, [-1]]

    def termination(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros((obs.shape[0], 1)) + self.gamma

    def policy(self, state: np.ndarray) -> np.ndarray:
        pis = np.ones((state.shape[0], 1, self.n_actions))
        return pis

    def impt_sampling_ratio(self, state: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Return importance sampling ratios for actions left and right.
        """
        is_ratios = super(SimpleChainGVF, self).impt_sampling_ratio(state, b)
        return is_ratios[:, np.arange(is_ratios.shape[1]), [0]]
