import numpy as np
from typing import Union

from unc.envs.lobster import LobsterFishing
from unc.envs.wrappers.lobster import LobsterFishingWrapper
from .base import GeneralValueFunction


class LobsterGVFs(GeneralValueFunction):
    """
    Two GVFs with policies that always do left and right respectively.
    """
    def __init__(self,
                 n_actions: int,
                 gamma: float):
        super(LobsterGVFs, self).__init__(n_actions)
        self.action_idxes = np.array([0, 1], dtype=int)
        self.gamma = gamma

    def cumulant(self, obs: np.ndarray) -> np.ndarray:
        """
        Our cumulants for this GVF is whether or not the rewards in
        node 1 or 2 are visible and present.
        """
        return obs[[4, 7]]

    def termination(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(2) + self.gamma

    def policy(self, state: np.ndarray) -> np.ndarray:
        pis = np.zeros((2, self.n_actions))
        pis[np.arange(pis.shape[0]), [0, 1]] = 1
        return pis

