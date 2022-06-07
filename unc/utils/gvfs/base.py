import numpy as np
import gym
from typing import Union

from unc.envs import Environment


class GeneralValueFunction:
    """
    A GVF! Defines a cumulant, termination, policy,
    and importance sampling ratio (given a policy)
    all at a given state.

    GVFs are dependent on a policy and a given state.
    Policy should be defined within the GVF itself.
    """
    def __init__(self,
                 n_actions: int):
        self.n_actions = n_actions

    def cumulant(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def termination(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def policy(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def impt_sampling_ratio(self, state: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Assume the policy returned is batch_size x num_gvfs x n_actions,
        while the behavior policy is batch_size x n_actions

        return importance sampling ratios of size batch_size x num_gvfs x n_actions
        """
        pis = self.policy(state)
        pis_n_gvfs_first = pis.transpose((1, 0, 2))
        transposed_is_ratios = pis_n_gvfs_first / b
        return transposed_is_ratios.transpose((1, 0, 2))



