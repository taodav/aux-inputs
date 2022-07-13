import numpy as np
from typing import Union


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

    @property
    def n(self) -> int:
        raise NotImplementedError

    def cumulant(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def termination(self, obs: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError

    def policy(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def impt_sampling_ratio(self, state: np.ndarray, b: np.ndarray, action: np.ndarray):
        raise NotImplementedError

    def all_impt_sampling_ratios(self, state: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Assume the policy returned is batch_size x num_gvfs x n_actions,
        while the behavior policy is batch_size x n_actions

        return importance sampling ratios of size batch_size x num_gvfs x n_actions
        """
        pis = self.policy(state)
        pis_n_gvfs_first = pis.transpose((1, 0, 2))
        transposed_is_ratios = pis_n_gvfs_first / b

        # we deal with nans here
        zero_b = b == 0
        zero_b_repeated = np.expand_dims(zero_b, 0).repeat(pis_n_gvfs_first.shape[0], 0)
        transposed_is_ratios[zero_b_repeated] = 0

        return transposed_is_ratios.transpose((1, 0, 2))



