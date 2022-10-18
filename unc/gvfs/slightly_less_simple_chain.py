import numpy as np
from typing import Union

from unc.envs.slightly_less_simple_chain import SlightlyLessSimpleChain
from .base import GeneralValueFunction



class SlightlyLessSimpleChainGVF(GeneralValueFunction):
    """
    One GVF with a policy that always goes right.
    """
    def __init__(self,
                 n_actions: int,
                 gamma: float):
        super(SlightlyLessSimpleChainGVF, self).__init__(n_actions)
        self.gamma = gamma

    @property
    def n(self):
        return 1

    def cumulant(self, state: np.ndarray) -> np.ndarray:
        """
        Cumulant for this is just whether or not we're at the last node
        state: batch_size x state_size
        """
        return state[:, [-1]]

    def termination(self, state: np.ndarray) -> Union[float, np.ndarray]:
        return state[:, -1] == 1

    def policy(self, state: np.ndarray) -> np.ndarray:
        pis = np.zeros((state.shape[0], 1, self.n_actions))
        pis[:, :, 1] = 1
        return pis

    def impt_sampling_ratio(self, state: np.ndarray, b: np.ndarray, actions: np.ndarray) -> np.ndarray:
        all_is_ratios = super(SlightlyLessSimpleChainGVF, self).all_impt_sampling_ratios(state, b)
        actions_mask = np.zeros((actions.shape[0], self.n_actions), dtype=bool)
        actions_mask[np.arange(actions.shape[0]), actions] = True
        actions_mask_n_gvfs_repeat = np.expand_dims(actions_mask, 1).repeat(all_is_ratios.shape[1], 1)
        flattened_all_is_ratios = all_is_ratios[actions_mask_n_gvfs_repeat]
        is_ratios = flattened_all_is_ratios.reshape(all_is_ratios.shape[0], all_is_ratios.shape[1])
        return is_ratios
