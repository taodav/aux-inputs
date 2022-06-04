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
        return self.policy(state) / b



