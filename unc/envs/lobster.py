import numpy as np
import gym

from typing import Tuple
from .base import Environment


class LobsterFishing(Environment):
    """
    See env spec sheet for a description of this environment:
    https://docs.google.com/document/d/16srtrtyKE40GXQNTx7VCR_leesarrukrg03vqr9MO8k/edit

    """
    reward_inverse_rates = np.array([10, 10])

    def __init__(self,
                 rng: np.random.RandomState = np.random.RandomState(),
                 traverse_prob: float = 0.3):
        super(LobsterFishing, self).__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9), high=np.ones(9)
        )
        self.action_space = gym.spaces.Discrete(3)

        self.traverse_prob = traverse_prob
        self.lambs = 1 / self.reward_inverse_rates
        self.pmfs_1 = self.lambs * np.exp(-self.lambs)
        self.rng = rng

        self.position = 0
        self.cages_full = np.zeros(2, dtype=int)

    @property
    def state(self):
        """
        Return underlying state of the environment. State consists of 3 features.
        1 for position
        2 for whether the cages are full
        IN THIS ORDER
        """
        state_features = np.zeros(3)
        state_features[0] = self.position
        state_features[1:] = self.cages_full.copy()
        return state_features

    @state.setter
    def state(self, state: np.ndarray):
        self.position = state[0]
        self.cages_full = state[1:]

    def get_terminal(self) -> bool:
        """
        Currently no terminal... is this an issue?
        """
        return False

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(9)

        # set position
        obs[state[0]] = 1

        # first set all rewards to unobservable
        obs[5] = 1
        obs[8] = 1

        if state[0] == 1:
            # reward in state 2 is unobservable
            obs[5] = 0

            obs[3:6][state[1]] = 1

        elif state[0] == 2:
            # reward in state 1 is unobservable
            obs[8] = 0

            obs[6:][state[2]] = 1

        return obs

    def get_reward(self, prev_state: np.ndarray) -> int:
        collected_reward = (self.state[1:] - prev_state[1:]) == 1
        if np.any(collected_reward):
            return 1
        return 0

    def reset(self):
        self.position = 0
        self.cages_full = np.ones(2)
        return self.get_obs(self.state)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()

        # if we're at home and we traverse successfully
        if state[0] == 0 and action < 2:
            if self.rng.random() < self.traverse_prob:
                new_state[0] += (action + 1)

        pos = state[0]
        if state[pos] == 1:
            # if cage is full
            new_state[1] = 0
        else:
            # if the cage was previously empty, w.p. self.pmfs_1[0] lobsters fill the cage again
            if self.rng.random() < self.pmfs_1[0]:
                new_state[1] = 1
        elif state[0] == 2:
            if state[2] == 1:
                # if the cage is full
                new_state[2] = 0
            else:
                if self.rng.random() < self.pmfs_1[1]:
                    new_state[2] = 1






