import numpy as np
import gym

from typing import Tuple
from .base import Environment


class LobsterFishing(Environment):
    """
    Also known as "Two Room"
    See env spec sheet for a description of this environment:
    https://docs.google.com/document/d/16srtrtyKE40GXQNTx7VCR_leesarrukrg03vqr9MO8k/edit

    """
    reward_inverse_rates = np.array([10, 10])

    def __init__(self,
                 rng: np.random.RandomState,
                 traverse_prob: float = 0.3):
        super(LobsterFishing, self).__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9), high=np.ones(9)
        )
        # actions are go right, go left, collect
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

    def all_states(self):
        positions = [0, 1, 2]
        cages = []

    def get_terminal(self) -> bool:
        """
        Currently no terminal... is this an issue?
        """
        return False

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(9)

        # set position
        obs[state[0].astype(int)] = 1

        # first set all rewards to unobservable
        obs[5] = 1
        obs[8] = 1

        if state[0] == 1:
            # reward in state 1 is observable
            obs[5] = 0

            obs[3:6][state[1].astype(int)] = 1

        elif state[0] == 2:
            # reward in state 1 is unobservable
            obs[8] = 0

            obs[6:][state[2].astype(int)] = 1

        return obs

    def get_reward(self, prev_state: np.ndarray) -> int:
        collected_reward = (prev_state[1:] - self.state[1:]) == 1
        if np.any(collected_reward):
            return 1
        return 0

    def reset(self):
        self.position = 0
        self.cages_full = np.ones(2)
        return self.get_obs(self.state)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        pos = int(state[0])

        left_staying = (pos == 1 and action == 0)
        right_staying = (pos == 2 and action == 1)

        # See if we MOVE or not
        if action < 2:
            make_it = self.rng.random() < self.traverse_prob
            if not left_staying and not right_staying and make_it:
                if pos == 0:
                    new_state[0] += (action + 1)

                else:
                    # since we're here, this means we are going home
                    new_state[0] = 0

        # We clear reward if we collect in either states 1 or 2.
        # Since reward is calculated based on diff of prev state and current state,
        # rewards are given if prev state cage was full, but current state cage is empty.

        if state[0] != 0 and action == 2:
            # we need to deal with resetting rewards if there are any
            new_pos = int(new_state[0])
            new_state[new_pos] = 0

        # we tick all the rewards that have been collected
        to_tick = new_state[1:] == 0
        reset_mask = self.rng.binomial(1, p=self.pmfs_1)

        new_state[1:][to_tick] = reset_mask[to_tick]

        return new_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_state = self.state

        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(prev_state), self.get_terminal(), {}










