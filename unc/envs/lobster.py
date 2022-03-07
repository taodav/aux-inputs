import numpy as np
import gym

from typing import Tuple
from .base import Environment


class LobsterFishing(Environment):
    """
    See env spec sheet for a description of this environment:
    https://docs.google.com/document/d/16srtrtyKE40GXQNTx7VCR_leesarrukrg03vqr9MO8k/edit

    """
    def __init__(self,
                 rng: np.random.RandomState = np.random.RandomState(),
                 traverse_prob: float = 0.4):
        super(LobsterFishing, self).__init__()

        self.observation_space = gym.spaces.Box(
            low=np.zeros(9), high=np.ones(9)
        )
        self.action_space = gym.spaces.Discrete(2)

        self.traverse_prob = traverse_prob
        self.rng = rng


