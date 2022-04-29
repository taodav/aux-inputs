import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import SimpleChainWrapper, SimpleChain


class PartiallyObservableWrapper(SimpleChainWrapper):
    """
    In this environment, you only see 1's in the first observation after resetting.
    """
    priority = 3

    def __init__(self, env: Union[SimpleChainWrapper, SimpleChain],
                 obs_shape: Tuple[int, ...] = (1,)):
        super(PartiallyObservableWrapper, self).__init__(env)

        low = np.zeros(obs_shape)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(low=low, high=high)

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        if state[0] == 1:
            return np.ones(self.observation_space.shape)
        return np.zeros(self.observation_space.shape)

    def reset(self):
        super(PartiallyObservableWrapper, self).reset()
        return self.get_obs(self.state)

    def step(self, action: int):
        self.state = self.transition(self.state.copy(), action)
        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}
