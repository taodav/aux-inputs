import gym
import numpy as np
from .compass_reward import RewardingCompassWorld

class StateRewardingCompassWorld(RewardingCompassWorld):

    def __init__(self, size: int = 8,
                 seed: int = 2021,
                 random_start: bool = False):
        super(RewardingCompassWorld, self).__init__(size, seed, random_start)

        self.observation_space = gym.spaces.Box(low=np.array([1, 1, 0, 0, 0, 0, 0, 0]), high=np.array([6, 6, 3, 1, 1, 1, 1, 1]), dtype=np.int16)

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = super(StateRewardingCompassWorld, self).get_obs(state)
        state = self.state.copy()
        
        return np.concatenate((state, obs), axis=0)


