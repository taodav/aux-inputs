import numpy as np
import gym
from typing import Union, Tuple

from unc.envs.tiger import Tiger

from .wrapper import TigerWrapper


class BeliefStateObservationWrapper(TigerWrapper):
    priority = 3

    def __init__(self, env: Union[Tiger, TigerWrapper], ground_truth: bool = False):
        super(BeliefStateObservationWrapper, self).__init__(env)

        self.ground_truth = ground_truth

        self.use_pf = hasattr(self, 'particles') and hasattr(self, 'weights')

        low = np.zeros(2)
        high = np.ones(2)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def get_obs(self, state: np.ndarray, particles: np.ndarray = None, weights: np.ndarray = None) -> np.ndarray:
        obs = np.zeros(2)
        if self.ground_truth:
            obs[state[0]] = 1
        else:
            for p, w in zip(self.particles, self.weights):
                obs[p] += w

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info

