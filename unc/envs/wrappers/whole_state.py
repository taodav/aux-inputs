import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs import CompassWorld


class WholeStateObservationWrapper(CompassWorldWrapper):
    priority = 3

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper],
                 ground_truth: bool = False):
        """
        Do we encode the particles/state with an entire array of
        uncertainties/one-hot encoding?
        :param env: environment to wrap. NOTE: if ground_truth == False, then the
        particle filter wrapper MUST be part of env.
        :param ground_truth: Do we encode the ground truth state or the particles?
        """
        super(WholeStateObservationWrapper, self).__init__(env)

        self.ground_truth = ground_truth

        if not self.ground_truth:
            assert hasattr(self, 'particles') and hasattr(self, 'weights')

        self.observation_space = gym.spaces.Box(
            low=np.zeros(((self.env.size - 2) * (self.env.size - 2) * 4) + 5),
            high=np.ones(((self.env.size - 2) * (self.env.size - 2) * 4) + 5)
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(((self.env.size - 2), (self.env.size - 2), 4))
        if self.ground_truth:
            obs[state[0] - 1, state[1] - 1, state[2]] = 1.
        else:
            for p, w in zip(self.particles, self.weights):
                obs[p[0] - 1, p[1] - 1, p[2]] += w
        obs = obs.flatten()

        # We assume that the color observation is always the last five observations
        color_obs = self.env.get_obs(state)[-5:]
        obs = np.concatenate((obs, color_obs))

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info

