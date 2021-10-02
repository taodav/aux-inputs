import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import RockSampleWrapper
from unc.envs.rocksample import RockSample


class LocalStateObservationWrapper(RockSampleWrapper):
    priority = 3

    def __init__(self, env: Union[RockSample, RockSampleWrapper],
                 ground_truth: bool = False):
        """
        Misleading title. Position is still encoded globally.

        In this case, we encode rock observations based on mean and variance.
        We build agent-state as follows

        First env.size * env.size elements are a one-hot encoding of position
        Last env.rocks * 2 features are either the current rock observations, or if
        ground_truth = True, then it's the ground truth rock morality (goodness/badness)
        :param env:
        :param ground_truth:
        """
        super(LocalStateObservationWrapper, self).__init__(env)

        self.ground_truth = ground_truth

        self.use_pf = hasattr(self, 'particles') and hasattr(self, 'weights')

        low = np.zeros(self.size * self.size + 2 * self.rocks)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def get_obs(self, state: np.ndarray, particles: np.ndarray = None, weights: np.ndarray = None) -> np.ndarray:
        position, rock_morality, _, current_rocks_obs = self.unwrapped.unpack_state(state)
        position_obs = np.zeros((self.size, self.size))
        position_obs[position[0], position[1]] = 1
        position_obs = np.concatenate(position_obs)

        rock_obs = rock_morality.copy()
        if not self.ground_truth:
            if self.use_pf:
                if particles is None:
                    particles = self.particles
                if weights is None:
                    weights = self.weights
                rock_obs = np.zeros(self.rocks * 2, dtype=np.float)
                for i in range(self.rocks):
                    rock_obs[2 * i] = (particles[:, i] * weights).sum()
                    rock_obs[2 * i + 1] = (weights * (particles[:, i] - rock_obs[2 * i])**2).sum()
            else:
                rock_obs = current_rocks_obs

        return np.concatenate([position_obs, rock_obs])

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
