import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import RockSampleWrapper
from unc.envs.rocksample import RockSample


class GlobalStateObservationWrapper(RockSampleWrapper):
    priority = 3

    def __init__(self, env: Union[RockSample, RockSampleWrapper],
                 ground_truth: bool = False):
        """
        Do we encode position globally?
        If we do, then we build agent-state as follows

        First env.size * env.size elements are a one-hot encoding of position
        Last env.rocks features are either the current rock observations, or if
        ground_truth = True, then it's the ground truth rock morality (goodness/badness)
        :param env:
        :param ground_truth:
        """
        super(GlobalStateObservationWrapper, self).__init__(env)

        self.ground_truth = ground_truth

        self.use_pf = hasattr(self, 'particles') and hasattr(self, 'weights')

        low = np.zeros(self.size * self.size + self.rocks)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        position, rock_morality, _, current_rocks_obs = self.unwrapped.unpack_state(state)
        position_obs = np.zeros((self.size, self.size))
        position_obs[position[0], position[1]] = 1
        position_obs = np.concatenate(position_obs)

        rock_obs = rock_morality.copy()
        if not self.ground_truth:
            if self.use_pf:
                rock_obs = np.zeros_like(rock_obs, dtype=np.float)
                for p, w in zip(self.particles, self.weights):
                    rock_obs += p * w
            else:
                rock_obs = current_rocks_obs

        return np.concatenate([position_obs, rock_obs])

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
