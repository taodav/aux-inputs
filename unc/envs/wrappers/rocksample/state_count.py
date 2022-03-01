import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import RockSampleWrapper
from unc.envs.rocksample import RockSample


class StateCountObservationWrapper(RockSampleWrapper):
    """
    Observation based count observations

    Code: c

    Terrible naming, I know.

    Instead of a state based count, we separate position and rock observations.
    So our observation returned is self.size ** 2 + 2 ** self.rocks.

    We also include a decay rate of previous counts at every step.
    """
    priority = 2

    def __init__(self, env: Union[RockSample, RockSampleWrapper], decay: float = 1.,
                 normalize: bool = True):
        super(StateCountObservationWrapper, self).__init__(env)
        low = np.zeros(self.size * self.size + 2**self.rocks)

        high = np.ones_like(low) * 1000
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )
        self.decay = decay
        self.normalize = normalize

        self.position_counts = None
        self.rock_obs_counts = None

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        if self.normalize:
            return 1 / (np.concatenate([self.position_counts.flatten(), self.rock_obs_counts.flatten()]) + 1)
        return np.concatenate([self.position_counts.flatten(), self.rock_obs_counts.flatten()])

    def _update_counts(self):
        if self.decay < 1.:
            self.position_counts *= self.decay
            self.rock_obs_counts *= self.decay
        pos, _, _, rock_obs = self.env.unpack_state(self.state)
        robs_as_int = int(np.array2string(rock_obs.astype(int), separator='')[1:-1], 2)
        self.position_counts[pos[0], pos[1]] += 1
        self.rock_obs_counts[robs_as_int] += 1

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.position_counts = np.zeros((self.size, self.size))
        self.rock_obs_counts = np.zeros(2 ** self.rocks)
        self._update_counts()
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        self._update_counts()

        return self.get_obs(self.state), reward, done, info
