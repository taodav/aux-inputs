import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import RockSampleWrapper
from unc.envs.rocksample import RockSample


class ObsCountObservationWrapper(RockSampleWrapper):
    """
    Observation based count observations

    Terrible naming, I know.

    Instead of a state based count, we separate position and rock observations.
    So our observation returned is self.size ** 2 + self.rocks. Note we are encoding position
    through a one-hot encoding

    We also include a decay rate of previous counts at every step.
    """
    priority = 2

    def __init__(self, env: Union[RockSample, RockSampleWrapper], decay: float = 1.,
                 normalize: bool = True, count_init: float = 10.):
        super(ObsCountObservationWrapper, self).__init__(env)
        low = np.zeros(self.size * self.size + self.rocks)

        high = np.ones_like(low)
        high[-self.rocks:] *= 1000
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )
        self.decay = decay
        self.normalize = normalize
        self.count_init = count_init

        self.counts = None

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        position, rock_morality, _, current_rocks_obs = self.unwrapped.unpack_state(state)
        position_obs = np.zeros((self.size, self.size))
        position_obs[int(position[0]), int(position[1])] = 1
        position_obs = np.concatenate(position_obs)

        count_obs = self.counts
        obs = self.env.get_obs(state)

        if self.normalize:
            count_obs = 1 / (np.sqrt(self.counts) + 1)
        # return np.concatenate([position_obs, count_obs])
        return np.concatenate([position_obs, count_obs, obs])

    def _update_counts(self):
        if self.decay < 1.:
            self.counts *= self.decay
        obs = self.env.get_obs(self.state)
        self.counts += obs[-self.rocks:]
        # self.counts += (1 - self.decay) * obs[-self.rocks:]

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.counts = np.zeros(self.rocks) + self.count_init
        self._update_counts()
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        self._update_counts()

        return self.get_obs(self.state), reward, done, info
