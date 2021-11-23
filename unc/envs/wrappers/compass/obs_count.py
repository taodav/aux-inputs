import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs.compass import CompassWorld


class ObsCountObservationWrapper(CompassWorldWrapper):
    """
    State based count observations
    Essentially our observations are of size self.env.size * self.env.size * 4 + 5,
    reflecting a count of all states since the beginning of the episode.

    We also include a decay rate of previous counts at every step.
    """
    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper], decay: float = 1.,
                 normalize: bool = True, count_init: float = 10.):
        super(ObsCountObservationWrapper, self).__init__(env)

        self.observation_space = gym.spaces.Box(
            low=np.zeros(5),
            high=np.ones(5) * 1000
        )
        self.decay = decay
        self.normalize = normalize
        self.count_init = count_init

        self.counts = None

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        if self.normalize:
            return 1 / (self.counts.flatten() + 1)
        return self.counts.flatten()

    def _update_counts(self):
        if self.decay < 1.:
            self.counts *= self.decay
        # self.counts += self.env.get_obs(self.state)
        self.counts += (1 - self.decay) * self.env.get_obs(self.state)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.counts = np.zeros(5)
        self._update_counts()
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        self._update_counts()

        return self.get_obs(self.state), reward, done, info



