import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing


class BoundedDecayingTraceObservationWrapper(LobsterFishingWrapper):
    """
    A bounded decaying trace for the reward observation.

    Code: o
    (We use code o here because it's the most similar to observation count)

    Our reward observations are augmented as follows:
    o_t = min(o(s_t) + beta * o_{t - 1}, \bm{1})

    where the \bm{1} here is a vector of ones
    """
    priority = 2

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper], decay: float = 0.9):
        super(BoundedDecayingTraceObservationWrapper, self).__init__(env)

        self.decay = decay
        self.max_obs = np.ones(2)

        self.trace = None

    def _update_trace(self, obs: np.ndarray):
        self.trace *= self.decay

        # if we can see either reward
        if obs[5] == 0:
            self.trace[0] = obs[3]
        else:
            self.trace[0] += obs[3]

        if obs[8] == 0:
            self.trace[1] = obs[6]
        else:
            self.trace[1] += obs[6]


    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        obs = self.env.get_obs(state).copy()
        obs[3] = self.trace[0]
        obs[6] = self.trace[1]

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)

        self.trace = np.zeros(2)
        self._update_trace(obs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._update_trace(obs)

        return self.get_obs(self.state), reward, done, info



