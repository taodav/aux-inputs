import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing


class BoundedDecayingTraceObservationWrapper(LobsterFishingWrapper):
    """
    TODO: CHANGE THIS FOR LOBSTER OBS
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

    def _update_obs(self, rew_obs: np.ndarray):
        self.trace *= self.decay
        self.trace = np.minimum(self.trace + rew_obs, 1)

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        original_obs = self.env.get_obs(state)
        pos, rew_obs = original_obs[:2], original_obs[2:]

        return np.concatenate((pos, self.trace))

    def reset(self, **kwargs) -> np.ndarray:
        original_obs = self.env.reset(**kwargs)
        rew_obs = original_obs[2:]

        self.trace = np.zeros(self.n_rewards)
        self._update_obs(rew_obs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        rew_obs = obs[2:]
        self._update_obs(rew_obs)
        return self.get_obs(self.state), reward, done, info



