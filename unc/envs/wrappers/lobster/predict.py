import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing


class PredictionObservationWrapper(LobsterFishingWrapper):
    """
    Code: e
    (We use code e here b/c a poisson distribution w/ increasing rate
    is essentially an exponential function)
    """
    priority = 2

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(PredictionObservationWrapper, self).__init__(env)

        self.max_obs = np.ones(2)

        self.r = env.unwrapped.r

        # this is a count of the time since last seen
        self.counts = None

    def _tick(self, obs: np.ndarray):
        self.counts += 1

        # if we can see either reward
        if obs[5] == 0:
            self.counts[0] = 0

        if obs[8] == 0:
            self.counts[1] = 0

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        obs = self.env.get_obs(state).copy()

        # we only replace observations if rewards are unobservable
        r1_unobservable = obs[5] == 1
        r2_unobservable = obs[8] == 1
        if r1_unobservable or r2_unobservable:
            rts = self.counts * self.r
            pmfs = rts * np.exp(-rts)

            if r1_unobservable:
                obs[4] = pmfs[0]

            if r2_unobservable:
                obs[7] = pmfs[1]

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)

        self.counts = np.zeros(2)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._tick(obs)

        return self.get_obs(self.state), reward, done, info



