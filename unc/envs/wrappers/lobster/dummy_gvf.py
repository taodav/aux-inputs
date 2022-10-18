import numpy as np
import gym
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from .predict import PredictionObservationWrapper
from unc.envs.lobster import LobsterFishing


class DummyGVFWrapper(PredictionObservationWrapper):
    """
    Code: d
    Given however many (delta) steps since I've seen r_i missing,
    if I take an average number of steps to try and get to r_i,
    what's the likelihood that r_i is present?
    """
    priority = 2

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        obs = self.env.get_obs(state).copy()

        # we only replace observations if rewards are unobservable
        r1_unobservable = obs[5] == 1
        r2_unobservable = obs[8] == 1
        if r1_unobservable or r2_unobservable:
            add_steps = [1, 1]
            if state[0] == 1:
                add_steps = [0, 2]
            elif state[0] == 2:
                add_steps = [2, 0]
            add_steps = np.array(add_steps)
            rts = (self.counts + add_steps / self.traverse_prob) * self.r
            pmfs = np.exp(-rts)

            if r1_unobservable:
                obs[4] = 1 - pmfs[0]

            if r2_unobservable:
                obs[7] = 1 - pmfs[1]

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)

        self.counts = np.zeros(2)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)
        self._tick(obs)

        return self.get_obs(self.state), reward, done, info



