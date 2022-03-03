import gym
import numpy as np
from typing import Union, Tuple
from collections import deque

from .wrapper import RockSampleWrapper
from unc.envs.rocksample import RockSample


class RockObservationStatsWrapper(RockSampleWrapper):
    """
    Rock Observation statistics wrapper.

    Code: n

    We essentially calculate the mean and variance of each rock morality given
    past observations.

    For now, if we have n observations for rock 1 in this episode,
    we take statistics over the previous min(n, 10) elements.

    We also normalize our variance to be between 0 and 1.

    NOTE: THIS ALREADY USES GLOBAL STATE OBS.
    there is no need to add that.
    """
    priority = 3

    def __init__(self, env: Union[RockSample, RockSampleWrapper], *args,
                 mean_only: bool = False, vars_only: bool = False,
                 stats_over_n: int = 20, **kwargs):
        super(RockObservationStatsWrapper, self).__init__(env, *args, **kwargs)

        self.mean_only = mean_only
        self.vars_only = vars_only

        # normalizing range for mean (binary, not used right now
        # )
        # self.element_mean_range = 1

        self.stats_over_n = stats_over_n

        # normalizing range for variance (bernoulli r.v.)
        # self.element_var_range = np.sqrt(0.25) / 2
        max_var_sample = np.zeros(self.stats_over_n)
        max_var_sample[::2] = 1
        self.element_var_range = max_var_sample.var(ddof=1)

        low = np.zeros(self.size * self.size + 2 * self.rocks)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

        # self.rock_clipped_check_counts = np.zeros(self.rocks, dtype=int)
        # self.buffer = np.zeros((self.rocks, self.stats_over_n), dtype=int)
        self.buffer = self.rng.choice([0, 1], size=(self.rocks, self.stats_over_n)).astype(int)

    def _update_buffer(self, obs: np.ndarray, action: int):
        rock_idx = action - 5
        if rock_idx > 0:
            rock_obs = obs[2:][rock_idx].astype(int)
            self.buffer[rock_idx][:-1] = self.buffer[rock_idx][1:]
            self.buffer[rock_idx][-1] = rock_obs
            # self.rock_clipped_check_counts[rock_idx] = min(self.stats_over_n, self.rock_clipped_check_counts[rock_idx] + 1)

    def get_obs(self, state: np.ndarray):
        position, rock_morality, _, current_rocks_obs = self.unwrapped.unpack_state(state)
        position_obs = np.zeros((self.size, self.size))
        position_obs[position[0], position[1]] = 1
        position_obs = np.concatenate(position_obs)

        stats = np.zeros(self.rocks * 2)

        stats[::2] = self.buffer.mean(axis=-1)
        stats[1::2] = self.buffer.var(axis=-1, ddof=1) / self.element_var_range

        # for i in range(self.rocks):
            # if self.rock_clipped_check_counts[i] > 1:
            #     samples = self.buffer[i, -self.rock_clipped_check_counts[i]:]
            #     stats[2 * i] = samples.mean()
            #     stats[2 * i + 1] = (samples.std(ddof=1) / self.rock_clipped_check_counts[i]) / self.element_var_range
            # else:
            #     stats[2 * i] = self.buffer[i, -self.rock_clipped_check_counts[i]]
            #     stats[2 * i + 1] = 1

        return np.concatenate([position_obs, stats])



    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, done, info = self.env.step(action)

        self._update_buffer(obs, action)

        return self.get_obs(self.state), reward, done, info
