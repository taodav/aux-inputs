import gym
import numpy as np
from typing import Union, Tuple

from unc.envs.ocean_nav import OceanNav
from unc.utils.data import ind_to_one_hot
from .wrapper import OceanNavWrapper


class VectorStateObservationWrapper(OceanNavWrapper):
    priority = 3

    def __init__(self, env: Union[OceanNav, OceanNavWrapper]):
        """
        Vector state observation wrapper.
        Instead of a 3D tensor of observations, we have a 1D vector of our ground-truth
        state.

        This is for debugging purposes, to see if the env is providing enough information
        to solve, OR if there's an issue with the CNN FA.

        First 2 * self.size features - one-hot encoding of both y and x coords.
        next 4 * self.size * self.size features - one-hot encoding of currents
        final 2 * self.size features - one-hot encoding of y and x coords of reward
        """

        super(VectorStateObservationWrapper, self).__init__(env)

        low = np.zeros(2 * self.size + 4 * self.size * self.size + 2 * self.size)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        current_map, position, reward_pos = self.unpack_state(state)

        y_pos_one_hot = np.zeros(self.size)
        x_pos_one_hot = np.zeros(self.size)
        y_pos_one_hot[position[0]] = 1
        x_pos_one_hot[position[1]] = 1

        pos_one_hot = np.concatenate((y_pos_one_hot, x_pos_one_hot))

        current_map_one_hot = ind_to_one_hot(current_map, max_val=4)
        current_map_one_hot = current_map_one_hot[:, :, 1:]
        current_flat = current_map_one_hot.flatten()

        y_rew_one_hot = np.zeros(self.size)
        x_rew_one_hot = np.zeros(self.size)
        y_rew_one_hot[reward_pos[0]] = 1
        x_rew_one_hot[reward_pos[1]] = 1
        rew_one_hot = np.concatenate((y_rew_one_hot, x_rew_one_hot))

        return np.concatenate((pos_one_hot, current_flat, rew_one_hot))

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
