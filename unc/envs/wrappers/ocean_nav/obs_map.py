import gym
import numpy as np
from typing import Union, Tuple

from .partial_obs import PartiallyObservableWrapper, OceanNavWrapper, OceanNav


class ObservationMapWrapper(PartiallyObservableWrapper):
    priority = 3

    def __init__(self, env: Union[OceanNav, OceanNavWrapper],
                 window_size: int = 5,
                 distance_noise: bool = False,
                 prob_levels: Tuple[int, int, int] = (1, 0.8, 0.65),
                 uncertainty_decay: float = 1.):
        """
        Observation Map for our Ocean Nav environment.

        While the agent only receives a square of window_size x window_size around itself
        (and it's position), everything beyond this is occluded, except for everything
        memorized in the map.

        At every step, the agent takes this observation and incorporates it into
        it's own map.

        TODO: add uncertainty channel and a decay variable

        """
        super(ObservationMapWrapper, self).__init__(env, window_size=window_size,
                                                    distance_noise=distance_noise,
                                                    prob_levels=prob_levels)
        self.uncertainty_decay = uncertainty_decay
        window_obs_size = self.observation_space.shape

        channels = window_obs_size[-1]
        if self.uncertainty_decay < 1:
            channels += 1

        self.obs_map_buffer = window_obs_size[0] // 2

        self.obs_map_size = self.map_size + 2 * self.obs_map_buffer

        self.observation_map = np.zeros((self.obs_map_size, self.obs_map_size, channels), dtype=np.half)
        self.expanded_obs_map = np.zeros((self.expanded_map_size, self.expanded_map_size, channels), dtype=np.half)

        low = np.zeros_like(self.expanded_obs_map, dtype=np.half)
        high = np.ones_like(low, dtype=np.half)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def incorporate_obs(self, window_obs: np.ndarray, pos: np.ndarray) -> None:
        """
        Incorporate an observation given a position.
        It turns out, with the edge buffers in window_obs, the position we need to incorporate
        is exactly the start position of the window to add.
        """
        if self.uncertainty_decay < 1.:
            if self.distance_noise:
                raise NotImplementedError("Haven't decided on how to incorporate distance noise yet into uncertainty")
            window_shape = window_obs.shape[:-1]
            certainty = np.ones(window_shape + (1,))
            window_obs = np.concatenate((window_obs, certainty), axis=-1)
        self.observation_map[pos[0]:pos[0] + window_obs.shape[0], pos[1]:pos[1] + window_obs.shape[1]] = window_obs

    def tick_uncertainties(self):
        self.observation_map[:, :, -1] *= self.uncertainty_decay

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        We return the agent-centric version of our map, at position given by
        the state variable.
        """
        _, pos, _ = self.unpack_state(state)

        # first we get rid of our buffer
        map_centric_map = self.observation_map[self.obs_map_buffer:-self.obs_map_buffer, self.obs_map_buffer:-self.obs_map_buffer]

        expanded_map = self.expanded_obs_map.copy()
        y_start = self.expanded_map_agent_pos[0] - pos[0]
        x_start = self.expanded_map_agent_pos[1] - pos[1]
        expanded_map[y_start:y_start + self.map_size, x_start:x_start + self.map_size] = map_centric_map
        expanded_map = expanded_map.astype(float)

        return expanded_map

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        window_obs = super(ObservationMapWrapper, self).get_obs(self.state)
        self.incorporate_obs(window_obs, self.position)

        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        window_obs = super(ObservationMapWrapper, self).get_obs(self.state)

        if self.uncertainty_decay < 1.:
            self.tick_uncertainties()

        self.incorporate_obs(window_obs, self.position)

        return self.get_obs(self.state), reward, done, info

