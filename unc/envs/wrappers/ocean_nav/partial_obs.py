import gym
import numpy as np
from typing import Union, Tuple

from .agent_centric import AgentCentricObservationWrapper, OceanNavWrapper, OceanNav


class PartiallyObservableWrapper(AgentCentricObservationWrapper):
    priority = 3

    def __init__(self, env: Union[OceanNav, OceanNavWrapper], window_size: int = 5):
        """
        Partially observable OceanNav environment.

        The agent can only see a square of window_size x window_size around itself,
        and if there's an obstacle in the way, everything beyond that is occluded.

        We also have a probability of obstacles, rewards and current showing the
        wrong observations, that has probability proportional to (given that
        obs is observation at position away from agent)
        max(abs(pos[0] - obs[0]), abs(pos[1] - obs[1]))
        """
        assert not isinstance(env, AgentCentricObservationWrapper), "Cannot have PartiallyObservable wrapper around AgentCentric"
        super(PartiallyObservableWrapper, self).__init__(env)

        self.window_size = window_size
        assert self.window_size % 2 != 0, "window_size must be odd number"

        if self.window_size > 5:
            # TODO: implement this
            raise NotImplementedError("Haven't implemented occlusion for anything larger than 5 yet.")

        half = self.window_size // 2

        larger_obs_shape = super(PartiallyObservableWrapper, self).observation_space.shape
        agent_pos = larger_obs_shape // 2 - 1
        
        y_start = agent_pos[0] - half
        self.y_range = [y_start, y_start + self.window_size]

        x_start = agent_pos[1] - half
        self.x_range = [x_start, x_start + self.window_size]

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        expanded_ac_map = super(PartiallyObservableWrapper, self).get_obs(state, *args, **kwargs)

        # get our map where we can see through walls
        see_thru_po_map = expanded_ac_map[self.y_range[0]:self.y_range[1], self.x_range[0]:self.x_range[1]]



    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
