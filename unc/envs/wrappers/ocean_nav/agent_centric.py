import gym
import numpy as np
from typing import Union, Tuple

from unc.envs.ocean_nav import OceanNav
from .wrapper import OceanNavWrapper


class AgentCentricObservationWrapper(OceanNavWrapper):
    priority = 3

    def __init__(self, env: Union[OceanNav, OceanNavWrapper]):
        """
        Agent-centric observation.

        We take as input a previous get_obs w x h x 7 tensor and make it agent-centric.
        """
        super(AgentCentricObservationWrapper, self).__init__(env)
        self.map_size = self.observation_space.shape[0]
        assert self.map_size == self.observation_space.shape[1] and len(self.observation_space.shape) == 3, "We can't currently deal with differing height/width"
        assert self.map_size % 2 != 0, "Wrapper requires an odd size for centering."

        self.expanded_map_size = self.map_size + self.map_size - 1

        self.obstacle_filler_idx = 1
        self.position_filler_idx = 0
        self.reward_filler_idx = 0
        self.glass_filler_idx = 0
        self.current_filler = np.array([0, 0, 0, 0], dtype=np.int16)

        self.expanded_obstacle_map = np.zeros((self.expanded_map_size, self.expanded_map_size, 1), dtype=np.int16)
        self.expanded_obstacle_map += self.obstacle_filler_idx

        self.expanded_reward_map = np.zeros_like(self.expanded_obstacle_map)
        self.expanded_reward_map += self.reward_filler_idx

        self.expanded_current_map = np.zeros((self.expanded_map_size, self.expanded_map_size, 4), dtype=np.int16)
        self.expanded_current_map[:, :] = self.current_filler

        self.expanded_glass_map = np.zeros_like(self.expanded_obstacle_map)
        self.expanded_glass_map += self.glass_filler_idx

        self.expanded_map_template = np.concatenate((self.expanded_obstacle_map, self.expanded_current_map, self.expanded_reward_map), axis=-1)
        self.expanded_map_agent_pos = np.array([self.expanded_map_size // 2, self.expanded_map_size // 2], dtype=np.int16)

        low = np.zeros_like(self.expanded_map_template)
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )

    def get_obs(self, state: np.ndarray, *args,
                return_expanded_glass_map: bool = False, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        obs = super(AgentCentricObservationWrapper, self).get_obs(state, *args, **kwargs)

        # we still assume our obs has the same structure as the unwrapped env get_obs in terms of channels.
        position_map = obs[:, :, 5]
        pos = np.concatenate(np.nonzero(position_map))

        map_to_paste = np.delete(obs, 5, axis=2)

        expanded_map = self.expanded_map_template.copy()
        y_start = self.expanded_map_agent_pos[0] - pos[0]
        x_start = self.expanded_map_agent_pos[1] - pos[1]
        expanded_map[y_start:y_start + self.map_size, x_start:x_start + self.map_size] = map_to_paste
        expanded_map = expanded_map.astype(float)

        if return_expanded_glass_map:
            expanded_glass_map = self.expanded_glass_map.copy()
            expanded_glass_map[y_start:y_start + self.map_size, x_start:x_start + self.map_size, 0] = self.glass_map
            return expanded_map, expanded_glass_map

        return expanded_map

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
