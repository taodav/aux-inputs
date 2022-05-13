import numpy as np
from typing import Union

from unc.utils.viz.ocean_nav import arr_to_viz
from unc.utils.viz import append_text
from .wrapper import OceanNavWrapper
from .obs_map import ObservationMapWrapper, OceanNav


class OceanNavRenderWrapper(OceanNavWrapper):
    priority = float('inf')

    action_map = ["North", "East", "South", "West"]

    def get_observation_map_layer(self) -> Union[ObservationMapWrapper, None]:
        obs_map_layer = None
        found = False

        # peel the onion
        while not isinstance(obs_map_layer, OceanNav):
            obs_map_layer = self.env
            if isinstance(obs_map_layer, ObservationMapWrapper):
                found = True
                break
        if found:
            return obs_map_layer

        return None

    def render(self, mode: str = 'rgb_array', action: int = None,
               q_vals: np.ndarray = None,
               render_map: bool = False, **kwargs):
        assert mode == 'rgb_array', "Not implemented render type"
        obs = None
        certainty_map = None
        glass_map = None

        # this means we render our partially observable map
        if render_map:
            obs_map_env = self.get_observation_map_layer()
            if obs_map_env is not None:
                obs = obs_map_env.get_obs(self.state)
                if obs_map_env.uncertainty_decay < 1.:
                    certainty_map = obs[:, :, 6]
                obstacle_map = obs[:, :, 0]
                current_map = obs[:, :, 1:5]
                position_map = np.zeros_like(obstacle_map)
                position_map[obs.shape[0] // 2, obs.shape[1] // 2] = 1
                reward_map = obs[:, :, 5]

                if np.any(self.glass_map):
                    _, pos, _ = self.unwrapped.unpack_state(self.state)
                    y_start = self.expanded_map_agent_pos[0] - pos[0]
                    x_start = self.expanded_map_agent_pos[1] - pos[1]
                    glass_map = obs_map_env.expanded_glass_map.copy()
                    glass_map[y_start:y_start + obs_map_env.map_size, x_start:x_start + obs_map_env.map_size, 0] = self.glass_map

        if obs is None:
            obs = self.unwrapped.get_obs(self.state)
            obstacle_map = obs[:, :, 0]
            current_map = obs[:, :, 1:5]
            position_map = obs[:, :, 5]
            reward_map = obs[:, :, 6]

        if glass_map is None:
            glass_map = self.glass_map if np.any(self.glass_map) else None

        kelp_map = self.kelp_map if np.any(self.kelp_map) else None

        viz = arr_to_viz(obstacle_map, current_map, position_map, reward_map,
                         glass_map=glass_map, kelp_map=kelp_map,
                         certainty_map=certainty_map)

        strs_to_attach = []

        # Either plot single action or all q-values.
        if action is not None:
            strs_to_attach.append(self.action_map[action])

        if q_vals is not None:
            assert q_vals.shape[0] == len(self.action_map)
            str_to_attach = ""
            for act, val in zip(self.action_map, q_vals):
                str_to_attach += f"{act}: {val:.4f}\n"

            strs_to_attach.append(str_to_attach)

        if strs_to_attach:
            viz = append_text(viz, strs_to_attach)

        return viz
