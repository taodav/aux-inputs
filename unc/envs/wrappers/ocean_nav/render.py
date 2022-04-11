import numpy as np

from unc.utils.viz.ocean_nav import arr_to_viz
from unc.utils.viz import append_text
from .wrapper import OceanNavWrapper


class OceanNavRenderWrapper(OceanNavWrapper):
    priority = float('inf')

    action_map = ["North", "East", "South", "West"]

    def render(self, mode: str = 'rgb_array', action: int = None,
               q_vals: np.ndarray = None, **kwargs):
        assert mode == 'rgb_array', "Not implemented render type"

        obs = self.unwrapped.get_obs(self.state)
        obstacle_map = obs[:, :, 0]
        current_map = obs[:, :, 1:5]
        position_map = obs[:, :, 5]
        reward_map = obs[:, :, 6]

        viz = arr_to_viz(obstacle_map, current_map, position_map, reward_map,
                         glass_map=self.glass_map)

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
