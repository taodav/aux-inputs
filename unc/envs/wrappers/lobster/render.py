import numpy as np

from unc.utils.viz.lobster import lobster_fishing_viz
from unc.utils.viz import append_text
from .wrapper import LobsterFishingWrapper


class LobsterFishingRenderWrapper(LobsterFishingWrapper):
    priority = float('inf')
    action_map = ["left", "right", "collect"]

    def render(self, mode='rgb_array',
               action: int = None,
               q_vals: np.ndarray = None,
               show_obs: bool = False,
               scale: int = 50,
               **kwargs):

        full_viz = lobster_fishing_viz(self.state, scale=scale)

        strs_to_attach = [""]
        if action is not None:
            strs_to_attach[0] += f"Action: {self.action_map[action]} "

        if q_vals is not None:
            assert q_vals.shape[0] == len(self.action_map)
            str_to_attach = ""
            for act, val in zip(self.action_map, q_vals):
                str_to_attach += f"{act}: {val:.4f}\n"

            strs_to_attach.append(str_to_attach)

        if show_obs:
            obs = self.get_obs(self.state)
            obs_str = f"""
Obs - Position: {int(self.state[0])} 
r1 o+t: {obs[3]:.2f}, r1 o+a: {obs[4]:.2f}, r1 not-o: {obs[5]:.2f}
r2 o+t: {obs[6]:.2f}, r2 o+a: {obs[7]:.2f}, r2 not-o: {obs[8]:.2f}"""
            strs_to_attach[0] += obs_str

        full_viz = append_text(full_viz, strs_to_attach)

        return full_viz
