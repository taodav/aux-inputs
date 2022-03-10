import numpy as np
from unc.utils.viz.rocksample import rocksample_arr_to_viz
from unc.utils.viz import append_text
from .wrapper import RockSampleWrapper


class RockRenderWrapper(RockSampleWrapper):
    priority = float('inf')

    def __init__(self, *args, **kwargs):
        super(RockRenderWrapper, self).__init__(*args, **kwargs)

        self.action_map = ["North", "East", "South", "West", "Sample"] + [f"Check_{i}" for i in range(1, self.rocks + 1)]

    def _generate_weighted_background(self) -> np.ndarray:
        background = np.zeros((self.size, self.size))
        if hasattr(self, 'weights') and hasattr(self, 'particles'):
            for p, w in zip(self.particles, self.weights):
                for i in range(p.shape[0]):
                    pos = self.rock_positions[i]
                    background[pos[0], pos[1]] += p[i] * w
        else:
            for mor, pos in zip(self.rock_morality, self.rock_positions):
                if mor > 0:
                    background[pos[0], pos[1]] = 1

        return background

    def _rock_weights(self) -> np.ndarray:
        rock_obs = np.zeros(self.rocks, dtype=np.float)
        for p, w in zip(self.particles, self.weights):
            rock_obs += p * w
        return rock_obs

    def render(self, mode: str = 'rgb_array', action: int = None,
               q_vals: np.ndarray = None, show_rock_info: bool = False,
               greedy_actions: np.ndarray = None,
               show_weights: bool = False, **kwargs) -> np.ndarray:
        assert mode == 'rgb_array'

        arr = self.env.generate_array()

        background_weights = None
        if show_weights:
            background_weights = self._generate_weighted_background()

        str_greedy_actions = None
        if greedy_actions is not None:
            str_greedy_actions = np.array(self.action_map)[greedy_actions]

        viz = rocksample_arr_to_viz(arr, scale=100, background_weights=background_weights,
                                    greedy_actions=str_greedy_actions)

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

        if show_rock_info:
            # Rock weights are at the end
            rock_weights = self._rock_weights()
            str_to_attach = ""
            for pos, w in zip(self.rock_positions, rock_weights):
                str_to_attach += f"{pos}: {w:.4f}\n"

            strs_to_attach.append(str_to_attach)

        viz = append_text(viz, strs_to_attach)

        return viz
