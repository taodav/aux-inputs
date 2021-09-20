import numpy as np
from unc.utils import rocksample_arr_to_viz, append_text
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

    def render(self, mode: str = 'rgb_array', action: int = None,
               show_weights: bool = False, **kwargs) -> np.ndarray:
        assert mode == 'rgb_array'

        arr = self.env.generate_array()

        background_weights = None
        if show_weights:
            background_weights = self._generate_weighted_background()

        viz = rocksample_arr_to_viz(arr, scale=100, background_weights=background_weights)
        if action is not None:
            viz = append_text(viz, self.action_map[action])

        return viz
