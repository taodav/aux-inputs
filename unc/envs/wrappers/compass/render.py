import numpy as np
from unc.utils.viz.compass import compass_arr_to_viz, append_text
from .wrapper import CompassWorldWrapper


class CompassRenderWrapper(CompassWorldWrapper):
    priority = float('inf')

    def _generate_weighted_background(self) -> np.ndarray:
        """
        Generate a background array with all the particle weights summed
        in each position.
        :return:
        """
        assert self.weights is not None and self.particles is not None
        background = np.zeros((self.size, self.size, 4))
        for p, w in zip(self.particles, self.weights):
            background[p[0], p[1], p[2]] += w

        return background


    def render(self, mode: str = 'rgb_array', show_obs: bool = True,
               show_weights: bool = True, **kwargs) -> np.ndarray:
        """
        Generates a
        :param mode:
        :return:
        """
        assert mode == 'rgb_array'

        arr = self.generate_array()

        background_weights = None
        if show_weights:
            background_weights = self._generate_weighted_background()

        viz = compass_arr_to_viz(arr, scale=100, background_weights=background_weights)
        if show_obs:
            obs = self.get_obs(self.state)
            np.set_printoptions(precision=2)
            viz = append_text(viz, str(obs))
        return viz
