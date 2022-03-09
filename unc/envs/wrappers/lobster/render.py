import numpy as np


class LobsterFishingRenderWrapper:
    priority = float('inf')

    def render(self, mode='rgb_array',
               q_vals: np.ndarray = None,
               obs: np.ndarray = None,
               **kwargs):