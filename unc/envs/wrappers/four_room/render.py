import numpy as np

from unc.utils.viz.four_room import four_room_arr_to_viz
from .wrapper import FourRoomWrapper


class FourRoomRenderWrapper(FourRoomWrapper):
    priority = float('inf')

    def render(self, mode='rgb_array', **kwargs):
        arr = self.generate_array()

        viz = four_room_arr_to_viz(arr, scale=100)

        return viz