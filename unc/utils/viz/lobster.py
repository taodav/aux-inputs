# ==========================================================
#                      LOBSTER FISHING
# ==========================================================

import numpy as np
from typing import Tuple

from .four_room import generate_agent_rgb, generate_four_room_agent, generate_four_room_reward


class Room:
    space_color = np.array([255, 255, 255], dtype=np.uint8)
    agent_color = np.array([0, 0, 0], dtype=np.uint8)
    reward_color = np.array([0, 255, 0], dtype=np.uint8)
    empty_reward_color = np.array([255, 239, 0], dtype=np.uint8)

    def __init__(self, agent_present: bool, reward_present: bool,
                 reward_active: bool = None,
                 scale: int = 30):
        """
        A room. Represented by a 3 x 3 x scale x scale array.
        Each of the 3 x 3 cells can be filled with anything (ie. the agent sprite).
        """
        self.scale = scale
        self.agent_present = agent_present
        self.reward_present = reward_present
        self.reward_active = reward_active

        self.cells = np.zeros((3, 3, self.scale, self.scale, 3), dtype=np.uint8)
        self.cells[:, :, :, :] = self.space_color
        self.draw_room()

    def draw_room(self):
        if self.agent_present:
            agent = generate_four_room_agent(self.scale, self.agent_color)
            to_fill = generate_agent_rgb(agent)
            self.place_in_center((1, 1), to_fill)

        if self.reward_present:
            reward = generate_four_room_reward(self.scale, self.agent_color)
            color = self.reward_color if self.reward_active else self.empty_reward_color
            to_fill = generate_agent_rgb(reward, val=0)
            to_fill += (to_fill == 0) * color
            self.place_in_center((0, 2), to_fill)

    def place_in_center(self, coords: Tuple[int, int], img: np.ndarray):
        y, x = coords
        height, width = img.shape[:2]
        assert y < self.cells.shape[0] and x < self.cells.shape[1]
        assert height <= self.scale and width <= self.scale

        row = self.scale // 2 - height // 2
        col = self.scale // 2 - width // 2
        self.cells[y, x, row:row + height, col:col + width] = img

    def rgb(self):
        rows_concat = np.concatenate(self.cells, axis=1)
        all_concat = np.concatenate(rows_concat, axis=1)
        return all_concat


def lobster_fishing_viz(state: np.ndarray, scale: int = 30):
    """
    Given a state, create a vizualization of the state.
    """
    pos = state[0]
    cages_full = state[1:]
    wall_color = np.array([255, 167, 23], dtype=np.uint8)

    room_0 = Room(pos == 0, False, scale=scale)
    room_1 = Room(pos == 1, True, cages_full[0] == 1, scale=scale)
    room_2 = Room(pos == 2, True, cages_full[1] == 1, scale=scale)

    border_thickness = 4
    room_border = np.zeros((scale * 3, border_thickness, 3), dtype=np.uint8)
    room_border[:, :] = wall_color

    final_viz_array = np.concatenate((room_1.rgb(), room_border, room_0.rgb(), room_border, room_2.rgb()), axis=1)
    return final_viz_array
