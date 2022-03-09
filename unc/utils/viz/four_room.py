import numpy as np
from .rocksample import create_rocksample_agent
from .compass import generate_agent_rgb

# ==========================================================
#                      FOUR ROOM
# ==========================================================


def generate_four_room_agent(size: int, agent_color: np.ndarray) -> np.ndarray:
    """
    Taken from https://stackoverflow.com/questions/58348401/numpy-array-filled-in-diamond-shape
    """
    a = np.arange(size)
    b = np.minimum(a, a[::-1])
    return (b[:, None] + b) >= (size - 1) // 2


def generate_four_room_reward(size: int, agent_color: np.ndarray) -> np.ndarray:
    return create_rocksample_agent(size, size, radius=size // 3, thickness=4)


def four_room_arr_to_viz(arr: np.ndarray, scale: int = 10, grid_lines: bool = True) -> np.ndarray:
    """
    Convert array representation of Four Room state to
    a scaled RGB array.
    Refer to FourRoom.generate_array for a color mapping.
    :param arr: Array representation of FourRoom (ref. FourRoom.generate_array)
    :param scale: Scale in which to make visualization. Each grid will be scalexscale pixels wide.
    :param grid_lines: Do we draw grid lines or not?
    :return: numpy array which you can plot.
    """
    space_color = np.array([255, 255, 255], dtype=np.uint8)
    wall_color = np.array([255, 167, 23], dtype=np.uint8)
    agent_color = np.array([0, 0, 0], dtype=np.uint8)
    reward_color = np.array([0, 255, 0], dtype=np.uint8)
    empty_reward_color = np.array([255, 239, 0], dtype=np.uint8)
    grid_color = None

    color_map = [space_color, wall_color, agent_color, reward_color, empty_reward_color]

    size = arr.shape[0] * scale
    if grid_lines:
        size += arr.shape[0] + 1
        grid_color = np.array([150, 150, 150], dtype=np.uint8)

    final_viz_array = np.zeros((size, size, 3), dtype=np.uint8)

    if grid_lines:
        final_viz_array[::(scale + 1)] = grid_color
        final_viz_array[:, ::(scale + 1)] = grid_color

    for y, row in enumerate(arr):
        for x, val in enumerate(row):
            assert val <= 4, "index out of range for image"
            background = None

            if val == 2:
                agent = generate_four_room_agent(scale, agent_color)
                to_fill = generate_agent_rgb(agent)
            elif val == 3 or val == 4:
                reward = generate_four_room_reward(scale, agent_color)
                to_fill = generate_agent_rgb(reward, val=0)
                to_fill += (to_fill == 0) * color_map[val]
            else:
                to_fill = np.copy(color_map[val])
                if val == 0 and background is not None:
                    to_fill = background

            if grid_lines:
                final_viz_array[y * (scale + 1) + 1:(y + 1) * (scale + 1),
                x * (scale + 1) + 1:(x + 1) * (scale + 1)] = to_fill
            else:
                final_viz_array[y * scale:(y + 1) * scale,
                x * scale:(x + 1) * scale] = to_fill

    return final_viz_array
