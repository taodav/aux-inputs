import numpy as np
from pathlib import Path
from PIL import Image


def save_info(results_path: Path, info: dict):
    np.save(results_path, info)


def load_info(results_path: Path):
    return np.load(results_path, allow_pickle=True).item()


def save_gif(arr: np.ndarray, path: Path, duration=400):
    gif = [Image.fromarray(img) for img in arr]

    gif[0].save(path, save_all=True, append_images=gif[1:], duration=duration, loop=0)


def west_facing_triangle(size: int) -> np.ndarray:
    grid = np.zeros((size, size))
    left_tri = np.ones_like(grid[:grid.shape[0] // 2, :])
    bottom_half = np.triu(left_tri, k=size // 4 + 1)
    top_half = np.flip(bottom_half, axis=0)
    return np.concatenate((top_half, bottom_half), axis=0)


def north_facing_triangle(size: int) -> np.ndarray:
    west = west_facing_triangle(size)
    return west.T


def east_facing_triangle(size: int) -> np.ndarray:
    return np.flip(west_facing_triangle(size), axis=1)


def south_facing_triangle(size: int) -> np.ndarray:
    return np.flip(north_facing_triangle(size), axis=0)


def generate_agent_rgb(one_d_array: np.ndarray, val: int = 0):
    rgb = np.repeat(one_d_array[..., np.newaxis], 3, axis=-1)
    rgb[rgb == 0] = 255
    rgb[rgb == 1] = val

    return rgb


def arr_to_viz(arr: np.ndarray, scale: int = 10, grid_lines: bool = True) -> np.ndarray:
    """
    Convert array representation of Compass World state to
    a scaled RGB array.
    Refer to CompassWorld.generate_array for a color mapping.
    :param arr: Array representation of Compass World (ref. CompassWorld.render)
    :param scale: Scale in which to make visualization. Each grid will be scalexscale pixels wide.
    :param grid_lines: Do we draw grid lines or not?
    :return: numpy array which you can plot.
    """
    space_color = np.array([255, 255, 255], dtype=np.uint8)
    orange_color = np.array([255, 167, 0], dtype=np.uint8)
    yellow_color = np.array([255, 239, 0], dtype=np.uint8)
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    blue_color = np.array([0, 0, 255], dtype=np.uint8)
    green_color = np.array([0, 255, 0], dtype=np.uint8)
    agent_color = np.array([0, 0, 0], dtype=np.uint8)

    color_map = [space_color, orange_color, yellow_color, red_color, blue_color, green_color, agent_color]

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
            assert val <= 9, "index out of range for image"
            if val == 6:
                north = north_facing_triangle(scale)
                to_fill = generate_agent_rgb(north, val=0)
            elif val == 7:
                east = east_facing_triangle(scale)
                to_fill = generate_agent_rgb(east, val=0)
            elif val == 8:
                south = south_facing_triangle(scale)
                to_fill = generate_agent_rgb(south, val=0)
            elif val == 9:
                west = west_facing_triangle(scale)
                to_fill = generate_agent_rgb(west, val=0)
            else:
                to_fill = color_map[val]
            if grid_lines:
                final_viz_array[y * (scale + 1) + 1:(y + 1) * (scale + 1),
                                x * (scale + 1) + 1:(x + 1) * (scale + 1)] = to_fill
            else:
                final_viz_array[y * scale:(y + 1) * scale,
                                x * scale:(x + 1) * scale] = to_fill

    return final_viz_array


