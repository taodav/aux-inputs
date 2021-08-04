import numpy as np
from PIL import Image, ImageDraw, ImageFont

def east_triangle(size: int) -> np.ndarray:
    grid = np.ones((size, size))
    up_tri = np.triu(grid)
    east = np.flip(up_tri, axis=0) * up_tri
    return east


def south_triangle(size: int) -> np.ndarray:
    return east_triangle(size).T


def west_triangle(size: int) -> np.ndarray:
    return np.flip(east_triangle(size), axis=1)


def north_triangle(size: int) -> np.ndarray:
    return np.flip(south_triangle(size), axis=0)


def cross(size: int) -> np.ndarray:
    grid = np.zeros((size, size))
    indices = np.arange(size)
    grid[indices, indices] = 1
    rev_diag = np.flip(grid, axis=0)
    cross = np.maximum(grid, rev_diag)
    return cross


def west_facing_agent(size: int, width: int = 2) -> np.ndarray:
    grid = np.zeros((size, size))
    left_tri = np.ones_like(grid[:grid.shape[0] // 2, :])
    bottom_half = np.triu(left_tri, k=size // 4 + 1)
    top_half = np.flip(bottom_half, axis=0)

    bs = np.triu(left_tri, k=size // 4 + 1 + width)
    ts = np.flip(bs, axis=0)
    smaller = np.concatenate((ts, bs), axis=0)
    larger = np.concatenate((top_half, bottom_half), axis=0)
    return larger - smaller


def north_facing_agent(size: int, width: int = 2) -> np.ndarray:
    west = west_facing_agent(size, width=width)
    return west.T


def east_facing_agent(size: int, width: int = 2) -> np.ndarray:
    return np.flip(west_facing_agent(size, width=width), axis=1)


def south_facing_agent(size: int, width: int = 2) -> np.ndarray:
    return np.flip(north_facing_agent(size, width=width), axis=0)


def plot_arr(arr: np.ndarray):
    Image.fromarray(np.uint8(arr)).show()


def generate_agent_rgb(one_d_array: np.ndarray, val: int = 0, w_p: int = 0):
    rgb = np.repeat(one_d_array[..., np.newaxis], 3, axis=-1)
    background = (rgb == 0)
    rgb[background] = 255
    rgb[:, :, :2] -= w_p
    rgb[(1 - background).astype(np.bool)] = val

    return rgb


def arr_to_viz(arr: np.ndarray, scale: int = 10, grid_lines: bool = True,
               background_weights: np.ndarray = None) -> np.ndarray:
    """
    Convert array representation of Compass World state to
    a scaled RGB array.
    Refer to CompassWorld.generate_array for a color mapping.
    :param arr: Array representation of Compass World (ref. CompassWorld.render)
    :param scale: Scale in which to make visualization. Each grid will be scalexscale pixels wide.
    :param grid_lines: Do we draw grid lines or not?
    :param background_weights: Weights to color the background.
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
            w_p = 0
            if background_weights is not None:
                # w is between [0, 1]
                w = background_weights[y, x]

                # 1 needs to be nearly solid blue w' ~ 127
                # 0 needs to be nearly white w' ~ 0
                # we then subtract the RG channels by w'
                if w > 0:
                    w_p = int(w * 155) + 30

            if val == 6:
                north = north_facing_agent(scale)
                to_fill = generate_agent_rgb(north, val=0, w_p=w_p)
            elif val == 7:
                east = east_facing_agent(scale)
                to_fill = generate_agent_rgb(east, val=0, w_p=w_p)
            elif val == 8:
                south = south_facing_agent(scale)
                to_fill = generate_agent_rgb(south, val=0, w_p=w_p)
            elif val == 9:
                west = west_facing_agent(scale)
                to_fill = generate_agent_rgb(west, val=0, w_p=w_p)
            else:
                to_fill = np.copy(color_map[val])
                if val == 0 and w_p > 0:
                    to_fill[:2] -= w_p
            if grid_lines:
                final_viz_array[y * (scale + 1) + 1:(y + 1) * (scale + 1),
                x * (scale + 1) + 1:(x + 1) * (scale + 1)] = to_fill
            else:
                final_viz_array[y * scale:(y + 1) * scale,
                x * scale:(x + 1) * scale] = to_fill

    return final_viz_array


def append_text(viz_array: np.ndarray, to_append: str) -> np.ndarray:
    h, w, _ = viz_array.shape
    font = ImageFont.truetype("Ubuntu-B.ttf", 24)

    img_to_guide = Image.new('RGB', (w, h // 2), (255, 255, 255))

    d = ImageDraw.Draw(img_to_guide)
    d.text((0, 0), to_append, (0, 0, 0), font=font)

    text_w, text_h = d.textsize(to_append)

    img_to_append = Image.new('RGB', (w, h // 2), (255, 255, 255))
    d_actual = ImageDraw.Draw(img_to_append)
    d_actual.text(((w - text_w) // 2, (h // 2 - text_h) // 2), to_append, fill=(0, 0, 0), font=font)
    arr_to_append = np.array(img_to_append)

    final_image = np.concatenate((viz_array, arr_to_append), axis=0)

    return final_image

