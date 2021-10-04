import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List

# ==========================================================
#                        ROCKSAMPLE
# ==========================================================


def create_circular_mask(h: int, w: int, center: Tuple[int, int] = None, radius: int = None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def create_rocksample_agent(h: int, w: int, thickness: int = 3, radius: int = None):
    grid = create_circular_mask(h, w, radius=radius)
    grid_inner = create_circular_mask(h, w, radius=radius - thickness)
    return np.bitwise_xor(grid, grid_inner)


def generate_rock_agent_rgb(size: int, agent_color: np.ndarray) -> np.ndarray:
    agent_mask = create_rocksample_agent(size, size, radius=size // 3)
    rgb = np.repeat(agent_mask[..., np.newaxis], 3, axis=-1)
    rgb = np.ones_like(rgb) * 255
    rgb[agent_mask.astype(bool)] = agent_color
    return rgb


def create_rectangle(h: int, w: int, thickness: int = 2, length: int = None, width: int = None):
    grid = np.zeros((h, w))
    if length is None:
        length = h - 2
    if width is None:
        width = w - 2

    assert h > length and w > width
    h_space = (h - length) // 2
    w_space = (w - width) // 2

    # East/West
    w_range_pos = np.arange(thickness) + w_space
    w_range = np.concatenate([w_range_pos, -(w_range_pos + 1)])
    grid[h_space:-h_space, w_range] = 1

    # North/South
    h_range_pos = np.arange(thickness) + h_space
    h_range = np.concatenate([h_range_pos, -(h_range_pos + 1)])
    grid[h_range, w_space:-w_space] = 1
    return grid


def generate_rock_rgb(size: int, rock_color: np.ndarray, weight: float = None,
                      background_color: np.ndarray = None):
    if weight is not None and background_color is None:
        w_p = int(weight * 155) + 30 if weight > 0 else 0
        background_color = np.array([255 - w_p, 255 - w_p, 255])
    l = size - size // 4
    rock_mask = create_rectangle(size, size, thickness=5, length=l, width=l)
    rgb = np.repeat(rock_mask[..., np.newaxis], 3, axis=-1)
    if weight is not None:
        rgb[:, :] = background_color
    else:
        rgb = np.ones_like(rgb) * 255
    rgb[rock_mask.astype(bool)] = rock_color
    return rgb


def generate_label(h: int, w: int, str_label: str,
                   font_size: int = 12):
    """
    Generate an h x w x 3 RGB label with str_label in the centre printed.
    :param h: height
    :param w: width
    :param str_label: string to label
    :param color: color for font
    :param font_size: what size font do we use
    :return:
    """
    # Start with an all white label
    label = Image.new('RGB', (w, h), (255, 255, 255))
    d_actual = ImageDraw.Draw(label)
    font = ImageFont.truetype("FreeMono.ttf", font_size)

    img_to_guide = Image.new('RGB', (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img_to_guide)
    d.text((0, 0), str_label, (0, 0, 0), font=font)

    text_w, text_h = d.textsize(str_label, font)
    offset_x, offset_y = font.getoffset(str_label)
    text_w += offset_x
    text_h += offset_y

    pos = ((w - text_w) // 2, (h - text_h) // 2)

    d_actual.text(pos, str_label, fill=(0, 0, 0), font=font)
    return np.array(label)


def rocksample_arr_to_viz(arr: np.ndarray, scale: int = 10, grid_lines: bool = True,
                          background_weights: np.ndarray = None, greedy_actions: np.ndarray = None) -> np.ndarray:
    """
    Make a pixel representation of rock sample state/array
    :param arr: array representation of the environment
    :param scale: How large do we scale up?
    :param grid_lines: Do we show grid lines?
    :param background_weights: What are the weights for each grid?
    :param greedy_actions: Do we show optimal actions for each position?
    :return: array to plot
    """
    space_color = np.array([255, 255, 255], dtype=np.uint8)
    rock_color = np.array([255, 167, 0], dtype=np.uint8)
    goal_color = np.array([0, 150, 0])
    agent_color = np.array([0, 0, 0], dtype=np.uint8)
    grid_color = None

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
            if val == 1:
                # AGENT
                to_fill = generate_rock_agent_rgb(scale, agent_color)
            elif val == 2:
                # ROCK
                to_fill = generate_rock_rgb(scale, rock_color, weight=background_weights[y, x] if background_weights is not None else None)
            elif val == 3:
                # AGENT + ROCK
                background = generate_rock_rgb(scale, rock_color, weight=background_weights[y, x] if background_weights is not None else None)
                agent = generate_rock_agent_rgb(scale, agent_color)
                background[(agent != 255).astype(bool)] = agent[(agent != 255).astype(bool)]
                to_fill = background
            elif val == 4:
                to_fill = np.zeros((scale, scale, 3))
                to_fill[:, :] = np.copy(goal_color)
            else:
                to_fill = np.zeros((scale, scale, 3))
                to_fill[:, :] = np.copy(space_color)

            if greedy_actions is not None and y < greedy_actions.shape[0] and x < greedy_actions.shape[1]:
                action_str = greedy_actions[y, x]
                label = generate_label(scale, scale, action_str)
                to_fill[(label != 255).astype(bool)] = label[(label != 255).astype(bool)]

            if grid_lines:
                final_viz_array[y * (scale + 1) + 1:(y + 1) * (scale + 1),
                x * (scale + 1) + 1:(x + 1) * (scale + 1)] = to_fill
            else:
                final_viz_array[y * scale:(y + 1) * scale,
                x * scale:(x + 1) * scale] = to_fill

    return final_viz_array


def generate_greedy_action_array(env, agent):
    """
    NOTE: for ROCKSAMPLE only.
    :return:
    """
    all_pos_states = env.sample_all_states()
    obses = []
    for state in all_pos_states:
        obses.append(env.get_obs(state))

    obses = np.stack(obses)
    qs = agent.Qs(obses, agent.network_params)
    actions = np.argmax(qs, axis=1)

    arr = np.zeros((env.size, env.size - 1), dtype=np.int16)
    for act, state in zip(actions, all_pos_states):
        pos = state[:2]
        arr[pos[0], pos[1]] = int(act)

    return arr


# ==========================================================
#                      COMPASS WORLD
# ==========================================================

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


def triangle(size: int, direction: int):
    assert direction >= 0 and direction < 4
    if direction == 0:
        return north_triangle(size)
    elif direction == 1:
        return east_triangle(size)
    elif direction == 2:
        return south_triangle(size)
    elif direction == 3:
        return west_triangle(size)

    raise NotImplementedError()


def cross(size: int) -> np.ndarray:
    grid = np.zeros((size, size))
    indices = np.arange(size)
    grid[indices, indices] = 1
    rev_diag = np.flip(grid, axis=0)
    cross = np.maximum(grid, rev_diag)
    return cross


def west_facing_agent(size: int, width: int = 5) -> np.ndarray:
    grid = np.zeros((size, size))
    left_tri = np.ones_like(grid[:grid.shape[0] // 2, :])
    bottom_half = np.triu(left_tri, k=size // 4 + 1)
    top_half = np.flip(bottom_half, axis=0)

    bs = np.triu(left_tri, k=size // 4 + 1 + width)
    ts = np.flip(bs, axis=0)
    smaller = np.concatenate((ts, bs), axis=0)
    larger = np.concatenate((top_half, bottom_half), axis=0)
    return larger - smaller


def north_facing_agent(size: int, width: int = 5) -> np.ndarray:
    west = west_facing_agent(size, width=width)
    return west.T


def east_facing_agent(size: int, width: int = 5) -> np.ndarray:
    return np.flip(west_facing_agent(size, width=width), axis=1)


def south_facing_agent(size: int, width: int = 5) -> np.ndarray:
    return np.flip(north_facing_agent(size, width=width), axis=0)


def plot_arr(arr: np.ndarray):
    Image.fromarray(np.uint8(arr)).show()


def generate_agent_rgb(one_d_array: np.ndarray, val: int = 0, background_weights: np.ndarray = None):
    rgb = np.repeat(one_d_array[..., np.newaxis], 3, axis=-1)
    agent = (rgb == 1)
    if background_weights is not None:
        rgb = background_weights
    else:
        rgb = np.ones_like(rgb) * 255
    rgb[agent.astype(bool)] = val

    return rgb


def generate_background_tile(size: int, background_weights: np.ndarray, grid_color: np.ndarray = None):
    """
    Create a background tile that's split by cardinal directions.
    Each cardinal direction has a weight defined in background_weights
    :param background_weights: size 4 weight vector
    :param grid_color:
    :return:
    """
    grid = np.zeros((size, size, 3)) + 255
    for i, w in enumerate(background_weights):
        if w > 0:
            w_p = int(w * 155) + 30
            color = np.array([255 - w_p, 255 - w_p, 255])
            tile = triangle(size, i)
            grid[tile.astype(bool)] = color

    c = cross(size)
    grid[c.astype(bool)] = grid_color
    return grid


def compass_arr_to_viz(arr: np.ndarray, scale: int = 10, grid_lines: bool = True,
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
    grid_color = None

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
            background = None
            if background_weights is not None:
                # w is between [0, 1]
                w = background_weights[y, x]

                # 1 needs to be nearly solid blue w' ~ 127
                # 0 needs to be nearly white w' ~ 0
                # we then subtract the RG channels by w'
                background = generate_background_tile(scale, w, grid_color)

            if val == 6:
                north = north_facing_agent(scale)
                to_fill = generate_agent_rgb(north, val=0, background_weights=background)
            elif val == 7:
                east = east_facing_agent(scale)
                to_fill = generate_agent_rgb(east, val=0, background_weights=background)
            elif val == 8:
                south = south_facing_agent(scale)
                to_fill = generate_agent_rgb(south, val=0, background_weights=background)
            elif val == 9:
                west = west_facing_agent(scale)
                to_fill = generate_agent_rgb(west, val=0, background_weights=background)
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


def create_blank(viz_array: np.ndarray):
    """
    Create a blank slate below the image, of size viz_array.shape[0] // 2, viz_array.shape[1]
    :param viz_array:
    :return:
    """
    h, w, _ = viz_array.shape
    blank_image = Image.new('RGB', (w, h // 2), (255, 255, 255))
    return blank_image


def write_on_image(canvas: Image, text_cols: List[str], start_pos: Tuple[int, int] = (0, 0)):
    """
    Write multiple columns of text onto the given image.
    :param canvas: PIL image to write on
    :param text_cols: list of strings, one for each column
    :param start_pos: starting position
    :return:
    """
    w, h = canvas.size
    font = ImageFont.truetype("FreeMono.ttf", 24)

    next_position_to_add = start_pos
    d_actual = ImageDraw.Draw(canvas)
    for text in text_cols:
        img_to_guide = Image.new('RGB', (w, h), (255, 255, 255))

        d = ImageDraw.Draw(img_to_guide)
        d.text((0, 0), text, (0, 0, 0), font=font)

        text_w, text_h = d.textsize(text, font)
        offset_x, offset_y = font.getoffset(text)
        text_w += offset_x
        text_h += offset_y

        d_actual.text(next_position_to_add, text, fill=(0, 0, 0), font=font)
        next_position_to_add = (next_position_to_add[0] + text_w + 20, next_position_to_add[1])

    return canvas


def append_text(viz_array: np.ndarray, to_append: List[str]) -> np.ndarray:
    img_to_append = create_blank(viz_array)
    img_to_append = write_on_image(img_to_append, to_append, start_pos=(10, 10))
    arr_to_append = np.array(img_to_append)

    final_image = np.concatenate((viz_array, arr_to_append), axis=0)

    return final_image


# FOR DEBUGGING
def plot_current_state(env, agent):
    obs = np.array([env.get_obs(env.state)])
    qs = agent.Qs(obs, agent.network_params)
    action = np.argmax(qs, axis=1)
    qs = qs[0]

    greedy_action_arr = generate_greedy_action_array(env, agent)

    render = env.render(action=action, q_vals=qs, show_weights=True, show_rock_info=True,
                             greedy_actions=greedy_action_arr)
    plot_arr(render)


def stringify_actions_q_vals(action_map: List, q_vals: np.ndarray) -> str:
    assert len(action_map) == q_vals.shape[0]
    str = ""
    for a, q in zip(action_map, q_vals):
        str += f"{a}: {q:.3f}\n"
    return str
