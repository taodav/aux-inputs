import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

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
