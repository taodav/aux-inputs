import numpy as np

from .compass import north_facing_agent, east_facing_agent, south_facing_agent, west_facing_agent
from .rocksample import generate_rock_rgb, generate_rock_agent_rgb


def get_current_rgb(current_direction: int, size: int, val: int = 0):
    one_d_array = [north_facing_agent, east_facing_agent, south_facing_agent, west_facing_agent][current_direction](size)

    rgb = np.repeat(one_d_array[..., np.newaxis], 3, axis=-1)
    agent = (rgb == 1)
    rgb = np.ones_like(rgb) * 255
    rgb[agent.astype(bool)] = val

    return rgb


def arr_to_viz(obstacle_map: np.ndarray, current_map: np.ndarray,
               position_map: np.ndarray, reward_map: np.ndarray,
               kelp_map: np.ndarray = None,
               glass_map: np.ndarray = None,
               scale: int = 50, grid_lines: bool = True):
    """
    Make a pixel representation of the OceanNav env
    """
    space_color = np.array([255, 255, 255], dtype=np.uint8)
    obstacle_color = np.array([255, 167, 0], dtype=np.uint8)
    glass_color = np.array([102, 178, 255], dtype=np.uint8)
    kelp_color = np.array([167, 246, 185], dtype=np.uint8)
    goal_color = np.array([0, 150, 0], dtype=np.uint8)
    agent_color = np.array([0, 0, 0], dtype=np.uint8)
    grid_color = None

    env_size = obstacle_map.shape[0]

    size = env_size * scale

    if grid_lines:
        size += env_size + 1
        grid_color = np.array([150, 150, 150], dtype=np.uint8)


    final_viz_array = np.zeros((size, size, 3), dtype=np.uint8)

    if grid_lines:
        final_viz_array[::(scale + 1)] = grid_color
        final_viz_array[:, ::(scale + 1)] = grid_color

    for y in range(env_size):
        for x in range(env_size):
            to_fill = None
            if obstacle_map[y, x] != 0:
                if glass_map is not None and glass_map[y, x] != 0:
                    to_fill = np.zeros((scale, scale, 3)) + glass_color
                else:
                    to_fill = np.zeros((scale, scale, 3)) + obstacle_color
            else:
                maybe_white = False
                if current_map[y, x].sum() > 0:
                    current_direction = np.nonzero(current_map[y, x])[0].item()
                    to_fill = get_current_rgb(current_direction, scale)
                elif kelp_map is not None and kelp_map[y, x] > 0:
                    to_fill = np.zeros((scale, scale, 3)) + kelp_color
                elif reward_map[y, x] > 0:
                    to_fill = generate_rock_rgb(scale, goal_color)
                else:
                    maybe_white = True

                if position_map[y, x] != 0:
                    agent_fill = generate_rock_agent_rgb(scale, agent_color)
                    if to_fill is not None:
                        to_fill[(agent_fill != 255).astype(bool)] = agent_fill[(agent_fill != 255).astype(bool)]
                    else:
                        to_fill = agent_fill
                elif maybe_white:
                    to_fill = np.zeros((scale, scale, 3))
                    to_fill[:, :] = np.copy(space_color)

            if grid_lines:
                final_viz_array[y * (scale + 1) + 1:(y + 1) * (scale + 1),
                x * (scale + 1) + 1:(x + 1) * (scale + 1)] = to_fill
            else:
                final_viz_array[y * scale:(y + 1) * scale,
                x * scale:(x + 1) * scale] = to_fill

    return final_viz_array

