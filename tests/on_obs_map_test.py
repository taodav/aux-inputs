import numpy as np
from jax import random
from typing import List

from unc.envs import get_env


if __name__ == "__main__":
    """
    This is mainly a visual test.
    """
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="u1m", distance_noise=False)
    env.reset()

    # we go around the whole environment
    actions = [2, 2, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1]
    for a in actions:
        env.step(a)

    env.position[0] = 2
    env.position[1] = 4

    actions = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 0]
    for a in actions:
        obs, rew, done, info = env.step(a)

    expected_obstacle_map = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    observation_map = env.observation_map[env.obs_map_buffer:-env.obs_map_buffer, env.obs_map_buffer:-env.obs_map_buffer]
    obstacle_map = observation_map[:, :, 0]
    assert np.all(obstacle_map == expected_obstacle_map), "discrepancy in obstacle maps."