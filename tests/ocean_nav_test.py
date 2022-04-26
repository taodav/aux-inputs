import numpy as np
from jax import random
from typing import List

from unc.envs import get_env


def unpack_obs(obs: np.ndarray):
    current_map = obs[:, :, 1:5]
    position_map = obs[:, :, 5]
    pos = np.concatenate(np.nonzero(position_map))
    reward_map = obs[:, :, 6]
    rew_pos = np.nonzero(reward_map)
    rew_pos = np.concatenate(rew_pos)
    return current_map, pos, rew_pos


def check_current_map(one_hot_current_map: np.ndarray, current_map: np.ndarray, currents: List):
    """
    Checks to see if a current map is correct
    """
    for group_info in currents:
        for pos in group_info['mapping']:
            current = current_map[pos[0], pos[1]]
            assert current - 1 in group_info['directions']
            pos_current_vector = one_hot_current_map[pos[0], pos[1]]
            assert pos_current_vector[current - 1] == 1
            assert pos_current_vector.sum() == 1


if __name__ == "__main__":
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="u1")

    obs = env.reset()
    position = env.position
    reward = env.rewards[0]

    # our obs is quite big.
    one_hot_current_map, obs_pos, obs_rew_pos = unpack_obs(obs)

    # essentially unit tests for get_obs
    assert np.all(position == obs_pos)
    assert np.all(reward == obs_rew_pos)

    check_current_map(one_hot_current_map, env.current_map, env.currents)

    # test bumping
    position = env.position
    obs, rew, done, info = env.step(0)
    new_position = env.position
    assert np.all(position == new_position)
    assert rew == 0

    check_current_map(obs[:, :, 1:5], env.current_map, env.currents)

    for _ in range(2):
        env.step(2)

    position = env.position
    obs, rew, done, info = env.step(2)
    new_position = env.position
    assert np.all(position == new_position)
    assert rew == 0

    check_current_map(obs[:, :, 1:5], env.current_map, env.currents)

    # Test current
    for _ in range(4):
        env.step(1)

    for _ in range(6):
        env.step(2)

    for _ in range(2):
        env.step(3)

    position = env.position
    obs, rew, done, info = env.step(3)
    check_current_map(obs[:, :, 1:5], env.current_map, env.currents)

    new_position = env.position

    # assert np.all(position == new_position), "Current doesn't push us back"
    assert rew == 0, "Wrong reward scheme for current"

    # now we test moving currents
    env.position[1] = 6

    env.step(3)
    assert np.all(env.position == np.array([8, 4], dtype=np.int16)), "Current doesn't move you to the correct position"

    obs, rew, done, info = env.step(0)

    assert rew == 0, "not banging into wall yet"

    obs, rew, done, info = env.step(3)
    assert rew == -.1, "not receiving our wall-banging reward"

    # now we test reward collecting and termination
    for _ in range(3):
        env.step(0)

    obs, rew, done, info = env.step(1)

    assert rew == 1., "not receiving the correct reward"
    assert done, "terminal is not correct"

    # test current instantiation
    env = get_env(rng, rand_key, env_str="u0")

    direction_counts = {}
    total = 100000
    current_position = [5, 0]

    for _ in range(total):
        env.reset()
        current = env.current_map[current_position[0], current_position[1]] - 1
        if current not in direction_counts:
            direction_counts[current] = 0
        direction_counts[current] += 1
    current = env.currents[-2]
    for d, prob in zip(current['directions'], current['start_probs']):
        assert np.isclose(direction_counts[d] / total, prob, atol=1e-2)

    print("All tests passed.")




