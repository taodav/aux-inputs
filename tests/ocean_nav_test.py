import numpy as np
from jax import random

from unc.envs import get_env


def unpack_obs(obs: np.ndarray):
    current_map = obs[1:5]
    position_map = obs[5]
    pos = np.concatenate(np.nonzero(position_map))
    reward_map = obs[6]
    rew_pos = np.nonzero(reward_map)
    rew_pos = np.concatenate(rew_pos)
    return current_map, pos, rew_pos


if __name__ == "__main__":
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="u1")

    obs = env.reset()
    position = env.position
    reward = env.reward

    # our obs is quite big.
    current_map, obs_pos, obs_rew_pos = unpack_obs(obs)

    # essentially unit tests for get_obs
    assert np.all(position == obs_pos)
    assert np.all(reward == obs_rew_pos)

    for g_info in env.currents:
        current = None
        for c_pos in g_info['mapping']:
            if current is None:
                current = np.nonzero(current_map[:, c_pos[0], c_pos[1]])[0].item()
                assert current in g_info['directions']
            else:
                new_current = np.nonzero(current_map[:, c_pos[0], c_pos[1]])[0].item()
                assert current == new_current

    # test bumping
    position = env.position
    obs, rew, done, info = env.step(0)
    new_position = env.position
    assert np.all(position == new_position)
    assert rew == 0

    for _ in range(2):
        env.step(2)

    position = env.position
    obs, rew, done, info = env.step(2)
    new_position = env.position
    assert np.all(position == new_position)
    assert rew == 0

    # Test current
    for _ in range(4):
        env.step(1)

    for _ in range(6):
        env.step(2)

    for _ in range(2):
        env.step(3)

    position = env.position
    obs, rew, done, info = env.step(3)
    new_position = env.position

    assert np.all(position == new_position), "Current doesn't push us back"
    assert rew == 0, "Wrong reward scheme for current"

    # now we test moving currents
    env.position[1] = 2

    env.step(1)
    assert np.all(env.position == np.array([8, 4], dtype=np.int16)), "Current doesn't move you to the correct position"


    obs, rew, done, info = env.step(0)

    assert rew == 0, "not banging into wall yet"

    obs, rew, done, info = env.step(1)
    assert rew == -.1, "not receiving our wall-banging reward"

    # now we test reward collecting and termination
    for _ in range(3):
        env.step(0)

    obs, rew, done, info = env.step(3)

    assert rew == 1., "not receiving the correct reward"
    assert done, "terminal is not correct"

    print("All tests passed.")




