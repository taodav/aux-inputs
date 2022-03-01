import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, env_str="4")

    # first some unit tests
    assert env.get_room_idx(np.array([0, 0], dtype=int)) == 0
    assert env.get_room_idx(np.array([env.size - 1, env.size - 1], dtype=int)) == 2

    obs = env.reset()
    # set our starting position to [2, 4]
    env.state = np.array([2, 4, 0, 0, 0])
    assert obs[2:].sum() == 0, "I can see no rewards in the starting room"

    # with this seed (2022), our starting position is [2, 4]
    actions_to_rew = [0, 3, 3, 3]
    rew = 0
    for a in actions_to_rew:
        obs, rew, done, info = env.step(a)

    assert np.all(obs[2:] == 0), "I can see a reward when I'm not supposed to"
    assert rew == 0, "I'm not meant to receive rewards"

    # now we test our corridors
    actions_to_corr = [2, 1, 1, 1, 1]
    for a in actions_to_corr:
        obs, rew, done, info = env.step(a)
    assert np.all(obs[2:] == 0)

    # test the second room
    obs, rew, done, info = env.step(1)
    assert obs[2:][0] == 1

    # here we test reward regeneration
    actions_to_second_reward = [2, 2, 1, 1, 1]
    for a in actions_to_second_reward:
        obs, rew, done, info = env.step(a)

    assert obs[2:][0] == 0

    times_to_regen = []
    for _ in range(100):
        t = 0
        while obs[2:][0] == 0:
            obs, rew, done, info = env.step(a)
            t += 1

        times_to_regen.append(t)

        obs, rew, done, info = env.step(3)

    assert np.mean(times_to_regen) - env.reward_inverse_rates[1] < 1.
    print("All tests passed.")


