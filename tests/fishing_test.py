import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="uf0", task_fname="fishing_{}_config.json", render=False)

    obs = env.reset()

    obs, rew, done, info = env.step(2)
    for _ in range(3):
        obs, rew, done, info = env.step(3)

    assert rew == 1

    # make sure our reward doesn't regenerate right away
    obs, rew, done, info = env.step(3)
    assert rew == 0

    # Now we test that our wrappers are working
    env = get_env(rng, rand_key, env_str="uf0a", task_fname="fishing_{}_config.json", render=False)
    obs = env.reset()

    size = env.size
    offset = size - 1
    assert obs[offset + 1, offset, 4] == 1
    assert obs[offset + 1, offset - 5, -1] == 1

    print("All tests passed for fishing")
