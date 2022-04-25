import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    """
    This is mainly a visual test.
    """
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="u1a")

    obs = env.reset()
    halfway = obs.shape[0] // 2
    position = env.position
    assert obs[halfway, halfway, 0] == 0, "No obstacles in the way"
    assert obs[halfway, halfway, 5] == 0, "No reward in the starting position in the way"
    assert np.all(obs[halfway, halfway, 1:5] == 0), "No currents in the starting position in the way"

    for _ in range(2):
        env.step(2)

    for _ in range(4):
        env.step(1)

    for _ in range(6):
        obs, rew, done, info = env.step(2)

    # Test kelp filled env here
    env = get_env(rng, rand_key, env_str="u0p", distance_noise=False)

    env.reset()

    env.position[0] = 4
    env.position[1] = 6

    obs = env.get_obs(env.state)
    size = env.observation_space.shape[0]

    assert obs[4, 4, 1 + 3] == 1 and obs[4, 4, 1:5].sum() == 1

    print("All tests passed")

