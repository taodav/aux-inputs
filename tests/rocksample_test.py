import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    seed = 2021
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, env_str="r", render=False)
    obs = env.reset()

    # Basic transition
    actions_to_rock = [2, 2, 2, 2, 2, 4]
    rew = 0
    for a in actions_to_rock:
        obs, rew, _, _ = env.step(a)

    assert rew < 0, "In the wrong place!"

    # now we check checking
    rock_to_check = 1
    obs, _, _, _ = env.step(4 + rock_to_check)

    assert obs[1 + rock_to_check] == 0

    # Now go to a good rock and check
    good_rock_to_check = 2

    actions_to_good_rock = [0, 0, 0, 3, 3, 3, 3]
    for a in actions_to_good_rock:
        obs, _, _, _ = env.step(a)

    obs, _, _, _ = env.step(4 + good_rock_to_check)

    assert obs[1 + good_rock_to_check] == 1

    # Now we see if terminal is behaving correctly
    actions_to_terminal = [1, 1, 1, 1, 1, 1]
    rew, term = 0, False
    for a in actions_to_terminal:
        obs, rew, term, _ = env.step(a)

    assert term and rew > 0

    print("All tests pass.")


