import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    seed = 2023
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, env_str="2")

    obs = env.reset()

    assert np.all(obs == np.array([1, 0, 0,
                                   0, 0, 1,
                                   0, 0, 1]))

    # first we test collecting nothing
    obs, rew, done, info = env.step(2)

    assert rew == 0, "Can't collect nothing!"

    # Let's go to room 1
    steps_to_1 = 0
    while not obs[1]:
        obs, rew, done, info = env.step(0)
        steps_to_1 += 1

    print(f"Steps to room 1: {steps_to_1}")
    assert obs[4] == 1, "I can't see the reward in state 1"

    # now we collect our lobsters
    obs, rew, done, info = env.step(2)

    assert rew == 1, "I didn't get the reward from collecting in state 1"
    assert obs[3] == 1, "State 1 reward is still observable"

    # now let's go back
    while not obs[0]:
        obs, rew, done, info = env.step(1)

    assert obs[5] == 1 and obs[3] == 0 and obs[4] == 0

    # Let's go to room 2
    steps_to_2 = 0
    while not obs[2]:
        obs, rew, done, info = env.step(1)
        steps_to_2 += 1

    print(f"Steps to room 2: {steps_to_2}")
    assert obs[7] == 1, "I can't see the reward in state 2"

    # now we collect our lobsters
    obs, rew, done, info = env.step(2)

    assert rew == 1, "Not collecting the reward from state 2"
    assert obs[6] == 1, "State 2 reward still observable"

