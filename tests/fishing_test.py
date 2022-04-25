import numpy as np
from jax import random

from unc.envs import get_env


if __name__ == "__main__":
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="uf0")
