import numpy as np
from jax import random

from unc.envs import get_env
from unc.envs.wrappers.compass import CompassWorldWrapper


def test_observable_env(env: CompassWorldWrapper):
    env.reset()
    env.state = np.array([1, 1, 0])
    assert np.all(env.get_obs(env.state) == np.array([1, 0, 0, 0, 0]))

    env.state = np.array([5, 5, 1])
    assert np.all(env.get_obs(env.state) == np.array([0, 1, 0, 0, 0]))


def test_unobservable_env(env: CompassWorldWrapper):
    env.reset()
    env.state = np.array([1, 1, 0])
    assert np.all(env.get_obs(env.state) == np.array([1, 0, 0, 0, 0]))

    env.state = np.array([2, 2, 0])
    assert np.all(env.get_obs(env.state) == np.array([0, 0, 0, 0, 0]))


if __name__ == "__main__":
    rng = np.random.RandomState(2022)
    rand_key = random.PRNGKey(2022)

    observable_env = get_env(rng, rand_key, env_str='fd', po_degree=0.)
    test_observable_env(observable_env)

    unobservable_env = get_env(rng, rand_key, env_str='fd', po_degree=10.)
    test_unobservable_env(unobservable_env)
