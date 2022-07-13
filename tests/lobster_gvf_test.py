import numpy as np
from jax import random

from unc.utils.gvfs.lobster import LobsterGVFs
from unc.envs import get_env
from unc.args import Args


if __name__ == "__main__":

    parser = Args()
    args = parser.parse_args(['--env', '2', '--discounting', '0.9'])

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)

    env = get_env(rng, rand_key, args)

    gvfs = LobsterGVFs(env.action_space.n, args.discounting)

    obs = np.expand_dims(env.reset(), 0)

    c, beta = gvfs.cumulant(obs), gvfs.termination(obs)
    assert np.all(c == 0)
    assert np.all(beta == args.discounting)

    # We first test cumulants and terminals
    for act in [0, 1]:
        while True:
            obs, _, _, _ = env.step(act)
            obs = np.expand_dims(obs, 0)
            c, beta = gvfs.cumulant(obs), gvfs.termination(obs)

            if env.state[0] == (act + 1):
                break

            assert c[0, act] == 0
            assert beta[0, act] == args.discounting

            policy = np.zeros((1, 3))
            # policy = np.zeros((2, 3))
            policy[:, act] = 1
            impt_sampling_ratios = gvfs.impt_sampling_ratio(obs, policy, np.array([act]))
            # impt_sampling_ratios = gvfs.impt_sampling_ratio(obs.repeat(2, 0), policy, np.array([act, act]))
            zero_actions = [0, 1]
            zero_actions.remove(act)
            assert np.all(impt_sampling_ratios[0, zero_actions[0]] == 0)
            assert np.all(impt_sampling_ratios[0, act] == 1)

        assert c[0, act] == 1
        assert beta[0, act] == 0
    print("done")


