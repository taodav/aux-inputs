import numpy as np
from jax import random

from unc.args import Args
from unc.envs import get_env
from .gather_lobster import init_and_train


if __name__ == "__main__":
    discounting = 0.9
    step_size = 1e-3
    total_steps = 1000000
    max_episode_steps = 200
    gather_n_episodes = 30

    parser = Args()
    gt_args = parser.parse_args()

    # Seeding
    np.random.seed(gt_args.seed)
    rng = np.random.RandomState(gt_args.seed)
    rand_key = random.PRNGKey(gt_args.seed)

    gt_args.algo = "sarsa"
    gt_args.arch = "linear"
    gt_args.env = "2e"
    gt_args.discounting = discounting
    gt_args.step_size = step_size
    gt_args.total_steps = total_steps
    gt_args.n_hidden = 20
    gt_args.max_episode_steps = max_episode_steps
    gt_args.seed = 2023

    print(f"Training 2e agent")
    gt_agent, gt_env = init_and_train(gt_args)

    parser = Args()
    trace_args = parser.parse_args()
    trace_args.env = "2o"
    trace_env = get_env(rng, rand_key, trace_args)

    parser = Args()
    pf_args = parser.parse_args()
    pf_args.env = "2pb"
    pf_env = get_env(rng, rand_key, pf_args)

    parser = Args()
    gvf_args = parser.parse_args()
    gvf_args.env = "2t"
    gvf_args.algo = "sarsa"
    gvf_args.arch = "nn"
    gvf_args.n_hidden = 20
    gvf_args.discounting = discounting
    gvf_args.step_size = step_size
    gvf_args.total_steps = total_steps
    gvf_args.max_episode_steps = max_episode_steps
    gvf_args.seed = 2023

    print(f"Training 2t agent")
    gvf_agent, gvf_env = init_and_train(gt_args)

    all_obs = {
        '2e': [],
        '2o': [],
        '2pb': [],
        '2t': [],
    }

    # now we roll out gather_n_episodes episodes based on our gt agent
    for ep in range(gather_n_episodes):
        all_eps_obs = {
            '2e': [],
            '2o': [],
            '2pb': [],
            '2t': [],
        }

        # This is okay for now, since we don't have a stochastic start
        all_eps_obs['2e'].append(gt_env.reset())
        all_eps_obs['2o'].append(trace_env.reset())
        all_eps_obs['2pb'].append(pf_env.reset())
        all_eps_obs['2t'].append(gvf_env.reset())




