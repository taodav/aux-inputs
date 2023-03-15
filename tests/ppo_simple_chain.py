import numpy as np
from jax import random
from jax.config import config

from unc.args import Args
from unc.envs.simple_chain import FullyObservableSimpleChain
from unc.models import build_network
from unc.optim import get_optimizer
from unc.agents import get_agent
from unc.trainers import get_or_load_trainer

def test_ppo_simple_chain():
    seed = 2020
    n = 10

    config.update('jax_platform_name', 'cpu')
    parser = Args()
    args = parser.parse_args()
    args.algo = 'ppo'
    args.arch = 'actor_critic'
    args.layers = 2
    args.n_hidden = 10
    args.step_size = 1e-2

    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    train_env = FullyObservableSimpleChain(n=n)
    test_env = FullyObservableSimpleChain(n=n)
    n_actions = train_env.action_space.n
    features_shape = train_env.observation_space.shape

    with_bias = not (('g' in args.env or 's' in args.env) and args.arch == 'linear')
    network = build_network(args.n_hidden, n_actions, model_str=args.arch, with_bias=with_bias,
                            init=args.weight_init, layers=args.layers, action_cond=args.action_cond)
    optimizer = get_optimizer(args.optim, args.step_size)

    # Initialize agent
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer)

    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env)

    trainer.train()

    all_obs = np.eye(n)
    predicted_vs = agent.V(all_obs, agent.critic_network_params)
    ground_truth_vs = args.discounting**np.arange(n - 1)
    print("Final values learnt:")
    print(predicted_vs)

    print("Ground-truth values:")
    print(ground_truth_vs)


if __name__ == "__main__":
    test_ppo_simple_chain()

