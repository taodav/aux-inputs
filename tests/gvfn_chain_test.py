import jax
import numpy as np
from jax import random

from unc.models import build_network
from unc.optim import get_optimizer
from unc.agents import get_agent
from unc.trainers.prediction_trainer import PredictionTrainer
from unc.gvfs.slightly_less_simple_chain import SlightlyLessSimpleChainGVF
from unc.envs.slightly_less_simple_chain import SlightlyLessSimpleChain
from unc.eval import slightly_less_simple_chain_gvf_eval
from unc.utils.files import init_files
from unc.args import Args


if __name__ == "__main__":
    gvf_str_args = [
        '--algo', 'sarsa',
        '--arch', 'nn',
        '--discounting', 0.9,
        '--step_size', 0.00001,
        '--total_steps', int(5e5),
        '--max_episode_steps', 200,
        '--seed', 2022,
        '--epsilon', 0.1,
        '--offline_eval_freq', 1000,
        '--action_cond', 'mult',
        '--gvf_trainer', 'prediction',
        # '--tile_code_gvfs'
    ]
    gvf_str_args = [str(s) for s in gvf_str_args]
    parser = Args()
    args = parser.parse_args(gvf_str_args)

    # Seeding
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)

    # Some filesystem initialization
    results_path, checkpoint_dir = init_files(args)

    # Setting our platform
    # TODO: GPU determinism?
    jax.config.update('jax_platform_name', args.platform)

    test_rng = np.random.RandomState(args.seed + 10)
    test_rand_key = random.PRNGKey(args.seed + 10)

    train_env_key, test_env_key, rand_key = random.split(rand_key, 3)

    # Initializing our environment, args we need are filtered out in get_env
    train_env = SlightlyLessSimpleChain()
    test_env = SlightlyLessSimpleChain()

    gvf = SlightlyLessSimpleChainGVF(train_env.action_space.n, args.discounting)

    model_str = args.arch
    features_shape = train_env.observation_space.shape
    n_predictions = gvf.n

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not (('g' in args.env or 's' in args.env) and model_str == 'linear')
    network = build_network(args.n_hidden, train_env.action_space.n, model_str=model_str, with_bias=with_bias,
                            init=args.weight_init, layers=args.layers, action_cond=args.action_cond, n_predictions=n_predictions)
    optimizer = get_optimizer(args.optim, args.step_size)

    # Initialize agent
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer,
                                n_predictions=n_predictions, gvf_trainer=args.gvf_trainer)

    trainer = PredictionTrainer(args, agent, train_env, test_env, checkpoint_dir, gvf,
                                eval_function=slightly_less_simple_chain_gvf_eval)
    trainer.reset()

    trainer.train()

    info = trainer.get_info()




