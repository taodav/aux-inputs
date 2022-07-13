import numpy as np
import jax
from jax import random
from typing import Tuple
from pathlib import Path
from time import time, ctime

from unc.envs import get_env
from unc.args import Args, hash_training_args
from unc.trainers import get_or_load_trainer
from unc.utils.gvfs import get_gvfs
from unc.utils import save_info
from unc.models import build_network
from unc.agents import get_agent
from unc.optim import get_optimizer


def init_prediction_files(args: Args) -> Tuple[Path, Path]:
    time_str = ctime(time())
    hashed_args = hash_training_args(args)
    results_fname_npy = f"{hashed_args}_{time_str}.npy"
    results_dir = args.results_dir.parent
    original_results_dir = args.results_dir.name
    prediction_results_dir = results_dir / f"{original_results_dir}_prediction"

    results_path = prediction_results_dir / results_fname_npy

    checkpoints_dir = prediction_results_dir / "checkpoints" / hashed_args

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return results_path, checkpoints_dir


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

    # Seeding
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)

    # Some filesystem initialization
    results_path, checkpoint_dir = init_prediction_files(args)

    # Setting our platform
    # TODO: GPU determinism?
    jax.config.update('jax_platform_name', args.platform)

    test_rng = np.random.RandomState(args.seed + 10)
    test_rand_key = random.PRNGKey(args.seed + 10)

    train_env_key, test_env_key, rand_key = random.split(rand_key, 3)

    # Initializing our environment, args we need are filtered out in get_env
    train_env = get_env(rng, train_env_key, args)
    test_env = get_env(test_rng, test_env_key, args)

    # GVFs for Lobster env.
    gvf = get_gvfs(train_env, gamma=args.discounting)
    n_actions_gvfs = 0
    output_size = gvf.n
    gvf_idxes = train_env.gvf_idxes

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    features_shape = train_env.observation_space.shape

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not (('g' in args.env or 's' in args.env) and model_str == 'linear')
    network = build_network(args.n_hidden, output_size, model_str=model_str, with_bias=with_bias,
                            init=args.weight_init, n_actions_gvfs=n_actions_gvfs)
    optimizer = get_optimizer(args.optim, args.step_size)

    # Initialize agent
    n_actions = 0
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer,
                                gvf_idxes=gvf_idxes)

    # Initialize our trainer
    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env, checkpoint_dir,
                                            gvf=gvf)

    trainer.train()

    # Checkpoint
    trainer.checkpoint()

    print(f"finished training for checkpoint {checkpoint_dir}")

    # Save results
    info = trainer.get_info()
    print(f"Saving results to {results_path}")
    save_info(results_path, info)


