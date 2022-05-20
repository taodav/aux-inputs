import numpy as np
from jax import random
from pathlib import Path

from unc.args import Args, hash_training_args
from unc.envs import get_env
from unc.models import build_network
from unc.optim import get_optimizer
from unc.agents import get_agent
from unc.trainers import get_or_load_trainer
from unc.utils.files import init_files


def get_uf_example_args(seed: int) -> Args:
    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 1
    args.trunc = 10
    args.epsilon = 0.1
    args.step_size = 0.001
    args.seed = seed
    args.replay = True
    args.max_episode_steps = 200
    args.buffer_size = 1000
    args.total_steps = 2000
    args.checkpoint_freq = 500
    args.platform = "cpu"
    args.arch = "nn"
    # we use uf6p here b/c no one uses uf6p for any experiment..
    args.env = "uf6p"
    args.process_args()
    return args


def init_everything(args: Args, rng: np.random.RandomState, rand_key: random.PRNGKey,
                    checkpoint_dir: Path):
    train_env = get_env(rng, rand_key, args)
    test_env = get_env(rng, rand_key, args)

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    output_size = train_env.action_space.n
    if args.distributional:
        output_size = train_env.action_space.n * args.atoms

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not ('g' in args.env and model_str == 'linear')
    network = build_network(args.n_hidden, output_size, model_str=model_str, with_bias=with_bias)
    optimizer = get_optimizer(args.optim, args.step_size)

    # for both lstm and cnn_lstm
    features_shape = train_env.observation_space.shape
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer)

    # Initialize our trainer
    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env, checkpoint_dir)
    return trainer, rand_key


if __name__ == "__main__":
    # FIRST we test hashing arguments, to see if they're working correctly
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    args = get_uf_example_args(seed)
    args_1 = get_uf_example_args(seed)
    args_1.total_steps = 3000

    args_1_hash = hash_training_args(args)
    args_2_hash = hash_training_args(args_1)
    assert args_1_hash == args_2_hash, "Hashes are different"

    # Now we test checkpointing

    # Some filesystem initialization
    results_path, checkpoint_dir = init_files(args)
    for ckpt in checkpoint_dir.iterdir():
        ckpt.unlink(missing_ok=True)

    rand_key, prev_rand_key = random.split(rand_key, 2)
    trainer, rand_key = init_everything(args, rng, prev_rand_key, checkpoint_dir)

    # train for a tiny bit, save a few checkpoints
    trainer.train()

    loaded_trainer, rand_key = init_everything(args_1, rng, rand_key, checkpoint_dir)

    assert trainer.num_steps == loaded_trainer.num_steps

    loaded_trainer.train()

    print("All tests passed")



