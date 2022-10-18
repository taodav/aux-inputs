import numpy as np
import jax
from jax import random
from pathlib import Path

from unc.envs import get_env
from unc.args import Args
from unc.trainers import get_or_load_trainer
from unc.models import build_network
from unc.sampler import Sampler
from unc.agents import get_agent
from unc.utils import save_info, save_video
from unc.utils.files import init_files
from unc.optim import get_optimizer
from unc.eval import test_episodes
from unc.gvfs import get_gvfs
from definitions import ROOT_DIR


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

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
    train_env = get_env(rng, train_env_key, args)
    test_env = get_env(test_rng, test_env_key, args)

    # Getting our pre-filled replay buffer if we need it.
    prefilled_buffer = None
    if args.replay and args.p_prefilled > 0:
        buffer_path = Path(ROOT_DIR, 'data', f'buffer_{args.env}_{args.seed}.pkl')
        replay_dict = Sampler.load(buffer_path)
        prefilled_buffer = replay_dict['buffer']
        rock_positions = replay_dict['rock_positions']
        train_env.rock_positions = rock_positions

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    n_actions = train_env.action_space.n
    features_shape = train_env.observation_space.shape

    if args.distributional:
        n_actions = train_env.action_space.n * args.atoms

    # GVFs for Lobster env.
    gvf, n_predictions = None, 0
    if '2' in args.env and ('g' in args.env or 't' in args.env):
        gvf = get_gvfs(train_env, args.gvf_type, gamma=args.discounting)
        n_predictions = gvf.n

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not (('g' in args.env or 's' in args.env) and model_str == 'linear')
    network = build_network(args.n_hidden, n_actions, model_str=model_str, with_bias=with_bias,
                            init=args.weight_init, layers=args.layers, action_cond=args.action_cond, n_predictions=n_predictions)
    optimizer = get_optimizer(args.optim, args.step_size)


    # Initialize agent
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer,
                                n_predictions=n_predictions, gvf_trainer=args.gvf_trainer)

    # Initialize our trainer
    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env, checkpoint_dir,
                                            prefilled_buffer=prefilled_buffer, gvf=gvf, gvf_trainer=args.gvf_trainer)

    # Train!
    trainer.train()

    # Save results
    info = trainer.get_info()

    # Potentially run a test episode

    if args.view_test_ep:
        imgs, rews = test_episodes(agent, test_env, n_episodes=args.test_episodes,
                                   render=True, test_eps=args.test_eps,
                                   max_episode_steps=args.max_episode_steps)
        vod_path = results_path.parents[0] / f"{results_path.stem}.mp4"

        print(f"Saving render of test episode(s) to {vod_path}")

        # save_gif(imgs, gif_path)
        save_video(imgs, vod_path)

        info['test_rews'] = rews

    if args.save_model:
        model_path = results_path.parents[0] / f"{results_path.stem}.pth"
        print(f"Saving model parameters to {model_path}")

        agent.save(model_path)


    print(f"Saving results to {results_path}")
    save_info(results_path, info)

