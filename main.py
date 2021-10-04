import numpy as np
import optax
from jax import random

from unc.envs import get_env
from unc.args import Args, get_results_fname
from unc.trainers.trainer import Trainer
from unc.models import build_network
from unc.agents import LearningAgent
from unc.utils import save_info, save_video
from unc.eval import test_episodes


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

    # Some argument post-processing
    results_fname, results_fname_npy = get_results_fname(args)
    args.results_fname = results_fname_npy

    # Seeding
    np.random.seed(args.seed)
    rand_key = random.PRNGKey(args.seed)

    # Initializing our environment
    train_env = get_env(args.seed,
                        env_str=args.env,
                        blur_prob=args.blur_prob,
                        random_start=args.random_start,
                        slip_prob=args.slip_prob,
                        slip_turn=args.slip_turn,
                        size=args.size,
                        n_particles=args.n_particles,
                        update_weight_interval=args.update_weight_interval)

    # Initialize model, optimizer and agent
    network = build_network(args.n_hidden, train_env.action_space.n)
    optimizer = optax.adam(args.step_size)

    agent = LearningAgent(network, optimizer, train_env.observation_space.shape[0], train_env.action_space.n, rand_key,
                          args)

    # Initialize our trainer
    trainer = Trainer(args, agent, train_env)
    trainer.reset()

    # Train!
    trainer.train()

    # Save results
    results_path = args.results_dir / args.results_fname
    info = trainer.get_info()

    # Potentially run a test episode
    if args.view_test_ep:
        imgs, rews = test_episodes(agent, train_env, n_episodes=5,
                                   render=True, test_eps=args.test_eps,
                                   max_episode_steps=args.max_episode_steps)
        vod_path = args.results_dir / f"{results_fname}.mp4"

        print(f"Saving render of test episode(s) to {vod_path}")

        # save_gif(imgs, gif_path)
        save_video(imgs, vod_path)

        info['test_rews'] = rews

    if args.save_model:
        model_path = args.results_dir / f"{results_fname}.pth"
        print(f"Saving model parameters to {model_path}")

        agent.save(model_path)


    print(f"Saving results to {results_path}")
    save_info(results_path, info)

