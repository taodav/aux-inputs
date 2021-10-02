import torch
import numpy as np

from unc.envs import get_env
from unc.args import Args, get_results_fname
from unc.trainers import DoubleBufferTrainer
from unc.models import QNetwork
from unc.agents import get_agent
from unc.utils import save_info, save_video
from unc.sampler import Sampler
from unc.eval import test_episodes

from definitions import ROOT_DIR
from pathlib import Path


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    buffer_path = Path(ROOT_DIR, 'data', f'buffer_{args.env}_2021.pkl')

    # Some argument post-processing
    results_fname, results_fname_npy = get_results_fname(args)
    args.results_fname = results_fname_npy

    # Seeding
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    replay_dict = Sampler.load(buffer_path)

    rock_positions = replay_dict['rock_positions']
    prefilled_buffer = replay_dict['buffer']

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

    # Here we have to manually set the same rock positions
    train_env.rock_positions = rock_positions

    # Initialize model, optimizer and agent
    model = QNetwork(train_env.observation_space.shape[0], args.n_hidden, train_env.action_space.n).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.step_size)
    agent_class = get_agent(args.algo)
    agent = agent_class(model, optimizer, train_env.action_space.n, rng,
                        args)

    # Initialize our trainer
    trainer = DoubleBufferTrainer(args, agent, train_env, prefilled_buffer)
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

