import numpy as np
import optax
import jax
from jax import random
from pathlib import Path

from unc.envs import get_env
from unc.args import Args, get_results_fname
from unc.trainers import Trainer, BufferTrainer
from unc.models import build_network
from unc.sampler import Sampler
from unc.agents import DQNAgent, NoisyNetAgent, LSTMAgent, kLSTMAgent, DistributionalLSTMAgent
from unc.utils import save_info, save_video
from unc.eval import test_episodes
from definitions import ROOT_DIR


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

    # Some argument post-processing
    results_fname, results_fname_npy = get_results_fname(args)
    args.results_fname = results_fname_npy

    # Setting our platform
    # TODO: GPU determinism?
    jax.config.update('jax_platform_name', args.platform)

    # Seeding
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)
    # TODO: when we do GPU jobs, make sure JAX CuDNN backend has determinism and seeding done

    # Initializing our environment
    train_env = get_env(rng,
                        rand_key,
                        env_str=args.env,
                        blur_prob=args.blur_prob,
                        random_start=args.random_start,
                        slip_prob=args.slip_prob,
                        slip_turn=args.slip_turn,
                        size=args.size,
                        n_particles=args.n_particles,
                        update_weight_interval=args.update_weight_interval,
                        rock_obs_init=args.rock_obs_init,
                        half_efficiency_distance=args.half_efficiency_distance,
                        count_decay=args.count_decay,
                        trace_decay=args.trace_decay,
                        unnormalized_counts=args.unnormalized_counts,
                        po_degree=args.po_degree)

    prefilled_buffer = None
    if args.replay and args.p_prefilled > 0:
        # Getting our pre-filled replay buffer if we need it.
        buffer_path = Path(ROOT_DIR, 'data', f'buffer_{args.env}_{args.seed}.pkl')
        replay_dict = Sampler.load(buffer_path)
        prefilled_buffer = replay_dict['buffer']
        rock_positions = replay_dict['rock_positions']
        train_env.rock_positions = rock_positions

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    output_size = train_env.action_space.n
    if args.distributional:
        output_size = train_env.action_space.n * args.atoms
    network = build_network(args.n_hidden, output_size, model_str=model_str)
    optimizer = optax.adam(args.step_size)


    if args.arch == 'lstm':
        n_features = train_env.observation_space.shape[0]
        n_actions = train_env.action_space.n

        # Currently we only do action conditioning with the LSTM agent.
        if args.action_cond == 'cat':
            n_features += n_actions
        if args.k_rnn_hs > 1:
            # value network takes as input mean + variance of hidden states and cell states.
            value_network = build_network(args.n_hidden, train_env.action_space.n, model_str="seq_value")
            value_optimizer = optax.adam(args.value_step_size)
            agent = kLSTMAgent(network, value_network, optimizer, value_optimizer,
                               n_features, n_actions, rand_key, args)
        elif args.distributional:
            agent = DistributionalLSTMAgent(network, optimizer, n_features,
                                            n_actions, rand_key, args)
        else:
            agent = LSTMAgent(network, optimizer, n_features,
                              n_actions, rand_key, args)
    elif args.arch == 'nn' and args.exploration == 'noisy':
        agent = NoisyNetAgent(network, optimizer, train_env.observation_space.shape,
                              train_env.action_space.n, rand_key, args)
    else:
        agent = DQNAgent(network, optimizer, train_env.observation_space.shape,
                         train_env.action_space.n, rand_key, args)

    # Initialize our trainer
    if args.replay:
        trainer = BufferTrainer(args, agent, train_env, prefilled_buffer=prefilled_buffer)
    else:
        trainer = Trainer(args, agent, train_env)
    trainer.reset()

    # Train!
    trainer.train()

    # Save results
    results_path = args.results_dir / args.results_fname
    info = trainer.get_info()

    # Potentially run a test episode
    if args.view_test_ep:
        imgs, rews = test_episodes(agent, train_env, n_episodes=1,
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

