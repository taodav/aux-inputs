import numpy as np
import optax
import jax
from jax import random
from pathlib import Path

from unc.envs import get_env, Environment
from unc.args import Args, get_results_fname
from unc.trainers import Trainer, BufferTrainer
from unc.models import build_network
from unc.agents import Agent, DQNAgent
from unc.utils import preprocess_step, get_action_encoding, save_info

from definitions import ROOT_DIR


def collect_observations(agent: Agent, env: Environment, n_episodes: int,
                         max_episode_steps: int, test_eps: float = 0.):

    all_obs = []
    all_rews = []

    for ep in range(n_episodes):
        episode_obs = []
        rews = []
        obs = env.reset()

        # Action conditioning
        if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
            action_encoding = get_action_encoding(agent.features_shape, -1, env.action_space.n)
            obs = np.concatenate([obs, action_encoding], axis=-1)

        episode_obs.append(obs)

        obs = np.array([obs])
        agent.reset()
        action = agent.act(obs).item()

        agent.set_eps(test_eps)

        for t in range(max_episode_steps):

            action = agent.act(obs).item()

            next_obs, reward, done, info = env.step(action)

            # Action conditioning
            if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
                action_encoding = get_action_encoding(agent.features_shape, action, env.action_space.n)
                next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

            episode_obs.append(next_obs)
            rews.append(reward)

            if done:
                break

            obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action)

        rews = np.array(rews)
        all_rews.append(rews)
        all_obs.append(np.stack(episode_obs))

    all_rews = np.stack(all_rews)
    all_obs = np.stack(all_obs)

    return all_obs, all_rews


def init_and_train(args: Args):

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

    test_rng = np.random.RandomState(args.seed + 10)
    test_rand_key = random.PRNGKey(args.seed + 10)
    # TODO: when we do GPU jobs, make sure JAX CuDNN backend has determinism and seeding done

    # Initializing our environment, args we need are filtered out in get_env
    train_env = get_env(rng,
                        rand_key,
                        args)

    test_env = get_env(test_rng,
                       test_rand_key,
                       args)

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    output_size = train_env.action_space.n
    if args.distributional:
        output_size = train_env.action_space.n * args.atoms
    network = build_network(args.n_hidden, output_size, model_str=model_str)
    optimizer = optax.adam(args.step_size)

    agent = DQNAgent(network, optimizer, train_env.observation_space.shape,
                     train_env.action_space.n, rand_key, args)

    # Initialize our trainer
    if args.replay:
        trainer = BufferTrainer(args, agent, train_env, test_env)
    else:
        trainer = Trainer(args, agent, train_env, test_env)

    trainer.reset()

    # Train!
    trainer.train()
    return agent, test_env


if __name__ == "__main__":
    parser = Args()
    obs_args = parser.parse_args()

    obs_args.algo = "sarsa"
    obs_args.arch = "linear"
    obs_args.env = "2"
    obs_args.discounting = 0.9
    obs_args.n_hidden = 5
    obs_args.step_size = 0.001
    obs_args.total_steps = 50000
    obs_args.max_episode_steps = 200
    obs_args.seed = 2022

    obs_agent, obs_test_env = init_and_train(obs_args)

    parser = Args()
    unc_args = parser.parse_args()

    unc_args.algo = "sarsa"
    unc_args.arch = "linear"
    unc_args.env = "2o"
    unc_args.discounting = 0.9
    unc_args.n_hidden = 5
    unc_args.step_size = 0.001
    unc_args.total_steps = 50000
    unc_args.max_episode_steps = 200
    unc_args.seed = 2022

    unc_agent, unc_test_env = init_and_train(unc_args)

    episodes_to_collect = 5

    obs_collected_obs, obs_collected_rew = collect_observations(obs_agent, obs_test_env,
                                                                n_episodes=episodes_to_collect,
                                                                max_episode_steps=obs_args.max_episode_steps,
                                                                test_eps=0.)

    unc_collected_obs, unc_collected_rew = collect_observations(unc_agent, unc_test_env,
                                                                n_episodes=episodes_to_collect,
                                                                max_episode_steps=unc_args.max_episode_steps,
                                                                test_eps=0.)

    results = {
        '2': {
            'args': obs_args.as_dict(),
            'obs': obs_collected_obs,
            'rews': obs_collected_rew
        },
        '2o': {
            'args': unc_args.as_dict(),
            'obs': unc_collected_obs,
            'rews': unc_collected_rew

        }
    }
    results_fname = Path(ROOT_DIR, 'results', 'lobster_data.npy')

    save_info(results_fname, results)
    print(f"Saved observations for test episodes in {results_fname}")

