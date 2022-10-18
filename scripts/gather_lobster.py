import numpy as np
import jax
from jax import random
from pathlib import Path

from unc.envs import get_env, Environment
from unc.envs.wrappers.lobster import FixedGVFTileCodingWrapper
from unc.args import Args
from unc.trainers import get_or_load_trainer
from unc.gvfs import get_gvfs
from unc.models import build_network
from unc.agents import Agent, get_agent
from unc.optim import get_optimizer
from unc.utils import preprocess_step, get_action_encoding, save_info
from unc.utils.files import init_files
from definitions import ROOT_DIR


def collect_observations(agent: Agent, env: Environment, n_episodes: int,
                         max_episode_steps: int, test_eps: float = 0.):

    all_obs = []
    all_rews = []
    all_others = []
    all_states = []

    for ep in range(n_episodes):
        episode_obs = []
        rews = []
        other_obs = []
        episode_states = []

        obs = env.reset()
        if isinstance(env, FixedGVFTileCodingWrapper):
            other_obs.append(env.predictions)

        episode_states.append(env.state)

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

            episode_states.append(env.state)
            if isinstance(env, FixedGVFTileCodingWrapper):
                other_obs.append(env.predictions)

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
        if other_obs:
            all_others.append(np.stack(other_obs))
        all_states.append(np.stack(episode_states))

    all_rews = np.stack(all_rews)
    all_obs = np.stack(all_obs)
    if all_others:
        all_others = np.stack(all_others)
    all_states = np.stack(all_states)

    return all_obs, all_rews, all_others, all_states


def init_and_train(args: Args):

    # Some filesystem initialization
    results_path, checkpoint_dir = init_files(args)

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

    train_env_key, test_env_key, rand_key = random.split(rand_key, 3)

    # Initializing our environment, args we need are filtered out in get_env
    train_env = get_env(rng, train_env_key, args)
    test_env = get_env(test_rng, test_env_key, args)

    # Initialize model, optimizer and agent
    model_str = args.arch
    if model_str == 'nn' and args.exploration == 'noisy':
        model_str = args.exploration
    output_size = train_env.action_space.n
    features_shape = train_env.observation_space.shape

    # GVFs for Lobster env.
    n_actions_gvfs = None
    gvf, gvf_idxes = None, None
    if '2' in args.env and ('g' in args.env or 't' in args.env):
        gvf = get_gvfs(train_env, gamma=args.discounting)
        n_actions_gvfs = train_env.action_space.n
        output_size += gvf.n
        gvf_idxes = train_env.gvf_idxes

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not (('g' in args.env or 's' in args.env) and model_str == 'linear')
    network = build_network(args.n_hidden, output_size, model_str=model_str, with_bias=with_bias,
                            init=args.weight_init, n_actions_gvfs=n_actions_gvfs)
    optimizer = get_optimizer(args.optim, args.step_size)


    # Initialize agent
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer,
                                gvf_idxes=gvf_idxes)

    # Initialize our trainer
    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env, checkpoint_dir, gvf=gvf)

    # Train!
    trainer.train()

    trainer.checkpoint()

    return trainer.agent, trainer.test_env



if __name__ == "__main__":
    discounting = 0.9
    step_size = 5e-4
    total_steps = 1000000
    max_episode_steps = 200

    parser = Args()

    gt_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2g',
        '--discounting', discounting,
        '--step_size', step_size,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2023,
        '--epsilon', 0.5
    ]
    gt_str_args = [str(s) for s in gt_str_args]
    gt_args = parser.parse_args(gt_str_args)

    # print(f"Training 2g agent")
    # gt_agent, gt_test_env = init_and_train(gt_args)

    parser = Args()
    obs_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2',
        '--discounting', discounting,
        '--step_size', step_size,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2022,
        '--epsilon', 0.5
    ]
    obs_str_args = [str(s) for s in obs_str_args]
    obs_args = parser.parse_args(obs_str_args)

    print(f"Training 2 agent")
    obs_agent, obs_test_env = init_and_train(obs_args)

    parser = Args()
    unc_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2o',
        '--discounting', discounting,
        '--step_size', step_size,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2022,
        '--epsilon', 0.5
    ]
    unc_str_args = [str(s) for s in unc_str_args]
    unc_args = parser.parse_args(unc_str_args)

    print(f"Training 2o agent")
    unc_agent, unc_test_env = init_and_train(unc_args)

    parser = Args()
    pb_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2pb',
        '--discounting', discounting,
        '--step_size', 0.0001,
        '--n_particles', 100,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2022,
        '--epsilon', 0.5
    ]
    pb_str_args = [str(s) for s in pb_str_args]
    pb_args = parser.parse_args(pb_str_args)

    print(f"Training 2pb agent")
    pb_agent, pb_test_env = init_and_train(pb_args)

    parser = Args()
    gvf_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2f',
        '--discounting', discounting,
        '--step_size', 0.001,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--fixed_gvf_path', "/Users/ruoyutao/Documents/uncertainty/results/2t_nn_prediction/checkpoints/2_layer_fd72f6afb33bb0c574cd3d55e94d9d12/500000.pkl",
        '--seed', 2060,
        '--epsilon', 0.1
    ]
    gvf_str_args = [str(s) for s in gvf_str_args]
    gvf_args = parser.parse_args(gvf_str_args)

    print(f"Training 2f agent")
    # gvf_agent, gvf_test_env = init_and_train(gvf_args)

    parser = Args()
    predict_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2e',
        '--discounting', discounting,
        '--step_size', 0.001,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2060,
        '--epsilon', 0.5
    ]
    predict_str_args = [str(s) for s in predict_str_args]
    predict_args = parser.parse_args(predict_str_args)

    print(f"Training 2e agent")

    predict_agent, predict_test_env = init_and_train(predict_args)

    episodes_to_collect = 500

    obs_collected_obs, obs_collected_rew, _, obs_all_states = collect_observations(obs_agent, obs_test_env,
                                                                n_episodes=episodes_to_collect,
                                                                max_episode_steps=obs_args.max_episode_steps,
                                                                test_eps=0.)

    unc_collected_obs, unc_collected_rew, _, unc_all_states = collect_observations(unc_agent, unc_test_env,
                                                                n_episodes=episodes_to_collect,
                                                                max_episode_steps=unc_args.max_episode_steps,
                                                                test_eps=0.)

    pb_collected_obs, pb_collected_rew, _, pb_all_states = collect_observations(pb_agent, pb_test_env,
                                                                n_episodes=episodes_to_collect,
                                                                max_episode_steps=pb_args.max_episode_steps,
                                                                test_eps=0.)

    predict_collected_obs, predict_collected_rew, _, predict_all_states = collect_observations(predict_agent, predict_test_env,
                                                                                n_episodes=episodes_to_collect,
                                                                                max_episode_steps=predict_args.max_episode_steps,
                                                                                test_eps=0.)

    # gvf_collected_obs, gvf_collected_rew, gvf_collected_predictions, gvf_all_states = collect_observations(
    #     gvf_agent, gvf_test_env,
    #                                                           n_episodes=episodes_to_collect,
    #                                                           max_episode_steps=gvf_args.max_episode_steps,
    #                                                           test_eps=0.5)

    results = {
        '2': {
            'args': obs_args.as_dict(),
            'obs': obs_collected_obs,
            'rews': obs_collected_rew,
            'states': obs_all_states
        },
        '2o': {
            'args': unc_args.as_dict(),
            'obs': unc_collected_obs,
            'rews': unc_collected_rew,
            'states': unc_all_states

        },
        '2pb': {
            'args': pb_args.as_dict(),
            'obs': pb_collected_obs,
            'rews': pb_collected_rew,
            'states': pb_all_states
        },
        '2e': {
            'args': predict_args.as_dict(),
            'obs': predict_collected_obs,
            'rews': predict_collected_rew,
            'states': predict_all_states
        },
        # '2f': {
        #     'args': gvf_args.as_dict(),
        #     'obs': gvf_collected_obs,
        #     'rews': gvf_collected_rew,
        #     'states': gvf_all_states,
        #     'predictions': gvf_collected_predictions
        # }
    }
    results_fname = Path(ROOT_DIR, 'results', 'lobster_data.npy')
    gt_agent_fname = Path(ROOT_DIR, 'results', f'2g_{gt_args.arch}_agent.pth')
    obs_agent_fname = Path(ROOT_DIR, 'results', f'2_{obs_args.arch}_agent.pth')
    unc_agent_fname = Path(ROOT_DIR, 'results', f'2o_{unc_args.arch}_agent.pth')
    pb_agent_fname = Path(ROOT_DIR, 'results', f'2pb_{pb_args.arch}_agent.pth')
    predict_agent_fname = Path(ROOT_DIR, 'results', f'2e_{predict_args.arch}_agent.pth')
    gvf_agent_fname = Path(ROOT_DIR, 'results', f'2f_{gvf_args.arch}_agent.pth')

    save_info(results_fname, results)
    print(f"Saved observations for test episodes in {results_fname}")

    # gt_agent.save(gt_agent_fname)
    # print(f"Saved 2g agent to {gt_agent_fname}")

    # obs_agent.save(obs_agent_fname)
    # print(f"Saved 2 agent to {obs_agent_fname}")

    # unc_agent.save(unc_agent_fname)
    # print(f"Saved 2o agent to {unc_agent_fname}")

    pb_agent.save(pb_agent_fname)
    print(f"Saved 2pb agent to {pb_agent_fname}")

    predict_agent.save(predict_agent_fname)
    print(f"Saved 2e agent to {predict_agent_fname}")

    # gvf_agent.save(gvf_agent_fname)
    # print(f"Saved 2f agent to {gvf_agent_fname}")

