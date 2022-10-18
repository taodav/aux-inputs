import numpy as np
import jax
from jax import random
from pathlib import Path

from definitions import ROOT_DIR
from unc.args import Args
from unc.envs import get_env
from unc.trainers import get_or_load_trainer
from unc.gvfs import get_gvfs
from unc.models import build_network
from unc.optim import get_optimizer
from unc.agents import get_agent
from unc.utils import save_info
from unc.utils.files import init_files


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
    gvf, n_predictions = None, 0
    if '2' in args.env and ('g' in args.env or 't' in args.env):
        gvf = get_gvfs(train_env, gamma=args.discounting)
        n_predictions = gvf.n

    # we don't use a bias unit if we're using ground-truth states
    with_bias = not (('g' in args.env or 's' in args.env) and model_str == 'linear')
    network = build_network(args.n_hidden, output_size, model_str=model_str, with_bias=with_bias,
                            init=args.weight_init, layers=args.layers, action_cond=args.action_cond, n_predictions=n_predictions)
    optimizer = get_optimizer(args.optim, args.step_size)


    # Initialize agent
    n_actions = train_env.action_space.n
    agent, rand_key = get_agent(args, features_shape, n_actions, rand_key, network, optimizer,
                                n_predictions=n_predictions, gvf_trainer=args.gvf_trainer)

    # Initialize our trainer
    trainer, rand_key = get_or_load_trainer(args, rand_key, agent, train_env, test_env, checkpoint_dir,
                                            gvf=gvf, gvf_trainer=args.gvf_trainer)

    # Train!
    trainer.train()

    trainer.checkpoint()

    return trainer.agent, trainer.test_env


if __name__ == "__main__":
    discounting = 0.9
    step_size = 1e-3
    total_steps = 500000
    max_episode_steps = 200
    gather_n_episodes = 30

    results_fname = Path(ROOT_DIR, 'results', 'lobster_diff_obs.npy')

    parser = Args()
    gt_args = parser.parse_args()

    # Seeding
    np.random.seed(gt_args.seed)
    rng = np.random.RandomState(gt_args.seed)
    rand_key = random.PRNGKey(gt_args.seed)

    parser = Args()
    gt_str_args = [
        '--algo', 'sarsa',
        '--arch', 'linear',
        '--env', '2e',
        '--discounting', discounting,
        '--step_size', step_size,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--seed', 2060,
        '--epsilon', 0.5
    ]
    gt_str_args = [str(s) for s in gt_str_args]
    gt_args = parser.parse_args(gt_str_args)

    print(f"Training 2e agent")
    gt_agent, gt_env = init_and_train(gt_args)

    parser = Args()
    gvf_str_args = [
        '--algo', 'sarsa',
        '--arch', 'nn',
        '--env', '2',
        '--discounting', discounting,
        '--step_size', 0.00001,
        '--total_steps', total_steps,
        '--max_episode_steps', max_episode_steps,
        '--offline_eval_freq', 1000,
        '--action_cond', 'mult',
        '--gvf_trainer', 'prediction',
        '--seed', 2060,
    ]
    gvf_str_args = [str(s) for s in gvf_str_args]
    gvf_args = parser.parse_args(gvf_str_args)


    parser = Args()
    trace_args = parser.parse_args()
    trace_args.env = "2o"
    trace_env = get_env(rng, rand_key, trace_args)

    parser = Args()
    pf_args = parser.parse_args()
    pf_args.env = "2pb"
    pf_args.n_particles = 100
    pf_env = get_env(rng, rand_key, pf_args)

    all_obs = {
        '2e': [],
        '2o': [],
        '2pb': [],
    }

    # now we roll out gather_n_episodes episodes based on our gt agent
    for ep in range(gather_n_episodes):
        all_eps_obs = {
            '2e': [],
            '2o': [],
            '2pb': [],
        }

        # This is okay for now, since we don't have a stochastic start
        obs = gt_env.reset()

        all_eps_obs['2e'].append(obs)
        all_eps_obs['2o'].append(trace_env.reset())
        all_eps_obs['2pb'].append(pf_env.reset())

        obs = np.expand_dims(obs, 0)
        gvf_obs = np.expand_dims(gvf_obs, 0)

        for t in range(max_episode_steps):
            gt_agent.set_eps(gt_args.epsilon)

            action = gt_agent.act(obs)

            next_obs, reward, done, info = gt_env.step(action.item())

            # Now we have to "step" for the rest of the environments.
            # This also entails updating internal state of some of the wrappers.
            trace_env.state = gt_env.state
            pf_env.state = gt_env.state

            # get normal observations
            unwrapped_obs = gt_env.unwrapped.get_obs(gt_env.state)

            # append our gt predictions
            all_eps_obs['2e'].append(next_obs)

            # append our trace observations
            trace_env._update_trace(unwrapped_obs)
            all_eps_obs['2o'].append(trace_env.get_obs(trace_env.state))

            # append our pf observations
            pf_env.env._update_particles_weights(unwrapped_obs, action.item())
            all_eps_obs['2pb'].append(pf_env.get_obs(pf_env.state))

            obs = np.expand_dims(next_obs, 0)

        for k in all_obs.keys():
            all_obs[k].append(np.stack(all_eps_obs[k]))
        print(f"Finished episode {ep + 1}")


    for k in all_obs.keys():
        all_obs[k] = np.stack(all_obs[k])


    save_info(results_fname, all_obs)
    print(f"Saved all_obs to {results_fname}")




