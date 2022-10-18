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
from unc.utils.files import init_files, load_checkpoint


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
    step_size = 1e-3
    total_steps = 500000
    max_episode_steps = 200
    gather_n_episodes = 30

    results_fname = Path(ROOT_DIR, 'results', 'lobster_diff_obs_other_behavior.npy')

    # get gt gvf predictions
    gvf_predictions_checkpoints = Path(ROOT_DIR, 'results', '2t_nn_prediction', 'checkpoints')

    # we just grab the first one.
    gvf_predictions_fname = list(list(gvf_predictions_checkpoints.iterdir())[0].iterdir())[0]
    gvf_prediction_trainer = load_checkpoint(gvf_predictions_fname)
    gvf_prediction_agent = gvf_prediction_trainer.agent

    parser = Args()
    gt_args = parser.parse_args()

    # Seeding
    np.random.seed(gt_args.seed)
    rng = np.random.RandomState(gt_args.seed)
    rand_key = random.PRNGKey(gt_args.seed)

    parser = Args()
    trace_args = parser.parse_args()
    trace_args.env = "2o"
    trace_args.algo = "sarsa"
    trace_args.arch = "linear"
    trace_args.discounting = discounting
    trace_args.step_size = step_size
    trace_args.total_steps = total_steps
    trace_args.max_episode_steps = max_episode_steps
    trace_args.seed = 2023
    print(f"Training trace agent")
    trace_agent, trace_env = init_and_train(trace_args)

    parser = Args()
    pf_args = parser.parse_args()
    pf_args.env = "2pb"
    pf_args.algo = "sarsa"
    pf_args.arch = "linear"
    pf_args.n_particles = 100
    pf_args.discounting = discounting
    pf_args.step_size = step_size
    pf_args.total_steps = total_steps
    pf_args.max_episode_steps = max_episode_steps
    pf_args.seed = 2023
    print(f"Training trace agent")
    pf_agent, pf_env = init_and_train(pf_args)

    parser = Args()
    gvf_args = parser.parse_args()
    gvf_args.env = "2t"
    gvf_args.algo = "sarsa"
    gvf_args.arch = "nn"
    gvf_args.n_hidden = 20
    gvf_args.discounting = discounting
    gvf_args.step_size = 1e-4
    gvf_args.total_steps = total_steps
    gvf_args.max_episode_steps = max_episode_steps
    gvf_args.seed = 2023

    print(f"Training 2t agent")
    gvf_agent, gvf_env = init_and_train(gvf_args)

    parser = Args()
    gvf_non_t_args = parser.parse_args()
    gvf_non_t_args.env = "2g"
    gvf_non_t_env = get_env(rng, rand_key, gvf_non_t_args)

    gt_args.env = "2e"
    gt_args.seed = 2023
    gt_trace_env = get_env(rng, rand_key, gt_args)
    gt_pf_env = get_env(rng, rand_key, gt_args)
    gt_gvf_env = get_env(rng, rand_key, gt_args)

    all_obs = {
        '2e_2o': [],
        '2e_2pb': [],
        '2e_2g': [],
        '2o': [],
        '2pb': [],
        '2g': [],
        '2t': [],
        '2t_prediction': [],
        '2t_state': [],
    }

    # now we roll out gather_n_episodes episodes based on our gt agent
    for ep in range(gather_n_episodes):
        all_eps_obs = {
            '2e_2o': [],
            '2e_2pb': [],
            '2e_2g': [],
            '2o': [],
            '2pb': [],
            '2g': [],
            '2t': [],
            '2t_prediction': [],
            '2t_state': [],
        }
        gvf_agent.reset()
        gvf_prediction_agent.reset()
        gvf_env.predictions = gvf_agent.current_gvf_predictions[0]
        gvf_non_t_env.predictions = gvf_agent.current_gvf_predictions[0]

        # This is okay for now, since we don't have a stochastic start
        gt_trace_obs = gt_trace_env.reset()
        gt_pf_obs = gt_pf_env.reset()
        gt_gvf_obs = gt_gvf_env.reset()
        trace_obs = np.expand_dims(trace_env.reset(), 0)
        pf_obs = np.expand_dims(pf_env.reset(), 0)
        gvf_obs = np.expand_dims(gvf_env.reset(), 0)

        # all_eps_obs['2e_2o'].append(gt_trace_obs)
        # all_eps_obs['2e_2pb'].append(gt_pf_obs)
        # all_eps_obs['2e_2g'].append(gt_gvf_obs)
        # all_eps_obs['2o'].append(trace_obs[0])
        # all_eps_obs['2pb'].append(pf_obs[0])
        # all_eps_obs['2g'].append(gvf_obs[0])
        all_eps_obs['2t_state'].append(gvf_env.state)

        for t in range(max_episode_steps):
            trace_agent.set_eps(trace_args.epsilon)
            pf_agent.set_eps(pf_args.epsilon)
            gvf_agent.set_eps(gvf_args.epsilon)

            trace_action = trace_agent.act(trace_obs)
            pf_action = pf_agent.act(pf_obs)
            gvf_action = gvf_agent.act(gvf_obs)
            gvf_prediction_agent.act(gvf_obs)
            gvf_non_t_env.predictions = gvf_agent.current_gvf_predictions[0]
            gvf_env.predictions = gvf_agent.current_gvf_predictions[0]

            all_eps_obs['2t_prediction'].append(gvf_prediction_agent.current_gvf_predictions[0])

            # append our gt predictions
            gt_trace_env.state = trace_env.state
            u_trace_obs = gt_trace_env.unwrapped.get_obs(gt_trace_env.state)
            gt_trace_env._tick(u_trace_obs)

            gt_pf_env.state = pf_env.state
            u_pf_obs = gt_pf_env.unwrapped.get_obs(gt_pf_env.state)
            gt_pf_env._tick(u_pf_obs)

            gt_gvf_env.state = gvf_env.state
            u_gvf_obs = gt_gvf_env.unwrapped.get_obs(gt_gvf_env.state)
            gt_gvf_env._tick(u_gvf_obs)

            all_eps_obs['2e_2o'].append(gt_trace_env.get_obs(gt_trace_env.state))
            all_eps_obs['2e_2pb'].append(gt_pf_env.get_obs(gt_pf_env.state))
            all_eps_obs['2e_2g'].append(gt_gvf_env.get_obs(gt_gvf_env.state))

            # append our trace observations
            all_eps_obs['2o'].append(trace_obs[0])

            # append our pf observations
            all_eps_obs['2pb'].append(pf_obs[0])

            # Deal with our GVFs.
            all_eps_obs['2g'].append(gvf_non_t_env.get_obs(gvf_env.state))
            all_eps_obs['2t'].append(gvf_obs[0])
            all_eps_obs['2t_state'].append(gvf_env.state)

            next_trace_obs, trace_reward, done, info = trace_env.step(trace_action.item())
            next_pf_obs, pf_reward, done, info = pf_env.step(pf_action.item())
            # next_gvf_obs, gvf_reward, done, info = gvf_env.step(gvf_action.item())
            next_gvf_obs, gvf_reward, done, info = gvf_env.step(0)
            gvf_non_t_env.state = gvf_env.state

            trace_obs = np.expand_dims(next_trace_obs, 0)
            pf_obs = np.expand_dims(next_pf_obs, 0)
            gvf_obs = np.expand_dims(next_gvf_obs, 0)

        for k in all_obs.keys():
            all_obs[k].append(np.stack(all_eps_obs[k]))
        print(f"Finished episode {ep + 1}")


    for k in all_obs.keys():
        all_obs[k] = np.stack(all_obs[k])


    save_info(results_fname, all_obs)
    print(f"Saved all_obs to {results_fname}")




