import numpy as np
import optax
import dill
from jax import random
from pathlib import Path
from typing import Tuple

from unc.envs import get_env, Environment
from unc.args import Args
from unc.agents import LSTMAgent
from unc.models import build_network
from unc.utils import EpisodeBuffer
from unc.utils import Batch
from unc.particle_filter import state_stats


def pf_stats(env: Environment):
    assert hasattr(env, "particles") and hasattr(env, "weights")
    return state_stats(env.particles, env.weights)


if __name__ == "__main__":
    # Log info every 5 episodes
    log_episode_freq = 5
    all_info = {
        'pf_vars': [],
        'values': [],
        'ep_end_timesteps': []
    }
    info_path = Path('lstm_value_variance_results.pkl')

    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 20
    args.trunc = 10
    args.epsilon = 0.1
    args.seed = 2020
    args.total_steps = int(1e6)
    args.action_cond = "cat"
    args.replay = True
    args.step_size = 0.0001

    # Compass World variables
    args.env = "fp"
    args.size = 9

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)
    train_env = get_env(rng, rand_key, env_str=args.env, size=args.size)

    network = build_network(args.n_hidden, train_env.action_space.n, model_str="lstm")
    optimizer = optax.adam(args.step_size)

    n_features, n_actions = train_env.observation_space.shape[0], train_env.action_space.n
    if args.action_cond == 'cat':
        n_features += n_actions

    agent = LSTMAgent(network, optimizer, n_features, n_actions, rand_key, args)
    agent.set_eps(args.epsilon)

    buffer = EpisodeBuffer(args.buffer_size, train_env.rng, (n_features,),
                           state_size=agent.state_shape)

    print("Starting test for variance of value from LSTM agents")
    steps = 0

    eps = 0
    while steps < args.total_steps:

        ep_pf_vars = []
        ep_values = []

        obs = train_env.reset()
        if args.action_cond == "cat":
            obs = np.concatenate([obs, np.zeros(train_env.action_space.n)])

        agent.reset()
        hs = agent.state
        obs = np.expand_dims(obs, 0)


        # action = train_env.action_space.sample()
        # agent_action = agent.act(obs)
        action = agent.act(obs)

        pf_mean, pf_var = pf_stats(train_env)
        curr_q = agent.curr_q
        print(f"Initial stats: "
              f"current Q values: {curr_q}, "
              f"Q value variance: {curr_q[0, 0].var()}, "
              f"particle filter variance {pf_var.mean()}")

        for t in range(args.max_episode_steps):
            next_hs = agent.state
            next_obs, reward, done, info = train_env.step(action)

            # Action conditioning
            if args.action_cond == 'cat':
                one_hot_action = np.zeros(train_env.action_space.n)
                one_hot_action[action] = 1
                next_obs = np.concatenate([next_obs, one_hot_action])

            next_obs = np.array([next_obs])

            # next_action = train_env.action_space.sample()
            # next_agent_action = agent.act(next_obs).item()
            next_action = agent.act(next_obs).item()
            sample = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                           next_action=next_action, state=hs, next_state=next_hs,
                           end=done or (t == args.max_episode_steps - 1))
            buffer.push(sample)
            steps += 1

            if args.batch_size <= len(buffer):

                batch = buffer.sample(args.batch_size, seq_len=args.trunc)
                batch.gamma = (1 - batch.done) * args.discounting
                loss = agent.update(batch)
                loss = 0

            pf_mean, pf_var = pf_stats(train_env)
            curr_q = agent.curr_q
            ep_pf_vars.append(pf_var)
            ep_values.append(curr_q)

            if steps % 10 == 0:

                print(f"Total steps {steps}, "
                      f"Episode {eps}, "
                      f"steps {t}, "
                      f"value variance {curr_q[0, 0].var():.6f}, "
                      f"particle filter variance {pf_var.mean()}")
            if done:
                break

            obs = next_obs
            # action = next_action
            # agent_action = next_agent_action
            action = next_action
            hs = next_hs

        eps += 1
        if eps % log_episode_freq == 0:
            all_info['pf_vars'].append(ep_pf_vars)
            all_info['values'].append(ep_values)
            all_info['ep_end_timesteps'].append(steps)

        print(f"End of episode {eps}")
        print()

    with open(info_path, "wb") as f:
        dill.dump(all_info, f)


