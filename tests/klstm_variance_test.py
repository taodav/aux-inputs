import numpy as np
import gym
import optax
from jax import random
from typing import Any

from unc.envs import get_env
from unc.args import Args
from unc.agents import kLSTMAgent
from unc.models import build_network
from unc.utils import EpisodeBuffer
from unc.utils import Batch



if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 50
    args.trunc = 10
    args.epsilon = 0.1
    args.seed = 2020
    args.total_steps = int(1e6)
    args.k_rnn_hs = 10
    args.action_cond = "cat"

    # Compass World variables
    args.env = "f"
    args.size = 9

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)
    train_env = get_env(rng, rand_key, env_str=args.env, size=args.size)

    network = build_network(args.n_hidden, train_env.action_space.n, model_str="lstm")
    optimizer = optax.adam(args.step_size)
    value_network = build_network(args.n_hidden, train_env.action_space.n, model_str="nn")
    value_optimizer = optax.adam(args.value_step_size)

    n_features = train_env.observation_space.shape[0]
    if args.action_cond == 'cat':
        n_features += train_env.action_space.n

    agent = kLSTMAgent(network, value_network, optimizer, value_optimizer,
                       n_features,
                       train_env.action_space.n, rand_key, args)
    agent.set_eps(args.epsilon)

    buffer = EpisodeBuffer(args.buffer_size, train_env.rng, (n_features,),
                           state_size=agent.state_shape)

    print("Starting test for variance of kLSTM agent hidden states")
    steps = 0

    eps = 0
    while steps < args.total_steps:

        obs = train_env.reset()
        if args.action_cond == "cat":
            obs = np.concatenate([obs, np.zeros(train_env.action_space.n)])

        agent.reset()
        hs = agent.state
        obs = np.expand_dims(obs, 0)

        action = train_env.action_space.sample()
        agent_action = agent.act(obs)
        for t in range(args.max_episode_steps):
            next_hs = agent.state
            next_obs, reward, done, info = train_env.step(action)

            # Action conditioning
            if args.action_cond == 'cat':
                one_hot_action = np.zeros(train_env.action_space.n)
                one_hot_action[action] = 1
                next_obs = np.concatenate([next_obs, one_hot_action])

            next_obs = np.array([next_obs])

            next_action = train_env.action_space.sample()
            sample = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                           next_action=next_action, state=hs, next_state=next_hs,
                           end=done or (t == args.max_episode_steps - 1))
            buffer.push(sample)
            steps += 1

            next_agent_action = agent.act(next_obs).item()
            if args.batch_size <= len(buffer):

                batch = buffer.sample_k(args.batch_size, seq_len=args.trunc, k=args.k_rnn_hs)
                batch.gamma = (1 - batch.done) * args.discounting
                loss = agent.update(batch)

            if done:
                break

            obs = next_obs
            action = next_action
            agent_action = next_agent_action
            hs = next_hs

        eps += 1


        if steps % 1000 == 0:
            lstm_state = agent._rewrap_hidden(batch.state[:, 0])
            hist_q_vals, final_hs = agent.Qs(batch.obs, lstm_state, agent.network_params)
            hist_q_vals = hist_q_vals[0, :, 0]
            actual_vals = args.discounting ** np.arange(hist_q_vals.shape[0])
            msve = np.mean(0.5 * (hist_q_vals - actual_vals) ** 2)
            print(f"Episode {eps}, "
                  f"Loss {loss:.6f}, "
                  f"MSVE {msve}, "
                  f"History Q-values: {hist_q_vals}\n")


