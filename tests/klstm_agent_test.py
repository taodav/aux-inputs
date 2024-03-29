import numpy as np
import gym
import optax
from jax import random
from typing import Any

from unc.envs import SimpleChain
from unc.args import Args
from unc.agents import kLSTMAgent
from unc.models import build_network
from unc.trainers import Trainer
from unc.utils import Batch, preprocess_step



if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 1
    args.k_rnn_hs = 5
    args.same_k_rnn_params = True
    args.trunc = 10
    args.epsilon = 0.
    args.step_size = 0.01
    args.value_step_size = 0.025
    args.seed = 2020
    args.total_steps = int(1e6)

    rand_key = random.PRNGKey(args.seed)
    train_env = SimpleChain(n=args.trunc)

    network = build_network(args.n_hidden, train_env.action_space.n, model_str="lstm")
    optimizer = optax.adam(args.step_size)

    n_features, n_actions = train_env.observation_space.shape[0], train_env.action_space.n

    value_network = build_network(args.n_hidden, train_env.action_space.n, model_str="seq_value")
    value_optimizer = optax.adam(args.value_step_size)
    agent = kLSTMAgent(network, value_network, optimizer, value_optimizer,
                       n_features, n_actions, rand_key, args)

    agent.set_eps(args.epsilon)

    print("Starting test for LSTM on SingleChain environment")
    steps = 0

    eps = 0
    while steps < args.total_steps:
        all_current_obs, all_actions, all_rewards, all_dones, all_next_obs = [], [], [], [], []
        all_hidden_states, all_next_hidden_states = [], []

        obs = np.expand_dims(train_env.reset(), 0)
        all_current_obs.append(obs)
        agent.reset()
        all_hidden_states.append(agent.state)

        for t in range(args.trunc):
            action = agent.act(obs).item()
            next_obs, reward, done, info = train_env.step(action)
            next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action)
            steps += 1
            all_next_obs.append(next_obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_next_hidden_states.append(agent.state)
            if done.item():
                break

            obs = next_obs
            all_current_obs.append(obs)
            all_hidden_states.append(agent.state)

        eps += 1
        batch = Batch(obs=np.concatenate(all_current_obs)[None, :], reward=np.concatenate(all_rewards)[None, :],
                      next_obs=np.concatenate(all_next_obs)[None, :], action=np.concatenate(all_actions)[None, :],
                      done=np.concatenate(all_dones)[None, :], state=np.stack(all_hidden_states)[None, :],
                      next_state=np.stack(all_next_hidden_states)[None, :])
        batch.gamma = (1 - batch.done) * args.discounting
        batch.next_action = batch.action.copy()
        batch.zero_mask = np.ones_like(batch.done)

        loss, other = agent.update(batch)

        if steps % 100 == 0:
            lstm_state = agent._rewrap_hidden(batch.state[:, 0])
            hist_q_vals, all_hidden, final_hs = agent.Qs(batch.obs, lstm_state, agent.rnn_network_params, agent.value_network_params)
            hist_q_vals = hist_q_vals[0, :, 0]
            actual_vals = args.discounting ** np.arange(hist_q_vals.shape[0])
            msve = np.mean(0.5 * (hist_q_vals - actual_vals) ** 2)
            print(f"Episode {eps}, "
                  f"Loss {loss:.6f}, "
                  f"MSVE {msve}, "
                  f"History Q-values: {hist_q_vals}\n")


