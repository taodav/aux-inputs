import numpy as np
import optax
from jax import random

from unc.envs import DynamicChain, SimpleChain
from unc.args import Args
from unc.agents import LSTMAgent
from unc.models import build_network
from unc.utils import Batch
from unc.utils.replay import EpisodeBuffer


if __name__ == "__main__":
    """
    DynamicChain is not well founded given our agent state.
    """
    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 1
    args.trunc = 10
    args.epsilon = 0.
    args.step_size = 0.001
    args.seed = 2020
    args.total_steps = int(1e6)
    args.buffer_size = int(1e5)

    rand_key = random.PRNGKey(args.seed)
    rng = np.random.RandomState(args.seed)
    train_env = SimpleChain(n=args.trunc)

    network = build_network(args.n_hidden, train_env.action_space.n, model_str="lstm")
    optimizer = optax.adam(args.step_size)

    agent = LSTMAgent(network, optimizer, train_env.observation_space.shape[0],
                      train_env.action_space.n, rand_key, args)
    agent.set_eps(args.epsilon)

    buffer = EpisodeBuffer(args.buffer_size, rng, train_env.observation_space.shape,
                           state_size=agent.state_shape)

    print("Starting test for LSTM + Buffer on SimpleChain environment")
    steps = 0
    eps = 0

    while steps < args.total_steps:
        all_current_obs, all_actions, all_rewards, all_dones, all_next_obs = [], [], [], [], []
        all_hidden_states, all_next_hidden_states = [], []

        obs = np.expand_dims(train_env.reset(), 0)
        agent.reset()
        hs = agent.state
        eps_loss = 0

        for t in range(args.trunc):
            action = agent.act(obs)
            next_obs, reward, done, info = train_env.step(action)
            next_obs = np.array([next_obs])
            next_action = action
            steps += 1

            next_hs = agent.state
            b = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                      next_action=next_action, state=hs, next_state=next_hs, end=done or (t == args.trunc - 1))
            buffer.push(b)

            if len(buffer) > args.batch_size:
                sample = buffer.sample(args.batch_size, seq_len=args.trunc)

                sample.gamma = (1 - sample.done) * args.discounting
                eps_loss += agent.update(sample)

            all_current_obs.append(obs)
            all_next_obs.append(next_obs)
            all_rewards.append(np.array([reward]))
            all_actions.append(action)
            all_dones.append(np.array([done]))
            all_hidden_states.append(hs)
            all_next_hidden_states.append(next_hs)

            if done:
                break

            obs = next_obs
            hs = next_hs

        eps += 1

        if eps % 10 == 0:
            batch = Batch(obs=np.concatenate(all_current_obs)[None, :], reward=np.concatenate(all_rewards)[None, :],
                          next_obs=np.concatenate(all_next_obs)[None, :], action=np.concatenate(all_actions)[None, :],
                          done=np.concatenate(all_dones)[None, :], state=np.stack(all_hidden_states)[None, :],
                          next_state=np.stack(all_next_hidden_states)[None, :])
            batch.gamma = (1 - batch.done) * args.discounting
            batch.next_action = batch.action.copy()
            batch.zero_mask = np.ones_like(batch.done)

            lstm_state = agent._rewrap_hidden(batch.state[:, 0])
            hist_q_vals, final_hs = agent.Qs(batch.obs, lstm_state, agent.network_params)
            hist_q_vals = hist_q_vals[0, :, 0]
            actual_vals = args.discounting ** np.arange(hist_q_vals.shape[0] - 1, -1, -1)
            msve = np.mean(0.5 * (hist_q_vals - actual_vals) ** 2)
            print(f"Episode {eps}, "
                  f"Loss {eps_loss/(t + 1):.6f}, "
                  f"MSVE {msve:.6f}, "
                  f"History Q-values: {hist_q_vals}\n")


