import numpy as np
import optax
from jax import random

from unc.utils.data import Batch, preprocess_step
from unc.utils.gvfs import get_gvfs
from unc.envs import SimpleChain
from unc.args import Args
from unc.agents import GVFAgent
from unc.models import build_network



if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.n_hidden = 2
    args.epsilon = 0.
    args.step_size = 0.0001
    args.seed = 2020
    args.arch = 'linear'
    # args.weight_init = 'zero'
    args.total_steps = int(1e6)
    args.gvf_features = 1
    args.discounting = 0.9
    chain_length = 10

    rand_key = random.PRNGKey(args.seed)
    train_env = SimpleChain(n=chain_length, reward_in_obs=True)
    n_actions = train_env.action_space.n

    output_size = n_actions
    output_size += args.gvf_features
    features_shape = train_env.observation_space.shape
    features_shape = (features_shape[0] + args.gvf_features,)

    network = build_network(args.n_hidden, output_size, model_str=args.arch, init=args.weight_init)
    optimizer = optax.adam(args.step_size)

    # Initialize GVFs if we have any
    gvf = None
    if args.gvf_features > 0:
        gvf = get_gvfs(train_env, gamma=args.discounting)

    agent = GVFAgent(gvf, network, optimizer, features_shape,
                      train_env.action_space.n, rand_key, args)
    agent.set_eps(args.epsilon)

    print("Starting test for GVFAgent on SingleChain environment")
    gvf_predictions = None
    steps = 0

    eps = 0
    while steps < args.total_steps:
        all_current_obs, all_actions, all_rewards, all_dones, all_next_obs = [], [], [], [], []
        all_predictions, all_qs = [], []
        agent.reset()

        obs = np.expand_dims(train_env.reset(), 0)
        all_current_obs.append(obs)
        gvf_predictions = agent.current_gvf_predictions
        all_predictions.append(gvf_predictions)

        action = agent.act(obs)

        for t in range(10):
            # GVF predictions and policy
            qs = agent.curr_q
            all_qs.append(qs)

            greedy_action = np.argmax(agent.curr_q, axis=1)
            current_pi = np.zeros(n_actions) + (agent.get_eps() / n_actions)
            current_pi[greedy_action] += (1 - agent.get_eps())
            next_gvf_predictions = agent.current_gvf_predictions
            all_predictions.append(next_gvf_predictions)

            next_obs, reward, done, info = train_env.step(action.item())
            next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

            gamma = (1 - done) * args.discounting

            next_action = agent.act(next_obs)
            batch = Batch(obs=obs, action=action, next_obs=next_obs,
                          gamma=gamma, reward=reward, next_action=next_action,
                          predictions=gvf_predictions, next_predictions=next_gvf_predictions,
                          policy=current_pi)

            loss, other_info = agent.update(batch)

            steps += 1
            all_next_obs.append(next_obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            if done.item():
                break

            obs = next_obs
            all_current_obs.append(obs)
            gvf_predictions = next_gvf_predictions

        eps += 1

        # TODO: DO THIS
        if steps % 100 == 0:
            hist_q_vals = np.array([q.item() for q in all_qs])
            gvf_vals = np.array([q.item() for q in all_predictions])
            actual_vals = args.discounting ** np.arange(hist_q_vals.shape[0])
            msve = np.mean(0.5 * (hist_q_vals - actual_vals) ** 2)
            print(f"Episode {eps}, "
                  f"Loss {loss:.6f}, "
                  f"MSVE {msve}, \n"
                  f"History Q-values: {hist_q_vals}\n"
                  f"GVF values: {gvf_vals[1:]}\n")
