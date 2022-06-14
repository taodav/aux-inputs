import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from unc.utils.data import Batch, preprocess_step
from unc.utils.gvfs import get_gvfs
from unc.envs import SimpleChain
from unc.envs.wrappers.simple_chain.gvf_tc import GVFTileCodingWrapper
from unc.args import Args
from unc.agents import GVFAgent
from unc.models import build_network
from unc.optim import get_optimizer


def gvf_td_unit_test(agent: GVFAgent, b: Batch, step: int = -1, total_steps: int = -1):
    state = b.obs
    next_state = b.next_obs
    reward = b.reward
    cumulants, cumulant_termination = b.cumulants, b.cumulant_terminations
    is_ratio = b.impt_sampling_ratio

    cumulants = jnp.concatenate([jnp.expand_dims(reward, -1), cumulants], axis=-1)
    cumulant_termination = jnp.concatenate([jnp.expand_dims(b.gamma, -1), cumulant_termination], axis=-1)

    outputs = agent.network.apply(agent.network_params, state)
    next_outputs = agent.network.apply(agent.network_params, next_state)

    td_err = agent.gvf_td_sarsa_error(outputs[0], action[0], cumulants[0], cumulant_termination[0],
                                      next_outputs[0], next_action[0], is_ratio[0])

    print(f"total_steps: {total_steps}, step: {step}, cumulants: {cumulants}, "
          f"obs: {state}, next_obs: {next_state}, "
          f"gammas: {cumulant_termination}\n"
          f"\toutputs: {outputs}, next_outputs: {next_outputs}, "
          f"td_err: {td_err}")


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    # args.n_hidden = 20
    args.n_hidden = 10
    args.epsilon = 0.
    args.step_size = 0.01
    args.seed = 2021
    args.env = '2g'
    # args.arch = 'linear'
    # args.weight_init = 'zero'
    args.optim = 'sgd'
    args.total_steps = int(1e6)
    args.discounting = 0.9
    chain_length = 10

    rand_key = random.PRNGKey(args.seed)
    # train_env = GVFWrapper(SimpleChain(n=chain_length, reward_in_obs=True))
    train_env = GVFTileCodingWrapper(SimpleChain(n=chain_length, reward_in_obs=True))
    n_actions = train_env.action_space.n

    gvf = get_gvfs(train_env, gamma=args.discounting)
    output_size = n_actions
    n_actions_gvfs = train_env.action_space.n
    output_size += gvf.n
    features_shape = train_env.observation_space.shape

    # network = build_network(args.n_hidden, output_size, n_actions_gvfs=n_actions_gvfs,
    #                         model_str=args.arch, init=args.weight_init, with_bias=args.arch != 'linear')
    network = build_network(args.n_hidden, output_size,
                            model_str=args.arch, init=args.weight_init, with_bias=args.arch != 'linear')
    optimizer = get_optimizer(args.optim, args.step_size)

    # Initialize GVFs if we have any

    gvf_idxes = train_env.gvf_idxes
    agent = GVFAgent(gvf_idxes, network, optimizer, features_shape,
                      train_env.action_space.n, rand_key, args)
    agent.set_eps(args.epsilon)

    actual_vals = args.discounting ** np.arange(9 - 1, -1, -1)

    print("Starting test for GVFAgent on SingleChain environment")
    gvf_predictions = None
    steps = 0

    eps = 0
    while steps < args.total_steps:
        all_current_obs, all_actions, all_rewards, all_dones, all_next_obs = [], [], [], [], []
        all_predictions, all_qs, all_losses = [], [], []
        ep_batches = []
        agent.reset()
        train_env.predictions = agent.current_gvf_predictions[0]

        obs = np.expand_dims(train_env.reset(), 0)
        all_current_obs.append(obs)

        # Set gvf_predictions for our batch
        all_predictions.append(agent.current_gvf_predictions)


        action = agent.act(obs)
        all_qs.append(agent.curr_q)
        # train_env.predictions = agent.current_gvf_predictions[0]
        train_env.predictions = actual_vals[0:1]

        for t in range(9):
            next_obs, reward, done, info = train_env.step(action.item())
            next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

            gamma = (1 - done) * args.discounting
            gvf.gamma = gamma

            next_action = agent.act(next_obs)
            all_qs.append(agent.curr_q)
            all_predictions.append(agent.current_gvf_predictions)

            batch = Batch(obs=obs, action=action, next_obs=next_obs,
                          gamma=gamma, reward=reward, next_action=next_action)

            # GVF predictions and policy
            greedy_action = np.argmax(agent.curr_q, axis=1)
            current_pi = np.zeros(n_actions) + (agent.get_eps() / n_actions)
            current_pi[greedy_action] += (1 - agent.get_eps())
            batch.impt_sampling_ratio = gvf.impt_sampling_ratio(batch.next_obs, current_pi)

            # After acting, agent.current_gvf_predictions is set to the predictions.
            # train_env.predictions = agent.current_gvf_predictions[0]
            if t + 2 < actual_vals.shape[0]:
                # set the ground truth gvf predictions for the next action
                # normally we don't need this.
                # next_gvf_predictions = np.expand_dims(actual_vals[t + 1:t + 2], 0)
                train_env.predictions = actual_vals[t + 1:t + 2]
            else:
                train_env.predictions = np.ones(1)

            batch.cumulants = gvf.cumulant(batch.next_obs)
            batch.cumulant_terminations = gvf.termination(batch.next_obs)

            ep_batches.append(batch)

            loss, other_info = agent.update(batch)
            all_losses.append(loss.item())

            steps += 1
            all_next_obs.append(next_obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            if done.item():
                break

            obs = next_obs
            all_current_obs.append(obs)

        eps += 1

        if steps % 100 == 0 or eps == 1:
            for t, b in enumerate(ep_batches):
                gvf_td_unit_test(agent, b, step=t, total_steps=steps)

            hist_q_vals = np.array([q.item() for q in all_qs[:-1]])
            gvf_vals = np.array([q.item() for q in all_predictions[:-1]])
            msve = np.mean(0.5 * (hist_q_vals - actual_vals) ** 2)
            string_losses = [f'{l:.4f}' for l in all_losses]
            print(f"Episode {eps}, "
                  f"Loss {loss:.6f}, "
                  f"MSVE {msve}, \n"
                  f"Actual vals: {actual_vals}\n"
                  f"History Q-values: {hist_q_vals}\n"
                  f"GVF values: {gvf_vals[1:]}\n"
                  f"Losses: {string_losses}")
