import numpy as np
from jax import random
from itertools import product
from functools import partial

from unc.envs import get_env
from unc.utils.lobster import all_lobster_states, transition_prob, batch_reward
from unc.args import Args
from unc.agents import Agent


class OptimalLobsterFisher(Agent):
    def __init__(self, q: np.ndarray):
        super(OptimalLobsterFisher, self).__init__()
        self.q = q

    def act(self, state: np.ndarray):
        state_idx = np.nonzero(state)[0].item()
        qs = self.q[state_idx]
        return np.argmax(qs)


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.env = "2p"
    args.n_particles = 100
    args.max_episode_steps = 200
    args.discounting = 0.9
    episodes = 1000

    seed = 2024
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    pf_env = get_env(rng, rand_key, args)

    args.env = "2s"
    state_env = get_env(rng, rand_key, args)

    all_states = all_lobster_states()
    num_states = all_states.shape[0]
    num_actions = state_env.action_space.n
    state_idxes = np.arange(num_states)
    action_idxes = np.arange(num_actions)
    product_sas = np.stack(product(state_idxes, action_idxes, state_idxes))

    state_to_idx = {}

    idx_to_state = np.zeros_like(all_states)
    idx_to_one_hot = np.zeros((all_states.shape[0], num_states))
    for s in all_states:
        one_hot = state_env.get_obs(s)
        idx = np.nonzero(one_hot)[0].item()
        idx_to_state[idx] = s
        idx_to_one_hot[idx] = one_hot
        state_to_idx[str(s)] = idx

    all_repeated_states = idx_to_state[product_sas[:, 0]]
    all_repeated_actions = product_sas[:, 1]
    all_repeated_next_states = idx_to_state[product_sas[:, 2]]

    def values(states: np.ndarray, vs: np.ndarray):
        state_idxes = np.stack([state_to_idx[str(s)] for s in states])
        return vs[state_idxes]

    instance_transition_probs = partial(transition_prob, state_env.traverse_prob, state_env.pmfs_1)
    all_transition_probs = instance_transition_probs(idx_to_state[product_sas[:, 0]], product_sas[:, 1], idx_to_state[product_sas[:, 2]])

    delta = 0
    tol = 1e-10
    v = np.zeros(12)
    iterations = 0

    qs = None
    # VALUE ITERATION
    while True:
        rewards = batch_reward(all_repeated_states, all_repeated_actions)

        # g = r + gamma * v(s')
        g = rewards + args.discounting * values(all_repeated_next_states, v)
        # g * p(s' | s, a)
        new_v_flat = all_transition_probs * g

        # sum over all next states
        new_v = np.sum(new_v_flat.reshape(num_states, num_actions, num_states), axis=-1)

        # now we find the max over actions
        max_new_v = np.max(new_v, axis=-1)

        deltas = np.abs(v - max_new_v)

        v = max_new_v

        delta = deltas.max()

        iterations += 1

        if delta < tol:
            qs = new_v
            break

    print(f"Done with value iteration. Final delta: {delta}")

    agent = OptimalLobsterFisher(qs)
    episode_rewards = []

    for ep in range(episodes):
        returns = 0

        state = state_env.reset()

        for step in range(args.max_episode_steps):
            action = agent.act(state)

            next_state, reward, done, info = state_env.step(action)
            returns += reward

            state = next_state

        episode_rewards.append(returns)
        print(f"Episode: {ep}, return: {returns}")

    episode_rewards = np.array(episode_rewards)
    mean = episode_rewards.mean()
    std_err = episode_rewards.std() / np.sqrt(episode_rewards.shape[0])
    print(f"Finished testing optimal policy on {episodes} episodes. Average return: {mean} ± {std_err}")
