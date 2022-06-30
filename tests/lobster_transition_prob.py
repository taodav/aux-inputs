import numpy as np
from jax import random

from unc.envs import get_env
from unc.utils.lobster import all_lobster_states, transition_prob, batch_reward
from unc.args import Args


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.env = "2p"
    args.n_particles = 100
    # outer_samples = 100
    # samples = 50000
    indv_samples = 10000

    seed = 2024
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    pf_env = get_env(rng, rand_key, args)

    args.env = "2s"
    state_env = get_env(rng, rand_key, args)

    all_states = all_lobster_states()
    num_states = all_states.shape[0]
    product_sas = np.array(np.meshgrid(np.arange(num_states), np.arange(state_env.action_space.n), np.arange(num_states))).T.reshape(-1, 3)

    state_to_idx = {}

    idx_to_state = np.zeros_like(all_states)
    idx_to_one_hot = np.zeros((all_states.shape[0], num_states))
    for s in all_states:
        one_hot = state_env.get_obs(s)
        idx = np.nonzero(one_hot)[0].item()
        idx_to_state[idx] = s
        idx_to_one_hot[idx] = one_hot

    all_repeated_states = idx_to_state[product_sas[:, 0]]
    all_repeated_actions = product_sas[:, 1]
    all_repeated_next_states = idx_to_state[product_sas[:, 2]]
    all_transition_probs = transition_prob(state_env.traverse_prob, state_env.pmfs_1,
                                           all_repeated_states, all_repeated_actions, all_repeated_next_states)

    all_rewards = batch_reward(all_repeated_states, all_repeated_actions)
    for s, a, r in zip(all_repeated_states, all_repeated_actions, all_rewards):
        non_batch_r = state_env.get_reward(s, a)
        assert r == non_batch_r, "rewards don't equal!"

    print("Batch reward tests passed.")

    for s, a, ns, tp in zip(all_repeated_states, all_repeated_actions, all_repeated_next_states, all_transition_probs):
        match_count = 0
        for n in range(indv_samples):
            sampled_ns = state_env.transition(s, a)

            if n > 9998 and abs((match_count / n) - tp) > 0.01:
            # if tp == 1. and not np.all(ns == sampled_ns):
                ntp = transition_prob(state_env.traverse_prob, state_env.pmfs_1,
                                s[None, :], np.array([a]), ns[None, :])
                print("hi")

            if np.all(ns == sampled_ns):
                match_count += 1
        match_ratio = match_count / indv_samples
        atol = abs(match_ratio - tp)
        assert atol < 0.01


    # all_states = all_lobster_states()
    # all_actions = np.arange(pf_env.action_space.n)
    # all_actions_repeat = np.tile(all_actions, all_states.shape[0])
    # all_states_repeat = np.repeat(all_states, all_actions.shape[0], axis=0)
    #
    # for s in range(outer_samples):
    #     all_next_states = pf_env.batch_transition(all_states_repeat.copy(), all_actions_repeat)
    #     all_transition_probs = transition_prob(pf_env.traverse_prob, pf_env.pmfs_1,
    #                                            all_states_repeat, all_actions_repeat, all_next_states)
    #
    #     # we test if these transition probs are correct.
    #     count_matches = np.zeros_like(all_transition_probs).astype(int)
    #     for _ in range(samples):
    #         all_sample_next_states = pf_env.batch_transition(all_states_repeat.copy(), all_actions_repeat)
    #         matches = np.all(all_next_states == all_sample_next_states, axis=-1)
    #         count_matches += matches.astype(int)
    #
    #     ratios = count_matches / samples
    #
    #     assert np.all(np.isclose(all_transition_probs, ratios, atol=1e-1)), "transition probabilities are off"
    #
    # print("All tests passed.")
