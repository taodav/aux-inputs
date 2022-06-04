import numpy as np
from jax import random

from unc.envs import get_env
from unc.envs.lobster import all_lobster_states
from unc.args import Args


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()
    args.env = "2p"
    args.n_particles = 100
    samples = 50000

    seed = 2024
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, args)

    all_states = all_lobster_states()
    all_actions = np.arange(env.action_space.n)
    all_actions_repeat = np.tile(all_actions, all_states.shape[0])
    all_states_repeat = np.repeat(all_states, all_actions.shape[0], axis=0)

    batch_exp_states = np.zeros_like(all_states_repeat)
    for _ in range(samples):
        batch_exp_states += env.batch_transition(all_states_repeat.copy(), all_actions_repeat)
    batch_exp_states = batch_exp_states.astype(float)
    batch_exp_states /= samples

    exp_states = np.zeros_like(all_states_repeat)
    for _ in range(samples):
        for i in range(all_states_repeat.shape[0]):

            transition_state = env.transition(all_states_repeat[i], all_actions_repeat[i])
            exp_states[i] += transition_state
            # assert np.all(transition_state == batch_transition_states[i])

    exp_states = exp_states.astype(float)
    exp_states /= samples
    assert np.all(np.isclose(batch_exp_states, exp_states, atol=1e-2))

    batch_all_obs = env.batch_get_obs(all_states)

    all_obs = []
    for i, s in enumerate(all_states):
        all_obs.append(env.get_obs(s))

    all_obs = np.stack(all_obs)
    assert np.all(all_obs == batch_all_obs)

