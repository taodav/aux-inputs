import numpy as np
from tqdm import trange
from jax import random

from unc.envs import RockSample, get_env
from unc.utils import euclidian_dist, half_dist_prob


def transition(env: RockSample, state: np.ndarray, action: int) -> np.ndarray:
    position, rock_morality, sampled_rocks, current_rocks_obs = env.unpack_state(state)

    if action > 4:
        # CHECK
        new_rocks_obs = current_rocks_obs.copy()
        rock_idx = action - 5
        dist = euclidian_dist(position, env.rock_positions[rock_idx])
        prob = half_dist_prob(dist, env.half_efficiency_distance)

        # w.p. prob we return correct rock observation.
        # rock_obs = rock_morality[rock_idx]
        # if env.rng.random() > prob:
        #     rock_obs = 1 - rock_obs

        rock_obs = rock_morality[rock_idx]
        choices = np.array([rock_obs, 1 - rock_obs])
        probs = np.array([prob, 1 - prob])
        key, subkey = random.split(env.rand_key)

        rock_obs = random.choice(subkey, choices, (1, ), p=probs)[0]
        new_rocks_obs[rock_idx] = rock_obs
        current_rocks_obs = new_rocks_obs
    elif action == 4:
        # SAMPLING
        ele = (env.rock_positions == position)
        idx = np.nonzero(ele[:, 0] & ele[:, 1])[0]

        if idx.shape[0] > 0:
            # If we're on a rock
            idx = idx[0]
            new_sampled_rocks = sampled_rocks.copy()
            new_rocks_obs = current_rocks_obs.copy()
            new_rock_morality = rock_morality.copy()

            new_sampled_rocks[idx] = 1

            # If this rock was actually good, we sampled it now it turns bad.
            # Elif this rock is bad, we sample a bad rock and return 0
            new_rocks_obs[idx] = 0
            new_rock_morality[idx] = 0

            sampled_rocks = new_sampled_rocks
            current_rocks_obs = new_rocks_obs
            rock_morality = new_rock_morality

        # If we sample a space with no rocks, nothing happens for transition.
    else:
        # MOVING
        new_pos = position + env.direction_mapping[action]
        position = np.maximum(np.minimum(new_pos, env.position_max), env.position_min)

    return env.pack_state(position, rock_morality, sampled_rocks, current_rocks_obs)

if __name__ == "__main__":
    seed = 2021
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, env_str="r", render=False)

    steps = 100

    states = []
    env.reset()
    for i in trange(1, steps + 1):
        state = env.state
        states.append(state)
        action = env.action_space.sample()
        current_rand_key = env.rand_key

        obs, _, done, _ = env.step(action)
        new_state = env.state
        new_rand_key = env.rand_key

        env.rand_key = current_rand_key
        gt_state = transition(env, state, action)
        env.rand_key = new_rand_key

        assert np.all(new_state == gt_state), "Mismatch states"
        if done:
            env.reset()

    print("Individual transitions test passed.")
    state_idxes = rng.choice(np.arange(len(states)), 10)
    states = np.stack(states)[state_idxes]

    for a in trange(env.action_space.n):
        env.reset()
        current_rand_key = env.rand_key

        rand_keys = random.split(env.rand_key, num=len(states) + 1)
        new_rand_key, rand_keys = rand_keys[0], rand_keys[1:]
        actions = np.ones(len(states), dtype=int) * a

        gt_states = []
        for state, action, rand_key in zip(states, actions, rand_keys):
            env.rand_key = rand_key
            gt_state = transition(env, state, action)
            gt_states.append(gt_state)

        env.rand_key = current_rand_key
        batch_states = env.batch_transition(np.array(states), actions)
        gt_states = np.array(gt_states)

        assert np.all(gt_states == batch_states), "mismatch in batch states"

    print("Batch transition tests passed.")

    print("All tests pass.")
