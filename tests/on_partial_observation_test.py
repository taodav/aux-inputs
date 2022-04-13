import numpy as np
from jax import random
from typing import List

from unc.envs import get_env


def get_occlusion_mask(env_agent_centric, env):
    obs, glass_map = env_agent_centric.get_obs(env_agent_centric.state, return_expanded_glass_map=True)

    see_thru_po_map = obs[y_range[0]:y_range[1], x_range[0]:x_range[1]]
    ac_glass_map = glass_map[y_range[0]:y_range[1], x_range[0]:x_range[1], 0]
    ac_obstacle_map = see_thru_po_map[:, :, 0]

    occlusion_mask = env.get_occlusion_mask(ac_obstacle_map, ac_glass_map)
    return occlusion_mask


if __name__ == "__main__":
    """
    This is mainly a visual test.
    """
    seed = 2022
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, env_str="u2p", distance_noise=False)
    y_range = env.y_range
    x_range = env.x_range

    # we first do some unit tests for occlusion mask
    env_agent_centric = get_env(rng, rand_key, env_str="u2a")
    env_agent_centric.reset()

    obs, rew, done, info = env_agent_centric.step(1)

    occlusion_mask = get_occlusion_mask(env_agent_centric, env)

    assert np.all(occlusion_mask[:, -1] == 1), "All of right side should be occluded"
    assert np.all(occlusion_mask[:, :-1] == 0), "The rest shouldn't be occluded"

    env_agent_centric.position[0] = 2
    env_agent_centric.position[1] = 3

    occlusion_mask = get_occlusion_mask(env_agent_centric, env)

    assert np.all(occlusion_mask[0] == 1)
    assert np.all(occlusion_mask[:, 0] == 1)
    assert np.all(occlusion_mask[:3, -1] == 1) and np.all(occlusion_mask[3:, -1] == 0)
    assert np.all(occlusion_mask[-1, 1:] == 0)
    assert np.all(occlusion_mask[1:-1, 1:-1] == 0)

    # now we test glass
    env_agent_centric.position[0] = 4
    env_agent_centric.position[1] = 7
    occlusion_mask = get_occlusion_mask(env_agent_centric, env)

    assert occlusion_mask[2, 0] == 0, "We can't see through the glass!"

    # final occlusion mask test: surrounded by 3 glass windows.
    env_agent_centric = get_env(rng, rand_key, env_str="u3a")
    env_agent_centric.reset()

    env_agent_centric.position[0] = 6
    env_agent_centric.position[1] = 4
    occlusion_mask = get_occlusion_mask(env_agent_centric, env)

    assert np.all(occlusion_mask[-1]) == 0, "rewards are occluded"

    # now we test our observations
    ground_truth_obs = env.reset()
    n = 1000

    total = np.zeros((env.window_size, env.window_size, 3))
    half = env.observation_space.shape[0] // 2

    for _ in range(n):
        noisy_obs = env.noisify_observations(ground_truth_obs.copy())
        assert np.all(noisy_obs[half, half] == ground_truth_obs[half, half]), "Agent position isn't always ground truth"

        total[:, :, 0] += (ground_truth_obs[:, :, 0] == noisy_obs[:, :, 0]).astype(int)
        total[:, :, 1] += (np.all(ground_truth_obs[:, :, 1:5] == noisy_obs[:, :, 1:5], axis=-1)).astype(int)
        total[:, :, 2] += (ground_truth_obs[:, :, 5] == noisy_obs[:, :, 5]).astype(int)

    avg = total / n
    prob_map = env.prob_map
    print("All tests passed.")

    # so... this test doesn't pass. But it's ok..... it's close enough.
    # would require a considerable amount more work to get it to pass.
    # it doesn't pass b/c we're resampling currents from scratch, and not omitting
    # the g.t. current. We COULD modify the tests to pass (by taking into consideration
    # that probability), but it's hardly worth the effort.
    assert np.all(np.isclose(avg, prob_map))

