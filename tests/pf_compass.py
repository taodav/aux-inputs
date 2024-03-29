"""
Particle filter based on 'A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking' by Arulampalam et al.
Word of warning - there are a few errors and typos in the tutorial.
"""
import numpy as np
from typing import Tuple

from unc.envs import CompassWorld, get_env
from unc.envs.wrappers import BlurryWrapper
from unc.particle_filter import step, state_stats, resample


def effective_sample_size(weights: np.ndarray) -> float:
    squared_sum = np.sum(weights ** 2)
    ess = 1 / squared_sum
    assert not np.isnan(ess), "NaN effective sample size."
    return ess


def pidxes(state: np.ndarray, particles: np.ndarray) -> list:
    idxes = []
    for i, p in enumerate(particles):
        if (p == state).all():
            idxes.append(i)

    return idxes


def run_pf_random_policy(env: CompassWorld,
                         steps: int = 10000,
                         log_interval: int = 1000,
                         weight_update_interval: int = 1,
                         resample_every: int = None,
                         n_particles: int = -1) -> Tuple[int, int]:
    """
    Run a particle filter on a random policy in Compass World.
    :param steps: Maximum steps to take
    :param log_interval: After each interval of log_interval, we log stats.
    :return: Number of steps taken before effective sample size == 1.
    """

    obs = env.reset()
    if n_particles == -1:
        particles = env.sample_all_states()
    else:
        particles = env.sample_states(n=n_particles)
    # particles = np.array([env.state])
    weights = np.ones(len(particles)) / len(particles)
    action = None
    esses = []
    total_steps = 0
    update_steps = 0
    particle_to_follow = pidxes(env.state, particles)[0]
    resample_every = resample_every if resample_every is not None else float('inf')

    for s in range(1, steps + 1):
        update_weights = s % weight_update_interval == 0
        weights, particles = step(weights, particles, obs, env.transition, env.emit_prob,
                                  action=action, update_weights=update_weights)
        if weights is None:
            print("=============RESETTING WEIGHTS=============")
            # If all our weights are 0, we get None for weights and have to
            # re-initialize particles
            if n_particles == -1:
                particles = env.sample_all_states()
            else:
                particles = env.sample_states(n=n_particles)

            weights = np.ones(particles.shape[0]) / particles.shape[0]

        # state_in_particles = False
        # print(particles[particle_to_follow], env.state, action)
        # current_state_prob = 0
        # for i, part in enumerate(particles):
        #     if (env.state == part).all():
        #         state_in_particles = True
        #         current_state_prob += weights[i]
        # print(f"Step {s} current state weight: {current_state_prob}")
        # assert state_in_particles
        ess = effective_sample_size(weights)
        esses.append(ess)
        means, var = state_stats(particles, weights)

        if np.isclose(ess, 1.0) or np.isclose(var.sum(), 0.0):
            print(f"Only one particle remaining at step {s}. Terminating.\n")
            break

        if s % log_interval == 0:
            print(f"Step {s}, "
                  f"Effective sample size: {esses[-1]:.2f}")

        if resample_every > 0 and s % resample_every == 0:
            weights, particles = resample(weights, particles, env.rng)

        action = env.action_space.sample()
        obs, rew, done, info = env.step(action)
        total_steps += 1
        update_steps += 1 if update_weights else 0


    return total_steps, update_steps


def test_CW():
    # n_runs = 30
    n_runs = 1
    weight_update_interval = 100
    log_interval = weight_update_interval
    seeds = np.arange(0, n_runs) + 2022
    steps_to_one = np.zeros_like(seeds)
    updates_to_one = np.zeros_like(seeds)

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        env = CompassWorld(seed=seed, random_start=True)
        steps_to_one[i], updates_to_one[i] = run_pf_random_policy(env, log_interval=log_interval, weight_update_interval=weight_update_interval)

    print(f"Finished running {n_runs}. "
          f"Average number of steps to converge to 1 particle: "
          f"{steps_to_one.mean():.4f} +/- {steps_to_one.std() / n_runs:.2f}, "
          f"Average number of updates to one particle: "
          f"{updates_to_one.mean():.4f} +/- {updates_to_one.std() / n_runs:.2f}")


def test_BlurryCW():

    n_runs = 30
    weight_update_interval = 1
    log_interval = 10
    seeds = np.arange(0, n_runs) + 2022
    steps_to_one = np.zeros_like(seeds)
    updates_to_one = np.zeros_like(seeds)
    blur_prob = 0.3

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        env = CompassWorld(seed=seed, random_start=True)
        env = BlurryWrapper(env, blur_prob=blur_prob)
        steps_to_one[i], updates_to_one[i] = run_pf_random_policy(env, log_interval=log_interval, weight_update_interval=weight_update_interval)

    print(f"Finished running {n_runs}. "
          f"Average number of steps to converge to 1 particle: "
          f"{steps_to_one.mean():.4f} +/- {steps_to_one.std() / n_runs:.2f}, "
          f"Average number of updates to one particle: "
          f"{updates_to_one.mean():.4f} +/- {updates_to_one.std() / n_runs:.2f}")


def test_SlipCW():
    n_runs = 30
    weight_update_interval = 1
    log_interval = 10
    seeds = np.arange(0, n_runs) + 2022
    steps_to_one = np.zeros_like(seeds)
    updates_to_one = np.zeros_like(seeds)
    slip_prob = 0.1
    env_str = "fi"

    for i, seed in enumerate(seeds):
        np.random.seed(seed)
        env = get_env(seed, env_str, slip_prob=slip_prob)
        steps_to_one[i], updates_to_one[i] = \
            run_pf_random_policy(env, log_interval=log_interval,
                                 weight_update_interval=weight_update_interval,
                                 resample_every=None,
                                 n_particles=1000)



if __name__ == "__main__":
    test_SlipCW()
