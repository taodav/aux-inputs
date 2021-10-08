import numpy as np
from typing import Callable, Tuple


def state_stats(particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the mean and std dev of the three state variables
    :param particles:
    :param weights:
    :return:
    """
    mean = np.zeros_like(particles[0], dtype=np.float32)
    variance = np.zeros_like(particles[0], dtype=np.float32)

    for i, (p, w) in enumerate(zip(particles, weights)):
        mean += w.astype(np.float32) * p.astype(np.float32)

    for i, (p, w) in enumerate(zip(particles, weights)):
        variance += w.astype(np.float32) * ((p.astype(np.float32) - mean) ** 2)

    return mean, variance


def batch_step(weights: np.ndarray, particles: np.ndarray, next_obs: np.ndarray,
         batch_transition_fn: Callable,
         emit_prob: Callable,
         action: int = None,
         update_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    assert weights.shape[0] == particles.shape[0]
    updated_particles = particles.copy()
    unnormalized_updated_weights = weights.copy()

    # First we propagate the particles
    if action is not None:
        batch_actions = np.zeros(particles.shape[0], dtype=int) + action
        updated_particles = batch_transition_fn(particles, batch_actions)
        # for i, p in enumerate(particles):
        #     # Some small optimization: If particle weights are 0, we don't apply the
        #     # transition function.
        #     if weights[i] > 10e-10:
        #         updated_particles[i] = transition_fn(p, action)

    # Now we re-weight based on emission probabilities
    if update_weights:
        unnormalized_updated_weights = weights * emit_prob(updated_particles, next_obs)
        # for i, p in enumerate(updated_particles):
        #     if 1 - weights[i] < 10e-10:
        #         continue
        #     unnormalized_updated_weights[i] = weights[i] * emit_prob(p, next_obs)

        # Normalize our weights again
        sum_weights = np.sum(unnormalized_updated_weights)
        if sum_weights == 0:
            updated_weights = None
        else:
            updated_weights = unnormalized_updated_weights / sum_weights

    else:
        updated_weights = unnormalized_updated_weights

    return updated_weights, updated_particles


def step(weights: np.ndarray, particles: np.ndarray, next_obs: np.ndarray,
       transition_fn: Callable,
       emit_prob: Callable,
       action: int = None,
       update_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    assert weights.shape[0] == particles.shape[0]
    updated_particles = particles.copy()
    unnormalized_updated_weights = weights.copy()

    # First we propagate the particles
    if action is not None:
        batch_actions = np.ones(particles.shape[0]) * action
        for i, p in enumerate(particles):
            # Some small optimization: If particle weights are 0, we don't apply the
            # transition function.
            if weights[i] > 10e-10:
                updated_particles[i] = transition_fn(p, action)

    # Now we re-weight based on emission probabilities
    if update_weights:
        for i, p in enumerate(updated_particles):
            if 1 - weights[i] < 10e-10:
                continue
            unnormalized_updated_weights[i] = weights[i] * emit_prob(p, next_obs)[0]

        # Normalize our weights again
        sum_weights = np.sum(unnormalized_updated_weights)
        if sum_weights == 0:
            updated_weights = None
        else:
            updated_weights = unnormalized_updated_weights / sum_weights

    else:
        updated_weights = unnormalized_updated_weights

    return updated_weights, updated_particles

def resample(weights: np.ndarray, particles: np.ndarray, rng: np.random.RandomState = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample particles w.r.t. given weights.
    Assume we sample the same number of particles that exist in the input particles.
    :param weights: weights to resample from
    :param particles: particles to resample from
    :return: Tuple of (new_weights, new_particles).
    """
    sampler = rng if rng is not None else np.random
    new_particle_idxes = sampler.choice(np.arange(particles.shape[0]), p=weights, size=len(particles), replace=True)
    new_particles = particles[new_particle_idxes]
    # new_particles_p = np.array([sampler.choice(particles, p=weights, replace=True) for i in range(len(particles))])
    new_weights = np.ones_like(weights) / len(weights)

    return new_weights, new_particles

