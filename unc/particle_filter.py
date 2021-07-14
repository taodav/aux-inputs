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
        for i, p in enumerate(particles):
            updated_particles[i] = transition_fn(p, action)

    # Now we re-weight based on emission probabilities
    if update_weights:
        for i, p in enumerate(updated_particles):
            if 1 - weights[i] < 10e-10:
                continue
            unnormalized_updated_weights[i] = weights[i] * emit_prob(p, next_obs)

        # Normalize our weights again
        updated_weights = unnormalized_updated_weights / np.sum(unnormalized_updated_weights)

        assert not np.isnan(updated_weights).any()
    else:
        updated_weights = unnormalized_updated_weights

    return updated_weights, updated_particles
