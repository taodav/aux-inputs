import numpy as np
from typing import Union

from unc.envs.lobster import LobsterFishing
from unc.particle_filter import batch_step, resample
from .wrapper import LobsterFishingWrapper


class LobsterParticleFilterWrapper(LobsterFishingWrapper):
    """
    Particle filter (not incl observations)

    """
    priority = 2

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper], *args,
                 update_weight_interval: int = 1, resample_interval: int = None,
                 n_particles: int = 100,
                 **kwargs):
        super(LobsterParticleFilterWrapper, self).__init__(env, *args, **kwargs)
        self.particles = None
        self.weights = None
        self.env_step = 0

        self.update_weight_interval = update_weight_interval
        self.n_particles = n_particles
        assert self.n_particles > 1, "Non-stochastic start states. Need more than one particle to start."
        self.resample_interval = resample_interval if resample_interval is not None else float("inf")

    def emit_prob(self, states: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Get the probability of emitting a batch of observations given states
        :param state: underlying state, size batch_size x 3
        :param obs: observation, size batch_size x 5
        :return: probability of emitting (either 0 or 1 for this deterministic environment).
        """
        ground_truth_obs = self.batch_get_obs(states)
        return (ground_truth_obs == obs).min(axis=-1).astype(np.float)

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        obs = np.zeros((states.shape[0], 9))

        # Set position
        obs[np.arange(states.shape[0]), states[:, 0]] = 1

        # first set all rewards to unobservable
        obs[:, [5, 8]] = 1

        # all observable b/c we're in state 1
        in_state_1 = states[:, 0] == 1

        obs[in_state_1, 5] = 0
        # idx 3: observable, not there; idx 4: observable, there.
        # if reward_1_present[i] = 0 (not present), then idx 3. If present, 4.
        obs[in_state_1, 3 + states[in_state_1][:, 1]] = 1

        # all observable b/c we're in state 2
        in_state_2 = states[:, 0] == 2

        obs[in_state_2, 8] = 0
        # idx 6: observable, not there; idx 7: observable, there.
        # if reward_2_present[i] = 0 (not present), then idx 6. If present, 7.
        obs[in_state_2, 6 + states[in_state_2][:, 2]] = 1

        return obs

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        new_states = states.copy()

        # We move if we aren't trying to move left/right in state 1/2 respectively
        pos_0_mask = states[:, 0] == 0
        other_pos_mask = ~pos_0_mask
        move_left_mask = (actions == 0) * (states[:, 0] != 1)
        move_right_mask = (actions == 1) * (states[:, 0] != 2)

        # Who's moving from pos 0?
        moving = np.logical_or(move_left_mask, move_right_mask)
        move_from_pos_0 = pos_0_mask * moving
        move_from_other_pos = other_pos_mask * moving

        # How many are moving?
        n_to_move = move_left_mask.sum() + move_right_mask.sum()
        # How many make the move?
        make_it_mask = self.rng.random(size=n_to_move) < self.traverse_prob
        # These are the dudes ACTUALLY moving
        move_mask = np.zeros_like(moving, dtype=bool)
        move_mask[moving] = make_it_mask

        # finally they MOVE
        pos_0_move = move_from_pos_0 * move_mask
        new_states[:, 0][pos_0_move] = actions[pos_0_move] + 1

        pos_other_move = move_from_other_pos * move_mask
        new_states[:, 0][pos_other_move] = 0

        # Collect
        # indices to collect
        actions_collect_mask = actions == 2
        actually_collect = other_pos_mask * actions_collect_mask

        # for each index, we collect at the given position (either 1 or 2)
        new_states[np.nonzero(actually_collect), new_states[:, 0][actually_collect]] = 0

        # Tick collected rewards
        to_tick = new_states[:, 1:] == 0
        repeat_pmfs = np.expand_dims(self.pmfs_1, 0).repeat(new_states.shape[0], axis=0)
        reset_mask = self.rng.binomial(1, p=repeat_pmfs)
        new_states[:, 1:][to_tick] = reset_mask[to_tick]

        return new_states

    def reset(self, **kwargs) -> np.ndarray:
        obs = self.env.reset(**kwargs)

        self.particles = self.sample_start_states(n=self.n_particles)
        self.weights = np.ones(self.n_particles) / self.n_particles

        # Update them based on the first observation
        self.weights, self.particles = batch_step(self.weights, self.particles, obs,
                                                  self.batch_transition, self.emit_prob)

        return obs

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        self.env_step += 1

        # Update our particles and weights after doing a transition
        self.weights, self.particles = batch_step(self.weights, self.particles, obs,
                                                  self.batch_transition, self.emit_prob, action=action,
                                                  update_weights=self.env_step % self.update_weight_interval == 0)

        if self.weights is None:
            # If all our weights are 0, we get None for weights and have to
            # re-initialize particles
            self.particles = self.sample_all_states(n=self.n_particles)
            self.weights = np.ones(self.n_particles) / self.n_particles

        if self.env_step % self.resample_interval == 0:
            self.weights, self.particles = resample(self.weights, self.particles, rng=self.rng)

        return obs, reward, done, info


