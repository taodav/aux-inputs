import gym
import numpy as np
from typing import Union, Tuple

from .agent_centric import AgentCentricObservationWrapper, OceanNavWrapper, OceanNav


class PartiallyObservableWrapper(AgentCentricObservationWrapper):
    priority = 3

    def __init__(self, env: Union[OceanNav, OceanNavWrapper],
                 window_size: int = 5,
                 distance_noise: bool = False,
                 prob_levels: Tuple[int, int, int] = (1, 0.85, 0.7)):
        """
        Partially observable OceanNav environment.

        The agent can only see a square of window_size x window_size around itself,
        and if there's an obstacle in the way, everything beyond that is occluded.

        We also have a probability of obstacles, rewards and current showing the
        wrong observations, that has probability proportional to (given that
        obs is observation at position away from agent)
        max(abs(pos[0] - obs[0]), abs(pos[1] - obs[1]))
        """
        assert not isinstance(env, AgentCentricObservationWrapper), "Cannot have PartiallyObservable wrapper around AgentCentric"
        super(PartiallyObservableWrapper, self).__init__(env)

        self.distance_noise = distance_noise
        self.window_size = window_size
        assert self.window_size % 2 != 0, "window_size must be odd number"

        if self.window_size > 5:
            # TODO: implement this
            raise NotImplementedError("Haven't implemented occlusion for anything larger than 5 yet.")

        half = self.window_size // 2
        if distance_noise:
            assert len(prob_levels) == half + 1, "Probabilities don't match window_size"

        larger_obs_shape = super(PartiallyObservableWrapper, self).observation_space.shape
        agent_pos = np.array(larger_obs_shape[:-1]) // 2

        y_start = agent_pos[0] - half
        self.y_range = [y_start, y_start + self.window_size]

        x_start = agent_pos[1] - half
        self.x_range = [x_start, x_start + self.window_size]

        low = np.zeros((self.window_size, self.window_size, larger_obs_shape[-1]))
        high = np.ones_like(low)
        self.observation_space = gym.spaces.Box(
            low=low, high=high
        )
        self.prob_levels = prob_levels

        # Our prob noise map only has 3 channels.
        # 1 for obstacles, 1 for current and 1 for reward position.
        self.prob_map = np.repeat(self.generate_prob_map()[:, :, None], 3)

        # self.potential_reward_map

    def generate_prob_map(self) -> np.ndarray:
        prob_map = np.zeros((self.window_size, self.window_size))

        middle = self.window_size // 2
        prob_map[middle - 1:-(middle - 1), middle - 1:-(middle - 1)] = self.prob_levels[1]
        prob_map[middle, middle] = self.prob_levels[0]

        prev = prob_map[middle - 1:-(middle - 1), middle - 1:-(middle - 1)]
        for i, prob in enumerate(self.prob_levels[2:], start=2):
            prob_map[middle - i:-(middle - i), middle - i:-(middle - i)] = prob
            prev_idx = middle - i + 1
            prob_map[prev_idx:-prev_idx, prev_idx:-prev_idx] = prev
            prev = prob_map[middle - i:-(middle - i), middle - i:-(middle - i)]

        return prob_map

    def noisify_observations(self, obs: np.ndarray) -> np.ndarray:
        # we invert here, since the prob_map shows probability of accurate observations
        to_flip_mask = np.invert(self.rng.binomial(1, p=self.prob_map).astype(bool))

        obstacle_mask = to_flip_mask[:, :, 0]
        current_mask = to_flip_mask[:, :, 1]
        reward_mask = to_flip_mask[:, :, 2]

        # all the incorrect ones, we flip
        obs[:, :, 0][obstacle_mask] = 1 - obs[:, :, 0][obstacle_mask]

        # for current it's a bit more complicated. For every bit we need to flip,
        # we randomly sample a current (or no current)
        currents_to_sample = obs[:, :, 1:5][current_mask].shape[0]
        random_currents = self.rng.choice(np.arange(5), size=currents_to_sample)
        non_zero_currents = random_currents != 0
        random_nonzero_currents_idx = random_currents[non_zero_currents] - 1

        # we first zero out all the currents we want to flip
        obs[:, :, 1:5][current_mask] = 0

        # now we set the corresponding observation
        obs[:, :, 1:5][current_mask][non_zero_currents][np.arange(random_nonzero_currents_idx.shape[0]), random_nonzero_currents_idx] = 1


    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        expanded_ac_map, expanded_glass_map = \
            super(PartiallyObservableWrapper, self).get_obs(state, *args, return_expanded_glass_map=True, **kwargs)

        # get our map where we can see through walls
        see_thru_po_map = expanded_ac_map[self.y_range[0]:self.y_range[1], self.x_range[0]:self.x_range[1]]
        ac_glass_map = expanded_glass_map[self.y_range[0]:self.y_range[1], self.x_range[0]:self.x_range[1]]
        ac_obstacle_map = see_thru_po_map[:, :, 0]

        occlusion_mask = self.get_occlusion_mask(ac_obstacle_map, ac_glass_map)
        final_map = see_thru_po_map

        final_map[:, :, 0][occlusion_mask] = self.obstacle_filler_idx
        final_map[:, :, 1:5][occlusion_mask] = self.current_filler
        final_map[:, :, 5][occlusion_mask] = self.self.reward_filler_idx

        # we randomly flip our binary observations
        if self.distance_noise:
            final_map = self.noisify_observations(final_map)

        return final_map

    @staticmethod
    def get_occlusion_mask(obstacle_map: np.ndarray, glass_map: np.ndarray) -> np.ndarray:
        """
        Given an agent-centric obstacle map and an agent-centric glass map,
        calculate an occlusion mask.
        """
        middle = obstacle_map.shape[0] // 2
        layers = range(1, middle)
        curr_occlusion_mask = np.zeros_like(obstacle_map)
        curr_obstacle_map = obstacle_map.copy()
        curr_glass_map = glass_map.copy()

        def get_occlusion_row(obstacle: np.ndarray, glass: np.ndarray, occlusion: np.ndarray, idx: int) -> np.ndarray:
            next_row_occlusion = np.zeros(obstacle.shape[1])
            obstacle_row = obstacle[idx]
            glass_row = glass[idx]
            occlusion_row = occlusion[idx]

            # we figure our occlusion for ends first
            next_row_occlusion[0] = obstacle_row[1] * (1 - glass_row[1])
            next_row_occlusion[-1] = obstacle_row[-2] * (1 - glass_row[-2])

            # now for the middle-sides
            # TODO: ONLY WORKS FOR WINDOW SIZE 5 NOW
            next_row_occlusion[1] = obstacle_row[1] * obstacle_row[2] *\
                                    ((1 - glass_row[1]) * (1 - glass_row[2]))
            next_row_occlusion[-2] = obstacle_row[-2] * obstacle_row[-3] *\
                                     ((1 - glass_row[-2]) * (1 - glass_row[-3]))

            # Now for the middle
            middle_idx = next_row_occlusion.shape[0] // 2
            next_row_occlusion[middle_idx] = obstacle_row[middle_idx] * (1 - glass_row[middle_idx])

            return next_row_occlusion

        # we get a partial mask for each direction.
        # we rotate our maps 3 times to get every single direction
        for layer in layers:
            row_idx = middle - layer
            curr_occlusion_mask[row_idx - 1] = np.maximum(curr_occlusion_mask[row_idx - 1],
                                                          get_occlusion_row(curr_obstacle_map, curr_glass_map,
                                                                            curr_occlusion_mask, row_idx))

            for _ in range(3):
                curr_occlusion_mask = np.rot90(curr_occlusion_mask)
                curr_obstacle_map = np.rot90(curr_obstacle_map)
                curr_glass_map = np.rot90(curr_glass_map)
                curr_occlusion_mask[row_idx - 1] = np.maximum(curr_occlusion_mask[row_idx - 1],
                                                              get_occlusion_row(curr_obstacle_map, curr_glass_map,
                                                                                curr_occlusion_mask, row_idx))

            curr_occlusion_mask = np.rot90(curr_occlusion_mask)
            curr_obstacle_map = np.rot90(curr_obstacle_map)
            curr_glass_map = np.rot90(curr_glass_map)

        return curr_occlusion_mask

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
