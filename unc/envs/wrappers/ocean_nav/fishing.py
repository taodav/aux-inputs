import numpy as np
from typing import Union, Tuple

from unc.envs.ocean_nav import OceanNav
from .wrapper import OceanNavWrapper


class FishingWrapper(OceanNavWrapper):
    priority = 1

    """
    For Fishing Wrapper, this is a continual learning task 
    analogous to foraging. The biggest differences to this and the OceanNav
    environment is the reward and terminals:
    
    Reward (TODO) - the list of rewards corresponds to regenerating rewards
    sampled around the given position.
    
    The way we track whether or not a reward has been taken is to check for
    -1's for both x and y positions for all reward positions.
    
    Terminals - Always False in this case.
    """

    def __init__(self, env: Union[OceanNav, OceanNavWrapper],
                 random_reward_start: bool = False):
        super(FishingWrapper, self).__init__(env)
        rew_pos_list = [rew_dict['position'] for rew_dict in self.config['rewards']]
        self.reward_centroids = np.array(rew_pos_list, dtype=np.int16)
        self.random_reward_start = random_reward_start

        # Set a default here if we don't include it in config
        # self.reward_change_rate = np.ones(len(rew_pos_list)) * 40
        self.reward_change_rate = np.array([rew_dict['change_rate'] for rew_dict in self.config['rewards']])
        reward_lambs = 1 / self.reward_change_rate

        # TODO: For now we assume all rewards have the same change rates
        self.rewards_pmfs_1 = reward_lambs * np.exp(-reward_lambs)

        self.rewards = None

    @staticmethod
    def generate_possible_rewards(centroids: np.ndarray):
        rewards = np.array(centroids, dtype=np.int16)
        return rewards

    def reset_rewards(self, centroids: np.ndarray) -> np.ndarray:
        rewards = self.generate_possible_rewards(centroids)
        if self.random_reward_start:
            missing_mask = self.rng.randint(2, size=len(centroids)).astype(bool)
            rewards[missing_mask] = np.array([-1, -1])

        return rewards

    def reset(self):
        self.reset_currents()
        self.position = self.start_positions[self.rng.choice(range(len(self.start_positions)))]
        self.rewards = self.reset_rewards(self.reward_centroids)
        return self.get_obs(self.state)

    def get_terminal(self) -> bool:
        return False

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state_less_reward = super(FishingWrapper, self).transition(state, action)

        current_map, position, reward_positions = self.unpack_state(new_state_less_reward)
        new_reward_positions = reward_positions.copy()
        for i, rew_pos in enumerate(reward_positions):
            if np.all(rew_pos == position):
                new_reward_positions[i] = np.array([-1, -1], dtype=np.int16)
            elif np.all(rew_pos == -1) and self.rng.rand() < self.rewards_pmfs_1[i]:
                # now we need to tick the rewards
                new_rew_pos = self.generate_possible_rewards(self.reward_centroids[i:i+1])[0]
                new_reward_positions[i] = new_rew_pos

        flattened_new_current_map = current_map.flatten()
        flattened_new_reward_positions = new_reward_positions.flatten()
        return np.concatenate((flattened_new_current_map, position, flattened_new_reward_positions), axis=0)

    def get_reward(self, prev_state: np.ndarray, action: int) -> float:
        current_map, position, reward_pos = self.unpack_state(self.state)
        prev_current_map, prev_position, prev_reward_pos = self.unpack_state(prev_state)

        reward = 0
        for i, rew in enumerate(reward_pos):
            if np.all(rew == -1) and np.all(prev_reward_pos[i] != -1):
                reward += 1
        assert reward < 2

        reward += self.get_current_reward(self.state, prev_state, action)
        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_state = self.state
        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(prev_state, action), self.get_terminal(), {'position': self.position.copy()}
