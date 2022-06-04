import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import LobsterFishingWrapper
from unc.envs.lobster import LobsterFishing, all_lobster_states


def get_lobster_state_map():
    flat_map = [
        # node 0
        [[0, 1], [2, 3]],  # r1, r2

        # node 1
        [[4, 5], [6, 7]],

        # node 2
        [[8, 9], [10, 11]]
    ]
    return np.array(flat_map)


class BeliefStateWrapper(LobsterFishingWrapper):
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(BeliefStateWrapper, self).__init__(env)

        assert hasattr(self, 'particles') and hasattr(self, 'weights')
        self.all_states = all_lobster_states()

        self.observation_space = gym.spaces.Box(
            low=np.zeros(self.all_states.shape[0]),
            high=np.ones(self.all_states.shape[0])
        )
        self.state_map = get_lobster_state_map()

    def get_state_idx(self, state: np.ndarray) -> int:
        return self.state_map[state[0], state[1], state[2]]

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(self.all_states.shape[0])

        for p, w in zip(self.particles, self.weights):
            obs[self.get_state_idx(p)] += w

        return obs

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
