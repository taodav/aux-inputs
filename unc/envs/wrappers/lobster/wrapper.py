from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.lobster import LobsterFishing


class LobsterFishingWrapper(gym.Wrapper):
    priority = 1

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper]):
        super(LobsterFishingWrapper, self).__init__(env)
        self.rng = env.rng

        if isinstance(self.env, LobsterFishingWrapper):
            assert self.env.priority <= self.priority

    @property
    def predictions(self):
        return self.env.predictions

    @predictions.setter
    def predictions(self, predictions: np.ndarray):
        self.env.predictions = predictions

    @property
    def gvf_idxes(self):
        return self.env.gvf_idxes

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def n_rewards(self) -> int:
        return self.unwrapped.n_rewards

    @property
    def traverse_prob(self) -> float:
        return self.unwrapped.traverse_prob

    @property
    def pmfs_1(self) -> float:
        return self.unwrapped.pmfs_1

    @state.setter
    def state(self, state) -> None:
        self.env.state = state

    # def generate_array(self) -> np.ndarray:
    #     return self.unwrapped.generate_array()

    # def unpack_state(self, state: np.ndarray):
    #     return self.unwrapped.unpack_state(state)

    # we do these assertions here b/c LobsterFishing is an env where we don't want
    # to use particle filters.
    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        return self.env.batch_get_obs(states)

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.env.get_obs(state)

    def get_reward(self, *args, **kwargs) -> float:
        return self.env.get_reward(*args, **kwargs)

    def get_terminal(self) -> bool:
        return self.env.get_terminal()

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.env.batch_transition(states, actions)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        return self.env.transition(state, action)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        assert NotImplementedError

    def sample_start_states(self, n: int = 10) -> np.ndarray:
        return self.env.sample_start_states(n)

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode=mode, **kwargs)
