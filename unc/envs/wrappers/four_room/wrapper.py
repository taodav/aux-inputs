from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.four_room import FourRoom


class FourRoomWrapper(gym.Wrapper):
    priority = 1

    def __init__(self, env: Union[FourRoom, FourRoomWrapper]):
        super(FourRoomWrapper, self).__init__(env)
        self.random_start = env.random_start
        self.rng = env.rng

        if isinstance(self.env, FourRoomWrapper):
            assert self.env.priority <= self.priority

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def size(self) -> int:
        return self.unwrapped.size

    @property
    def n_rewards(self) -> int:
        return self.unwrapped.n_rewards

    @state.setter
    def state(self, state) -> None:
        self.env.state = state

    def generate_array(self) -> np.ndarray:
        return self.unwrapped.generate_array()

    def unpack_state(self, state: np.ndarray):
        return self.unwrapped.unpack_state(state)

    # we do these assertions here b/c FourRoom is an env where we don't want
    # to use particle filters.
    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        assert NotImplementedError

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.env.get_obs(state)

    def get_reward(self) -> float:
        return self.env.get_reward()

    def get_terminal(self) -> bool:
        return self.env.get_terminal()

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        assert NotImplementedError

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        return self.env.transition(state, action)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        assert NotImplementedError

    def sample_states(self, n: int = 10) -> np.ndarray:
        return self.env.sample_states(n)

    def sample_all_states(self) -> np.ndarray:
        return self.env.sample_all_states()

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode=mode, **kwargs)
