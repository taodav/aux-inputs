from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.simple_chain import SimpleChain


class SimpleChainWrapper(gym.Wrapper):
    priority = 1

    def __init__(self, env: Union[SimpleChain, SimpleChainWrapper]):
        super(SimpleChainWrapper, self).__init__(env)

        if isinstance(self.env, SimpleChainWrapper):
            assert self.env.priority <= self.priority

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def n_rewards(self) -> int:
        return self.unwrapped.n_rewards

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
        return self.env.emit_prob(state, obs)

    def sample_states(self, n: int = 10) -> np.ndarray:
        assert NotImplementedError

    def sample_all_states(self) -> np.ndarray:
        assert NotImplementedError

    def render(self, mode='rgb_array', **kwargs):
        assert NotImplementedError
