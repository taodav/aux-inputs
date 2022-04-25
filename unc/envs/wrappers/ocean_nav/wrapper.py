from __future__ import annotations
import gym
import numpy as np
from typing import Union

from unc.envs.ocean_nav import OceanNav


class OceanNavWrapper(gym.Wrapper):
    priority = 1
    groups_info = None

    def __init__(self, env: Union[OceanNav, OceanNavWrapper]):
        super(OceanNavWrapper, self).__init__(env)
        self.rng = env.rng

        if isinstance(self.env, OceanNavWrapper):
            assert self.env.priority <= self.priority

    def unpack_state(self, state: np.ndarray):
        return self.env.unpack_state(state)

    def get_current_reward(self, state: np.ndarray, prev_state: np.ndarray, action: int) -> float:
        return self.env.get_current_reward(state, prev_state, action)

    @property
    def obstacle_map(self):
        return self.env.obstacle_map

    @property
    def start_positions(self):
        return self.env.start_positions

    @property
    def config(self):
        return self.env.config

    @property
    def glass_map(self):
        return self.env.glass_map

    @property
    def kelp_map(self):
        return self.env.kelp_map

    @property
    def current_map(self):
        return self.env.current_map

    @property
    def size(self):
        return self.env.size

    @property
    def state_space(self):
        return self.env.state_space

    @property
    def rewards(self) -> np.ndarray:
        return self.env.rewards

    @rewards.setter
    def rewards(self, rewards: np.ndarray) -> None:
        self.env.rewards = rewards

    @property
    def position(self) -> np.ndarray:
        return self.env.position

    @position.setter
    def position(self, position: np.ndarray) -> None:
        self.env.position = position

    @property
    def state(self) -> np.ndarray:
        return self.env.state

    @property
    def size(self) -> int:
        return self.env.size

    @state.setter
    def state(self, state) -> None:
        self.env.state = state

    def reset_currents(self) -> None:
        self.env.reset_currents()

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        return self.env.batch_get_obs(states)

    def get_obs(self, state: np.ndarray, *args, **kwargs) -> np.ndarray:
        return self.env.get_obs(state, *args, **kwargs)

    def get_reward(self, *args, **kwargs) -> float:
        return self.env.get_reward(*args, **kwargs)

    def get_terminal(self) -> bool:
        return self.env.get_terminal()

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        return self.env.batch_transition(states, actions)

    def transition(self, state: np.ndarray, action: int, *args, **kwargs) -> np.ndarray:
        return self.env.transition(state, action, *args, **kwargs)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        return self.env.emit_prob(state, obs)

    def sample_position(self, n: int = 10) -> np.ndarray:
        return self.env.sample_position(n)

    def sample_all_states(self) -> np.ndarray:
        return self.env.sample_all_states()

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode=mode, **kwargs)
