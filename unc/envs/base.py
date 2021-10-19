import gym
import numpy as np
from typing import Any, Tuple


class Environment(gym.Env):

    @property
    def state(self):
        return self.state

    @state.setter
    def state(self, state: np.ndarray):
        self.state = state

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_reward(self, prev_state: np.ndarray = None, action: int = None) -> int:
        raise NotImplementedError()

    def get_terminal(self) -> bool:
        raise NotImplementedError()
    
    def reset(self) -> np.ndarray:
        return super(Environment, self).reset()

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        raise NotImplementedError()

    def emit_prob(self, state: Any, obs: np.ndarray) -> float:
        pass
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        return super(Environment, self).step(action)

