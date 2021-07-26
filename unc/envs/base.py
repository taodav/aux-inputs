import gym
import numpy as np


class Environment(gym.Env):

    @property
    def state(self):
        return self.state

    @state.setter
    def state(self, state: np.ndarray):
        self.state = state

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def get_reward(self) -> int:
        raise NotImplementedError()

    def get_terminal(self) -> bool:
        raise NotImplementedError()
    
    def reset(self):
        super(Environment, self).reset()

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        raise NotImplementedError()

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:
        pass
    
    def step(self, action: int):
        super(Environment, self).step(action)

