import gym
import numpy as np
from typing import Any, Tuple

from .base import Environment


class DirectionalTMaze(Environment):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, rng: np.random.RandomState, size: int = 10):
        self.size = size

        self.observation_space = gym.spaces.MultiBinary(2)
        self.action_space = gym.spaces.Discrete(3)

        self.rng = rng
        self.map = np.ones((3 + 2, self.size + 2), dtype=np.uint8)
        self.map[2, 1:-1] = 0
        self.map[1:-1, -2] = 0

        self.position = np.array([0, 0])
        self.pose = None
        self.goal = np.zeros(2)  # Either 1 or 2
        self.first_step = True

        # TODO: time is also in markov state since timestep 0 you reveal goal position

    @property
    def state(self):
        return np.array([self.position[0], self.position[1],
                         self.pose,
                         self.goal[0], self.goal[1],
                         int(self.first_step)], dtype=np.uint8)

    @state.setter
    def state(self, state: np.ndarray):
        self.position = state[:2]
        self.pose = state[2]
        self.goal = state[3:5]
        self.first_step = bool(state[-1])

    def get_terminal(self) -> bool:
        return self.position[1] == (self.size - 1) and self.position[0] in [1, -1]

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        position = state[:2]
        pose = state[2]
        goal = state[3:5]
        first_step = bool(state[-1])

        goal_obs = goal if first_step else np.zeros_like(goal)

        in_front = position + self.direction_mapping[pose] + 1
        see_wall = 1 - self.map[in_front[0], in_front[1]]

        return np.array([int(see_wall), *goal_obs], dtype=np.int16)

    def get_reward(self, prev_state: np.ndarray = None, action: int = None) -> float:
        if self.get_terminal():
            if (self.goal == 1 and self.position[0] == 1) or (self.goal == 2 and self.position[0] == -1):
                return 4.
            return -1.

        return -0.1

    def reset(self) -> np.ndarray:
        goal_idx = self.rng.choice([0, 1])
        self.goal = np.zeros(2)
        self.goal[goal_idx] = 1
        self.position = np.array([0, 0], dtype=np.uint8)
        self.pose = self.rng.choice(np.arange(4))
        self.first_step = True

        return self.get_obs(self.state)

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        next_state = state.copy()
        if action == 0:
            new_pos = state[:2] + self.direction_mapping[state[2]]
            new_pos_shifted = new_pos + 1
            if self.map[new_pos_shifted[0], new_pos_shifted[1]] == 0:
                next_state[:2] = new_pos
        elif action == 1:
            next_state[2] = (next_state[2] + 1) % 4
        elif action == 2:
            next_state[2] = (next_state[2] - 1) % 4
        next_state[-1] = 0

        return next_state

    def emit_prob(self, state: Any, obs: np.ndarray) -> float:
        return 1.

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        new_state = self.transition(self.state, action)
        self.state = new_state

        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}

