from typing import Any


class Agent:
    def act(self, state: Any):
        raise NotImplementedError()

    def update(self, state: Any, action: Any, next_state: Any, gamma: Any, reward: Any):
        pass

    def set_eps(self, eps: float):
        pass

