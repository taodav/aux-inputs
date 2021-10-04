from typing import Any

from unc.utils.data import Batch


class Agent:
    def act(self, state: Any):
        raise NotImplementedError()

    def update(self, batch: Batch):
        pass

    def set_eps(self, eps: float):
        pass

