import numpy as np

from .simple_chain import SimpleChain


class DynamicChain(SimpleChain):
    def __init__(self, rng: np.random.RandomState, n: int = 10, ):
        """
        Simple func. approx single chain. Always returns an observation of 1.
        :param n: length of chain
        """
        super(DynamicChain, self).__init__(n)
        self._state = None
        self.rng = rng

    @property
    def state(self):
        return self._state

    def reset(self):
        self._state = np.zeros(self.rng.choice(np.arange(2, self.n)))
        self.current_idx = 0
        self._state[self.current_idx] = 1
        return self.get_obs(self._state)
