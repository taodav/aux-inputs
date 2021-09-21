import numpy as np
from typing import Tuple
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.RandomState,
                 obs_size: Tuple, state_size: Tuple = None):
        """
        Replay buffer that saves both observation and state.
        :param capacity:
        :param rng:
        """

        self.capacity = capacity
        self.rng = rng
        self.state_size = state_size
        self.obs_size = obs_size

        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))
        self.obs = np.zeros((self.capacity, *self.obs_size), dtype=np.int16)
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._age = 0
        self._filled = False

        # We do this b/c cursor + 1, cursor + 2, ..., cursor + stack - 1
        # cannot be sampled.
        self.eligible_idxes = deque(maxlen=self.capacity - 1)

    def reset(self):
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))
        self.obs = np.zeros((self.capacity, *self.obs_size), dtype=np.int16)
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._age = 0
        self._filled = False

    def push_initial(self, obs: np.ndarray, state: np.ndarray = None):
        if self.state_size is not None and state is not None:
            self.s[self._cursor] = state
        self.obs[self._cursor] = obs

    def push(self, batch: dict):
        next_cursor = (self._cursor + 1) % self.capacity
        self.a[self._cursor] = batch['action']
        if self.state_size is not None and 'state' in batch:
            self.s[next_cursor] = batch['state']
        self.obs[next_cursor] = batch['obs']
        self.d[self._cursor] = batch['done']
        self.r[self._cursor] = batch['reward']

        self.eligible_idxes.append(self._cursor)
        self._cursor = next_cursor

    def __len__(self):
        return len(self.eligible_idxes)

    def sample(self, batch_size: int) -> dict:
        if len(self.eligible_idxes) < batch_size:
            batch_size = len(self.eligible_idxes)

        sample_idx = self.rng.choice(self.eligible_idxes, size=batch_size)
        batch = {}
        if self.state_size is not None:
            batch['state'] = self.s[sample_idx]
            batch['next_state'] = self.s[(sample_idx + 1) % self.capacity]
        batch['obs'] = self.obs[sample_idx]
        batch['next_obs'] = self.obs[(sample_idx + 1) % self.capacity]
        batch['action'] = self.a[sample_idx]
        batch['done'] = self.d[sample_idx]
        batch['reward'] = self.r[sample_idx]

        return batch






