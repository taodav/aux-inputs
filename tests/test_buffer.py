import numpy as np
from typing import Tuple
from collections import deque

from unc.utils.data import Batch


class OldReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.RandomState,
                 obs_size: Tuple,
                 obs_dtype: type,
                 state_size: Tuple = None):
        """
        Replay buffer that saves both observation and state.
        :param capacity:
        :param rng:
        """

        self.capacity = capacity
        self.rng = rng
        self.state_size = state_size
        self.obs_size = obs_size

        # TODO: change this to half precision to save GPU memory.
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))
        if obs_dtype is not None:
            self.obs = np.zeros((self.capacity, *self.obs_size), dtype=obs_dtype)
        else:
            self.obs = np.zeros((self.capacity, *self.obs_size))
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.next_a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._filled = False

        # We have the -1 here b/c we write next_obs as well.
        self.eligible_idxes = deque(maxlen=self.capacity - 1)

    def reset(self):
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))
        self.obs = np.zeros((self.capacity, *self.obs_size))
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.next_a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self._cursor = 0
        self._filled = False

    def push(self, batch: Batch):
        next_cursor = (self._cursor + 1) % self.capacity

        self.a[self._cursor] = batch.action
        if self.state_size is not None and batch.state is not None and batch.next_state is not None:
            self.s[self._cursor] = batch.state
            self.s[next_cursor] = batch.next_state

        self.obs[self._cursor] = batch.obs
        self.d[self._cursor] = batch.done
        self.r[self._cursor] = batch.reward
        if batch.next_action is not None:
            self.next_a[self._cursor] = batch.next_action

        self.obs[next_cursor] = batch.next_obs

        self.eligible_idxes.append(self._cursor)
        self._cursor = next_cursor

    def __len__(self):
        return len(self.eligible_idxes)

    def sample(self, batch_size: int, **kwargs) -> Batch:
        """
        NOTE: If done is True, then the next state returned is either all
        0s or the state from the start of the next episode. Either way
        it shouldn't matter for your target calculation.
        :param batch_size:
        :return:
        """
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
        batch['next_action'] = self.next_a[sample_idx]
        batch['done'] = self.d[sample_idx]
        batch['reward'] = self.r[sample_idx]
        batch['indices'] = sample_idx

        return Batch(**batch)
