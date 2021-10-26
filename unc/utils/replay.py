import numpy as np
from typing import Tuple
from collections import deque

from unc.utils import Batch


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


class EpisodeBuffer(ReplayBuffer):
    """
    For episode buffer, we return zero-padded batches back.

    We have to save "end" instead of done, to track if either an episode is finished
    or we reach the max number of steps

    How zero-padded batches work in our case is that the "done" tensor
    is essentially a mask for
    """
    def __init__(self, capacity: int, rng: np.random.RandomState,
                 obs_size: Tuple, state_size: Tuple = None):
        super(EpisodeBuffer, self).__init__(capacity, rng, obs_size, state_size=state_size)
        self.end = np.zeros_like(self.d, dtype=bool)

    def push(self, batch: Batch):
        self.end[self._cursor] = batch.end
        super(EpisodeBuffer, self).push(batch)

    def sample(self, batch_size: int, seq_len: int = 1) -> Batch:
        # TODO: Fix bug where last seq_len experiences don't have correct "ends"
        if len(self.eligible_idxes) < batch_size:
            batch_size = len(self.eligible_idxes)

        sample_idx = self.rng.choice(self.eligible_idxes, size=batch_size)
        sample_idx = (sample_idx + np.arange(seq_len)[:, None]).T % self.capacity

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
        ends = self.end[sample_idx]

        # Zero mask is essentially mask where we only learn if we're still within an episode.
        # To do this, we set everything AFTER done == True as 0, and any episode that ends
        # after max steps. The array ends accomplishes this.
        zero_mask = np.ones_like(ends)
        ys_ends, xs_ends = ends.nonzero()

        # Also, we don't want any experience beyond our current cursor.
        ys_cursor, xs_cursor = np.nonzero(sample_idx == self.eligible_idxes[-1])

        ys, xs = np.concatenate([ys_cursor, ys_ends]), np.concatenate([xs_cursor, xs_ends])

        if ys.shape[0] > 0:
            for y, x in zip(ys, xs):
                zero_mask[y, x + 1:] = 0

        batch['zero_mask'] = zero_mask

        return Batch(**batch)

