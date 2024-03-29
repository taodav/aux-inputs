import numpy as np
from jax import random, jit
from typing import Tuple, Union

from unc.utils import Batch, sample_idx_batch, sample_seq_idxes


class ReplayBuffer:
    def __init__(self, capacity: int, rand_key: random.PRNGKey,
                 obs_size: Tuple,
                 obs_dtype: type,
                 state_size: Tuple = None,
                 prediction_size: int = 0):
        """
        Replay buffer that saves both observation and state.
        :param capacity:
        :param rng:
        """

        self.capacity = capacity
        self.rand_key = rand_key
        self.state_size = state_size
        self.prediction_size = prediction_size
        self.obs_size = obs_size

        # TODO: change these to half precision to save GPU memory.
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))

        if self.prediction_size > 0:
            self.predictions = np.zeros((self.capacity, self.prediction_size))

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
        self.eligible_idxes = np.zeros(self.capacity - 1, dtype=int)
        self.n_eligible_idxes = 0
        self.ei_cursor = 0
        self.jitted_sampled_idx_batch = jit(sample_idx_batch, static_argnums=0)

    def reset(self):
        if self.state_size is not None:
            self.s = np.zeros((self.capacity, *self.state_size))
        if self.prediction_size > 0:
            self.predictions = np.zeros((self.capacity, self.prediction_size))

        self.obs = np.zeros((self.capacity, *self.obs_size))
        self.a = np.zeros(self.capacity, dtype=np.int16)
        self.next_a = np.zeros(self.capacity, dtype=np.int16)
        self.r = np.zeros(self.capacity, dtype=np.float)
        self.d = np.zeros(self.capacity, dtype=bool)

        self.ei_cursor = 0
        self._cursor = 0
        self._filled = False

    def push(self, batch: Batch):
        next_cursor = (self._cursor + 1) % self.capacity
        next_ei_cursor = (self.ei_cursor + 1) % (self.capacity - 1)

        self.a[self._cursor] = batch.action
        if self.state_size is not None and batch.state is not None and batch.next_state is not None:
            self.s[self._cursor] = batch.state
            self.s[next_cursor] = batch.next_state

        if self.prediction_size > 0 and batch.predictions is not None and batch.next_predictions is not None:
            self.predictions[self._cursor] = batch.predictions
            self.predictions[next_cursor] = batch.predictions

        self.obs[self._cursor] = batch.obs
        self.d[self._cursor] = batch.done
        self.r[self._cursor] = batch.reward
        if batch.next_action is not None:
            self.next_a[self._cursor] = batch.next_action

        self.obs[next_cursor] = batch.next_obs

        self.eligible_idxes[self.ei_cursor] = self._cursor
        self.n_eligible_idxes = min(self.capacity - 1, self.n_eligible_idxes + 1)
        self.ei_cursor = next_ei_cursor
        self._cursor = next_cursor

    def __len__(self):
        return len(self.eligible_idxes)

    def sample_eligible_idxes(self, batch_size: int):
        length = self.n_eligible_idxes
        idxes, self.rand_key = self.jitted_sampled_idx_batch(batch_size, length, self.rand_key)
        return self.eligible_idxes[idxes]

    def sample(self, batch_size: int, **kwargs) -> Batch:
        """
        NOTE: If done is True, then the next state returned is either all
        0s or the state from the start of the next episode. Either way
        it shouldn't matter for your target calculation.
        :param batch_size:
        :return:
        """

        sample_idx = self.sample_eligible_idxes(batch_size)
        batch = {}
        if self.state_size is not None:
            batch['state'] = self.s[sample_idx]
            batch['next_state'] = self.s[(sample_idx + 1) % self.capacity]

        if self.prediction_size > 0:
            batch['predictions'] = self.predictions[sample_idx]
            batch['next_predictions'] = self.predictions[(sample_idx + 1) % self.capacity]

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
    def __init__(self, capacity: int, rand_key: random.PRNGKey,
                 obs_size: Tuple, obs_dtype: type, state_size: Tuple = None):
        super(EpisodeBuffer, self).__init__(capacity, rand_key, obs_size, obs_dtype, state_size=state_size)
        self.jitted_sampled_seq_idxes = jit(sample_seq_idxes, static_argnums=(0, 1, 2))
        self.end = np.zeros_like(self.d, dtype=bool)

    def push(self, batch: Batch):
        self.end[self._cursor] = batch.end
        super(EpisodeBuffer, self).push(batch)

    def sample_eligible_idxes(self, batch_size: int, seq_len: int) -> np.ndarray:
        length = self.n_eligible_idxes
        sampled_eligible_idxes, self.rand_key = self.jitted_sampled_seq_idxes(batch_size, self.capacity, seq_len, length,
                                                                  self.eligible_idxes, self.rand_key)
        return sampled_eligible_idxes

    def sample(self, batch_size: int, seq_len: int = 1, as_dict: bool = False) -> Union[Batch, dict]:
        sample_idx = self.sample_eligible_idxes(batch_size, seq_len)

        batch = {}
        batch['state'] = self.s[sample_idx]
        batch['next_state'] = self.s[(sample_idx + 1) % self.capacity]
        batch['obs'] = self.obs[sample_idx]
        batch['next_obs'] = self.obs[(sample_idx + 1) % self.capacity]
        if hasattr(self, 'predictions') and self.predictions is not None:
            batch['predictions'] = self.predictions[sample_idx]
            batch['next_predictions'] = self.predictions[(sample_idx + 1) % self.capacity]

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
        ys_cursor, xs_cursor = np.nonzero(np.array(sample_idx) == self.eligible_idxes[self.ei_cursor])

        ys, xs = np.concatenate([ys_cursor, ys_ends]), np.concatenate([xs_cursor, xs_ends])

        if ys.shape[0] > 0:
            for y, x in zip(ys, xs):
                zero_mask[y, x + 1:] = 0

        batch['zero_mask'] = zero_mask

        if as_dict:
            return batch

        return Batch(**batch)

    def sample_k(self, batch_size: int, seq_len: int = 1, k: int = 1):
        batch = self.sample(batch_size * k, seq_len=seq_len, as_dict=True)
        for key, arr in batch.items():
            batch[key] = np.stack(np.split(arr, k, axis=0))

        return Batch(**batch)
