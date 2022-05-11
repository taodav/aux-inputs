import numpy as np
import time as tm
from jax import random
from typing import Tuple, Union
from collections import deque

from unc.envs import get_env
from unc.utils.data import Batch
from unc.utils.replay import ReplayBuffer, EpisodeBuffer
from unc.args import Args


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


class OldEpisodeBuffer(OldReplayBuffer):
    """
    For episode buffer, we return zero-padded batches back.
    We have to save "end" instead of done, to track if either an episode is finished
    or we reach the max number of steps
    How zero-padded batches work in our case is that the "done" tensor
    is essentially a mask for
    """

    def __init__(self, capacity: int, rng: np.random.RandomState,
                 obs_size: Tuple, obs_dtype: type, state_size: Tuple = None):
        super(OldEpisodeBuffer, self).__init__(capacity, rng, obs_size, obs_dtype, state_size=state_size)
        self.end = np.zeros_like(self.d, dtype=bool)

    def push(self, batch: Batch):
        self.end[self._cursor] = batch.end
        super(OldEpisodeBuffer, self).push(batch)

    def sample(self, batch_size: int, seq_len: int = 1, as_dict: bool = False) -> Union[Batch, dict]:
        sample_idx = self.rng.choice(self.eligible_idxes, size=batch_size)
        sample_idx = (sample_idx + np.arange(seq_len)[:, None]).T % self.capacity

        batch = {}
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

        if as_dict:
            return batch

        return Batch(**batch)

    def sample_k(self, batch_size: int, seq_len: int = 1, k: int = 1):
        batch = self.sample(batch_size * k, seq_len=seq_len, as_dict=True)
        for key, arr in batch.items():
            batch[key] = np.stack(np.split(arr, k, axis=0))

        return Batch(**batch)


if __name__ == "__main__":
    seed = 2020
    buffer_size = 40
    iterations = 20000

    parser = Args()
    args = parser.parse_args()
    args.seed = seed
    args.buffer_size = buffer_size
    args.env = "uf2a"
    args.batch_size = 32
    args.n_hidden = 10
    args.trunc = 5

    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)

    env = get_env(rng, rand_key, args)
    new_buffer = ReplayBuffer(args.buffer_size, rand_key, (1, ),
                              obs_dtype=env.observation_space.low.dtype)

    new_epi_buffer = EpisodeBuffer(args.buffer_size, rand_key, (1, ),
                                obs_dtype=env.observation_space.low.dtype,
                                state_size=(2, args.n_hidden))

    old_buffer = OldReplayBuffer(args.buffer_size, rng, (1, ),
                                  obs_dtype=env.observation_space.low.dtype)

    old_epi_buffer = OldEpisodeBuffer(args.buffer_size, rng, (1, ),
                                   obs_dtype=env.observation_space.low.dtype,
                                   state_size=(2, args.n_hidden))

    obs = np.array([0])
    state = rng.randint(0, 10, size=(2, args.n_hidden))
    offset = 40
    to_add = buffer_size + offset
    for i in range(1, to_add + 1):
        next_obs = np.array([i])
        next_state = rng.randint(0, 10, size=(2, args.n_hidden))
        sample = Batch(obs=obs, reward=0, next_obs=next_obs, action=0, done=i % 10 == 0,
                       next_action=0, state=state, next_state=next_state, end=i % 10 == 0)

        new_buffer.push(sample)
        old_buffer.push(sample)
        new_epi_buffer.push(sample)
        old_epi_buffer.push(sample)

        obs = next_obs
        state = next_state

    print(f"Pushed {to_add} samples to all buffers")

    new_buff_sample_counts = np.zeros(args.buffer_size)
    old_buff_sample_counts = np.zeros(args.buffer_size)
    new_epi_buff_sample_counts = np.zeros(args.buffer_size + 1)
    old_epi_buff_sample_counts = np.zeros(args.buffer_size + 1)

    new_seq_len = 2

    for _ in range(iterations):
        old_b = old_buffer.sample(args.batch_size)
        old_buff_sample_counts[old_b.obs.flatten().astype(int) - offset] += 1

        new_b = new_buffer.sample(args.batch_size)
        new_buff_sample_counts[new_b.obs.flatten().astype(int) - offset] += 1

        old_epi_b = old_epi_buffer.sample(args.batch_size, seq_len=new_seq_len)
        old_epi_buff_sample_counts[old_epi_b.obs.flatten().astype(int) - offset] += 1

        new_epi_b = new_epi_buffer.sample(args.batch_size, seq_len=new_seq_len)
        new_epi_buff_sample_counts[new_epi_b.obs.flatten().astype(int) - offset] += 1

    total_samples_buff = args.batch_size * iterations
    total_samples_epi_buff = total_samples_buff * new_seq_len

    norm_old_buff = old_buff_sample_counts / old_buff_sample_counts.sum()
    norm_new_buff = new_buff_sample_counts / new_buff_sample_counts.sum()

    norm_old_epi_buff = old_epi_buff_sample_counts / old_epi_buff_sample_counts.sum()
    norm_new_epi_buff = new_epi_buff_sample_counts / new_epi_buff_sample_counts.sum()

    print(f"Sampled {iterations}. Checking ratio.")
