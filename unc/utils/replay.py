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
        self._age = 0
        self._filled = False

        # We do this b/c cursor + 1, cursor + 2, ..., cursor + stack - 1
        # cannot be sampled.
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
        self._age = 0
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

        return Batch(**batch)


class Episode:
    def __init__(self, rng: np.random.RandomState):
        self.obs = []
        self.action = []
        self.next_action = []
        self.done = []
        self.reward = []
        self.current_obs = None
        self.rng = rng

    def push(self, batch: Batch):
        self.obs.append(batch.obs)
        self.action.append(batch.action)
        if batch.next_action is not None:
            self.next_action.append(batch.next_action)
        self.done.append(batch.done)
        self.reward.append(batch.reward)
        self.current_obs = batch.next_obs

    def __len__(self):
        return len(self.obs)

    def sample(self, seq_len: int) -> Batch:
        """
        Sample a sequence of experience of length seq_len
        :param seq_len: Length to sample. If the sequence we sample can't be this long,
        we do zero-padding.
        :return:
        """
        start = self.rng.choice(np.arange(len(self)))
        end = min(start + seq_len, len(self) - 1)

        obs = np.stack(self.obs[start:end])
        action = np.stack(self.action[start:end])

        next_action, next_action_pad = None, None
        if len(self.next_action) > 0:
            next_action = np.stack(self.next_action[start:end])
            next_action_pad = np.zeros((seq_len, *next_action.shape[1:]))

        done = np.stack(self.done[start:end])
        reward = np.stack(self.reward[start:end])

        batch = Batch(
            obs=np.zeros((seq_len, *obs.shape[1:])),
            action=np.zeros((seq_len, *action.shape[1:])),
            next_obs=np.zeros((seq_len, *obs.shape[1:])),
            next_action=next_action_pad,
            done=np.zeros((seq_len, *done.shape[1:])),
            reward=np.zeros((seq_len, *reward.shape[1:])),
        )

        batch.obs[:obs.shape[0]] = obs
        batch.action[:action.shape[0]] = action
        if next_action_pad is not None:
            batch.next_action[:next_action.shape[0]] = next_action

        batch.next_obs[:action.shape[0]] = action
        batch.done[:done.shape[0]] = done
        batch.reward[:reward.shape[0]] = reward

        return batch


# class EpisodeReplayBuffer(ReplayBuffer):
#     def __init__(self, capacity: int, rng: np.random.RandomState,
#                  obs_size: Tuple, state_size: Tuple = None):
#         self.capacity = capacity
#         self.rng = rng
#         self.state_size = state_size
#         self.obs_size = obs_size
# 
#         self.curr_episode = Episode(self.rng)
#         self.episodes = [self.curr_episode]
# 
#     def __len__(self):
#         return sum(len(ep) for ep in self.episodes)
# 
#     def reset(self):
#         self.curr_episode = Episode(self.rng)
#         self.episodes = [self.curr_episode]
# 
#     def push(self, batch: Batch):
#         self.curr_episode.push(batch)
# 
#         if batch.done.item():
#             self.curr_episode = Episode(self.rng)
#             self.episodes.append(self.curr_episode)
# 
#     def sample(self, batch_size: int, seq_len: int) -> Batch:
#


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

    # def get_current_episode(self):


    def push(self, batch: Batch):
        self.end[self._cursor] = batch.end
        super(EpisodeBuffer, self).push(batch)

    def sample(self, batch_size: int, seq_len: int = 1) -> Batch:
        if len(self.eligible_idxes) < batch_size:
            batch_size = len(self.eligible_idxes)

        sample_idx = self.rng.choice(self.eligible_idxes, size=batch_size)
        sample_idx = np.minimum(sample_idx + np.arange(seq_len)[:, None], self.eligible_idxes[-1]).T

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
        ends = self.end[sample_idx]

        # Zero mask is essentially mask where we only learn if we're still within an episode.
        # To do this, we set everything AFTER done == True as 0.
        zero_mask = np.ones_like(ends)
        ys, xs = ends.nonzero()
        if ys.shape[0] > 0:
            for y, x in zip(ys, xs):
                zero_mask[y, x + 1:] = 0

        batch['zero_mask'] = zero_mask

        return Batch(**batch)

