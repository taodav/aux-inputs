import numpy as np
from typing import Union, List
from time import time, ctime

from unc.args import Args
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper
from unc.agents import Agent, LSTMAgent
from unc.utils import ReplayBuffer, EpisodeBuffer
from unc.utils.data import Batch, zip_batches, get_action_encoding

from .trainer import Trainer

# DEBUGGING
from unc.debug import summarize_checks, all_unchecked_rock_q_vals


class BufferTrainer(Trainer):
    def __init__(self, args: Args, agent: Agent, env: Union[RockSample, RockSampleWrapper],
                 test_env: Union[RockSample, RockSampleWrapper], prefilled_buffer: ReplayBuffer = None):
        """
        Double buffer trainer. Essentially Sarsa except with two experience replay buffers.

        The first experience replay buffer is pre-filled with experience from some policy.
        In this case, the policy is the rock sampler agent.

        The second experience replay buffer is for the agent's own experiences.
        :param args: arguments
        :param agent: agent to train
        :param env: environment to train on (currently only supports rocksample)
        :param prefilled_buffer: buffer pre-filled from a certain policy
        """
        super(BufferTrainer, self).__init__(args, agent, env, test_env)

        self.batch_size = args.batch_size
        self.arch = args.arch
        self.action_cond = args.action_cond
        self.n_actions = self.env.action_space.n

        if 'lstm' in self.arch and isinstance(self.agent, LSTMAgent):
            # We save state for an LSTM agent
            obs_shape = self.env.observation_space.shape
            if self.action_cond == 'cat':
                obs_shape = obs_shape[:-1] + (obs_shape[-1] + self.n_actions,)

            self.buffer = EpisodeBuffer(args.buffer_size, self.env.rng, obs_shape,
                                        obs_dtype=self.env.observation_space.low.dtype,
                                        state_size=self.agent.state_shape)
        else:
            self.buffer = ReplayBuffer(args.buffer_size, self.env.rng, self.env.observation_space.shape,
                                       obs_dtype=self.env.observation_space.low.dtype)
        self.prefilled_buffer = prefilled_buffer

        self.p_prefilled = args.p_prefilled

        # Do we save our agent hidden state?
        self.save_hidden = 'lstm' in args.arch and hasattr(self.agent, 'hidden_state')

    def update_buffer_hidden(self, sample_idxs: np.ndarray, other_info: dict,
                             update_mask: np.ndarray = None):
        """
        [RNNS] update our buffer hidden states.
        :param sample_idxs: sample indices, of shape [batch x seq_len x *hidden_state_shape]
        :param other_info: this is populated with updated hidden states, depending on what kind of er updates we make.
        :return:
        """
        er_hidden_update = self.agent.er_hidden_update
        if er_hidden_update == "grad":
            # For grad, we only update the initial hidden state.
            first_hs = other_info['first_hidden_state']  # hk.LSTMState, cell and hidden both batch_size x n_hidden
            idxs_to_update = sample_idxs[:, 0]
            self.buffer.s[idxs_to_update] = np.stack([first_hs.hidden, first_hs.cell], axis=1)
        elif er_hidden_update == "update":
            """
            TODO: For this update, the hidden states we get is for the next state.
            BE SURE to get the proper indices for all next states,
            and also be sure to mask correctly with update_mask
            """
            next_hidden_states = other_info['next_hidden_states']
            body_to_update = sample_idxs[:, 1:]
            tail_to_update = (sample_idxs[:, -1] + 1) % self.buffer.capacity
            idxs_to_update = np.concatenate([body_to_update, tail_to_update[:, None]], axis=1)

            body_mask = update_mask[:, 1:]
            tail_mask = update_mask[:, -1][:, None]  # Assume that the ends continue
            next_mask = np.concatenate([body_mask, tail_mask], axis=1)
            self.buffer.s[idxs_to_update][next_mask.astype(bool)] = next_hidden_states[next_mask.astype(bool)]

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # Timing stuff
        prev_time = time_start
        log_interval = 1000
        total_target_updates = self.total_steps // log_interval
        num_logs = 0
        avg_time_per_log = 0

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0
            episode_info = {}

            # LSTM hidden state
            hs = None
            next_hs = None
            obs = self.env.reset()

            # Action conditioning
            if self.action_cond == 'cat':
                action_encoding = get_action_encoding(self.agent.features_shape, -1, self.n_actions)
                obs = np.concatenate([obs, action_encoding], axis=-1)

            obs = np.expand_dims(obs, 0)
            self.agent.reset()
            if self.save_hidden:
                hs = self.agent.state

            action = self.agent.act(obs).item()

            # DEBUGGING: if we check a rock that's never been sampled before
            # checked_rocks_info = {}
            # if time_to_check and 'p' in self.args.env:
            #     unchecked_q_vals = all_unchecked_rock_q_vals(self.env, self.agent, self.env.checked_rocks)
            #     checked_rocks_info[self.num_steps] = unchecked_q_vals
            #     all_mor, all_good_diff, all_bad_diff = summarize_checks(self.env, unchecked_q_vals)
            #     moralities.append(all_mor)
            #     good_diffs.append(all_good_diff)
            #     bad_diffs.append(all_bad_diff)
            #     time_to_check = False

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                if self.save_hidden:
                    next_hs = self.agent.state

                next_obs, reward, done, info = self.env.step(action)

                # Action conditioning
                if self.action_cond == 'cat':
                    action_encoding = get_action_encoding(self.agent.features_shape, action, self.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                next_obs = np.array([next_obs])

                next_action = self.agent.act(next_obs).item()

                sample = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                               next_action=next_action, state=hs, next_state=next_hs, end=done or (t == self.max_episode_steps - 1))
                self.buffer.push(sample)

                self.info['reward'].append(reward)

                prefilled_bs = int(self.batch_size * self.p_prefilled)
                online_bs = self.batch_size - prefilled_bs

                if online_bs <= len(self.buffer):
                    trunc = 0 if not hasattr(self.agent, 'trunc') else self.agent.trunc
                    sample = self.buffer.sample(online_bs, seq_len=trunc)

                    # If we also sample from a prefilled buffer
                    if self.p_prefilled > 0 and self.prefilled_buffer is not None:
                        sample = zip_batches(sample, self.prefilled_buffer.sample(prefilled_bs, seq_len=trunc))

                    sample.gamma = (1 - sample.done) * self.discounting

                    loss, other_info = self.agent.update(sample)

                    if self.save_hidden:
                        self.update_buffer_hidden(sample.indices, other_info, update_mask=sample.zero_mask)

                    # Logging
                    episode_loss += loss
                    if other_info:
                        for k, v in other_info.items():
                            if k not in episode_info:
                                episode_info[k] = v
                            else:
                                episode_info[k] += v

                episode_reward += reward
                self.num_steps += 1

                # Logging and timing
                if self.num_steps % log_interval == 0:
                    time_to_check = True

                    num_logs += 1
                    curr_time = time()
                    time_per_fix_freq = curr_time - prev_time
                    avg_time_per_log += (1 / num_logs) * (time_per_fix_freq - avg_time_per_log)
                    time_remaining = (total_target_updates - num_logs) * avg_time_per_log
                    self._print(f"Remaining time: {time_remaining / 60:.2f}")
                    prev_time = curr_time

                # Offline evaluation
                if self.offline_eval_freq > 0 and self.num_steps % self.offline_eval_freq == 0:
                    self.offline_evaluation()

                if done:
                    break

                obs = next_obs
                action = next_action
                hs = next_hs

            # FOR DEBUGGING
            # print()
            # print(f"Q-values at end of episode: {self.agent.curr_q}")
            # print(f"RNN Q-values at end of episode: {self.agent.rnn_curr_q}")

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t,
                                    additional_info=episode_info)

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
