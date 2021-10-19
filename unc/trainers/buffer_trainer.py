import numpy as np
from typing import Union, List
from time import time, ctime

from unc.args import Args
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper
from unc.agents import Agent, LSTMAgent
from unc.utils import ReplayBuffer, EpisodeBuffer
from unc.utils.data import Batch, zip_batches

from .trainer import Trainer

# DEBUGGING
from unc.utils import plot_arr
from unc.debug import summarize_checks, all_unchecked_rock_q_vals


class BufferTrainer(Trainer):
    def __init__(self, args: Args, agent: Agent, env: Union[RockSample, RockSampleWrapper],
                 prefilled_buffer: ReplayBuffer = None):
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
        super(BufferTrainer, self).__init__(args, agent, env)

        self.batch_size = args.batch_size

        if args.arch == 'lstm' and isinstance(self.agent, LSTMAgent):
            # We save state for an LSTM agent
            self.buffer = EpisodeBuffer(args.buffer_size, self.env.rng, self.env.observation_space.shape,
                                        state_size=self.agent.state_shape)
        else:
            self.buffer = ReplayBuffer(args.buffer_size, self.env.rng, self.env.observation_space.shape)
        self.prefilled_buffer = prefilled_buffer

        self.p_prefilled = args.p_prefilled

        # Do we save our agent hidden state?
        self.save_hidden = args.arch == 'lstm' and hasattr(self.agent, 'hidden_state')


    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # DEBUGGING
        time_to_check = False
        good_diffs = []
        bad_diffs = []
        moralities = []

        # Timing stuff
        prev_time = time_start
        log_interval = 1000
        total_target_updates = self.total_steps // log_interval
        num_logs = 0
        avg_time_per_log = 0


        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            # LSTM hidden state
            hs = None
            next_hs = None

            obs = np.expand_dims(self.env.reset(), 0)
            self.agent.reset()
            if self.save_hidden:
                hs = self.agent.state

            action = self.agent.act(obs).item()

            # DEBUGGING
            checked_rocks_info = {}

            # DEBUGGING: if we check a rock that's never been sampled before
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
                next_obs = np.array([next_obs])

                next_action = self.agent.act(next_obs).item()

                sample = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                               next_action=next_action, state=hs, next_state=next_hs, end=done or (t == self.max_episode_steps - 1))
                self.buffer.push(sample)

                self.info['reward'].append(reward)

                prefilled_bs = int(self.batch_size * self.p_prefilled)
                online_bs = self.batch_size - prefilled_bs

                if online_bs < len(self.buffer):
                    trunc = 0 if not hasattr(self.agent, 'trunc') else self.agent.trunc
                    sample = self.buffer.sample(online_bs, seq_len=trunc)

                    # If we also sample from a prefilled buffer
                    if self.p_prefilled > 0 and self.prefilled_buffer is not None:
                        sample = zip_batches(sample, self.prefilled_buffer.sample(prefilled_bs, seq_len=trunc))

                    sample.gamma = (1 - sample.done) * self.discounting

                    loss = self.agent.update(sample)

                    # Logging
                    episode_loss += loss
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

                if done:
                    break

                obs = next_obs
                action = next_action
                hs = next_hs

            # FOR DEBUGGING
            # print(f"Q-values at end of episode: {self.agent.curr_q}")

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t)

        # DEBUGGING
        self.info['moralities'] = moralities
        self.info['good_diffs'] = good_diffs
        self.info['bad_diffs'] = bad_diffs

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
