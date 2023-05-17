import numpy as np
from time import time, ctime
from jax import random
from pathlib import Path

from unc.args import Args
from unc.envs import Environment
from unc.agents import Agent
from unc.utils import ReplayBuffer
from unc.utils.data import Batch, get_action_encoding
from unc.gvfs import GeneralValueFunction

from unc.trainers.buffer_trainer import BufferTrainer


class BufferMCTrainer(BufferTrainer):
    def __init__(self, args: Args, agent: Agent,
                 env: Environment,
                 test_env: Environment,
                 rand_key: random.PRNGKey,
                 checkpoint_dir: Path = None,
                 prefilled_buffer: ReplayBuffer = None,
                 gvf: GeneralValueFunction = None):
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
        self.rand_key, buffer_key = random.split(rand_key)
        super(BufferMCTrainer, self).__init__(args, agent, env, test_env, buffer_key, checkpoint_dir=checkpoint_dir,
                                              prefilled_buffer=prefilled_buffer, gvf=gvf)
        self.reward_scale = args.reward_scale
        self.gamma_terminal = args.gamma_terminal

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # Timing stuff
        prev_time = time_start
        log_interval = 1000 if self.offline_eval_freq == 0 else self.offline_eval_freq
        total_target_updates = self.total_steps // log_interval
        num_logs = 0
        avg_time_per_log = 0

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0
            episode_info = {}
            episode_rewards = []
            episode_batches = []

            checkpoint_after_ep = False
            done = False

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

            t = 0
            while not done:
                self.agent.set_eps(self.get_epsilon())

                if self.save_hidden:
                    next_hs = self.agent.state

                next_obs, reward, done, info = self.env.step(action)
                t += 1
                if self.gamma_terminal:
                    gamma_term_key, self.rand_key = random.split(self.rand_key)
                    done = done or (random.uniform(gamma_term_key, minval=0, maxval=1) > self.discounting)

                # Action conditioning
                if self.action_cond == 'cat':
                    action_encoding = get_action_encoding(self.agent.features_shape, action, self.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                next_obs = np.array([next_obs])

                next_action = self.agent.act(next_obs).item()

                batch = Batch(obs=obs, reward=reward, next_obs=next_obs, action=action, done=done,
                               next_action=next_action, state=hs, next_state=next_hs,
                               end=done)

                episode_batches.append(batch)
                # self.buffer.push(batch)

                self.info['reward'].append(reward)

                if self.batch_size <= len(self.buffer):
                    trunc = 0 if not hasattr(self.agent, 'trunc') else self.agent.trunc
                    sample = self.buffer.sample(self.batch_size, seq_len=trunc)

                    sample.gamma = (1 - sample.done)
                    if not self.gamma_terminal:
                        sample.gamma *= self.discounting

                    loss, other_info = self.agent.update(sample)

                    # if self.save_hidden:
                    #     self.update_buffer_hidden(sample.indices, other_info, update_mask=sample.zero_mask)

                    # Logging
                    episode_loss += loss
                    if other_info:
                        for k, v in other_info.items():
                            if k not in episode_info:
                                episode_info[k] = v
                            else:
                                episode_info[k] += v

                episode_reward += reward
                episode_rewards.append(reward)

                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq > 0 and self.num_steps % self.offline_eval_freq == 0:
                    before_eval_time = time()
                    self.offline_evaluation()
                    self.info['eval_time'] += time() - before_eval_time

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

                if self.checkpoint_freq > 0 and self.num_steps % self.checkpoint_freq == 0:
                    checkpoint_after_ep = True

                if done or (not self.gamma_terminal and t > self.max_episode_steps):
                    break

                obs = next_obs
                action = next_action
                hs = next_hs

            episode_rewards = np.array(episode_rewards)
            if self.reward_scale != 1.:
                episode_rewards *= self.reward_scale

            if not self.gamma_terminal:
                discounts = self.discounting ** np.arange(len(episode_rewards))
                episode_rewards = episode_rewards * discounts

            returns = np.cumsum(episode_rewards[::-1])[::-1]

            for b, ret in zip(episode_batches, returns):
                b.returns = ret
                self.buffer.push(b)

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t,
                                    additional_info=episode_info)

            if checkpoint_after_ep:
                self.checkpoint()

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
        self.info['total_time'] = time_end - time_start
