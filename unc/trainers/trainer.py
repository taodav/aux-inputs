import gym
import logging
import dill
from time import time, ctime
import numpy as np
from typing import List, Any, Tuple, Union
from pathlib import Path
from collections import deque

from unc.args import Args
from unc.agents import Agent
from unc.utils.data import Batch, preprocess_step, get_action_encoding
from unc.utils.gvfs import GeneralValueFunction
from unc.eval import test_episodes


class Trainer:
    def __init__(self, args: Args, agent: Agent,
                 env: Union[gym.Env, gym.Wrapper],
                 test_env: Union[gym.Env, gym.Wrapper],
                 checkpoint_dir: Path = None,
                 gvf: GeneralValueFunction = None):
        self.args = args
        self.discounting = args.discounting

        self.epsilon = args.epsilon
        self.anneal_steps = args.anneal_steps
        self.epsilon_start = args.epsilon_start
        self.anneal_value = (self.epsilon_start - self.epsilon) / self.anneal_steps if self.anneal_steps > 0 else 0

        self.max_episode_steps = args.max_episode_steps

        self.agent = agent
        self.env = env
        self.n_actions = env.action_space.n

        # For LSTMS
        self.action_cond = args.action_cond

        # For GVFAgents
        self.gvf = gvf

        self.total_steps = args.total_steps
        self.test_env = test_env
        self.offline_eval_freq = args.offline_eval_freq
        self.test_eps = args.test_eps
        self.test_episodes = args.test_episodes
        self.checkpoint_freq = args.checkpoint_freq
        self.checkpoint_dir = checkpoint_dir
        self.save_all_checkpoints = args.save_all_checkpoints

        self.episode_num = 0
        self.num_steps = 0

        self.info = None
        self.trunc = args.trunc if 'lstm' in args.arch else 0

        logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

    def reset(self):
        """
        Reset all logging info.
        :return:
        """
        self.info = {
            'episode_reward': [],
            'episode_length': [],
            'reward': [],
            'avg_episode_loss': [],
            'total_time': 0,  # total time for training
            'eval_time': 0,  # total time spent on eval
            'args': self.args.as_dict()
        }
        if 'p' in self.args.env:
            self.info['pf_episodic_mean'] = []
            self.info['pf_episodic_var'] = []

        if self.offline_eval_freq > 0:
            self.info['offline_eval_returns'] = []
            self.info['offline_eval_discounted_returns'] = []

        self.num_steps = 0

    def get_epsilon(self):
        """
        Get the (potentially) epsilon annealed epsilon.
        :return: epsilon to set
        """
        epsilon = self.epsilon
        if self.anneal_steps > 0 and self.anneal_steps > self.num_steps:
            epsilon = self.epsilon_start - self.anneal_value * self.num_steps

        return epsilon

    def checkpoint(self):
        if self.checkpoint_dir is None:
            print("Can't save checkpoint with no path!")
            return

        checkpoint_path = self.checkpoint_dir / f"{self.num_steps}.pkl"
        with open(checkpoint_path, "wb") as f:
            dill.dump(self, f)

        print(f"Saved checkpoint to {checkpoint_path}")

        # we do this check AFTER in case our job time limit ends
        # as we're saving.
        if not self.save_all_checkpoints:
            for f in self.checkpoint_dir.iterdir():
                if f != checkpoint_path:
                    f.unlink()

    @staticmethod
    def load_checkpoint(checkpoint_path: Path):
        with open(checkpoint_path, "rb") as f:
            trainer = dill.load(f)

        return trainer

    def collect_rnn_batch(self, b: Batch, hs: np.ndarray, next_hs: np.ndarray, trunc_batch: Batch) -> Tuple[Batch, Batch]:
        trunc_batch.obs.append(b.obs), trunc_batch.action.append(b.action), trunc_batch.next_obs.append(b.next_obs)
        trunc_batch.gamma.append(b.gamma), trunc_batch.reward.append(b.reward), trunc_batch.next_action.append(b.next_action)
        trunc_batch.state.append(hs), trunc_batch.next_state.append(next_hs)

        gammas = np.concatenate(trunc_batch.gamma)[None, :]
        batch = Batch(obs=np.concatenate(trunc_batch.obs)[None, :],
                      action=np.concatenate(trunc_batch.action)[None, :],
                      next_obs=np.concatenate(trunc_batch.next_obs)[None, :],
                      gamma=gammas,
                      reward=np.concatenate(trunc_batch.reward)[None, :],
                      next_action=np.concatenate(trunc_batch.next_action)[None, :],
                      state=np.concatenate(trunc_batch.state)[None, :],
                      next_state=np.concatenate(trunc_batch.next_state)[None, :],
                      zero_mask=np.ones_like(gammas))
        return batch, trunc_batch

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

            checkpoint_after_ep = False
            self.agent.reset()

            # For RNN training
            trunc_batch = None
            if self.trunc > 0:
                trunc_batch = Batch(obs=deque(maxlen=self.trunc), action=deque(maxlen=self.trunc),
                                    next_obs=deque(maxlen=self.trunc), gamma=deque(maxlen=self.trunc),
                                    reward=deque(maxlen=self.trunc), next_action=deque(maxlen=self.trunc),
                                    state=deque(maxlen=self.trunc), next_state=deque(maxlen=self.trunc))

            # Cumulant predictions for GVF training
            if self.gvf is not None:
                self.env.predictions = self.agent.current_gvf_predictions[0]

            obs = self.env.reset()
            # Action conditioning
            if self.action_cond == 'cat':
                action_encoding = get_action_encoding(self.agent.features_shape, -1, self.n_actions)
                obs = np.concatenate([obs, action_encoding], axis=-1)

            obs = np.expand_dims(obs, 0)

            # Hidden state for RNN training
            hs, next_hs = None, None
            if self.trunc > 0:
                hs = self.agent.state[None, :]

            action = self.agent.act(obs)
            if self.agent.n_actions == 0:
                action = np.array([self.env.action_space.sample()])

            if self.gvf is not None:
                self.env.predictions = self.agent.current_gvf_predictions[0]

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                if self.trunc > 0:
                    next_hs = self.agent.state[None, :]

                next_obs, reward, done, info = self.env.step(action.item())

                # Action conditioning
                if self.action_cond == 'cat':
                    action_encoding = get_action_encoding(self.agent.features_shape, action, self.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                self.info['reward'].append(reward)

                # Preprocess everything for updating
                next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

                gamma = (1 - done) * self.discounting

                next_action = self.agent.act(next_obs)
                if self.agent.n_actions == 0:
                    next_action = np.array([self.env.action_space.sample()])

                batch = Batch(obs=obs, action=action, next_obs=next_obs,
                              gamma=gamma, reward=reward, next_action=next_action)

                if self.gvf is not None:
                    if self.agent.n_actions > 0:
                        greedy_action = np.argmax(self.agent.curr_q, axis=1)
                        current_pi = np.zeros(self.n_actions) + (self.agent.get_eps() / self.n_actions)
                        current_pi[greedy_action] += (1 - self.agent.get_eps())
                    else:
                        # prediction setting
                        current_pi = np.ones(self.n_actions) / self.n_actions
                    batch.impt_sampling_ratio = self.gvf.impt_sampling_ratio(batch.next_obs, current_pi)


                    self.env.predictions = self.agent.current_gvf_predictions[0]

                    batch.cumulants = self.gvf.cumulant(batch.obs)
                    batch.cumulant_terminations = self.gvf.termination(batch.obs)

                # This is for real-time RNN training
                if self.trunc > 0:
                    batch, trunc_batch = self.collect_rnn_batch(batch, hs, next_hs, trunc_batch)

                loss, other_info = self.agent.update(batch)

                # Logging
                episode_loss += loss
                episode_reward += reward[0]
                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq > 0 and self.num_steps % self.offline_eval_freq == 0:
                    self.offline_evaluation()

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

                if done.item():
                    break

                obs = next_obs
                action = next_action
                hs = next_hs

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t)

            if checkpoint_after_ep:
                self.checkpoint()

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")

    def offline_evaluation(self):
        _, eval_rews = test_episodes(self.agent, self.test_env, n_episodes=self.test_episodes,
                                     test_eps=self.test_eps, render=False,
                                     max_episode_steps=self.max_episode_steps)
        eval_returns = np.zeros(len(eval_rews))
        eval_discounted_returns = np.zeros_like(eval_returns)

        for i, rew in enumerate(eval_rews):
            eval_returns[i] = rew.sum()

            discounts = self.discounting ** np.arange(len(rew))
            eval_discounted_returns[i] = (rew * discounts).sum()

        # eval_returns = np.sum(eval_rews, axis=-1)
        self._print(f"Step {self.num_steps} avg. offline evaluation returns: {np.mean(eval_returns):.2f}")
        self.info['offline_eval_returns'].append(eval_returns)
        self.info['offline_eval_discounted_returns'].append(eval_discounted_returns)

    def post_episode_print(self, episode_reward: int, episode_loss: float, t: int,
                           additional_info: dict = None):
        self.info['episode_reward'].append(episode_reward)
        self.info['episode_length'].append(t + 1)
        self.info['avg_episode_loss'].append(episode_loss / (t + 1))

        # if use_pf:
        #     self.info['pf_episodic_mean'].append(pf_episode_means)
        #     self.info['pf_episodic_var'].append(pf_episode_vars)

        avg_over = min(self.episode_num, 30)
        print_str = (f"Episode {self.episode_num}, steps: {t + 1}, "
                    f"total steps: {self.num_steps}, "
                    f"moving avg steps: {sum(self.info['episode_length'][-avg_over:]) / avg_over:.3f}, "
                    f"moving avg returns: {sum(self.info['episode_reward'][-avg_over:]) / avg_over:.3f}, "
                    f"rewards: {episode_reward:.2f}, "
                    f"avg episode loss: {episode_loss / (t + 1):.4f}")

        if additional_info is not None:
            print_str += ", "
            for k, v in additional_info.items():
                print_str += f"{k}: {v / (t + 1):.4f}, "
        self._print(print_str)

    def _maybe_convert_rewards(self, rewards: List[Any]):
        """
        We do this in order to reduced the size of our saved array, as we
        take many many steps and save reward at each step.
        :param rewards: list of rewards over all steps.
        :return:
        """
        if hasattr(self.env, 'unique_rewards'):
            max_unique, min_unique = max(self.env.unique_rewards), min(self.env.unique_rewards)
            if max_unique <= 127 and min_unique >= -128:
                return np.array(rewards, dtype=np.int8)
            elif max_unique <= 32767 and min_unique >= -32768:
                return np.array(rewards, dtype=np.int16)

            return np.array(rewards, dtype=np.int32)

    def get_info(self):
        return_info = {}
        for k, v in self.info.items():
            if k == 'reward':
                return_info[k] = self._maybe_convert_rewards(self.info[k])
            else:
                return_info[k] = np.array(self.info[k])

        return return_info

    @staticmethod
    def _print(msg):
        print(msg)

