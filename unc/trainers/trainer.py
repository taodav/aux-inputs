import gym
import logging
from time import time, ctime
import numpy as np
from typing import List, Any, Tuple, Union
from collections import deque

from unc.args import Args
from unc.agents import Agent
from unc.utils.data import Batch, preprocess_step
from unc.eval import test_episodes


class Trainer:
    def __init__(self, args: Args, agent: Agent,
                 env: Union[gym.Env, gym.Wrapper],
                 test_env: Union[gym.Env, gym.Wrapper]):
        self.args = args
        self.discounting = args.discounting

        self.epsilon = args.epsilon
        self.anneal_steps = args.anneal_steps
        self.epsilon_start = args.epsilon_start
        self.anneal_value = (self.epsilon_start - self.epsilon) / self.anneal_steps if self.anneal_steps > 0 else 0

        self.total_steps = args.total_steps
        self.max_episode_steps = args.max_episode_steps

        self.agent = agent
        self.env = env
        self.n_actions = env.action_space.n
        self.action_cond = args.action_cond

        self.test_env = test_env
        self.offline_eval_freq = args.offline_eval_freq
        self.test_eps = args.test_eps
        self.test_episodes = args.test_episodes

        self.episode_num = 0
        self.num_steps = 0

        self.info = None
        self.trunc = args.trunc if args.arch in ['lstm', 'gru'] else 0

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
            'args': self.args.as_dict()
        }
        if 'p' in self.args.env:
            self.info['pf_episodic_mean'] = []
            self.info['pf_episodic_var'] = []

        if self.offline_eval_freq > 0:
            self.info['offline_eval_reward'] = []

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
        # use_pf = 'p' in self.args.env

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            # For RNN training
            trunc_batch = None
            if self.trunc > 0:
                trunc_batch = Batch(obs=deque(maxlen=self.trunc), action=deque(maxlen=self.trunc),
                                    next_obs=deque(maxlen=self.trunc), gamma=deque(maxlen=self.trunc),
                                    reward=deque(maxlen=self.trunc), next_action=deque(maxlen=self.trunc),
                                    state=deque(maxlen=self.trunc), next_state=deque(maxlen=self.trunc))

            # For logging particle filter statistics
            # pf_episode_means = []
            # pf_episode_vars = []

            obs = self.env.reset()
            # Action conditioning
            if self.action_cond == 'cat':
                obs = np.concatenate([obs, np.zeros(self.n_actions)])

            obs = np.expand_dims(obs, 0)
            self.agent.reset()

            # Hidden state for RNN training
            hs, next_hs = None, None
            if self.trunc > 0:
                hs = self.agent.state[None, :]

            action = self.agent.act(obs)

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                # Log particle means and variances
                # if use_pf:
                #     state_info = obs[0][:6]
                #     means = state_info[::2]
                #     vars = state_info[1::2]
                #     pf_episode_means.append(means)
                #     pf_episode_vars.append(vars)

                if self.trunc > 0:
                    next_hs = self.agent.state[None, :]

                next_obs, reward, done, info = self.env.step(action.item())

                # Action conditioning
                if self.action_cond == 'cat':
                    one_hot_action = np.zeros(self.n_actions)
                    one_hot_action[action] = 1
                    next_obs = np.concatenate([next_obs, one_hot_action])

                self.info['reward'].append(reward)

                # Preprocess everything for updating
                next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

                gamma = (1 - done) * self.discounting

                next_action = self.agent.act(next_obs)

                batch = Batch(obs=obs, action=action, next_obs=next_obs,
                              gamma=gamma, reward=reward, next_action=next_action)

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

                if done.item():
                    break

                obs = next_obs
                action = next_action
                hs = next_hs

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t)

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")

    def offline_evaluation(self):
        _, eval_rews = test_episodes(self.agent, self.test_env, n_episodes=self.test_episodes,
                                     test_eps=self.test_eps, render=False,
                                     max_episode_steps=self.max_episode_steps)
        eval_returns = np.sum(eval_rews, axis=-1)
        self._print(f"Step {self.num_steps} avg. offline evaluation returns: {np.mean(eval_returns):.2f}")
        self.info['offline_eval_reward'].append(eval_returns)

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

