import gym
import logging
from time import time, ctime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from unc.args import Args
from unc.agents import Agent


class Trainer:
    def __init__(self, args: Args, agent: Agent, env: gym.Env):
        self.args = args
        self.discounting = args.discounting
        self.epsilon = args.epsilon
        self.total_steps = args.total_steps
        self.max_episode_steps = args.max_episode_steps

        self.agent = agent
        self.env = env

        self.episode_num = 0
        self.num_steps = 0

        self._writer = SummaryWriter(args.log_dir)
        self.info = None

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
            'loss': [],
            'args': self.args.as_dict()
        }
        if 'p' in self.args.env:
            self.info['pf_episodic_mean'] = []
            self.info['pf_episodic_var'] = []

        self.num_steps = 0

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")
        use_pf = 'p' in self.args.env

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            # For logging particle filter statistics
            pf_episode_means = []
            pf_episode_vars = []

            obs = np.array([self.env.reset()])

            for t in range(self.max_episode_steps):
                with torch.no_grad():
                    action = self.agent.act(obs).item()

                # Log particle means and variances
                if use_pf:
                    state_info = obs[0][:6]
                    means = state_info[::2]
                    vars = state_info[1::2]
                    pf_episode_means.append(means)
                    pf_episode_vars.append(vars)

                next_obs, reward, done, info = self.env.step(action)

                self.info['reward'].append(reward)

                # Preprocess everything for updating
                next_obs, reward, done, info, action = self.preprocess_step(next_obs, reward, done, info, action)

                gamma = (1 - done) * self.discounting

                loss = self.agent.update(obs, action, next_obs, gamma, reward)

                # Logging
                episode_loss += loss
                episode_reward += reward[0]
                self.info['loss'].append(loss)
                self.num_steps += 1

                if done:
                    if use_pf:
                        state_info = obs[0][:6]
                        means = state_info[::2]
                        vars = state_info[1::2]
                        pf_episode_means.append(means)
                        pf_episode_vars.append(vars)
                    break

                obs = next_obs

            self.episode_num += 1
            self.info['episode_reward'].append(episode_reward)
            self.info['episode_length'].append(t + 1)

            if use_pf:
                self.info['pf_episodic_mean'].append(pf_episode_means)
                self.info['pf_episodic_var'].append(pf_episode_vars)

            avg_over = min(self.episode_num, 30)
            self._print(f"Episode {self.episode_num}, steps: {t + 1}, "
                        f"total steps: {self.num_steps}, "
                        f"moving avg steps: {sum(self.info['episode_length'][-avg_over:]) / avg_over:.3f}, "
                        f"rewards: {episode_reward:.2f}, "
                        f"avg episode loss: {episode_loss / (t + 1):.4f}")

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")

    @staticmethod
    def preprocess_step(obs: np.ndarray,
                        reward: float,
                        done: bool,
                        info: dict,
                        action: int):
        obs = np.array([obs])
        reward = np.array([reward])
        done = np.array([done])
        action = np.array([action])
        return obs, reward, done, info, action

    def get_info(self):
        return_info = {}
        for k, v in self.info.items():
            return_info[k] = np.array(self.info[k])

        return return_info

    @staticmethod
    def _print(msg):
        logging.info(msg)

