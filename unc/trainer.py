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
            'loss': []
        }
        self.num_steps = 0

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            obs = np.array([self.env.reset()])

            for t in range(self.max_episode_steps):
                with torch.no_grad():
                    action = self.agent.act(obs).item()

                next_obs, reward, done, info = self.env.step(action)

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
                    break

                obs = next_obs

            self.episode_num += 1
            self.info['episode_reward'].append(episode_reward)
            self.info['episode_length'].append(t + 1)

            avg_over = min(self.episode_num, 30)
            self._print(f"Episode {self.episode_num}, steps: {t + 1}, "
                        f"Moving avg steps: {sum(self.info['episode_length'][-avg_over:]) / avg_over:.3f}, "
                        f"rewards: {episode_reward:.2f},"
                        f"Avg episode loss: {episode_loss / (t + 1):.4f}")

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")

    def preprocess_step(self, obs: np.ndarray,
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

