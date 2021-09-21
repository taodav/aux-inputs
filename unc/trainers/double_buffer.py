import torch
import numpy as np
from typing import Union
from time import time, ctime

from unc.args import Args
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper
from unc.agents import Agent
from unc.utils import ReplayBuffer

from .trainer import Trainer


class DoubleBufferTrainer(Trainer):
    def __init__(self, args: Args, agent: Agent, env: Union[RockSample, RockSampleWrapper],
                 prefilled_buffer: ReplayBuffer, p_prefilled: float = 0.5):
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
        super(DoubleBufferTrainer, self).__init__(args, agent, env)

        self.batch_size = args.batch_size

        self.buffer = ReplayBuffer(prefilled_buffer.capacity, self.env.rng, self.env.observation_space.shape)
        self.prefilled_buffer = prefilled_buffer

        self.p_prefilled = p_prefilled

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0
            obs = self.env.reset()
            self.buffer.push_initial(obs)

            obs = np.array([obs])

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())
                with torch.no_grad():
                    action = self.agent.act(obs).item()

                next_obs, reward, done, info = self.env.step(action)

                self.buffer.push({
                    'obs': next_obs, 'reward': reward, 'done': done, 'action': action
                })

                self.info['reward'].append(reward)

                prefilled_bs = int(self.batch_size * self.p_prefilled)
                prefilled_sample = self.prefilled_buffer.sample(prefilled_bs)
                sample = self.buffer.sample(self.batch_size - prefilled_bs)

                batch_obs = np.concatenate([prefilled_sample['obs'], sample['obs']])
                batch_reward = np.concatenate([prefilled_sample['reward'], sample['reward']])
                batch_action = np.concatenate([prefilled_sample['action'], sample['action']])
                batch_next_obs = np.concatenate([prefilled_sample['next_obs'], sample['next_obs']])
                batch_done = np.concatenate([prefilled_sample['done'], sample['done']])

                batch_gamma = (1 - batch_done) * self.discounting

                loss = self.agent.update(batch_obs, batch_action, batch_next_obs, batch_gamma, batch_reward)

                # Logging
                episode_loss += loss
                episode_reward += reward
                self.info['loss'].append(loss)
                self.num_steps += 1

                if done:
                    break

                obs = np.array([next_obs])

            self.episode_num += 1
            self.info['episode_reward'].append(episode_reward)
            self.info['episode_length'].append(t + 1)

            avg_over = min(self.episode_num, 30)
            self._print(f"Episode {self.episode_num}, steps: {t + 1}, "
                        f"total steps: {self.num_steps}, "
                        f"moving avg steps: {sum(self.info['episode_length'][-avg_over:]) / avg_over:.3f}, "
                        f"rewards: {episode_reward:.2f}, "
                        f"avg episode loss: {episode_loss / (t + 1):.4f}")

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
