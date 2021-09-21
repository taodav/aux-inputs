"""
Data sampler. Currently we support the following agent(s):
RockSamplerAgent

Saves data to a replay buffer for later sampling.
"""
import dill
import numpy as np
from typing import Union
from pathlib import Path

from unc.agents import RockSamplerAgent
from unc.utils import ReplayBuffer, save_video
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper


class Sampler:
    def __init__(self, env: Union[RockSample, RockSampleWrapper], agent: RockSamplerAgent,
                 steps_to_collect: int = 1000, render: bool = False):
        """
        Sampler samples every step and stores it in a buffer.
        For each step, save the following:
        action, reward, next_state, next_observation, done
        :param env:
        """
        self.env = env
        self.agent = agent
        self.steps_to_collect = steps_to_collect
        self.render = render
        self.imgs = []

        self.collected = 0

        self.buffer = ReplayBuffer(steps_to_collect + 1, env.rng,
                                   env.observation_space.shape, state_size=env.state_space.shape)

        self.filled = False

    def collect(self):
        """
        Collect steps_to_collect samples from the environment.
        :return:
        """
        obs = self.env.reset()
        target_idxes = []
        episode_rews = 0
        eps = 0
        while self.collected < self.steps_to_collect:
            if len(target_idxes) == 0:
                target_idxes = np.arange(len(self.env.rock_positions) + 1)
                self.env.rng.shuffle(target_idxes)
            target_idx = target_idxes[0]
            target_idxes = target_idxes[1:]
            target_str = "goal"
            target_position = np.array([0, 0])
            if target_idx < len(self.env.rock_positions):
                target_str = "rock"
                target_position = self.env.rock_positions[target_idx]

            self.agent.set_target_position(target_position, target_str)

            self.buffer.push_initial(obs, state=self.env.state)
            while True:
                # NOTE: This is using STATE as input, not observation
                action = self.agent.act(self.env.state)
                obs, rew, done, _ = self.env.step(action)
                episode_rews += rew
                self.collected += 1

                batch = {
                    'state': self.env.state,
                    'action': action,
                    'obs': obs,
                    'reward': rew,
                    'done': done
                }
                self.buffer.push(batch)

                if self.render:
                    self.imgs.append(self.env.render(show_weights=True, action=action))

                if done:
                    obs = self.env.reset()
                    if self.render:
                        self.imgs.append(self.env.render(show_weights=True, action=action))
                    target_idxes = []
                    print(f"Finished episode {eps}, episode reward is {episode_rews}")
                    episode_rews = 0
                    eps += 1
                    break

                # if self.collected % 100 == 0:
                #     print(f"Collected {self.collected}")

                if self.agent.finished_current_option or self.collected >= self.steps_to_collect:
                    break

        self.filled = True

    def save(self, path: Path, video_path: Path = None) -> None:
        """
        Save all the necessary information
        :return:
        """

        to_save = {
            'buffer': self.buffer,
            'rock_positions': self.env.rock_positions
        }
        with open(path, "wb") as f:
            dill.dump(to_save, f)

        if self.render:
            save_video(np.array(self.imgs), video_path)


    @staticmethod
    def load(path: Path):
        with open(path, "rb") as f:
            return dill.load(f)




