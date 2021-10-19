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
from unc.utils import ReplayBuffer, save_video, Batch
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
        all_episode_rews = []
        obs = self.env.reset()
        episode_rews = 0
        eps = 0

        while self.collected < self.steps_to_collect:
            target_idxes = np.arange(len(self.env.rock_positions) + 1)
            self.env.rng.shuffle(target_idxes)
            target_idx = target_idxes[0]
            target_idxes = target_idxes[1:]

            # If we are trying to get to the goal
            target_str = "goal"

            # If we sample a rock
            if target_idx < len(self.env.rock_positions):
                target_str = "rock"

            self.agent.set_target(target_idx, target_str)

            while True:
                # NOTE: This is using STATE as input, not observation
                action = self.agent.act(obs)
                state = self.env.state
                next_obs, rew, done, _ = self.env.step(action)
                next_state = self.env.state

                if self.agent.finished_current_option:
                    target_idx = target_idxes[0]
                    target_idxes = target_idxes[1:]

                    target_str = "goal"

                    if target_idx < len(self.env.rock_positions):
                        target_str = "rock"

                    self.agent.set_target(target_idx, target_str)

                next_action = 0
                if not done:
                    next_action = self.agent.act(obs)

                episode_rews += rew
                self.collected += 1

                sample = Batch(**{
                    'state': state,
                    'action': action,
                    'obs': obs,
                    'next_state': next_state,
                    'next_obs': next_obs,
                    'reward': rew,
                    'next_action': next_action,
                    'done': done
                })
                self.buffer.push(sample)

                if self.render:
                    self.imgs.append(self.env.render(show_weights=True, action=action))

                if done:
                    obs = self.env.reset()
                    if self.render:
                        self.imgs.append(self.env.render(show_weights=True, action=action))
                    print(f"Finished episode {eps}, episode reward is {episode_rews}")
                    all_episode_rews.append(episode_rews)
                    episode_rews = 0
                    eps += 1
                    break

                if self.collected >= self.steps_to_collect:
                    break

                obs = next_obs

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




