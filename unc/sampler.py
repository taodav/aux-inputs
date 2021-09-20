"""
Data sampler. Currently we support the following agent(s):
RockSamplerAgent

Saves data to a replay buffer for later sampling.
"""
import gym

from unc.agents import Agent

class Sampler:
    def __init__(self, env: gym.Env, agent: Agent):
        """
        Sampler samples every step and stores it in a buffer.
        For each step, save the following:
        action, reward, next_state, next_observation, done
        :param env:
        """

        self.env = env
        self.agent = Agent

        self.buffer = None


