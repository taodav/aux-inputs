from .base import Agent
from .sarsa import SarsaAgent, ExpectedSarsa
from .qlearning import QLearningAgent
from .rock_sampler import RockSamplerAgent


def get_agent(algo_str: str = 'sarsa'):
    if algo_str == 'sarsa':
        return SarsaAgent
    elif algo_str == 'esarsa':
        return ExpectedSarsa
    elif algo_str == 'qlearning':
        return QLearningAgent
    else:
        raise NotImplementedError
