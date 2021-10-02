"""
Sample rock policy. Randomly selects one of the following macro actions and executes.
Go to rock 1 and sample
...
Go to rock k and sample
Exit
"""
import numpy as np
from typing import Union

from unc.agents import Agent
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper


class RockSamplerAgent(Agent):

    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self, env: Union[RockSampleWrapper, RockSample], check_prob: float = 0.1,
                 ground_truth: bool = False):
        """
        Agent that samples a given rock. OR it will exit.
        :param check_prob: with what probability do we check a random rock?
        """
        super(RockSamplerAgent, self).__init__()
        self.rock_idx = None
        self.target_type = None
        self.check_prob = check_prob
        self.ground_truth = ground_truth
        self.env = env

    def set_target(self, rock_idx: int, target_type: str):
        assert target_type in ['rock', 'goal']
        self.rock_idx = rock_idx
        self.target_type = target_type

    @property
    def finished_current_option(self):
        if self.target_type == 'goal':
            return self.env.agent_position[1] == self.env.size
        elif self.target_type == 'rock':
            return np.all(self.env.agent_position == self.env.rock_positions[self.rock_idx]) and \
                   self.env.sampled_rocks[self.rock_idx]

    def act(self, obs: np.ndarray):
        """
        Act takes as input the underlying mdp state for rock sample
        :param obs:
        :return:
        """
        position = self.env.agent_position
        if not self.ground_truth:
            if self.env.rng.random() < self.check_prob:
                rock_idx = self.env.rng.choice(np.arange(self.env.rocks))
                return rock_idx + 5

        if self.target_type == 'goal' and self.rock_idx > (len(self.env.rock_positions) - 1):
            # EXIT
            return 1

        if not self.ground_truth:
            # We scan the target rock before moving towards it
            if not self.env.checked_rocks[self.rock_idx]:
                return self.rock_idx + 5

        rock_position = self.env.rock_positions[self.rock_idx]
        if np.all(rock_position == position):
            return 4

        # All our policies first deal with north/south, then east/west positions
        if rock_position[0] == position[0]:
            if rock_position[1] < position[1]:
                return 3

            return 1

        if rock_position[0] < position[0]:
            return 0
        return 2


