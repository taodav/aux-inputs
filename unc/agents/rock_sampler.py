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

    def __init__(self, env: Union[RockSampleWrapper, RockSample], check_prob: float = 0.1):
        """
        Agent that samples a given rock. OR it will exit.
        :param check_prob: with what probability do we check a random rock?
        """
        super(RockSamplerAgent, self).__init__()
        self.rock_idx = None
        self.target_type = None
        self.check_prob = check_prob
        self.env = env

        self.scanned = False

        self.finished_current_option = False

    def set_target(self, rock_idx: int, target_type: str):
        assert target_type in ['rock', 'goal']
        self.rock_idx = rock_idx
        self.target_type = target_type
        self.finished_current_option = False
        self.scanned = False

    def act(self, obs: np.ndarray):
        """
        Act takes as input the underlying mdp state for rock sample
        :param obs:
        :return:
        """
        position = self.env.agent_position

        if self.env.rng.random() < self.check_prob:
            rock_idx = self.env.rng.choice(np.arange(self.env.rocks))
            return rock_idx + 5

        if self.target_type == 'goal' and self.rock_idx > (len(self.env.rock_positions) - 1):
            # EXIT
            return 1

        # We scan the target rock before moving towards it
        if not self.scanned:
            self.scanned = True
            return self.rock_idx + 5

        rock_position = self.env.rock_positions[self.rock_idx]
        if np.all(rock_position == position):
            if not self.finished_current_option:
                # If we're in position but haven't finished, this means we need to sample
                self.finished_current_option = True
                return 4
            else:
                raise AssertionError("We need to set a new option!")

        # All our policies first deal with north/south, then east/west positions
        if rock_position[0] == position[0]:
            if rock_position[1] < position[1]:
                return 3

            return 1

        if rock_position[0] < position[0]:
            return 0
        return 2


