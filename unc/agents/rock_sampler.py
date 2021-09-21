"""
Sample rock policy. Randomly selects one of the following macro actions and executes.
Go to rock 1 and sample
...
Go to rock k and sample
Exit
"""
import numpy as np

from unc.agents import Agent


class RockSamplerAgent(Agent):

    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self):
        """
        Agent that samples a given rock. OR it will exit.
        """
        super(RockSamplerAgent, self).__init__()
        self.target_position = None
        self.target_type = None

        self.finished_current_option = False

    def set_target_position(self, target_position: np.ndarray, target_type: str):
        assert target_type in ['rock', 'goal']
        self.target_position = target_position
        self.target_type = target_type
        self.finished_current_option = False

    def act(self, state: np.ndarray):
        """
        Act takes as input the underlying mdp state for rock sample
        :param state:
        :return:
        """
        position = state[:2]
        if self.target_type == 'goal':
            # EXIT
            return 1

        if np.all(self.target_position == position):
            if not self.finished_current_option:
                # If we're in position but haven't finished, this means we need to sample
                self.finished_current_option = True
                return 4
            else:
                raise AssertionError("We need to set a new option!")

        # All our policies first deal with north/south, then east/west positions
        if self.target_position[0] == position[0]:
            if self.target_position[1] < position[1]:
                return 3

            return 1

        if self.target_position[0] < position[0]:
            return 0
        return 2


