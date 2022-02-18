import gym
import numpy as np
from typing import Union, Tuple

from .wrapper import CompassWorldWrapper
from unc.envs.compass import CompassWorld
from unc.utils.data import zero_dist_prob


class NoisyCorridorObservationWrapper(CompassWorldWrapper):
    """
    Noisy corridor observations.
    The agent sees the entire corridor in front of them instead of just the wall directly
    in front of them.

    The agent also has a noisy sensor, with noise depending on the DISTANCE of the obstruction/wall
    in front of them.

    So if the wall in front of the agent is 3 spaces away, it'll see the correct
    observation w.p. zero_dist_prob(3), and a {RANDOM OR NOTHING} otherwise.


    """

    priority = 2

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper],
                 po_degree: float = 1.0):
        """
        Here, po_degree implies the degree of partial observability, which
        corresponds to the rate at which distance effects the probability of
        seeing what's actually in front of you

        po_degree: 0 means always a prob. of 1 of seeing the wall.
        essentially anything > 10, means you can only see the wall in front of you (normal compass world)
        """
        super(NoisyCorridorObservationWrapper, self).__init__(env)
        self.po_degree = po_degree

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        # Distance to wall you're facing NORTH, EAST, SOUTH, WEST respectively.
        obs = [0, 0, 0, 0, 1]
        if state[-1] == 0:
            dist = state[0]
            obs = [1, 0, 0, 0, 0]
        elif state[-1] == 1:
            dist = self.size - state[1] - 1
            obs = [0, 1, 0, 0, 0]
        elif state[-1] == 2:
            dist = self.size - state[0] - 1
            obs = [0, 0, 1, 0, 0]
        else:
            dist = state[1]
            if state[0] != self.size // 2:
                obs = [0, 0, 0, 1, 0]

        obs = np.array(obs)

        p = zero_dist_prob(dist, self.po_degree)
        if self.rng.random() > p:
            obs = np.zeros_like(obs)

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
