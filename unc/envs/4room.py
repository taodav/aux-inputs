import gym.spaces
import jax.numpy as jnp
from jax import random

from .base import Environment


# TODO: REFACTOR TO MORE FUNCTIONAL


class FourRoom(Environment):
    """
    Our partially observable 4Room environment to demonstrate uncertainty over time.

    Each room has a reward in it, placed at a fixed position.

    After the reward is taken, each reward reappears after a random interval (dist is fixed for each reward).

    Observations are:
    obs[0] -> y-position
    obs[1] -> x-position
    obs[2-5] -> reward of the current room you're in.

    w.r.t. the doorways, doorways are in the room that's directly clockwise of itself.
    So if you're in a doorway, you can observe the room directly clockwise of yourself.
    Rooms are also ordered in a clockwise direction, starting at the top left.
    """
    reward_positions = jnp.array([[1, 1], [4, 9], [10, 10], [9, 0]])
    reward_inverse_rates = jnp.array([30, 35, 40, 45])
    def __init__(self,
                 rand_key: random.PRNGKey,
                 random_start: bool = True,
                 reward_val: float = 1.0):
        super(FourRoom, self).__init__()
        self.observation_space = gym.spaces.MultiDiscrete(6)
        self.action_space = gym.spaces.Discrete(4)
        self.rand_key = rand_key
        self.size = 11
        self.random_start = random_start
        self.reward_val = reward_val

        self.lambs = 1 / self.reward_inverse_rates
        self.pmf_1 = self.lambs * jnp.exp(-self.lambs)

        self.time_since_takens = jnp.zeros(4)
        self.position = jnp.zeros(2)

    def get_room_idx(self, position: jnp.ndarray):


