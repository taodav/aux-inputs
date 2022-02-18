import gym.spaces
import numpy as np

from .base import Environment


# TODO: REFACTOR TO MORE FUNCTIONAL

class TimeDependentReward:
    def __init__(self, rng: np.random.RandomState, inverse_rate: int, reward_val: float = 1.0):
        """
        Time dependent reward. Once the reward is taken, it "renews"
        based on a poisson distribution with lambda = 1 / inverse_rate.

        The inverse_rate is essentially on average, how many time steps
        until this reward appears again?
        """
        assert inverse_rate >= 1, "Can't have a rate of appearance less than 1. We're in the discrete time setting."
        self.lamb = 1 / inverse_rate
        self.pmf_1 = self.lamb * np.exp(-self.lamb)

        self.reward_val = reward_val
        self.rng = rng

        # if time since taken == 0, that means the reward hasn't been taken.
        self.time_since_taken = 0

    def tick(self):
        if self.time_since_taken > 0:
            # If the reward has already been taken, re-appear w.p. self.pmf_1
            if self.rng.random() > self.pmf_1:
                self.time_since_taken = 0

    def get_reward(self):
        if self.time_since_taken == 0:
            return self.reward_val
        return 0.


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
    reward_positions = np.array([[1, 1], [4, 9], [10, 10], [9, 0]])
    reward_inverse_rates = np.array([30, 35, 40, 45])
    def __init__(self,
                 rng: np.random.RandomState = np.random.RandomState(),
                 random_start: bool = True,
                 reward_val: float = 1.0):
        super(FourRoom, self).__init__()
        self.observation_space = gym.spaces.MultiDiscrete(6)
        self.action_space = gym.spaces.Discrete(4)
        self.rng = rng
        self.size = 11
        self.random_start = random_start
        self.reward_val = self.reward_val

        self.state_max = [self.size - 1, self.size - 1]
        self.state_min = [1, 1, 0]

        self.rewards = []
        self.init_rewards()

    def init_rewards(self):
        self.rewards = [TimeDependentReward(self.rng, l) for l in self.reward_inverse_rates]

    def get_room_idx(self):

