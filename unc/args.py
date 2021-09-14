import torch
import hashlib
from time import time, ctime
from typing import Union
from tap import Tap
from pathlib import Path

from definitions import ROOT_DIR

class Args(Tap):
    env: str = "s"
    """
    What environment do we use? combine the following keys in any order:
    (order will be dictated by priority of components (check priorities in their corresponding wrappers))
    s = Ground-truth state concatenated to observation.
    b = With some prob., sample a random observation over the ground-truth. "Blurry observations" .
    i = SLIP (i for ice) - When you move forward, with some prob., stay in the same spot.
    p = Particle filter observations, where the mean and variance of the particles are prepended to observation.
    m = Particle filter with only mean of particles + observations + reward. This will only work if "p" is in env string.
    v = Particle filter with only variance of particles + observations + reward. This will only work if "p" is in env string.
        If m or v is in string without p, nothing happens.
    f = Fixed Compass World where the green terminal state is in the middle of the west wall.
    g = global-state observations + color observation. This encodes all particles/states in a single array.
    """
    size: int = 8  # How large do we want each dimension of our gridworld to be?
    slip_prob: float = 0.1  # [STOCHASTICITY] With what probability do we slip and stay in the same grid when moving forward?
    slip_turn: bool = False  # If we're in the slip setting, do we slip on turns as well?
    total_steps: int = 60000  # Total number of steps to take
    max_episode_steps: int = 1000  # Maximum number of steps in an episode
    blur_prob: float = 0.3  # If b is in env (blurry env), what is the probability that we see a random observation?

    update_weight_interval: int = 1  # How often do we update our particle weights?
    resample_interval: int = 1  # [STOCHASTICITY] How often do we resample our particles?
    n_particles: int = -1  # How many particles do we sample? If -1, assign one particle per state.

    step_size: float = 0.0001  # Step size for our neural network
    n_hidden: int = 100
    """
    How many nodes in our hidden layer of our neural network?
    0 for linear function approximation.
    """
    discounting: float = 0.9  # Discount factor
    epsilon: float = 0.1  # Epsilon random action sampling probability
    random_start: bool = True  # Do we have a random initial state distribution?

    seed: int = 2021  # Random seed
    device: Union[str, torch.device] = "cuda"  # What device do we use? (cpu | cuda)

    test_eps: float = 0.0  # What's our test epsilon?
    log_dir: Path = Path(ROOT_DIR, 'log')  # For tensorboard logging. Where do we log our files?
    results_dir: Path = Path(ROOT_DIR, 'results')  # What directory do we save our results in?
    results_fname: str = "default.npy"  # What file name do we save results to? If nothing filled, we use a hash + time.
    view_test_ep: bool = False  # Do we create a gif of a test episode after training?
    save_model: bool = False  # Do we save our model after finishing training?

    def process_args(self) -> None:
        # Set our device
        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')

        # Create our log and results directories if it doesn't exist
        # We also save the different environments in different folders
        self.log_dir /= self.env
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir /= self.env
        self.results_dir /= str(self.size)
        self.results_dir.mkdir(parents=True, exist_ok=True)


def md5(args: Args) -> str:
    return hashlib.md5(str(args).encode('utf-8')).hexdigest()


def get_results_fname(args: Args):
    time_str = ctime(time())
    results_fname_npy = f"{md5(args)}_{time_str}.npy"
    results_fname = f"{md5(args)}_{time_str}"
    return results_fname, results_fname_npy

