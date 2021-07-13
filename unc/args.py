import torch
import hashlib
from time import time, ctime
from typing import Union
from tap import Tap
from pathlib import Path

from definitions import ROOT_DIR

class Args(Tap):
    env: str = "sr"
    """
    What environment do we use? combine keys as follows:
    r = reward
    sr = state observations + reward
    (NOT IMPLEMENTED) pr = particle filter + reward
    """
    total_steps: int = 10000  # Total number of steps to take
    max_episode_steps: int = 500  # Maximum number of steps in an episode

    step_size: float = 0.0001  # Step size for our neural network
    n_hidden: int = 100  # How many nodes in our hidden layer of our neural network?
    discounting: float = 0.9  # Discount factor
    epsilon: float = 0.1  # Epsilon random action sampling probability
    random_start: bool = True  # Do we have a random initial state distribution?

    seed: int = 2021  # Random seed
    device: Union[str, torch.device] = "cuda"  # What device do we use? (cpu | cuda)

    log_dir: Path = Path(ROOT_DIR, 'log')  # For tensorboard logging. Where do we log our files?
    results_dir: Path = Path(ROOT_DIR, 'results')  # What directory do we save our results in?
    results_fname: str = "default.npy"  # What file name do we save results to? If nothing filled, we use a hash + time.

    def process_args(self) -> None:
        # Set our device
        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')

        # Create our log and results directories if it doesn't exist
        # We also save the different environments in different folders
        self.log_dir /= self.env
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir /= self.env
        self.results_dir.mkdir(parents=True, exist_ok=True)


def md5(args: Args) -> str:
    return hashlib.md5(str(args).encode('utf-8')).hexdigest()


def get_results_fname(args: Args):
    time_str = ctime(time())
    results_fname = f"{md5(args)}_{time_str}.npy"
    return results_fname

