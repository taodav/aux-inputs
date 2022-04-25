import hashlib
from jaxlib.xla_extension import Device
from time import time, ctime
from typing import Union
from tap import Tap
from pathlib import Path

from definitions import ROOT_DIR

class Args(Tap):
    env: str = "f"
    """
    What environment do we use? 
    r = RockSample
    t = directional t-maze
    4 = Four room
    2 = two room (lobster fishing)
    "" = (default) compass world
    
    Combine the following keys in any order for additional add-ons for COMPASS WORLD:
    (order will be dictated by priority of components (check priorities in their corresponding wrappers))
    s = Ground-truth state concatenated to observation.
    b = With some prob., sample a random observation over the ground-truth. "Blurry observations" .
    i = SLIP (i for ice) - When you move forward, with some prob., stay in the same spot.
    p = [STATE-BASED ENVS] Particle filter observations, where the mean and variance of the particles are prepended to observation.
    p = [OBS-BASED ENVS] Partially observable observations
    m = Particle filter with only mean of particles + observations + reward. This will only work if "p" is in env string.
    v = Particle filter with only variance of particles + observations + reward. This will only work if "p" is in env string.
        If m or v is in string without p, nothing happens.
    f = Fixed Compass World where the green terminal state is in the middle of the west wall.
    g = global-state observations + color observation. This encodes all particles/states in a single array.
    x = [ROCKSAMPLE] Perfect sensor (X marks the spot) for the RockSample agent.
    c = State count-based uncertainty estimation.
    o = Observation count-based uncertainty estimation.
    d = [COMPASSWORLD] Noisy corridor environment.
    4 = [FOUR ROOM] Four room environment.
    """
    algo: str = 'sarsa'  # Which learning algorithm do we use? (sarsa | qlearning | esarsa)
    arch: str = 'nn'  # What kind of model architecture do we use? (nn | lstm | linear)
    exploration: str = 'eps'  # Which exploration method do we use? (eps | noisy)
    size: int = 9  # How large do we want each dimension of our gridworld to be?
    slip_prob: float = 0.1  # [STOCHASTICITY] With what probability do we slip and stay in the same grid when moving forward?
    slip_turn: bool = False  # If we're in the slip setting, do we slip on turns as well?
    total_steps: int = 60000  # Total number of steps to take
    max_episode_steps: int = 1000  # Maximum number of steps in an episode
    blur_prob: float = 0.3  # If b is in env (blurry env), what is the probability that we see a random observation?

    distance_noise: bool = False  # [OCEANNAV] For our OceanNav partially observable wrappers, do we use distance noise?
    uncertainty_decay: float = 1.  # [OCEANNAV] How quickly do we decay our uncertainty?
    task_fname: str = "task_{}_config.json"  # [OCEANNAV] What's our task config file name?

    rock_obs_init: float = 0.  # [ROCKSAMPLE] What value do we initialize our rock observations to?
    half_efficiency_distance: float = 20.  # [ROCKSAMPLE] Half efficiency distance for checking rocks

    update_weight_interval: int = 1  # How often do we update our particle weights?
    resample_interval: int = 1  # [STOCHASTICITY] How often do we resample our particles?
    n_particles: int = -1  # How many particles do we sample? If -1, assign one particle per state.

    po_degree: float = 0.  # [COMPASSWORLD] Partial observability degree, for noisy corridor observations.

    trace_decay: float = 0.9   # [FOUR ROOM] for bounded trace decay obs, at what rate do we decay?

    count_decay: float = 1.  # If we use count observations, do we decay our counts? If so by what rate?
    unnormalized_counts: bool = False  # Do we normalize our count-based observations?

    step_size: float = 0.0001  # Step size for our neural network
    n_hidden: int = 100
    """
    How many nodes in our hidden layer of our neural network?
    0 for linear function approximation.
    """

    trunc: int = 10  # [RNN] truncation size
    action_cond: str = None  # [RNN] Action conditioning (None | cat)
    init_hidden_var: float = 0.  # [RNN] w/ what variance (of a zero mean normal) do we set initial hidden state to?
    er_hidden_update: str = None
    """
    [RNN] Do we update the hidden states in the replay? if we do how? (None | grad | update)
    None: Don't update. | grad: Update hidden state w.r.t. loss. | 
    update: Update hidden state based on the calculated batch and update.
    """

    k_rnn_hs: int = 1  # [k-RNN] How many RNN hidden states do we take statistics over?
    same_k_rnn_params: bool = False  # [k-RNN] Across our k RNNs, do we use the same parameters for each?
    value_step_size: float = 0.0001  # [k-RNN] What is our step-size for our value network?

    distributional: bool = False  # [dist-LSTM] Does our RNN output distributional values?
    atoms: int = 51  # [dist-LSTM] What support do we use?
    v_max: int = 100  # [dist-LSTM] Support max
    v_min: int = -10  # [dist-LSTM] Support min

    discounting: float = 0.9  # Discount factor
    epsilon: float = 0.1  # Epsilon random action sampling probability
    anneal_steps: int = 0  # If we do epsilon annealing, over how many steps do we anneal epsilon?
    epsilon_start: float = 1.0  # If we do epsilon annealing, where do we start the epsilon?
    random_start: bool = True  # Do we have a random initial state distribution?

    seed: int = 2020  # Random seed
    platform: str = "cpu"  # What platform do we use? (cpu | gpu)

    test_eps: float = 0.0  # What's our test epsilon?
    log_dir: Union[Path, str] = Path(ROOT_DIR, 'log')  # For tensorboard logging. Where do we log our files?
    results_dir: Union[Path, str] = Path(ROOT_DIR, 'results')  # What directory do we save our results in?
    results_fname: str = "default.npy"  # What file name do we save results to? If nothing filled, we use a hash + time.
    view_test_ep: bool = False  # Do we create a gif of a test episode after training?
    test_episodes: int = 5  # How many episodes do we test on at the end of training?
    save_model: bool = False  # Do we save our model after finishing training?

    replay: bool = False  # Do we use a replay buffer to learn?
    batch_size: int = 64  # Batch size for buffer training
    p_prefilled: float = 0.  # What percentage of each batch is sampled from our prefilled buffer?
    buffer_size: int = 20000  # How large is our "online" buffer?

    def configure(self) -> None:
        def to_path(str_path: str) -> Path:
            return Path(str_path)

        self.add_argument('--results_dir', type=to_path)
        self.add_argument('--log_dir', type=to_path)

    def process_args(self) -> None:
        # Create our log and results directories if it doesn't exist
        # We also save the different environments in different folders
        self.log_dir /= f"{self.env}_{self.arch}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir /= f"{self.env}_{self.arch}"
        # self.results_dir /= str(self.size)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.exploration == 'noisy':
            self.epsilon = 0.

        # if we're fishing and we're trying to use the default config
        if "u" in self.env and "f" in self.env and self.task_fname == "task_{}_config.json":
            self.task_fname = "fishing_{}_config.json"


def md5(args: Args) -> str:
    return hashlib.md5(str(args).encode('utf-8')).hexdigest()


def get_results_fname(args: Args):
    time_str = ctime(time())
    results_fname_npy = f"{md5(args)}_{time_str}.npy"
    results_fname = f"{md5(args)}_{time_str}"
    return results_fname, results_fname_npy

