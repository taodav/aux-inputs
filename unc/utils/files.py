from pathlib import Path
from time import time, ctime
from typing import Tuple

from unc.args import Args, hash_training_args


def init_files(args: Args) -> Tuple[Path, Path]:
    time_str = ctime(time())
    hashed_args = hash_training_args(args)
    results_fname_npy = f"{hashed_args}_{time_str}.npy"

    results_path = args.results_dir / results_fname_npy

    checkpoints_dir = args.results_dir / "checkpoints" / hashed_args

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return results_path, checkpoints_dir

