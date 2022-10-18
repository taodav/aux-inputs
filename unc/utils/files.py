import dill
from pathlib import Path
from time import time, ctime
from typing import Tuple

from unc.args import Args, hash_training_args


def init_files(args: Args) -> Tuple[Path, Path]:
    time_str = ctime(time())
    hashed_args = hash_training_args(args)
    results_fname_npy = f"{hashed_args}_{time_str}.npy"

    if args.gvf_trainer == 'prediction':
        results_dir = args.results_dir.parent
        original_results_dir = args.results_dir.name
        prediction_results_dir = results_dir / f"{original_results_dir}_prediction"
        results_path = prediction_results_dir / results_fname_npy
        checkpoints_dir = prediction_results_dir / "checkpoints" / hashed_args
    else:
        results_path = args.results_dir / results_fname_npy

        checkpoints_dir = args.results_dir / "checkpoints" / hashed_args

    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    return results_path, checkpoints_dir


def load_checkpoint(checkpoint_path: Path):
    with open(checkpoint_path, "rb") as f:
        trainer = dill.load(f)

    return trainer
