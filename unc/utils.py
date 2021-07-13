import numpy as np
from pathlib import Path


def save_info(results_path: Path, info: dict):
    np.save(results_path, info)


def load_info(results_path: Path):
    return np.load(results_path).item()