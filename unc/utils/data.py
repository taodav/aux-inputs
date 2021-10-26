import numpy as np
import jax.numpy as jnp
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Union, Iterable


@dataclass
class Batch:
    obs: Union[np.ndarray, Iterable]
    action: Union[np.ndarray, Iterable]
    next_obs: Union[np.ndarray, Iterable]
    reward: Union[np.ndarray, Iterable]
    done: Union[np.ndarray, Iterable] = None
    gamma: Union[np.ndarray, Iterable] = None
    next_action: Union[np.ndarray, Iterable] = None
    state: Union[np.ndarray, Iterable] = None
    next_state: Union[np.ndarray, Iterable] = None
    zero_mask: Union[np.ndarray, Iterable] = None
    end: Union[np.ndarray, Iterable] = None  # End is done or max_steps == timesteps
    indices: Union[np.ndarray, Iterable] = None  # Indices that were sampled


def zip_batches(b1: Batch, b2: Batch):
    zipped = {}
    for k, v1 in b1.__dict__.items():
        if v1 is not None:
            v2 = b2.__dict__[k]
            zipped[k] = np.concatenate([v1, v2])
    return Batch(**zipped)


def euclidian_dist(arr1: np.ndarray, arr2: np.ndarray):
    return jnp.linalg.norm(arr1 - arr2, 2)


def manhattan_dist(arr1: np.ndarray, arr2: np.ndarray):
    return np.linalg.norm(arr1 - arr2, 1)


def half_dist_prob(dist: float, max_dist: float):
    prob = (1 + jnp.power(2.0, -dist / max_dist)) * 0.5
    return prob


def save_info(results_path: Path, info: dict):
    np.save(results_path, info)


def load_info(results_path: Path):
    return np.load(results_path, allow_pickle=True).item()


def save_gif(arr: np.ndarray, path: Path, duration=400):
    gif = [Image.fromarray(img) for img in arr]

    gif[0].save(path, save_all=True, append_images=gif[1:], duration=duration, loop=0)


def save_video(arr: np.ndarray, path:Path, fps: int = 2):
    import cv2

    length, h, w, c = arr.shape
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), True)

    for i in range(length * fps):
        frame = cv2.cvtColor(arr[i // fps], cv2.COLOR_RGB2BGR)
        vw.write(frame)
    vw.release()


def batch_wall_split(states: np.ndarray, size: int, green_idx: int = 1):
    north_facing_idxes = np.argwhere(states[:, 2] == 0)[:, 0]
    relative_n_facing_wall_idxes = np.argwhere(states[north_facing_idxes][:, 0] == 1)[:, 0]
    north_facing_wall_idxes = north_facing_idxes[relative_n_facing_wall_idxes]

    east_facing_idxes = np.argwhere(states[:, 2] == 1)[:, 0]
    relative_e_facing_wall_idxes = np.argwhere(states[east_facing_idxes][:, 1] == size - 2)[:, 0]
    east_facing_wall_idxes = east_facing_idxes[relative_e_facing_wall_idxes]

    south_facing_idxes = np.argwhere(states[:, 2] == 2)[:, 0]
    relative_s_facing_wall_idxes = np.argwhere(states[south_facing_idxes][:, 0] == size - 2)[:, 0]
    south_facing_wall_idxes = south_facing_idxes[relative_s_facing_wall_idxes]

    west_facing_idxes = np.argwhere(states[:, 2] == 3)[:, 0]
    relative_w_facing_wall_idxes = np.argwhere(states[west_facing_idxes][:, 1] == 1)[:, 0]
    west_facing_wall_idxes = west_facing_idxes[relative_w_facing_wall_idxes]

    relative_w_facing_green_idxes = np.argwhere(states[west_facing_wall_idxes][:, 0] == green_idx)
    relative_w_facing_blue_idxes = np.argwhere(states[west_facing_wall_idxes][:, 0] != green_idx)
    green_facing_wall_idxes = west_facing_wall_idxes[relative_w_facing_green_idxes]
    blue_facing_wall_idxes = west_facing_wall_idxes[relative_w_facing_blue_idxes]

    return north_facing_wall_idxes, east_facing_wall_idxes, south_facing_wall_idxes, blue_facing_wall_idxes, green_facing_wall_idxes
