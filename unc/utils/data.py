import numpy as np
import jax.numpy as jnp
import haiku as hk
from pathlib import Path
from PIL import Image
from dataclasses import dataclass
from typing import Union, Iterable, List


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


def get_discounted_returns(ep_lengths: np.ndarray, ep_rews: np.ndarray, discount: float = 0.99, maxlen: int = 1000):
    assert ep_lengths.sum() == ep_rews.shape[
        0], f"Incompatible lengths and rews - lengths: {ep_lengths.sum()}, rews shape[0]: {ep_rews.shape[0]}"

    discounted_returns = []
    current_ep_disc_rew = 0
    idx = 0
    discounts = discount ** np.arange(maxlen)

    for ep_len in ep_lengths:
        curr_ep_rews = ep_rews[idx:idx + ep_len]
        current_ep_disc_rew = np.dot(curr_ep_rews, discounts[:ep_len])
        idx += ep_len
        discounted_returns.append(current_ep_disc_rew.astype(np.float32))

    discounted_returns = np.array(discounted_returns)

    assert discounted_returns.shape[0] == ep_lengths.shape[0]

    return discounted_returns


def average_returns_over_last(data: dict, eps_max_over: int = 100):
    """
    Get the average returns over the final eps_max_over episodes.
    :param data: dictionary with key = hparams, value = array with shape (samples x num_episodes)
    :param eps_max_over: number of episodes to average over.
    :return: List of tuples of (args, return_avg_over_final_eps_max_over).
    """
    hparam_results = []
    for args, all_dis_rew in data.items():
        final_max = np.mean([np.mean(dis_rew[-eps_max_over:]) for length, dis_rew, all_args in all_dis_rew])
        hparam_results.append((args, final_max))
    return hparam_results


def moving_avg(x, mode='valid', w=100):
    return np.convolve(x, np.ones(w), mode=mode) / w


def map_discounted_returns_to_steps(data: List, w: int = 1000, trunc: int = 1e6):
    """
    Map our discounted episodic returns back to steps. We assign the discounted reward achieved at the
    end of and episode to each timestep within the episode
    :param data: list of tuples of (episode_lengths, discounted_episodic_rewards, args)
    :param w: window to average over
    :param trunc: truncation length of entire sequence of experience.
    :return: An array of size (samples x trunc).
    """
    all_seeds = []
    for lengths, dis_rews, _ in data:
        current_seed = []
        for length, dis_rew in zip(lengths, dis_rews):
            current_seed.append(np.zeros(length) + dis_rew)
        all_seeds.append(moving_avg(np.concatenate(current_seed)[:trunc], w=w))
    return np.array(all_seeds)


def count_params(all_params: hk.Params) -> int:
    total_params = 0
    for _, layer in all_params.items():
        for _, w in layer.items():
            total_params += np.prod(w.shape)
    return total_params