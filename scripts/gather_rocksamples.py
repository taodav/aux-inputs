from pathlib import Path
from itertools import product
import numpy as np
from jax import random

from unc.envs import get_env
from unc.agents import RockSamplerAgent
from unc.sampler import Sampler
from definitions import ROOT_DIR

if __name__ == "__main__":
    seeds = [(i + 2020) for i in range(10)]
    # seeds = [2022]
    env_strs = ['rxg']
    n_particles = 100
    render = False
    steps_to_collect = 10000
    data_dir = Path(ROOT_DIR, 'data')
    data_dir.mkdir(exist_ok=True)

    for seed, env_str in product(seeds, env_strs):
        rng = np.random.RandomState(seed)
        rand_key = random.PRNGKey(seed)
        replay_save_path = Path(data_dir, f'buffer_{env_str}_{seed}.pkl')
        video_save_path = Path(data_dir, f'buffer_{env_str}_{seed}.mp4')

        env = get_env(rng, rand_key, env_str=env_str, n_particles=n_particles)
        agent = RockSamplerAgent(env, ground_truth='s' in env_str)

        sampler = Sampler(env, agent, steps_to_collect=steps_to_collect, render=render)

        sampler.collect()

        sampler.save(replay_save_path, video_save_path)

        print(f"Done collecting. Saving to {replay_save_path}")

