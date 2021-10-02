from pathlib import Path
from itertools import product

from unc.envs import get_env
from unc.agents import RockSamplerAgent
from unc.sampler import Sampler
from definitions import ROOT_DIR

if __name__ == "__main__":
    seeds = [(i + 2020) for i in range(10)]
    env_strs = ['rpg', 'rsg', 'r']
    n_particles = 100
    render = False
    steps_to_collect = 10000
    data_dir = Path(ROOT_DIR, 'data')
    data_dir.mkdir(exist_ok=True)

    for seed, env_str in product(seeds, env_strs):
        replay_save_path = Path(data_dir, f'buffer_{env_str}_{seed}.pkl')
        video_save_path = Path(data_dir, f'buffer_{env_str}_{seed}.mp4')

        env = get_env(seed, env_str=env_str, n_particles=n_particles)
        agent = RockSamplerAgent(env, ground_truth='s' in env_str)

        sampler = Sampler(env, agent, steps_to_collect=steps_to_collect, render=render)

        sampler.collect()

        sampler.save(replay_save_path, video_save_path)

        print(f"Done collecting. Saving to {replay_save_path}")

