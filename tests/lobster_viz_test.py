import numpy as np
from jax import random
from pathlib import Path

from unc.envs import get_env
from unc.utils import save_video
from definitions import ROOT_DIR

if __name__ == "__main__":
    seed = 2023
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    env = get_env(rng, rand_key, env_str="2", render=True)

    video_path = Path(ROOT_DIR, 'results', '2_nn', 'test.mp4')

    env.reset()

    imgs = []

    for i in range(10):
        action = env.action_space.sample()
        viz = env.render(action=action, show_obs=True)
        imgs.append(viz)

    imgs = np.stack(imgs)

    save_video(imgs, video_path)



