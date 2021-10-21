import numpy as np
from jax import random
from pathlib import Path

from unc.agents import LSTMAgent
from unc.envs import get_env
from unc.eval import test_episodes
from unc.utils import save_video
from definitions import ROOT_DIR

if __name__ == "__main__":
    exp_dir = Path(ROOT_DIR, 'results', 'rg')
    agent_path = exp_dir / "8426321e0223271d7848d30ce1000edc_Tue Oct 19 22:39:38 2021.pth"
    agent = LSTMAgent.load(agent_path, LSTMAgent)
    args = agent.args
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)

    env = get_env(rng,
                  rand_key,
                  env_str=args.env,
                  blur_prob=args.blur_prob,
                  random_start=args.random_start,
                  slip_prob=args.slip_prob,
                  slip_turn=args.slip_turn,
                  size=args.size,
                  n_particles=args.n_particles,
                  update_weight_interval=args.update_weight_interval,
                  rock_obs_init=args.rock_obs_init)

    imgs, rews = test_episodes(agent, env, n_episodes=5,
                               render=True, test_eps=args.test_eps,
                               max_episode_steps=args.max_episode_steps)
    vod_path = exp_dir / f"test.mp4"

    print(f"Saving render of test episode(s) to {vod_path}")

    # save_gif(imgs, gif_path)
    save_video(imgs, vod_path)
