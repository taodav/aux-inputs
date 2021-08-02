import torch
from pathlib import Path

from unc.agents import SarsaAgent
from unc.eval import test_episodes
from unc.envs import get_env
from unc.utils import save_video

from definitions import ROOT_DIR

if __name__ == "__main__":
    device = torch.device('cpu')
    model_path = Path(
        '/home/taodav/Documents/uncertainty/results/fp/8211abb9a4e561a0e818491470c665dc_Wed Jul 28 15:34:10 2021.pth')
    agent = SarsaAgent.load(model_path, device)

    args = agent.args
    test_env = get_env(args.seed,
                       env_str=args.env,
                       blur_prob=args.blur_prob,
                       random_start=args.random_start,
                       update_weight_interval=args.update_weight_interval,
                       render=True)

    imgs, rews = test_episodes(agent, test_env, n_episodes=5, render=True, test_eps=0.0, max_episode_steps=100)

    results_dir = Path(ROOT_DIR, 'results', args.env)
    vod_path = results_dir / 'test.mp4'
    if vod_path.is_file():
        vod_path.unlink()
    save_video(imgs, vod_path)
