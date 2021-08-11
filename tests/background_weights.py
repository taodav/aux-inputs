import torch
from pathlib import Path

from unc.agents import SarsaAgent
from unc.eval import test_episodes
from unc.envs import get_env
from unc.utils import save_video

from definitions import ROOT_DIR

if __name__ == "__main__":
    device = torch.device('cpu')
    # model_path = Path('/home/taodav/Documents/uncertainty/results/fipg/9/9d32e3a7c1e472a075a570d4a50e5a1e_Fri Aug  6 00:47:29 2021.pth')
    model_path = Path('/home/taodav/Documents/uncertainty/results/fip/9/3d8f9ef15b114320f6ab02c694618c1e_Fri Aug  6 00:47:29 2021.pth')
    agent = SarsaAgent.load(model_path, device)

    args = agent.args
    test_env = get_env(2000,
                       env_str=args.env,
                       blur_prob=args.blur_prob,
                       random_start=args.random_start,
                       slip_prob=args.slip_prob,
                       update_weight_interval=args.update_weight_interval,
                       size=args.size,
                       render=True)

    imgs, rews = test_episodes(agent, test_env, n_episodes=5, render=True, test_eps=0.0, max_episode_steps=100)

    results_dir = Path(ROOT_DIR, 'results', args.env, str(args.size))
    vod_path = results_dir / 'test.mp4'
    if vod_path.is_file():
        vod_path.unlink()
    save_video(imgs, vod_path)
