import numpy as np
from pathlib import Path

from unc.envs import get_env
from unc.agents import RockSamplerAgent
from unc.utils import save_video
from definitions import ROOT_DIR


if __name__ == "__main__":
    n_episodes = 1
    render = True
    env_str = "rpg"
    env = get_env(2021, env_str=env_str, render=render)

    agent = RockSamplerAgent()
    imgs = []
    eps = 0
    env.reset()
    while eps < n_episodes:

        target_idx = env.rng.choice(np.arange(len(env.rock_positions) + 1))
        target_str = "goal"
        target_position = np.array([0, 0])
        if target_idx < len(env.rock_positions):
            target_str = "rock"
            target_position = env.rock_positions[target_idx]

        agent.set_target_position(target_position, target_str)

        action = agent.act(env.state)
        done = False

        while True:
            if render:
                imgs.append(env.render(mode='rgb_array',
                                       show_weights=True,
                                       action=action))

            action = agent.act(env.state)
            obs, rew, done, _ = env.step(action)
            if done or agent.finished_current_option:
                if render:
                    imgs.append(env.render(mode='rgb_array',
                                           show_weights=True,
                                           action=action))
                if done:
                    eps += 1
                    obs = env.reset()
                break


    imgs = np.array(imgs)

    vod_path = Path(ROOT_DIR, "results", env_str, "test.mp4")

    print(f"Saving render of test episode(s) to {vod_path}")

    save_video(imgs, vod_path)
