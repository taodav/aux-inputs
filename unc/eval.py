import numpy as np
import gym
from typing import Union, Tuple

from unc.envs import CompassWorld
from unc.agents import Agent
from unc.trainers.trainer import Trainer


def test_episodes(agent: Agent, env: Union[CompassWorld, gym.Wrapper],
                 n_episodes: int = 1, test_eps: float = 0.0, render: bool = True,
                  max_episode_steps: int = 1000, show_obs: bool = True, show_weights: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    imgs = []
    all_rews = []

    for ep in range(n_episodes):
        rews = []
        obs = env.reset()

        # Action conditioning
        if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
            obs = np.concatenate([obs, np.zeros(env.action_space.n)])

        obs = np.array([obs])
        agent.reset()
        action = agent.act(obs).item()
        if render:
            imgs.append(env.render(mode='rgb_array',
                                   show_obs=show_obs,
                                   show_weights=show_weights,
                                   action=action))

        agent.set_eps(test_eps)

        for t in range(max_episode_steps):

            action = agent.act(obs).item()

            next_obs, reward, done, info = env.step(action)

            # Action conditioning
            if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
                one_hot_action = np.zeros(env.action_space.n)
                one_hot_action[action] = 1
                next_obs = np.concatenate([next_obs, one_hot_action])

            rews.append(reward)
            if render:
                imgs.append(env.render(mode='rgb_array',
                                       show_obs=show_obs,
                                       show_weights=show_weights,
                                       action=action))

            if done:
                break

            obs, reward, done, info, action = Trainer.preprocess_step(next_obs, reward, done, info, action)

        rews = np.array(rews)
        all_rews.append(rews)
    imgs = np.array(imgs)

    return imgs, all_rews
