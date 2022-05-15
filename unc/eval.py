import numpy as np
import gym
from typing import Union, Tuple, List

from unc.agents import Agent
from unc.utils import preprocess_step, get_action_encoding


def test_episodes(agent: Agent, env: Union[gym.Env, gym.Wrapper],
                  n_episodes: int = 1, test_eps: float = 0.0, render: bool = True,
                  max_episode_steps: int = 1000, show_obs: bool = True, show_weights: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    imgs = []
    all_rews = []
    render_map = "uf" in agent.args.env and "m" in agent.args.env
    if not hasattr(env, "weights") or not hasattr(env, "particles"):
        show_weights = False

    for ep in range(n_episodes):
        rews = []
        obs = env.reset()

        # Action conditioning
        if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
            action_encoding = get_action_encoding(agent.features_shape, -1, env.action_space.n)
            obs = np.concatenate([obs, action_encoding], axis=-1)

        obs = np.array([obs])
        agent.reset()
        action = agent.act(obs).item()
        if render:
            q_vals = None
            if hasattr(agent, "curr_q"):
                q_vals = agent.curr_q[0]
                if len(q_vals.shape) == 2:
                    q_vals = q_vals[0]
            imgs.append(env.render(mode='rgb_array',
                                   show_obs=show_obs,
                                   show_weights=show_weights,
                                   action=action,
                                   q_vals=q_vals,
                                   render_map=render_map))

        agent.set_eps(test_eps)

        for t in range(max_episode_steps):

            action = agent.act(obs).item()

            next_obs, reward, done, info = env.step(action)

            # Action conditioning
            if hasattr(agent.args, 'action_cond') and agent.args.action_cond == 'cat':
                action_encoding = get_action_encoding(agent.features_shape, action, env.action_space.n)
                next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

            rews.append(reward)
            if render:
                q_vals = None
                if hasattr(agent, "curr_q"):
                    q_vals = agent.curr_q[0]
                    if len(q_vals.shape) == 2:
                        q_vals = q_vals[0]
                imgs.append(env.render(mode='rgb_array',
                                       show_obs=show_obs,
                                       show_weights=show_weights,
                                       action=action,
                                       q_vals=q_vals,
                                       render_map=render_map))

            if done:
                break

            obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action)

        rews = np.array(rews)
        all_rews.append(rews)

    imgs = np.array(imgs)

    return imgs, all_rews
