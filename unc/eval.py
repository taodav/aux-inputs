import gym
import numpy as np
from jax import random
from typing import Union, Tuple, List

from unc.agents import Agent
from unc.envs import Environment
from unc.gvfs import GeneralValueFunction
from unc.utils import preprocess_step, get_action_encoding


def lobster_gvf_eval(
        agent: Agent,
        gvf: GeneralValueFunction,
        env: Environment,
        gamma: float = 0.9,
        eval_episodes: int = 5,
        max_episode_steps: int = 200) -> dict:

    def tick_and_update_counts(counts: np.ndarray, obs: np.ndarray):
        counts += 1

        if obs[5] == 0:
            counts[0] = 0

        if obs[8] == 0:
            counts[1] = 0

        return counts


    eval_dict = {
        'eval_predictions': [],
        'eval_reward_not_seen_counts': [],
        'eval_states': [],
    }
    for ep in range(eval_episodes):
        episode_reward_counts = []
        episode_predictions = []
        episode_states = []
        counts = np.zeros(2, dtype=int)

        prev_obs = np.expand_dims(env.reset(), 0)
        prev_pred = agent.reset()
        choice_key, agent._rand_key = random.split(agent._rand_key, 2)
        episode_states.append(env.state.copy())

        # follow the first GVF policy
        # policy = gvf.policy(prev_obs)[0, 0]
        policy = np.ones(env.action_space.n) / env.action_space.n
        prev_action = random.choice(choice_key, policy.shape[0], p=policy)

        obs, reward, done, info = env.step(prev_action)
        counts = tick_and_update_counts(counts, obs)
        obs, reward, done, info, prev_action = preprocess_step(obs, reward, done, info, prev_action.item())
        episode_states.append(env.state.copy())

        for t in range(max_episode_steps):
            predictions = agent.predictions(prev_pred, prev_action, obs)
            episode_predictions.append(predictions[0])

            choice_key, agent._rand_key = random.split(agent._rand_key, 2)
            # policy = gvf.policy(prev_obs)[0, 0]
            policy = np.ones(env.action_space.n) / env.action_space.n
            action = random.choice(choice_key, policy.shape[0], p=policy)

            next_obs, reward, done, info = env.step(action)
            episode_states.append(env.state.copy())
            counts = tick_and_update_counts(counts, next_obs)
            episode_reward_counts.append(counts.copy())

            next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

            obs = next_obs
            prev_action = action
            prev_pred = predictions

        eval_dict['eval_states'].append(np.stack(episode_states))
        eval_dict['eval_predictions'].append(np.stack(episode_predictions))
        eval_dict['eval_reward_not_seen_counts'].append(np.stack(episode_reward_counts))

    eval_dict['eval_states'] = np.stack(eval_dict['eval_states'])
    eval_dict['eval_predictions'] = np.stack(eval_dict['eval_predictions'])
    eval_dict['eval_reward_not_seen_counts'] = np.stack(eval_dict['eval_reward_not_seen_counts'])

    return eval_dict


def slightly_less_simple_chain_gvf_eval(
        agent: Agent,
        gvf: GeneralValueFunction,
        env: Environment,
        gamma: float = 0.9,
        eval_episodes: int = 10) -> dict:
    chain_length = env.n
    ground_truth_predictions = gamma ** np.arange(chain_length - 3, -1, -1)

    eval_dict = {
        'predictions': []
    }
    for ep in range(eval_episodes):
        episode_predictions = []
        prev_obs = np.expand_dims(env.reset(), 0)
        prev_pred = agent.reset()
        choice_key, agent._rand_key = random.split(agent._rand_key, 2)

        # follow the first GVF policy
        policy = gvf.policy(prev_obs)[0, 0]
        prev_action = random.choice(choice_key, policy.shape[0], p=policy)

        obs, reward, done, info = env.step(prev_action)
        obs, reward, done, info, prev_action = preprocess_step(obs, reward, done, info, prev_action.item())

        while not done:
            predictions = agent.predictions(prev_pred, prev_action, obs)
            episode_predictions.append(predictions.item())

            choice_key, agent._rand_key = random.split(agent._rand_key, 2)
            policy = gvf.policy(prev_obs)[0, 0]
            action = random.choice(choice_key, policy.shape[0], p=policy)

            next_obs, reward, done, info = env.step(action)

            next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

            obs = next_obs
            prev_action = action
            prev_pred = predictions

        eval_dict['predictions'].append(episode_predictions)

    eval_dict['predictions'] = np.stack(eval_dict['predictions'])
    eval_dict['gt_predictions'] = ground_truth_predictions
    eval_str = f"""
Ground-truth predictions: {ground_truth_predictions},
Avg. predictions over {eval_episodes} runs: {np.mean(eval_dict['predictions'], axis=0)} 
MSVE: {np.mean((eval_dict['predictions'] - eval_dict['gt_predictions']) ** 2)}
"""
    eval_dict['eval_str'] = eval_str

    return eval_dict


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
