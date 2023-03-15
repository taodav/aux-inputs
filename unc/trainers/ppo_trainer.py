from collections import deque
from dataclasses import fields
import gym
import numpy as np
from pathlib import Path
from time import time, ctime
from typing import Union

from unc.args import Args
from unc.agents import PPOAgent
from unc.utils.data import Batch, preprocess_step, get_action_encoding
from unc.gvfs import GeneralValueFunction
from .trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(self,
                 args: Args,
                 agent: PPOAgent,
                 env: Union[gym.Env, gym.Wrapper],
                 test_env: Union[gym.Env, gym.Wrapper],
                 checkpoint_dir: Path = None,
                 gvf: GeneralValueFunction = None,
                 n_step_returns: int = None
                 ):
        super().__init__(args, agent, env, test_env, checkpoint_dir, gvf)
        self.n_step_returns = n_step_returns if n_step_returns is not None else args.max_episode_steps
        if 'lstm' in args.arch:
            raise NotImplementedError('Currently RNN functionality is not implemented for PPO.')

    @staticmethod
    def update_batch(b: Batch, n_step_batch: Batch):
        for field in fields(b):
            val = getattr(b, field.name)
            if val is not None:
                getattr(n_step_batch, field.name).append(val)
        return n_step_batch

    @staticmethod
    def numpyify_batch(b: Batch):
        new_batch = {}
        for field in fields(b):
            val = getattr(b, field.name)
            if val is not None:
                new_batch[field.name] = np.concatenate(val)[None, :]
        return Batch(**new_batch)

    def train(self) -> None:

        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # Timing stuff
        prev_time = time_start
        avg_time_per_log = 0
        num_logs = 0
        log_interval = 1000 if self.offline_eval_freq == 0 else self.offline_eval_freq
        total_target_updates = self.total_steps // log_interval

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            checkpoint_after_ep = False
            self.agent.reset()

            n_step_batch = Batch(obs=deque(maxlen=self.n_step_returns),
                                 action=deque(maxlen=self.n_step_returns),
                                 gamma=deque(maxlen=self.n_step_returns),
                                 reward=deque(maxlen=self.n_step_returns),
                                 log_prob=deque(maxlen=self.n_step_returns),
                                 value=deque(maxlen=self.n_step_returns))

            obs = self.env.reset()
            # Action conditioning
            if self.action_cond == 'cat':
                action_encoding = get_action_encoding(self.agent.features_shape, -1, self.n_actions)
                obs = np.concatenate([obs, action_encoding], axis=-1)

            obs = np.expand_dims(obs, 0)

            action = self.agent.act(obs)
            log_prob = np.log(self.agent.curr_pi[action])
            value = self.agent.value(obs, self.agent.critic_network_params)[0]

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                next_obs, reward, done, info = self.env.step(action.item())

                # Action conditioning
                if self.action_cond == 'cat':
                    action_encoding = get_action_encoding(self.agent.features_shape, action, self.n_actions)
                    next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                self.info['reward'].append(reward)

                # Preprocess everything for updating
                next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

                gamma = (1 - done) * self.discounting

                next_action = self.agent.act(next_obs)
                next_log_prob = np.log(self.agent.curr_pi[next_action])
                next_value = self.agent.value(next_obs, self.agent.critic_network_params)[0]

                batch = Batch(obs=obs, action=action, gamma=gamma, reward=reward,
                              log_prob=log_prob, value=value)
                n_step_batch = self.update_batch(batch, n_step_batch)

                episode_reward += reward[0]
                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq > 0 and self.num_steps % self.offline_eval_freq == 0:
                    self.offline_evaluation()

                # Logging and timing
                # if self.num_steps % log_interval == 0:
                #     time_to_check = True
                #
                #     curr_time = time()
                #     time_per_fix_freq = curr_time - prev_time
                #     avg_time_per_log += (1 / num_logs) * (time_per_fix_freq - avg_time_per_log)
                #     time_remaining = (total_target_updates - num_logs) * avg_time_per_log
                #     self._print(f"Remaining time: {time_remaining / 60:.2f}")
                #     prev_time = curr_time

                if self.checkpoint_freq > 0 and self.num_steps % self.checkpoint_freq == 0:
                    checkpoint_after_ep = True

                if done.item():
                    break

                obs = next_obs
                action = next_action
                value = next_value
                log_prob = next_log_prob

            # deal with state and value for final state
            # gamma = 0 at terminal will deal with everything else.
            n_step_batch.obs = [o for o in n_step_batch.obs] + [next_obs]
            n_step_batch.value = [v for v in n_step_batch.value] + [next_value]

            numpy_batch = self.numpyify_batch(n_step_batch)
            loss, _ = self.agent.update(numpy_batch)

            self.episode_num += 1
            self.post_episode_print(episode_reward, loss, t)

            # # TODO: FOR DEBUGGING! delete
            if self.episode_num % 100 == 0:
                print(f"Learnt values after {self.episode_num} episodes:")
                vals = np.array(self.agent.value(numpy_batch.obs, self.agent.critic_network_params)[0, :-1, 0])
                print(vals.round(3))
                print()
                print(f"Learnt policy after {self.episode_num} episodes:")
                policy, _ = self.agent.policy(numpy_batch.obs, self.agent.actor_network_params)
                policy = np.array(policy)[:, :-1]
                print(policy.round(3))
                print()

            if checkpoint_after_ep:
                self.checkpoint()

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
