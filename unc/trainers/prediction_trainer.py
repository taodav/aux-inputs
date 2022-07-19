import gym
import numpy as np
from typing import Union
from time import time, ctime
from pathlib import Path
from collections import deque

from unc.args import Args
from unc.agents import GVFPredictionAgent
from unc.utils.data import Batch, preprocess_step, get_action_encoding
from unc.utils.gvfs import GeneralValueFunction
from .trainer import Trainer


class PredictionTrainer(Trainer):
    def __init__(self,
                 args: Args,
                 agent: GVFPredictionAgent,
                 env: Union[gym.Env, gym.Wrapper],
                 test_env: Union[gym.Env, gym.Wrapper],
                 checkpoint_dir: Path = None,
                 gvf: GeneralValueFunction = None
                 ):
        super(PredictionTrainer, self).__init__(args, agent, env, test_env,
                                                checkpoint_dir=checkpoint_dir, gvf=gvf)

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # Timing stuff
        prev_time = time_start
        log_interval = 1000 if self.offline_eval_freq == 0 else self.offline_eval_freq
        total_target_updates = self.total_steps // log_interval
        num_logs = 0
        avg_time_per_log = 0

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0

            checkpoint_after_ep = False

            # For RNN training
            # trunc_batch = None
            # if self.trunc > 0:
            #     trunc_batch = Batch(obs=deque(maxlen=self.trunc), action=deque(maxlen=self.trunc),
            #                         next_obs=deque(maxlen=self.trunc), gamma=deque(maxlen=self.trunc),
            #                         reward=deque(maxlen=self.trunc), next_action=deque(maxlen=self.trunc),
            #                         state=deque(maxlen=self.trunc), next_state=deque(maxlen=self.trunc))

            # Cumulant predictions for GVF training
            # s_1 predictions
            prev_predictions = self.agent.reset()

            # o_1 observations
            prev_obs = self.env.reset()

            # Action conditioning
            # if self.action_cond == 'cat':
            #     action_encoding = get_action_encoding(self.agent.features_shape, -1, self.n_actions)
            #     obs = np.concatenate([obs, action_encoding], axis=-1)

            prev_obs = np.expand_dims(prev_obs, 0)

            # Hidden state for RNN training
            # hs, next_hs = None, None
            # if self.trunc > 0:
            #     hs = self.agent.state[None, :]

            # a_1 action,
            prev_action = self.agent.act(prev_obs)

            self.agent.set_eps(self.get_epsilon())

            # we need o_2 observation here for our predictions
            obs, reward, done, info = self.env.step(prev_action.item())

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                # now we actually update our predictions
                # for predictions, we have v(s_{t - 1}, a_{t - 1}, o_t)
                predictions = self.agent.predictions(prev_predictions, prev_action, obs)

                action = self.agent.act(obs)

                # if self.trunc > 0:
                #     next_hs = self.agent.state[None, :]

                next_obs, reward, done, info = self.env.step(action.item())

                # Action conditioning
                # if self.action_cond == 'cat':
                #     action_encoding = get_action_encoding(self.agent.features_shape, action, self.n_actions)
                #     next_obs = np.concatenate([next_obs, action_encoding], axis=-1)

                self.info['reward'].append(reward)

                # Preprocess everything for updating
                next_obs, reward, done, info, action = preprocess_step(next_obs, reward, done, info, action.item())

                gamma = (1 - done) * self.discounting

                current_pi = self.agent.policy(obs)

                # so now batch holds
                # obs=o_t, prev_predictions=s_{t - 1}, action=a_{t - 1}
                # cumulant=c_{t + 1}
                # next_obs=o_{t + 1}, predictions=s_t, action=a_t
                batch = Batch(obs=obs, prev_predictions=prev_predictions, prev_action=prev_action,
                              cumulants=self.gvf.cumulant(next_obs),
                              cumulant_terminations=self.gvf.termination(next_obs),
                              next_obs=next_obs, predictions=predictions,
                              action=action,
                              gamma=gamma, reward=reward,
                              impt_sampling_ratio=self.gvf.impt_sampling_ratio(next_obs, current_pi, action))

                # This is for real-time RNN training
                # if self.trunc > 0:
                #     batch, trunc_batch = self.collect_rnn_batch(batch, hs, next_hs, trunc_batch)

                loss, other_info = self.agent.update(batch)

                # Logging
                episode_loss += loss
                episode_reward += reward[0]
                self.num_steps += 1

                # Offline evaluation
                if self.offline_eval_freq > 0 and self.num_steps % self.offline_eval_freq == 0:
                    self.offline_evaluation()

                # Logging and timing
                if self.num_steps % log_interval == 0:
                    time_to_check = True

                    num_logs += 1
                    curr_time = time()
                    time_per_fix_freq = curr_time - prev_time
                    avg_time_per_log += (1 / num_logs) * (time_per_fix_freq - avg_time_per_log)
                    time_remaining = (total_target_updates - num_logs) * avg_time_per_log
                    self._print(f"Remaining time: {time_remaining / 60:.2f}")
                    prev_time = curr_time

                if self.checkpoint_freq > 0 and self.num_steps % self.checkpoint_freq == 0:
                    checkpoint_after_ep = True

                if done.item():
                    break

                obs = next_obs
                prev_action = action
                prev_predictions = predictions
                # hs = next_hs

            self.episode_num += 1
            self.post_episode_print(episode_reward, episode_loss, t)

            if checkpoint_after_ep:
                self.checkpoint()

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
