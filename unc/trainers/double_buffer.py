import numpy as np
from typing import Union, List
from time import time, ctime

from unc.args import Args
from unc.envs import RockSample
from unc.envs.wrappers.rocksample import RockSampleWrapper
from unc.agents import Agent
from unc.utils import ReplayBuffer
from unc.utils.data import Batch
from unc.utils.viz import generate_greedy_action_array

from .trainer import Trainer

# DEBUGGING
from unc.utils import plot_arr
from unc.utils.viz import plot_current_state, stringify_actions_q_vals



class DoubleBufferTrainer(Trainer):
    def __init__(self, args: Args, agent: Agent, env: Union[RockSample, RockSampleWrapper],
                 prefilled_buffer: ReplayBuffer, p_prefilled: float = 0.0, buffer_size: int = 20000):
        """
        Double buffer trainer. Essentially Sarsa except with two experience replay buffers.

        The first experience replay buffer is pre-filled with experience from some policy.
        In this case, the policy is the rock sampler agent.

        The second experience replay buffer is for the agent's own experiences.
        :param args: arguments
        :param agent: agent to train
        :param env: environment to train on (currently only supports rocksample)
        :param prefilled_buffer: buffer pre-filled from a certain policy
        """
        super(DoubleBufferTrainer, self).__init__(args, agent, env)

        self.batch_size = args.batch_size

        # self.buffer = ReplayBuffer(args.total_steps, self.env.rng, self.env.observation_space.shape)
        self.buffer = ReplayBuffer(args.buffer_size, self.env.rng, self.env.observation_space.shape)
        self.prefilled_buffer = prefilled_buffer

        self.p_prefilled = args.p_prefilled

    # FOR DEBUGGING

    def teleported_rock_q_vals(self, rock_idx: int):
        # Now we teleport our state
        state = self.env.state
        new_state = state.copy()
        new_state[:2] = self.env.rock_positions[rock_idx].copy()
        new_obs = np.array([self.env.get_obs(new_state)])
        q_val = self.agent.Qs(new_obs, self.agent.network_params)
        return q_val, new_state


    def all_unchecked_rock_q_vals(self, checked: List[bool]):
        unchecked_rocks_info = {}
        for i, check in enumerate(checked):
            if not check:
                action = 5 + i
                before_check_qvals, teleported_state = self.teleported_rock_q_vals(i)
                checked_state, new_particles, new_weights = self.env.transition(teleported_state, action, self.env.particles, self.env.weights)
                checked_obs = np.array([self.env.get_obs(checked_state, particles=new_particles, weights=new_weights)])
                after_check_qvals = self.agent.Qs(checked_obs, self.agent.network_params)

                unchecked_rocks_info[tuple(self.env.rock_positions[i])] = {
                    'before': before_check_qvals.squeeze(0).numpy(),
                    'after': after_check_qvals.squeeze(0).numpy(),
                    'morality': self.env.rock_morality[i]
                }
        return unchecked_rocks_info

    def summarize_checks(self, unchecked: dict):
        all_mor = {}
        all_good_diff = np.zeros(self.env.action_space.n)
        all_bad_diff = np.zeros(self.env.action_space.n)
        good, bad = 0, 0
        for pos, info in unchecked.items():
            all_mor[pos] = info['morality']
            before = info['before']
            after = info['after']
            diff = after - before
            if all_mor[pos]:
                good += 1
                all_good_diff += diff
            else:
                bad += 1
                all_bad_diff += diff
        if good > 0:
            all_good_diff /= good
        if bad > 0:
            all_bad_diff /= bad
        print(f"Unchecked rock moralities: {all_mor}\n"
              f"Good rocks average Q value differences: ")
        print(stringify_actions_q_vals(self.env.action_map, all_good_diff))
        print(f"Bad rocks average Q value differences: ")
        print(stringify_actions_q_vals(self.env.action_map, all_bad_diff))

        return all_mor, all_good_diff, all_bad_diff

    # END DEBUGGING

    def train(self) -> None:
        assert self.info is not None, "Reset the trainer before training"
        time_start = time()

        self._print(f"Begin training at {ctime(time_start)}")

        # DEBUGGING
        time_to_check = False
        good_diffs = []
        bad_diffs = []
        moralities = []

        # Timing stuff
        prev_time = time_start
        log_interval = 1000
        total_target_updates = self.total_steps // log_interval
        num_logs = 0
        avg_time_per_log = 0

        while self.num_steps < self.total_steps:
            episode_reward = 0
            episode_loss = 0
            obs = np.expand_dims(self.env.reset(), 0)
            action = self.agent.act(obs).item()

            # DEBUGGING
            checked_rocks_info = {}

            # DEBUGGING: if we check a rock that's never been sampled before
            if time_to_check and 'p' in self.args.env:
                unchecked_q_vals = self.all_unchecked_rock_q_vals(self.env.checked_rocks)
                checked_rocks_info[self.num_steps] = unchecked_q_vals
                all_mor, all_good_diff, all_bad_diff = self.summarize_checks(unchecked_q_vals)
                moralities.append(all_mor)
                good_diffs.append(all_good_diff)
                bad_diffs.append(all_bad_diff)
                time_to_check = False

            for t in range(self.max_episode_steps):
                self.agent.set_eps(self.get_epsilon())

                next_obs, reward, done, info = self.env.step(action)
                next_obs = np.array([next_obs])

                next_action = self.agent.act(next_obs).item()

                self.buffer.push({
                    'obs': obs, 'reward': reward, 'done': done, 'action': action, 'next_obs': next_obs,
                    'next_action': next_action
                })

                self.info['reward'].append(reward)

                prefilled_bs = int(self.batch_size * self.p_prefilled)
                online_bs = self.batch_size - prefilled_bs
                if online_bs > len(self.buffer):
                    online_bs = len(self.buffer)
                    prefilled_bs = self.batch_size - online_bs

                prefilled_sample = self.prefilled_buffer.sample(prefilled_bs)
                sample = self.buffer.sample(online_bs)

                batch_obs = np.concatenate([prefilled_sample['obs'], sample['obs']])
                batch_reward = np.concatenate([prefilled_sample['reward'], sample['reward']])
                batch_action = np.concatenate([prefilled_sample['action'], sample['action']])
                batch_next_obs = np.concatenate([prefilled_sample['next_obs'], sample['next_obs']])
                batch_done = np.concatenate([prefilled_sample['done'], sample['done']])
                batch_next_action = np.concatenate([prefilled_sample['next_action'], sample['next_action']])

                batch_gamma = (1 - batch_done) * self.discounting

                loss = self.agent.update(Batch(batch_obs, batch_action, batch_next_obs, batch_gamma, batch_reward,
                                         batch_next_action))

                # Logging
                episode_loss += loss
                episode_reward += reward
                self.info['loss'].append(loss)
                self.num_steps += 1

                if self.num_steps % log_interval == 0:
                    time_to_check = True

                    num_logs += 1
                    curr_time = time()
                    time_per_fix_freq = curr_time - prev_time
                    avg_time_per_log += (1 / num_logs) * (time_per_fix_freq - avg_time_per_log)
                    time_remaining = (total_target_updates - num_logs) * avg_time_per_log
                    self._print(f"Remaining time: {time_remaining / 60:.2f}")
                    prev_time = curr_time

                if done:
                    break

                obs = next_obs
                action = next_action

            self.episode_num += 1
            self.info['episode_reward'].append(episode_reward)
            self.info['episode_length'].append(t + 1)

            avg_over = min(self.episode_num, 30)
            self._print(f"Episode {self.episode_num}, steps: {t + 1}, "
                        f"total steps: {self.num_steps}, "
                        f"moving avg steps: {sum(self.info['episode_length'][-avg_over:]) / avg_over:.3f}, "
                        f"rewards: {episode_reward:.2f}, "
                        f"avg episode loss: {episode_loss / (t + 1):.4f}")

        # DEBUGGING
        self.info['moralities'] = moralities
        self.info['good_diffs'] = good_diffs
        self.info['bad_diffs'] = bad_diffs

        time_end = time()
        self._print(f"Ending training at {ctime(time_end)}")
