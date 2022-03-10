import numpy as np
from typing import List
from unc.utils.viz import plot_arr, stringify_actions_q_vals
from unc.utils.viz.rocksample import generate_greedy_action_array

# FOR DEBUGGING rocksample
def plot_current_state(env, agent):
    obs = np.array([env.get_obs(env.state)])
    qs = agent.Qs(obs, agent.network_params)
    action = np.argmax(qs, axis=1)
    qs = qs[0]

    greedy_action_arr = generate_greedy_action_array(env, agent)

    render = env.render(action=action, q_vals=qs, show_weights=True, show_rock_info=True,
                        greedy_actions=greedy_action_arr)
    plot_arr(render)


def teleported_rock_q_vals(env, agent, rock_idx: int):
    # Now we teleport our state
    state = env.state
    new_state = state.copy()
    new_state[:2] = env.rock_positions[rock_idx].copy()
    new_obs = np.array([env.get_obs(new_state)])
    q_val = agent.Qs(new_obs, agent.network_params)
    return q_val, new_state


def all_unchecked_rock_q_vals(env, agent, checked: List[bool]):
    unchecked_rocks_info = {}
    for i, check in enumerate(checked):
        if not check:
            action = 5 + i
            before_check_qvals, teleported_state = teleported_rock_q_vals(env, agent, i)
            checked_state, new_particles, new_weights = env.transition(teleported_state, action,
                                                                       env.particles, env.weights)
            checked_obs = np.array([env.get_obs(checked_state, particles=new_particles, weights=new_weights)])
            after_check_qvals = agent.Qs(checked_obs, agent.network_params)

            unchecked_rocks_info[tuple(env.rock_positions[i])] = {
                'before': np.array(before_check_qvals[0]),
                'after': np.array(after_check_qvals[0]),
                'morality': env.rock_morality[i]
            }
    return unchecked_rocks_info


def summarize_checks(env, unchecked: dict):
    all_mor = {}
    all_good_diff = np.zeros(env.action_space.n)
    all_bad_diff = np.zeros(env.action_space.n)
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
    print(stringify_actions_q_vals(env.action_map, all_good_diff))
    print(f"Bad rocks average Q value differences: ")
    print(stringify_actions_q_vals(env.action_map, all_bad_diff))

    return all_mor, all_good_diff, all_bad_diff

# END DEBUGGING
