import numpy as np
from itertools import product


def all_lobster_states():
    all_states = np.array(list(product([0, 1, 2], [0, 1], [0, 1])))
    return all_states


def transition_prob(traverse_prob: float, pmfs_1: np.ndarray, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):

    transition_probs = np.ones_like(actions).astype(float)

    node_0_states = states[:, 0] == 0
    node_1_states = states[:, 0] == 1
    node_2_states = states[:, 0] == 2

    node_0_ns = next_states[:, 0] == 0
    node_1_ns = next_states[:, 0] == 1
    node_2_ns = next_states[:, 0] == 2

    # we first deal with impossible transitions
    impossible_move = ((states[:, 0] == 1) * (next_states[:, 0] == 2)) + ((states[:, 0] == 2) * (next_states[:, 0] == 1))

    # move then collecting
    collect_actions = actions == 2
    move_transitions = ~collect_actions
    indv_collected = next_states[:, 1:] < states[:, 1:]
    collected = np.any(indv_collected, axis=-1)
    impossible_collect = move_transitions * collected

    impossible = impossible_move | impossible_collect

    # Collecting in a wrong spot
    not_node_1_states = ~node_1_states
    collected_node_1 = indv_collected[:, 0]
    impossible |= (not_node_1_states & collected_node_1)

    not_node_2_states = ~node_2_states
    collected_node_2 = indv_collected[:, 1]
    impossible |= (not_node_2_states & collected_node_2)

    # Calculate movement probs.
    did_move = states[:, 0] != next_states[:, 0]

    # collecting then move
    impossible_collect_then_move = did_move * collect_actions

    impossible |= impossible_collect_then_move

    # moving in the wrong direction
    left_actions = actions == 0
    right_actions = actions == 1

    moved_left = (node_0_states & node_1_ns) | (node_2_states & node_0_ns)
    moved_right = (node_0_states & node_2_ns) | (node_1_states & node_0_ns)

    impossible_left = left_actions & moved_right
    impossible_right = right_actions & moved_left
    impossible |= impossible_left
    impossible |= impossible_right

    transition_probs[impossible] = 0.

    # If we're in node 1 and take a left, transition_prob == 1 for movement.
    might_move_left = left_actions * (~node_1_states)
    did_move_left = might_move_left & moved_left
    didnt_move_left = might_move_left * (~moved_left)

    # If we're in node 2 and take a right, transition_prob == 1 for movement.
    might_move_right = right_actions * (~node_2_states)
    did_move_right = might_move_right & moved_right
    didnt_move_right = might_move_right * (~moved_right)

    supposed_to_move = did_move_left | did_move_right
    failed_to_move = didnt_move_left | didnt_move_right

    # prob that you tried to move and did move
    transition_probs[supposed_to_move] *= traverse_prob

    # prob that you tried to move and didn't move
    transition_probs[failed_to_move] *= 1 - traverse_prob

    # now we figure out probabilities of rewards regenerating
    rewards_not_there = states[:, 1:] == 0
    rewards_still_not_there = next_states[:, 1:] == 0
    rewards_regenerated = rewards_not_there * (~rewards_still_not_there)  # b x 2
    rewards_not_regenerated = rewards_not_there * rewards_still_not_there

    r1_there = (~rewards_not_there)[:, 0]
    r2_there = (~rewards_not_there)[:, 1]

    # Probabilities for collecting.
    collect_r1 = node_1_states & collect_actions & r1_there
    collect_r1_no_regen = collect_r1 & rewards_still_not_there[:, 0]
    transition_probs[collect_r1_no_regen] *= (1 - pmfs_1[0])

    collect_r2 = node_2_states & collect_actions & r2_there
    collect_r2_no_regen = collect_r2 & rewards_still_not_there[:, 1]
    transition_probs[collect_r2_no_regen] *= (1 - pmfs_1[1])

    r1_still_there = (~rewards_still_not_there)[:, 0]
    collect_r1_then_regen = collect_r1 * r1_still_there
    rewards_regenerated[:, 0] = np.logical_or(rewards_regenerated[:, 0], collect_r1_then_regen)

    r2_still_there = (~rewards_still_not_there)[:, 1]
    collect_r2_then_regen = node_2_states * collect_actions * r2_there * r2_still_there
    rewards_regenerated[:, 1] = np.logical_or(rewards_regenerated[:, 1], collect_r2_then_regen)
    # transition_probs[collect_r1_then_regen | collect_r2_then_regen] *=

    rewards_regen_probs = rewards_regenerated * pmfs_1
    rewards_not_regen_probs = rewards_not_regenerated * (1 - pmfs_1)
    all_regen_probs = rewards_regen_probs + rewards_not_regen_probs

    # now we get our transition probabilities for reward regen
    all_regen_probs[all_regen_probs == 0] = 1
    prod_regen_probs = np.prod(all_regen_probs, axis=-1)

    transition_probs *= prod_regen_probs

    return transition_probs


def batch_reward(states: np.ndarray, actions: np.ndarray):
    node_1_states = states[:, 0] == 1
    node_2_states = states[:, 0] == 2

    collect_actions = actions == 2
    rewards_not_there = states[:, 1:] == 0
    r1_there = (~rewards_not_there)[:, 0]
    r2_there = (~rewards_not_there)[:, 1]
    collect_r1 = node_1_states & collect_actions & r1_there
    collect_r2 = node_2_states & collect_actions & r2_there
    return collect_r1.astype(int) + collect_r2.astype(int)

