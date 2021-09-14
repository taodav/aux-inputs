import timeit
import numpy as np
from unc.envs import CompassWorld, FixedCompassWorld


def iter_get_obs(states, get_obs):
    all_obs = []
    for s in states:
        all_obs.append(get_obs(s))
    return np.stack(all_obs)


def iter_transition(states, actions, transition):
    all_next_states = []
    for s, a in zip(states, actions):
        all_next_states.append(transition(s, a))
    return np.stack(all_next_states)

def iter_emit_prob(states, obs, get_obs):
    emit_probs = []
    for s in states:
        ground_truth_obs = get_obs(s)
        emit_probs.append(float((ground_truth_obs == obs).all()))
    return np.stack(emit_probs)


if __name__ == "__main__":
    env = FixedCompassWorld(size=9)
    states = env.sample_states(1000)

    # Correctness
    assert np.all(iter_get_obs(states, env.get_obs) == env.batch_get_obs(states)), "GET_OBS: Batch doesn't match iterative!"

    # # Timing batchify get_obs
    # iter_get_obs_time = timeit.timeit('iter_get_obs(states, env.get_obs)', globals=globals(), number=1000)
    # batch_get_obs_time = timeit.timeit('env.batch_get_obs(states)', globals=globals(), number=1000)
    # print(f"Iterative get_obs: {iter_get_obs_time} \t Batch get_obs: {batch_get_obs_time}")

    actions = np.random.choice(np.arange(3), states.shape[0])

    iter_transitions = iter_transition(states, actions, env.transition)
    batch_transitions = env.batch_transition(states, actions)
    assert np.all(iter_transitions == batch_transitions), "TRANSITION: Batch doesn't match iterative!"

    # iter_transition_time = timeit.timeit('iter_transition(states, actions, env.transition)', globals=globals(), number=1000)
    # batch_transition_time = timeit.timeit('env.batch_transition(states, actions)', globals=globals(), number=1000)
    #
    # print(f"Iterative transition: {iter_transition_time} \t Batch transition: {batch_transition_time}")

    all_possible_states = env.sample_all_states()
    all_possible_obs = np.concatenate((np.eye(5), np.zeros((1, 5))), axis=0)
    for obs in all_possible_obs:
        iter_emit = iter_emit_prob(all_possible_states, obs, env.get_obs)
        batch_emit = env.emit_prob(all_possible_states, obs)
        assert np.all(iter_emit == batch_emit), "EMIT PROB: Batch doesn't match iterative!"

    print("All tests passed.")

