import numpy as np
from unc.envs.directional_tmaze import DirectionalTMaze


if __name__ == "__main__":
    seed = 2020
    rng = np.random.RandomState(seed)

    env = DirectionalTMaze(rng)
    env.reset()

    assert env.state[-1] == 1

    env.state = np.array([0, 0, 0, 1, 0, 1])

    # Bump into a wall
    obs, _, _, _ = env.step(0)
    assert obs[0] == 1 and env.state[0] == 0

    # Turn right
    obs, rew, done, _ = env.step(1)
    assert np.all(obs == 0)

    for _ in range(9):
        obs, _, _, _ = env.step(0)

    assert obs[0] == 1

    obs, _, _, _ = env.step(2)
    assert obs[0] == 0

    obs, rew, done, _ = env.step(0)
    assert obs[0] == 1 and rew == 4 and done

    env.reset()

    env.state = np.array([0, 9, 1, 1, 0, 1])

    env.step(1)
    obs, rew, done, _ = env.step(0)

    assert obs[0] == 1 and rew == -1 and done

    print("Finished t-maze tests")



