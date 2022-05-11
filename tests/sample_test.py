import numpy as np
import time as tm
from jax import random, jit
import jax.numpy as jnp

from unc.utils.data import sample_idx_batch, sample_seq_idxes


def sample_uniform_batch(batch_size: int, rand_key: random.PRNGKey):
    new_rand_key, sample_rand_key = random.split(rand_key, 2)
    unif = random.uniform(sample_rand_key, (batch_size,))
    return unif, new_rand_key


def multiply(unif: jnp.ndarray, length: int):
    return unif * length


if __name__ == "__main__":
    seed = 2020
    batch_size = 32
    seq_len = 10
    rng = np.random.RandomState(seed)
    rand_key = random.PRNGKey(seed)
    random_idxes = rng.randint(0, 100000, size=100000)

    idx = None

    # t_start = tm.time()
    # # first we time how long choice takes
    # for _ in range(1000000):
    #     idx = rng.choice(random_idxes, size=batch_size)
    # t_end = tm.time()
    # print(f"Sequential choices: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")
    #
    # t_start = tm.time()
    # # now we try uniform sequential
    # for i in range(1000000):
    #     unif = rng.rand(batch_size)
    #     idx = np.floor(unif * i)
    #
    # t_end = tm.time()
    # print(f"Sequential uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")
    #
    # t_start = tm.time()
    # # now we try batch_uniform
    # unif_steps = rng.rand(1000000, batch_size)
    # for i in range(1000000):
    #     idx = np.round(unif_steps[i] * random_idxes.shape[0])
    #
    # t_end = tm.time()
    # print(f"Batch uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")

    # # Now we try things with JAX
    # t_start = tm.time()
    # # now we try uniform sequential
    # jitted_sample_uniform_batch = jit(sample_uniform_batch, static_argnums=0)
    #
    # for i in range(1000000):
    #     unif, rand_key = jitted_sample_uniform_batch(batch_size, rand_key)
    #     idx = multiply(unif, i)
    #
    # t_end = tm.time()
    # print(f"Sequential separate uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")
    #
    # t_start = tm.time()
    # # now we try uniform sequential
    #
    jitted_sample_idx_batch = jit(sample_idx_batch, static_argnums=0)
    # for i in range(1000000):
    #     idx, rand_key = jitted_sample_idx_batch(batch_size, i, rand_key)
    #
    # t_end = tm.time()
    # print(f"Sequential together uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")
    #
    # t_start = tm.time()
    # # now we try batch_uniform
    #
    # new_rand_key, sample_rand_key = random.split(rand_key, 2)
    # unif_steps = random.uniform(sample_rand_key, (1000000, batch_size))
    # jitted_multiply = jit(multiply)
    # for i in range(1000000):
    #     idx = jitted_multiply(unif_steps[i], i)
    #
    # t_end = tm.time()
    # print(f"Sequential batch uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {idx}")

    t_start = tm.time()
    # now we try batch_uniform with sequential indices, that we turn into sequences

    new_rand_key, sample_rand_key = random.split(rand_key, 2)

    for i in range(50000):
        idx, rand_key = jitted_sample_idx_batch(batch_size, i, rand_key)
        seq_range = jnp.arange(seq_len, dtype=int)[:, None]
        sample_seq_idx = (idx + seq_range).T % 100000

    t_end = tm.time()
    print(f"Sequential sequence split uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {sample_seq_idx}")

    t_start = tm.time()
    # now we try batch_uniform with sequential indices, that we turn into sequences

    new_rand_key, sample_rand_key = random.split(rand_key, 2)
    jitted_sample_seq_idxes = jit(sample_seq_idxes, static_argnums=(0, 1, 2))

    for i in range(50000):
        idx, rand_key = jitted_sample_seq_idxes(batch_size, 100000, seq_len, i, rand_key)

    t_end = tm.time()
    print(f"Sequential sequence combined uniform: {(t_end - t_start)} secs")
    # print(f"Sample idx: {sample_seq_idx}")
