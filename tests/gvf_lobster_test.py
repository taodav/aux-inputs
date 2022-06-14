import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from unc.utils.data import Batch, preprocess_step
from unc.utils.gvfs import get_gvfs
from unc.args import Args
from unc.envs import get_env
from unc.agents import get_agent
from unc.models import build_network
from unc.optim import get_optimizer


if __name__ == "__main__":
    parser = Args()
    args = parser.parse_args()

    args.env = '2t'
    # args.env = '2g'
    args.arch = 'nn'
    args.discounting = 0.9
    args.n_hidden = 5
    args.step_size = 0.01
    args.total_steps = 250000
    args.max_episode_steps = 200
    args.seed = 2020

    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    rand_key = random.PRNGKey(args.seed)

    jax.config.update('jax_platform_name', args.platform)

    train_env_key, rand_key = random.split(rand_key, 2)
    train_env = get_env(rng, train_env_key, args)

    output_size = train_env.action_space.n
    features_shape = train_env.observation_space.shape

    # GVFs for Lobster env.
    gvf = get_gvfs(train_env, gamma=args.discounting)
    n_actions_gvfs = train_env.action_space.n
    output_size += gvf.n
    gvf_idxes = train_env.gvf_idxes
