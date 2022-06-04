import haiku as hk
import optax
from jax import random
from typing import Tuple, List

from .base import Agent
from .dqn import DQNAgent
from .noisy import NoisyNetAgent
from .rock_sampler import RockSamplerAgent
from .lstm import LSTMAgent
from .k_lstm import kLSTMAgent
from .dist_lstm import DistributionalLSTMAgent
from .gvf import GVFAgent
from unc.models import build_network
from unc.optim import get_optimizer
from unc.args import Args
from unc.utils.gvfs import GeneralValueFunction


def get_agent(args: Args, features_shape: Tuple[int, ...], n_actions: int, rand_key: random.PRNGKey,
              network: hk.Transformed, optimizer: optax.GradientTransformation,
              gvfs: List[GeneralValueFunction] = None):
    """
    Get our agent!
    """
    agent_key, rand_key = random.split(rand_key, 2)

    if 'lstm' in args.arch:
        # Currently we only do action conditioning with the LSTM agent.
        if args.action_cond == 'cat':
            features_shape = features_shape[:-1] + (features_shape[-1] + n_actions,)

        if args.k_rnn_hs > 1:
            # value network takes as input mean + variance of hidden states and cell states.
            value_network = build_network(args.n_hidden, n_actions, model_str="seq_value")
            value_optimizer = get_optimizer(args.optim, args.value_step_size)
            agent = kLSTMAgent(network, value_network, optimizer, value_optimizer,
                               features_shape, n_actions, agent_key, args)
        elif args.distributional:
            agent = DistributionalLSTMAgent(network, optimizer, features_shape,
                                            n_actions, agent_key, args)
        else:
            agent = LSTMAgent(network, optimizer, features_shape,
                              n_actions, agent_key, args)
    elif args.arch == 'nn' and args.exploration == 'noisy':
        agent = NoisyNetAgent(network, optimizer, features_shape,
                              n_actions, agent_key, args)
    else:
        if args.gvf_features > 0 :
            agent = GVFAgent(gvfs, network, optimizer, features_shape, n_actions,
                             agent_key, args)
        else:
            agent = DQNAgent(network, optimizer, features_shape,
                             n_actions, agent_key, args)

    return agent, rand_key
