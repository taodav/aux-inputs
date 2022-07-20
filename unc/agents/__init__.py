import haiku as hk
import optax
import numpy as np
from jax import random
from typing import Tuple

from .base import Agent
from .dqn import DQNAgent
from .noisy import NoisyNetAgent
from .rock_sampler import RockSamplerAgent
from .lstm import LSTMAgent
from .k_lstm import kLSTMAgent
from .dist_lstm import DistributionalLSTMAgent
from .gvf_control import GVFControlAgent
from .gvf_prediction import GVFPredictionAgent
from unc.models import build_network
from unc.optim import get_optimizer
from unc.args import Args


def get_agent(args: Args, features_shape: Tuple[int, ...], n_actions: int, rand_key: random.PRNGKey,
              network: hk.Transformed, optimizer: optax.GradientTransformation,
              n_predictions: int = 0, gvf_trainer: str = 'control'):
    """
    Get our agent!
    """
    agent_key, rand_key = random.split(rand_key, 2)

    if args.action_cond == 'cat':
        features_shape = features_shape[:-1] + (features_shape[-1] + n_actions,)

    if 'lstm' in args.arch:
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
        if n_predictions > 0:
            if gvf_trainer == 'control':
                # TODO: incorporate a GVFPredictionAgent into the control agent.
                raise NotImplementedError
                # agent = GVFControlAgent(gvf_idxes, network, optimizer, features_shape, n_actions,
                #                         agent_key, args)
            elif gvf_trainer == 'prediction':
                agent = GVFPredictionAgent(network, optimizer, features_shape, n_actions, n_predictions, rand_key, args)

        else:
            agent = DQNAgent(network, optimizer, features_shape,
                             n_actions, agent_key, args)

    return agent, rand_key
