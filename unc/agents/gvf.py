import haiku as hk
from jax import random
from optax import GradientTransformation
from typing import Iterable, List

from unc.utils.gvfs import GeneralValueFunction
from unc.args import Args
from .dqn import DQNAgent


class GVFAgent(DQNAgent):
    def __init__(self, gvfs: List[GeneralValueFunction],
                 network: hk.Transformed,
                 optimizer: GradientTransformation,
                 features_shape: Iterable[int],
                 n_actions: int,
                 rand_key: random.PRNGKey,
                 args: Args):

        super(GVFAgent, self).__init__(network, optimizer, features_shape, n_actions, rand_key, args)
        self.gvfs = gvfs