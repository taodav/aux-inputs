from typing import Union

from unc.envs import Environment, LobsterFishing
from unc.envs.wrappers.lobster import LobsterFishingWrapper
from unc.envs.simple_chain import SimpleChain
from unc.envs.wrappers.simple_chain.wrapper import SimpleChainWrapper
from .base import GeneralValueFunction
from .lobster import LobsterGVFs
from .simple_chain import SimpleChainGVF


def get_gvfs(env: Union[Environment], gamma: float = None) -> GeneralValueFunction:
    if isinstance(env, LobsterFishing) or isinstance(env, LobsterFishingWrapper):
        return LobsterGVFs(env.action_space.n, gamma)
    elif isinstance(env, SimpleChain) or isinstance(env, SimpleChainWrapper):
        return SimpleChainGVF(env.action_space.n, gamma)
    else:
        raise NotImplementedError

