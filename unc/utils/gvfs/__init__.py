from typing import Union

from unc.envs.base import Environment
from unc.envs.lobster import LobsterFishing
from unc.envs.wrappers.lobster import LobsterFishingWrapper
from unc.envs.simple_chain import SimpleChain
from unc.envs.wrappers.simple_chain.wrapper import SimpleChainWrapper
from .base import GeneralValueFunction
from .lobster import LobsterGVFs, LobsterR1GVFs, LobsterR2GVFs
from .simple_chain import SimpleChainGVF


def get_gvfs(env: Union[Environment], gvf_type: str = 'both', gamma: float = None) -> GeneralValueFunction:
    if isinstance(env, LobsterFishing) or isinstance(env, LobsterFishingWrapper):
        if gvf_type == 'both':
            return LobsterGVFs(env.action_space.n, gamma)
        elif gvf_type == 'r1':
            return LobsterR1GVFs(env.action_space.n, gamma)
        elif gvf_type == 'r2':
            return LobsterR2GVFs(env.action_space.n, gamma)
    elif isinstance(env, SimpleChain) or isinstance(env, SimpleChainWrapper):
        return SimpleChainGVF(env.action_space.n, gamma)
    else:
        raise NotImplementedError

