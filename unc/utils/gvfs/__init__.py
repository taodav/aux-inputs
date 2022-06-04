from typing import List, Union

from unc.envs import Environment, LobsterFishing
from unc.envs.wrappers.lobster import LobsterFishingWrapper
from .base import GeneralValueFunction
from .lobster import LobsterGVFs


def get_gvfs(env: Union[Environment]) -> List[GeneralValueFunction]:
    gvfs = []
    if isinstance(env, LobsterFishing) or isinstance(env, LobsterFishingWrapper):
        gvfs.append(LobsterGVFs(env.action_space.n))
    else:
        raise NotImplementedError

    return gvfs
