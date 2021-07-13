from .compass import CompassWorld
from .compass_blurry import BlurryCompassWorld
from .compass_reward import RewardingCompassWorld
from .compass_state_reward import StateRewardingCompassWorld


def get_env(env_str: str = "sr"):
    if env_str == "sr":
        return StateRewardingCompassWorld
    elif env_str == "r":
        return RewardingCompassWorld
    else:
        raise NotImplementedError()
