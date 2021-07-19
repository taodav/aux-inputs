from .compass import CompassWorld
from unc.envs.wrappers import *

wrapper_map = {
    'r': RewardingWrapper,
    's': StateObservationWrapper,
    'b': BlurryWrapper,
    'p': ParticleFilterWrapper,
    'm': None
}

def get_env(seed: int, env_str: str = "sr",
            random_start: bool = True,
            blur_prob: float = 0.1,
            update_weight_interval: int = 1):
    list_w = list(set(env_str))
    wrapper_list = [wrapper_map[w] for w in list_w if wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    env = CompassWorld(seed=seed, random_start=random_start)

    for w in ordered_wrapper_list:
        if w == BlurryWrapper:
            env = w(env, blur_prob=blur_prob)
        elif w == ParticleFilterWrapper:
            env = w(env, update_weight_interval=update_weight_interval,
                    mean_only='m' in env_str)
        else:
            env = w(env)

    return env
