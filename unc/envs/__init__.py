from .compass import CompassWorld
from unc.envs.wrappers import *

wrapper_map = {
    'r': RewardingWrapper,
    's': StateObservationWrapper,
    'b': BlurryWrapper,
    'p': ParticleFilterWrapper
}

def get_env(seed: int, env_str: str = "sr",
            random_start: bool = True, blur_prob: float = 0.1):
    list_w = list(set(env_str))
    wrapper_list = [wrapper_map[w] for w in list_w]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    env = CompassWorld(seed=seed, random_start=random_start)

    for w in ordered_wrapper_list:
        if w == BlurryWrapper:
            env = w(env, blur_prob=blur_prob)
        else:
            env = w(env)

    return env
