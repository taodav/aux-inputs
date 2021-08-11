from .compass import CompassWorld
from .fixed import FixedCompassWorld
from .base import Environment
from unc.envs.wrappers import *

wrapper_map = {
    's': StateObservationWrapper,
    'i': SlipWrapper,
    'b': BlurryWrapper,
    'p': ParticleFilterWrapper,
    'g': GlobalStateObservationWrapper,
    'l': LocalStateObservationWrapper,
    'm': None,
    'v': None,
    'f': None
}

def get_env(seed: int, env_str: str = "s",
            random_start: bool = True,
            blur_prob: float = 0.1,
            slip_prob: float = 0.1,
            update_weight_interval: int = 1,
            size: int = 8,
            resample_interval: int = None,
            n_particles: int = -1,
            render: bool = True):

    ground_truth = False
    if "g" in env_str and "s" in env_str:
        ground_truth = True
        env_str = env_str.replace("s", "")

    if "p" in env_str and "g" not in env_str:
        env_str += "l"

    # Don't do any resampling in deterministic environments.
    if "i" not in env_str:
        resample_interval = None

    list_w = list(set(env_str))
    wrapper_list = [wrapper_map[w] for w in list_w if wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    if "f" in env_str:
        env = FixedCompassWorld(seed=seed, random_start=random_start, size=size)
    else:
        env = CompassWorld(seed=seed, random_start=random_start, size=size)

    for w in ordered_wrapper_list:
        if w == BlurryWrapper:
            env = w(env, blur_prob=blur_prob)
        elif w == SlipWrapper:
            env = w(env, slip_prob=slip_prob)
        elif w == ParticleFilterWrapper:
            env = w(env, update_weight_interval=update_weight_interval,
                    resample_interval=resample_interval, n_particles=n_particles)
        elif w == LocalStateObservationWrapper:
            env = w(env, mean_only='m' in env_str, vars_only='v' in env_str)
        elif w == GlobalStateObservationWrapper:
            assert "m" not in env_str and "v" not in env_str, "'m' or 'v' doesn't make sense with 'w'"
            env = w(env, ground_truth=ground_truth)
        else:
            env = w(env)

    if render:
        env = RenderWrapper(env)

    return env
