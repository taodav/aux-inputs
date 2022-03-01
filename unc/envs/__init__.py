import numpy as np
import jax
from pathlib import Path
import unc.envs.wrappers.compass as cw
import unc.envs.wrappers.rocksample as rw
import unc.envs.wrappers.tiger as tw
import unc.envs.wrappers.four_room as fr

from .compass import CompassWorld
from .fixed import FixedCompassWorld
from .rocksample import RockSample
from .tiger import Tiger
from .base import Environment
from .simple_chain import SimpleChain
from .dynamic_chain import DynamicChain
from .four_room import FourRoom
from definitions import ROOT_DIR


compass_wrapper_map = {
    's': cw.StateObservationWrapper,
    'i': cw.SlipWrapper,
    'b': cw.BlurryWrapper,
    'p': cw.CompassParticleFilterWrapper,
    'g': cw.GlobalStateObservationWrapper,
    'l': cw.LocalStateObservationWrapper,
    'c': cw.StateCountObservationWrapper,
    'o': cw.ObsCountObservationWrapper,
    'n': cw.ObservationStatsWrapper,
    'd': cw.NoisyCorridorObservationWrapper,
    'm': None,
    'v': None,
    'f': None
}

rocksample_wrapper_map = {
    'g': rw.GlobalStateObservationWrapper,
    'l': rw.LocalStateObservationWrapper,
    'p': rw.RocksParticleFilterWrapper,
    'x': rw.PerfectSensorWrapper,
    'c': rw.StateCountObservationWrapper,
    'o': rw.ObsCountObservationWrapper
}

tiger_wrapper_map = {
    'g': tw.BeliefStateObservationWrapper,
    'p': tw.TigerParticleFilterWrapper
}

four_room_wrapper_map = {
    'o': fr.BoundedDecayingTraceObservationWrapper
}


def get_env(rng: np.random.RandomState, rand_key: jax.random.PRNGKey, env_str: str = "r", *args, **kwargs):
    if "r" in env_str:
        env_str = env_str.replace('r', '')
        env = get_rocksample_env(rng, rand_key, env_str, *args, **kwargs)
    elif "t" in env_str:
        env = get_tiger_env(rng, env_str, *args, **kwargs)
    elif "4" in env_str:
        env_str = env_str.replace('4', '')
        env = get_four_room_env(rng, env_str, *args, **kwargs)
    else:
        env = get_compass_env(rng, *args, env_str=env_str, **kwargs)
    return env


def get_four_room_env(rng: np.random.RandomState,
                      env_str: str = "4",
                      render: bool = True,
                      trace_decay: float = 0.99,
                      *args, **kwargs):
    env = FourRoom(rng)
    list_w = list(set(env_str))
    wrapper_list = [four_room_wrapper_map[w] for w in list_w if four_room_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)
    for w in ordered_wrapper_list:
        if w == fr.BoundedDecayingTraceObservationWrapper:
            env = w(env, decay=trace_decay)
        else:
            env = w(env)

    if render:
        env = fr.FourRoomRenderWrapper(env)

    return env


def get_rocksample_env(rng: np.random.RandomState,
                       rand_key: jax.random.PRNGKey, env_str: str = "r",
                       config_path: Path = Path(ROOT_DIR, "unc", "envs", "configs", "rock_sample_config.json"),
                       *args,
                       update_weight_interval: int = 1,
                       half_efficiency_distance: float = 20.,
                       rock_obs_init: float = 0.,
                       resample_interval: int = None,
                       count_decay: float = 1.,
                       unnormalized_counts: bool = False,
                       n_particles: int = 100,
                       render: bool = True,
                       **kwargs):
    ground_truth = False
    if ("g" in env_str or "l" in env_str) and "s" in env_str:
        ground_truth = True
        env_str = env_str.replace("s", "")

    list_w = list(set(env_str))
    wrapper_list = [rocksample_wrapper_map[w] for w in list_w if rocksample_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    env = RockSample(config_path, rng, rand_key, rock_obs_init=rock_obs_init, half_efficiency_distance=half_efficiency_distance)
    for w in ordered_wrapper_list:
        if w == rw.RocksParticleFilterWrapper:
            env = w(env, update_weight_interval=update_weight_interval,
                    resample_interval=resample_interval,
                    n_particles=n_particles)
        elif w == rw.GlobalStateObservationWrapper:
            env = w(env, ground_truth=ground_truth)
        elif w == rw.LocalStateObservationWrapper:
            env = w(env, ground_truth=ground_truth)
        elif w == rw.StateCountObservationWrapper or w == rw.ObsCountObservationWrapper:
            env = w(env, decay=count_decay, normalize=not unnormalized_counts)
        else:
            env = w(env)

    if render:
        env = rw.RockRenderWrapper(env)

    return env


def get_compass_env(rng: np.random.RandomState, *args, env_str: str = "s",
                    random_start: bool = True,
                    blur_prob: float = 0.1,
                    slip_prob: float = 0.1,
                    slip_turn: bool = True,
                    update_weight_interval: int = 1,
                    size: int = 8,
                    resample_interval: int = None,
                    n_particles: int = -1,
                    count_decay: float = 1.,
                    po_degree: float = 0.,
                    unnormalized_counts: bool = False,
                    render: bool = True,
                    **kwargs):

    ground_truth = False
    if "g" in env_str and "s" in env_str:
        ground_truth = True
        env_str = env_str.replace("s", "")

    if "p" in env_str and "g" not in env_str and "n" not in env_str:
        env_str += "l"

    # Don't do any resampling in deterministic environments.
    if "i" not in env_str:
        resample_interval = None

    list_w = list(set(env_str))
    wrapper_list = [compass_wrapper_map[w] for w in list_w if compass_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    if "f" in env_str:
        env = FixedCompassWorld(rng, random_start=random_start, size=size)
    else:
        env = CompassWorld(rng, random_start=random_start, size=size)

    for w in ordered_wrapper_list:
        if w == cw.BlurryWrapper:
            env = w(env, blur_prob=blur_prob)
        elif w == cw.SlipWrapper:
            env = w(env, slip_prob=slip_prob, slip_turn=slip_turn)
        elif w == cw.CompassParticleFilterWrapper:
            env = w(env, update_weight_interval=update_weight_interval,
                    resample_interval=resample_interval, n_particles=n_particles)
        elif w == cw.LocalStateObservationWrapper:
            env = w(env, mean_only='m' in env_str, vars_only='v' in env_str)
        elif w == cw.GlobalStateObservationWrapper:
            assert "m" not in env_str and "v" not in env_str, "'m' or 'v' doesn't make sense with 'w'"
            env = w(env, ground_truth=ground_truth)
        elif w == cw.StateCountObservationWrapper or w == cw.ObsCountObservationWrapper:
            env = w(env, decay=count_decay, normalize=not unnormalized_counts)
        elif w == cw.NoisyCorridorObservationWrapper:
            env = w(env, po_degree=po_degree)
        else:
            env = w(env)

    if render:
        env = cw.CompassRenderWrapper(env)

    return env


def get_tiger_env(rng: np.random.RandomState, *args, env_str: str = "t",
                  update_weight_interval: int = 1,
                  resample_interval: int = None,
                  n_particles: int = -1,
                  render: bool = True,
                  **kwargs):

    ground_truth = False
    if "g" in env_str and "s" in env_str:
        ground_truth = True
        env_str = env_str.replace("s", "")

    list_w = list(set(env_str))
    wrapper_list = [tiger_wrapper_map[w] for w in list_w if tiger_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)

    env = Tiger(rng)

    for w in ordered_wrapper_list:
        if w == tw.TigerParticleFilterWrapper:
            env = w(env, update_weight_interval=update_weight_interval,
                    resample_interval=resample_interval, n_particles=n_particles)
        elif w == tw.BeliefStateObservationWrapper:
            env = w(env, ground_truth=ground_truth)
        else:
            env = w(env)

    # if render:
    #     env = cw.CompassRenderWrapper(env)

    return env
