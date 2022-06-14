import numpy as np
import jax
import json
from pathlib import Path

import unc.envs.wrappers.compass as cw
import unc.envs.wrappers.rocksample as rw
import unc.envs.wrappers.tiger as tw
import unc.envs.wrappers.four_room as fr
import unc.envs.wrappers.lobster as lf
import unc.envs.wrappers.ocean_nav as on

from unc.args import Args
from .compass import CompassWorld
from .fixed import FixedCompassWorld
from .rocksample import RockSample
from .tiger import Tiger
from .base import Environment
from .simple_chain import SimpleChain
from .dynamic_chain import DynamicChain
from .four_room import FourRoom
from .lobster import LobsterFishing
from .ocean_nav import OceanNav
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
    'o': rw.ObsCountObservationWrapper,
    'n': rw.RockObservationStatsWrapper,
}

tiger_wrapper_map = {
    'g': tw.BeliefStateObservationWrapper,
    'p': tw.TigerParticleFilterWrapper
}

four_room_wrapper_map = {
    'o': fr.BoundedDecayingTraceObservationWrapper
}

lobster_wrapper_map = {
    'o': lf.BoundedDecayingTraceObservationWrapper,
    's': lf.GroundTruthStateWrapper,
    'p': lf.LobsterParticleFilterWrapper,
    'b': lf.BeliefStateWrapper,
    'g': lf.GVFWrapper,
    't': lf.GVFTileCodingWrapper
}

ocean_nav_wrapper_map = {
    'v': on.VectorStateObservationWrapper,
    'a': on.AgentCentricObservationWrapper,
    'p': on.PartiallyObservableWrapper,
    'm': on.ObservationMapWrapper,
    'f': on.FishingWrapper
}


def get_env(rng: np.random.RandomState, rand_key: jax.random.PRNGKey, args: Args):
    env_str = args.env

    kwargs = dict(blur_prob=args.blur_prob,
                  random_start=args.random_start,
                  slip_prob=args.slip_prob,
                  slip_turn=args.slip_turn,
                  size=args.size,
                  n_particles=args.n_particles,
                  update_weight_interval=args.update_weight_interval,
                  rock_obs_init=args.rock_obs_init,
                  half_efficiency_distance=args.half_efficiency_distance,
                  count_decay=args.count_decay,
                  trace_decay=args.trace_decay,
                  unnormalized_counts=args.unnormalized_counts,
                  po_degree=args.po_degree,
                  distance_noise=args.distance_noise,
                  distance_unc_encoding=args.distance_unc_encoding,
                  uncertainty_decay=args.uncertainty_decay,
                  task_fname=args.task_fname,
                  random_reward_start=args.random_reward_start,
                  max_episode_steps=args.max_episode_steps)

    if "r" in env_str:
        # r for rocksample
        env_str = env_str.replace('r', '')
        env = get_rocksample_env(rng, rand_key, env_str, **kwargs)
    elif "2" in env_str:
        # 2 for 2 room / lobster env
        env_str = env_str.replace('2', '')
        env = get_lobster_env(rng, env_str, **kwargs)
    elif "t" in env_str:
        # t for tiger env
        env = get_tiger_env(rng, env_str, **kwargs)
    elif "u" in env_str:
        # u for underwater / ocean nav
        # u has to go before 4 or 2, cuz u1, u2, u3 == ocean nav 1, 2, 3 etc.
        env_str = env_str.replace('u', '')
        assert any(c.isdigit() for c in env_str), "Missing OceanNav task specification"
        env = get_ocean_nav_env(rng, env_str, **kwargs)
    elif "4" in env_str:
        # 4 for 4 room
        env_str = env_str.replace('4', '')
        env = get_four_room_env(rng, env_str, **kwargs)
    else:
        env = get_compass_env(rng, env_str=env_str, **kwargs)
    return env


def get_ocean_nav_env(rng: np.random.RandomState,
                      env_str: str,
                      task_fname: str = "task_{}_config.json",
                      config_dir: Path = Path(ROOT_DIR, 'unc', 'envs', 'configs', 'ocean_nav'),
                      distance_noise: bool = True,
                      render: bool = True,
                      uncertainty_decay: float = 1.,
                      slip_prob: float = 0.,
                      random_reward_start: bool = False,
                      **kwargs):
    # get the first digit
    task_num = None
    for c in env_str:
        if c.isdigit():
            assert task_num is None, "More than one digit in env string"
            task_num = c
    task_fname = task_fname.format(task_num)
    config_path = config_dir / task_fname
    with open(config_path, 'r+') as f:
        config = json.load(f)

    env_str = env_str.replace(task_num, '')
    env = OceanNav(rng, config, slip_prob=slip_prob)

    list_w = list(set(env_str))
    wrapper_list = [ocean_nav_wrapper_map[w] for w in list_w if ocean_nav_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)
    for w in ordered_wrapper_list:
        if w == on.PartiallyObservableWrapper:
            env = w(env, distance_noise=distance_noise)
        elif w == on.ObservationMapWrapper:
            env = w(env, distance_noise=distance_noise, uncertainty_decay=uncertainty_decay)
        elif w == on.FishingWrapper:
            env = w(env, random_reward_start=random_reward_start)
        else:
            env = w(env)

    if render:
        env = on.OceanNavRenderWrapper(env)

    return env


def get_lobster_env(rng: np.random.RandomState,
                    env_str: str,
                    traverse_prob: float = 0.6,
                    render: bool = True,
                    trace_decay: float = 0.8,
                    n_particles: int = 100,
                    max_episode_steps: int = 200,
                    **kwargs):

    env = LobsterFishing(rng, traverse_prob=traverse_prob)
    list_w = list(set(env_str))
    wrapper_list = [lobster_wrapper_map[w] for w in list_w if lobster_wrapper_map[w] is not None]

    ordered_wrapper_list = sorted(wrapper_list, key=lambda w: w.priority)
    for w in ordered_wrapper_list:
        if w == lf.BoundedDecayingTraceObservationWrapper:
            env = w(env, decay=trace_decay)
        elif w == lf.LobsterParticleFilterWrapper:
            env = w(env, n_particles=n_particles)
        elif w == lf.GVFTileCodingWrapper:
            env = w(env, max_episode_steps=max_episode_steps)
        else:
            env = w(env)

    if render:
        env = lf.LobsterFishingRenderWrapper(env)

    return env


def get_four_room_env(rng: np.random.RandomState,
                      env_str: str = "4",
                      render: bool = True,
                      trace_decay: float = 0.99,
                      **kwargs):
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


def get_compass_env(rng: np.random.RandomState, env_str: str = "s",
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


def get_tiger_env(rng: np.random.RandomState, env_str: str = "t",
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
