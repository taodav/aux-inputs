from tests.envs import StateRewardingCompassWorld
from unc.envs import CompassWorld
from unc.envs.wrappers import RewardingWrapper, StateObservationWrapper

if __name__ == "__main__":
    test_steps = 1000
    sr_env = StateRewardingCompassWorld()
    wrapper_sr_env = StateObservationWrapper(RewardingWrapper(CompassWorld()))
    sr_prev_obs = sr_env.reset()
    wr_prev_obs = wrapper_sr_env.reset()

    wrapper_sr_env.state = sr_env.state
    for step in range(test_steps):
        act = sr_env.action_space.sample()

        sr_obs, sr_rew, sr_done, sr_info = sr_env.step(act)
        wr_obs, wr_rew, wr_done, wr_info = wrapper_sr_env.step(act)

        print(f"Action: {act}")
        assert (sr_obs == wr_obs).all(), f"Obs are different. " \
                                         f"sr_prev_obs: {sr_prev_obs}, " \
                                         f"sr_obs: {sr_obs}, " \
                                         f"wr_prev_obs: {wr_prev_obs}, " \
                                         f"wr_obs: {wr_obs}"

        assert sr_rew == wr_rew, f"Rewards are different. " \
                                 f"sr_rew: {sr_rew}, " \
                                 f"wr_rew: {wr_rew}"

        assert sr_done == wr_done, f"Terminals are different. " \
                                   f"sr_done: {sr_done}, " \
                                   f"wr_done: {wr_done}"

        sr_prev_obs = sr_obs
        wr_prev_obs = wr_obs

    print("All tests passed.")