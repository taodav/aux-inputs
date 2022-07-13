import numpy as np
import gym
from typing import Union, Tuple
from pathlib import Path

from .wrapper import LobsterFishingWrapper
# from .gvf import GVFWrapper
from .gvf_tc import GVFTileCodingWrapper
from unc.envs.lobster import LobsterFishing
from unc.utils.files import load_checkpoint


class FixedGVFWrapper(GVFTileCodingWrapper):
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper],
                 checkpoint_fname: Path):
        super(FixedGVFWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(9 + 2),
            high=np.ones(9 + 2),
        )
        trainer = load_checkpoint(checkpoint_fname)
        self.prediction_agent = trainer.agent

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        unwrapped_obs = self.unwrapped.get_obs(state)

        return np.concatenate((unwrapped_obs, self.predictions), axis=-1)

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.prediction_agent.reset()
        self.predictions = self.prediction_agent.current_gvf_predictions[0]
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Update our predictions
        prev_gvf_tc_obs = np.expand_dims(super(FixedGVFWrapper, self).get_obs(self.state), 0)
        self.prediction_agent.act(prev_gvf_tc_obs)
        self.predictions = self.prediction_agent.current_gvf_predictions[0]

        # Step
        _, reward, done, info = self.env.step(action)

        # Get new obs with new predictions
        return self.get_obs(self.state), reward, done, info
