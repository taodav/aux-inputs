import gym
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from PyFixedReps import TileCoder

from .wrapper import LobsterFishingWrapper
from .gvf_tc import GVFTileCodingWrapper
from unc.envs.lobster import LobsterFishing
from unc.utils.files import load_checkpoint


class FixedGVFTileCodingWrapper(GVFTileCodingWrapper):
    priority = 3

    def __init__(self, env: Union[LobsterFishing, LobsterFishingWrapper],
                 checkpoint_fname: Path,
                 tile_code_obs: bool = False):
        super(FixedGVFTileCodingWrapper, self).__init__(env)
        trainer = load_checkpoint(checkpoint_fname)
        self.prediction_agent = trainer.agent
        self.tile_code_obs = tile_code_obs
        before_tc_features = 2
        if self.tile_code_obs:
            before_tc_features += 9

        self.tc = TileCoder({
            'tiles': 8,
            'tilings': 8,
            'dims': before_tc_features,

            'input_ranges': [(0, 1) for _ in range(before_tc_features)],
            'scale_output': False
        })

        after_tc_features = self.tc.features()
        if not self.tile_code_obs:
            after_tc_features += 9

        self.observation_space = gym.spaces.Box(
            low=np.zeros(after_tc_features),
            high=np.ones(after_tc_features),
        )

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        unwrapped_obs = self.unwrapped.get_obs(state).astype(float)
        if self.tile_code_obs:
            non_tc_obs = np.concatenate((unwrapped_obs, self.predictions))
            obs = self.tc.encode(non_tc_obs)
        else:
            tc_predictions = self.tc.encode(self.predictions).astype(float)
            obs = np.concatenate((unwrapped_obs, tc_predictions))
        return obs

    def reset(self, **kwargs) -> np.ndarray:
        self.env.reset(**kwargs)
        self.prediction_agent.reset()
        self.predictions = self.prediction_agent.current_gvf_predictions[0]
        return self.get_obs(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Update our predictions
        prev_gvf_obs = np.expand_dims(super(FixedGVFTileCodingWrapper, self).get_obs(self.state), 0)
        self.prediction_agent.act(prev_gvf_obs)
        self.predictions = self.prediction_agent.current_gvf_predictions[0]
        _, reward, done, info = self.env.step(action)

        return self.get_obs(self.state), reward, done, info
