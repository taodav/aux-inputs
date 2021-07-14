import numpy as np
from typing import Union

from unc.envs.wrappers import CompassWorldWrapper
from unc.envs.compass import CompassWorld


class BlurryWrapper(CompassWorldWrapper):
    priority = 1

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper],
                 blur_prob: float = 0.1, *args, **kwargs):
        """
        Stochastic version of Compass World where the observation emittance function is stochastic.
        We add stochasticity by changing our observation function to emit a different colored wall
        selected at random w.p. blur_prob.
        :param blur_prob:
        :param kwargs:
        """
        super(BlurryWrapper, self).__init__(env, *args, **kwargs)
        self.blur_prob = blur_prob

        assert self.blur_prob <= 1.0, "Probability exceeds 1."

    def get_obs(self, state: np.ndarray) -> np.ndarray:

        if self.rng.random() < self.blur_prob:
            # In this case, we select a random color to emit.
            obs = np.zeros(5)
            random_idx = self.rng.choice(np.arange(self.observation_space.shape[0]))
            obs[random_idx] = 1
            return obs

        return super(BlurryCompassWorld, self).get_obs(state)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:

        ground_truth_obs = super(BlurryCompassWorld, self).get_obs(state)
        if (ground_truth_obs == obs).all():
            return (1 - self.blur_prob) + self.blur_prob

        return self.blur_prob

