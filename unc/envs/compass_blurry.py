import numpy as np

from .compass import CompassWorld


class BlurryCompassWorld(CompassWorld):
    def __init__(self, blur_prob: float = 0.1, **kwargs):
        """
        Stochastic version of Compass World where the observation emittance function is stochastic.
        We add stochasticity by changing our observation function to emit a different colored wall
        selected at random w.p. blur_prob.
        :param blur_prob:
        :param kwargs:
        """
        super(BlurryCompassWorld, self).__init__(**kwargs)
        self.blur_prob = blur_prob

        assert self.blur_prob <= 1.0, "Probability exceeds 1."

    def get_obs(self, state: np.ndarray) -> np.ndarray:

        if self._rng.random() < self.blur_prob:
            # In this case, we select a random color to emit.
            obs = np.zeros(5)
            random_idx = self._rng.choice(np.arange(self.observation_space.shape[0]))
            obs[random_idx] = 1
            return obs

        return super(BlurryCompassWorld, self).get_obs(state)

    def emit_prob(self, state: np.ndarray, obs: np.ndarray) -> float:

        ground_truth_obs = super(BlurryCompassWorld, self).get_obs(state)
        if (ground_truth_obs == obs).all():
            return (1 - self.blur_prob) + self.blur_prob

        return self.blur_prob

