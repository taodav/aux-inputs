import numpy as np
from typing import Union, Tuple

from unc.envs.wrappers import CompassWorldWrapper
from unc.envs.compass import CompassWorld


class SlipWrapper(CompassWorldWrapper):
    priority = 1

    def __init__(self, env: Union[CompassWorld, CompassWorldWrapper],
                 slip_prob: float = 0.1, slip_turn: bool = True, *args, **kwargs):
        """
        Stochastic version of Compass World where you have a chance of "slipping" when
        moving forward and remaining in the same grid. The probability of this happening
        is defined by slip_prob.
        :param slip_prob:
        :param kwargs:
        """
        super(SlipWrapper, self).__init__(env, *args, **kwargs)
        self.slip_prob = slip_prob
        self.slip_turn = slip_turn

        assert self.slip_prob <= 1.0, "Probability exceeds 1."

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Action description:
        0 = Forward, 1 = Turn Right, 2 = Turn Left
        :param action: Action to take
        :return:
        """
        assert action in list(range(0, 4)), f"Invalid action: {action}"

        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), {}

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self.slip_turn:
            same_actions = np.arange(actions.shape[0])
        else:
            same_actions = np.argwhere((actions == 0))[:, 0]

        same_mask = self.rng.choice([0, 1], p=[1 - self.slip_prob, self.slip_prob], size=same_actions.shape[0])
        same_idx = same_actions[same_mask.astype(bool)]
        next_states = self.env.batch_transition(states, actions)
        next_states[same_idx] = states[same_idx]

        return next_states

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        if action == 0:
            if self.rng.random() > self.slip_prob:
                new_state[:-1] += self.direction_mapping[state[-1]]

        elif action == 1:
            new_state[-1] = (state[-1] + 1) % 4
        elif action == 2:
            new_state[-1] = (state[-1] - 1) % 4

        # Wall interactions
        new_state = np.maximum(np.minimum(new_state, self.state_max), self.state_min)

        return new_state
