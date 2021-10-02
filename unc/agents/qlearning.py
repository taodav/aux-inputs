import numpy as np
import torch
import torch.nn.functional as F

from .sarsa import SarsaAgent


class QLearningAgent(SarsaAgent):
    """
    TODO: Something something target networks & double Q learning
    """

    def update(self, state: np.ndarray,
               action: np.ndarray,
               next_state: np.ndarray,
               gamma: np.ndarray,
               reward: np.ndarray,
               **kwargs) -> float:
        """
        Update our model using the Sarsa(0) target.
        :param state: (b x *state.shape) State to update.
        :param action: (b) Action taken in the above state
        :param next_state: (b x *state.shape) Next state for targets
        :param gamma: (b) Discount factor. Our done masking is done here too (w/ gamma = 0.0 if terminal)
        :param reward: (b) Rewards
        :return: loss from update
        """
        q = self.Q(state, action)

        # Here we have our Q-learning target
        with torch.no_grad():
            q1s = self.model(self.preprocess_state(next_state))
            q1 = q1s.max(dim=1)[0]

        # Casting
        gamma = torch.tensor(gamma, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        loss = F.mse_loss(q, reward + gamma * q1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
