import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any

from unc.args import Args


class Agent:
    def act(self, state: Any):
        raise NotImplementedError

    def update(self, state: Any, action: Any, next_state: Any, gamma: Any, reward: Any):
        raise NotImplementedError


class SarsaAgent(Agent):
    def __init__(self, model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 n_actions: int,
                 rng: np.random.RandomState,
                 args: Args):
        self.model = model
        self.optimizer = optimizer
        self.n_actions = n_actions
        self.eps = args.epsilon
        self.device = args.device

        self._rng = rng

    def set_eps(self, eps: float):
        self.eps = eps

    def preprocess_state(self, obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def act(self, state: np.ndarray, model: nn.Module = None) -> np.ndarray:
        """
        Get epsilon-greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param model: Optional. Potentially use another model to find action-values.
        :return: (b) epsilon-greedy actions
        """
        if self._rng.random() > self.eps:
            return self.greedy_act(state, model)
        return self._rng.choice(np.arange(self.n_actions), size=state.shape[0])

    def greedy_act(self, state: np.ndarray, model: nn.Module = None) -> np.ndarray:
        """
        Get greedy actions given a state
        :param state: (b x *state.shape) State to find actions
        :param model: Optional. Potentially use another model to find action-values.
        :return: (b) Greedy actions
        """
        state = self.preprocess_state(state)
        if model is None:
            q = self.model(state)
        else:
            q = model(state)
        return torch.argmax(q, dim=1).cpu().numpy()

    def Q(self, state: np.ndarray, action: np.ndarray, model: nn.Module = None) -> torch.Tensor:
        """
        Get action-values given a state and action
        :param state: (b x *state.shape) State to find action-values
        :param action: (b) Actions for action-values
        :param model: Optional. Potenially use another model
        :return: (b) Action-values
        """
        if model is None:
            model = self.model
        state = self.preprocess_state(state)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        return torch.gather(model(state), dim=1, index=action).squeeze(dim=1)  # squeeze

    def update(self, state: np.ndarray,
               action: np.ndarray,
               next_state: np.ndarray,
               gamma: np.ndarray,
               reward: np.ndarray) -> float:
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

        # Here we have our epsilon-soft Sarsa target
        with torch.no_grad():
            next_action = self.act(next_state)
            q1 = self.Q(next_state, next_action)

        # Casting
        gamma = torch.tensor(gamma, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        loss = F.mse_loss(q, reward + gamma * q1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()






