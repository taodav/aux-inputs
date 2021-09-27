import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pathlib import Path

from unc.args import Args
from unc.models import QNetwork

from .base import Agent


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
        self.args = args

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
        qs = self.Qs(state, model=model)
        return torch.argmax(qs, dim=1).cpu().numpy()

    def Qs(self, state: np.ndarray, model: nn.Module = None) -> torch.tensor:
        """
        Get all Q-values given a state.
        :param state: (b x *state.shape) State to find action-values
        :param model: Optional. Potenially use another model
        :return: (b x actions) torch.tensor full of action-values.
        """
        if model is None:
            model = self.model
        state = self.preprocess_state(state)
        return model(state)

    def Q(self, state: np.ndarray, action: np.ndarray, model: nn.Module = None) -> torch.Tensor:
        """
        Get action-values given a state and action
        :param state: (b x *state.shape) State to find action-values
        :param action: (b) Actions for action-values
        :param model: Optional. Potenially use another model
        :return: (b) Action-values
        """
        qs = self.Qs(state, model=model)
        action = torch.tensor(action, dtype=torch.long, device=self.device).unsqueeze(-1)
        return torch.gather(qs, dim=1, index=action).squeeze(dim=1)  # squeeze

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
            # q1s = self.model(self.preprocess_state(next_state))
            # q1 = q1s.max(dim=1)[0]

        # Casting
        gamma = torch.tensor(gamma, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        loss = F.mse_loss(q, reward + gamma * q1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: Path):
        """
        Saves the agent's parameters in a given path.
        :param path: path to save to (including file name).
        """
        to_save = {
            'model': self.model.state_dict(),
            'n_features': self.model.n_features,
            'n_hidden': self.model.n_hidden,
            'n_actions': self.model.n_actions,
            'optimizer': self.optimizer.state_dict(),
            'args': self.args.as_dict(),
            'rng': self._rng
        }
        torch.save(to_save, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> Agent:
        # This is hardcoded for now. In the future we'd want to models & agents based
        # on some helper function.
        loaded = torch.load(path)

        model = QNetwork(loaded['n_features'], loaded['n_hidden'], loaded['n_actions']).to(device)
        optimizer = torch.optim.Adam(model.parameters())

        # Load our state dicts
        model.load_state_dict(loaded['model'])
        optimizer.load_state_dict(loaded['optimizer'])

        args = Args()
        args.from_dict(loaded['args'])
        agent = SarsaAgent(model, optimizer, loaded['n_actions'], loaded['rng'], args)

        return agent





