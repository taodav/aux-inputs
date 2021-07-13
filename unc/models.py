import torch
from torch import nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, n_features: int, n_hidden: int, n_actions: int):
        super(QNetwork, self).__init__()

        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_actions = n_actions

        self.l1 = nn.Linear(self.n_features, self.n_hidden)
        self.l2 = nn.Linear(self.n_hidden, self.n_actions)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.l1(x))
        out = self.l2(out)
        return out


