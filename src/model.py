"""A small multilayer perceptron for reward regression."""

import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)
