import torch
import torch.nn as nn


class EnergyConsequenceModel(nn.Module):
    """Action-conditioned high-level energy consequence predictor."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs, high_action):
        return self.net(torch.cat([obs, high_action], dim=-1))
