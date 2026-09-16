from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CollisionRiskEstimate:
    mean_probability: np.ndarray
    aleatoric_std: np.ndarray
    epistemic_std: np.ndarray
    uncalibrated_upper: np.ndarray


class _RiskMember(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class CollisionRiskEnsemble(nn.Module):
    def __init__(self, input_dim: int, *, members: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        if members < 2:
            raise ValueError("epistemic ensemble requires at least two members")
        self.members = nn.ModuleList(_RiskMember(input_dim, hidden_dim) for _ in range(members))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(features) for member in self.members], dim=0)

    def estimate(self, features: torch.Tensor, *, epistemic_scale: float = 2.0) -> CollisionRiskEstimate:
        if epistemic_scale < 0.0:
            raise ValueError("epistemic_scale must be nonnegative")
        with torch.no_grad():
            probabilities = torch.sigmoid(self.logits(features))
            mean = probabilities.mean(dim=0)
            epistemic = probabilities.std(dim=0, unbiased=True)
            aleatoric = torch.sqrt(torch.clamp(mean * (1.0 - mean), min=0.0))
            upper = torch.clamp(mean + epistemic_scale * epistemic, 0.0, 1.0)
        return CollisionRiskEstimate(
            mean_probability=mean.cpu().numpy(),
            aleatoric_std=aleatoric.cpu().numpy(),
            epistemic_std=epistemic.cpu().numpy(),
            uncalibrated_upper=upper.cpu().numpy(),
        )
