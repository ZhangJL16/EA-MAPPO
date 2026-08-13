from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class ScalarEnergyCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.softplus(self.network(features)).squeeze(-1)


class MonotoneQuantileCritic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        quantile_levels: tuple[float, ...] = (0.50, 0.90, 0.95, 0.99),
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not quantile_levels or any(not 0.0 < value < 1.0 for value in quantile_levels):
            raise ValueError("quantile levels must lie in (0, 1)")
        if tuple(sorted(quantile_levels)) != quantile_levels:
            raise ValueError("quantile levels must be strictly ordered")
        self.quantile_levels = quantile_levels
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.base_head = nn.Linear(hidden_dim, 1)
        self.increment_head = nn.Linear(hidden_dim, len(quantile_levels) - 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(features)
        base = functional.softplus(self.base_head(hidden))
        if len(self.quantile_levels) == 1:
            return base
        increments = functional.softplus(self.increment_head(hidden))
        return torch.cat((base, base + torch.cumsum(increments, dim=-1)), dim=-1)
