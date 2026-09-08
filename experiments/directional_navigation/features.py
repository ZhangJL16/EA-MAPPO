"""Versioned spatial readout; the historical R3 extractor is never modified."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import functional as F

from experiments.jacobian_energy_bridge.features import StructuredLidarFeatureExtractor


class DirectionalLidarExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        horizontal_sectors: int = 128,
        vertical_sectors: int = 8,
        grid: tuple[int, int] = (2, 16),
        remaining_time: bool = False,
        readout: str = "ordered",
    ) -> None:
        if readout not in {"ordered", "global_tiled"}:
            raise ValueError("readout must be ordered or global_tiled")
        if len(grid) != 2 or any(type(v) is not int or v <= 0 for v in grid):
            raise ValueError("grid must contain two positive integers")
        if vertical_sectors % grid[0] or horizontal_sectors % grid[1]:
            raise ValueError("grid must divide the input angular dimensions")
        self.base_dim = 7 + 2 * horizontal_sectors * vertical_sectors
        self.remaining_time = remaining_time
        self.grid = grid
        self.observation_dim = self.base_dim + int(remaining_time)
        self.readout = readout
        if observation_space.shape != (self.base_dim + int(remaining_time),):
            raise ValueError("unexpected directional observation contract")
        super().__init__(
            observation_space,
            features_dim=32 + 64 * grid[0] * grid[1] + int(remaining_time),
        )
        base_space = gym.spaces.Box(
            observation_space.low[: self.base_dim],
            observation_space.high[: self.base_dim],
            dtype=observation_space.dtype.type,
        )
        self.spatial = StructuredLidarFeatureExtractor(
            base_space,
            horizontal_sectors=horizontal_sectors,
            vertical_sectors=vertical_sectors,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError("observations must match the batched observation contract")
        goal, lidar = self.spatial.split_observation(observations[:, : self.base_dim])
        maps = self.spatial.lidar_feature_map(lidar)
        output_grid = self.grid if self.readout == "ordered" else (1, 1)
        mean = F.adaptive_avg_pool2d(maps, output_grid)
        maximum = F.adaptive_max_pool2d(maps, output_grid)
        if self.readout == "global_tiled":
            # Equal downstream parameter count, without adding bearing information.
            mean = mean.expand(-1, -1, *self.grid)
            maximum = maximum.expand(-1, -1, *self.grid)
        parts = [self.spatial.goal_encoder(goal), mean.flatten(1), maximum.flatten(1)]
        if self.remaining_time:
            parts.append(observations[:, -1:])
        return torch.cat(parts, dim=1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "version": "directional_readout_v1",
            "readout": self.readout,
            "grid": list(self.grid),
            "remaining_time": self.remaining_time,
            "features_dim": self.features_dim,
            "backbone": self.spatial.architecture_audit(),
            "parameters": sum(p.numel() for p in self.parameters()),
        }
