from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from torch.nn import functional as F


class StructuredLidarFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        horizontal_sectors: int,
        vertical_sectors: int,
        goal_embedding_dim: int = 32,
        lidar_channels: tuple[int, int] = (16, 32),
    ) -> None:
        if horizontal_sectors <= 0 or vertical_sectors <= 0:
            raise ValueError("LiDAR sector counts must be positive")
        if goal_embedding_dim <= 0 or any(channel <= 0 for channel in lidar_channels):
            raise ValueError("feature dimensions must be positive")
        self.horizontal_sectors = int(horizontal_sectors)
        self.vertical_sectors = int(vertical_sectors)
        self.lidar_sector_count = self.horizontal_sectors * self.vertical_sectors
        expected_dim = 7 + 2 * self.lidar_sector_count
        if observation_space.shape != (expected_dim,):
            raise ValueError(
                f"structured LiDAR extractor expected observation shape {(expected_dim,)}, "
                f"got {observation_space.shape}"
            )
        lidar_embedding_dim = 2 * int(lidar_channels[-1])
        super().__init__(
            observation_space,
            features_dim=int(goal_embedding_dim + lidar_embedding_dim),
        )
        self.goal_encoder = nn.Sequential(
            nn.Linear(7, goal_embedding_dim),
            nn.ReLU(),
            nn.Linear(goal_embedding_dim, goal_embedding_dim),
            nn.ReLU(),
        )
        self.lidar_convolutions = nn.ModuleList(
            [
                nn.Conv2d(2, lidar_channels[0], kernel_size=3),
                nn.Conv2d(lidar_channels[0], lidar_channels[1], kernel_size=3),
            ]
        )

    def split_observation(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[1] != self._observation_space.shape[0]:
            raise ValueError("observations must have shape (batch, observation_dim)")
        goal = observations[:, :7]
        ranges = observations[:, 7 : 7 + self.lidar_sector_count]
        valid = observations[:, 7 + self.lidar_sector_count :]
        lidar = torch.stack(
            (
                ranges.reshape(-1, self.vertical_sectors, self.horizontal_sectors),
                valid.reshape(-1, self.vertical_sectors, self.horizontal_sectors),
            ),
            dim=1,
        )
        return goal, lidar

    @staticmethod
    def _circular_azimuth_bounded_elevation_pad(values: torch.Tensor) -> torch.Tensor:
        values = F.pad(values, (1, 1, 0, 0), mode="circular")
        return F.pad(values, (0, 0, 1, 1), mode="replicate")

    def lidar_feature_map(self, lidar: torch.Tensor) -> torch.Tensor:
        features = lidar
        for convolution in self.lidar_convolutions:
            features = self._circular_azimuth_bounded_elevation_pad(features)
            features = F.relu(convolution(features))
        return features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        goal, lidar = self.split_observation(observations)
        goal_features = self.goal_encoder(goal)
        lidar_map = self.lidar_feature_map(lidar)
        lidar_mean = F.adaptive_avg_pool2d(lidar_map, output_size=1).flatten(1)
        lidar_max = F.adaptive_max_pool2d(lidar_map, output_size=1).flatten(1)
        return torch.cat((goal_features, lidar_mean, lidar_max), dim=1)

    def architecture_audit(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "observation_slicing": {
                "goal_velocity": [0, 7],
                "lidar_ranges": [7, 7 + self.lidar_sector_count],
                "lidar_valid": [
                    7 + self.lidar_sector_count,
                    7 + 2 * self.lidar_sector_count,
                ],
            },
            "lidar_tensor_shape": [2, self.vertical_sectors, self.horizontal_sectors],
            "azimuth_padding": "circular",
            "elevation_padding": "replicate_bounded",
            "features_dim": self.features_dim,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }
