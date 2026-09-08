"""One-factor v2 ablation: range-derived hit basis, no extra sensor data."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

from experiments.directional_navigation.deployable_observation import (
    DIM, FIELDS, DeployableRecovery, DeployableLidarExtractor,
)
from experiments.jacobian_energy_bridge.features import StructuredLidarFeatureExtractor

VERSION = "deployable_distance_derived_hit_v3"


class DerivedHitLidarExtractor(DeployableLidarExtractor):
    """Retain initial v2 function; learn the new channel from zero weights.

All v2 parameters and the global CPU RNG stream stay unchanged at construction.
Only144 first-convolution weights per extractor are added. With fixed float32
100m range normalization,1 denotes the max-range sentinel. This is a threshold
basis computed from observed distance, not simulator hit identities/confidence.
"""

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space)
        old = self.lidar_convolutions[0]
        # SB3 constructs extractors on CPU, then moves the policy to its device.
        # Do not shift RNG draws for subsequent policy heads/critic networks.
        with torch.random.fork_rng(devices=[]):
            replacement = nn.Conv2d(2, 16, kernel_size=3)
        with torch.no_grad():
            replacement.weight[:, :1].copy_(old.weight)
            replacement.weight[:, 1:].zero_()
            replacement.bias.copy_(old.bias)
        self.lidar_convolutions[0] = replacement

    @staticmethod
    def sensor_channels(observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != DIM:
            raise ValueError("derived-hit encoder requires the unchanged1039-D input")
        ranges = observations[:, 7:1031].reshape(-1, 1, 8, 128)
        return torch.cat((ranges, (ranges < 1.0).to(ranges.dtype)), dim=1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.sensor_channels(observations)
        for convolution in self.lidar_convolutions:
            features = StructuredLidarFeatureExtractor._circular_azimuth_bounded_elevation_pad(features)
            features = F.relu(convolution(features))
        return torch.cat((
            self.goal_encoder(observations[:, :7]),
            F.adaptive_avg_pool2d(features, (2, 16)).flatten(1),
            F.adaptive_max_pool2d(features, (2, 16)).flatten(1),
            observations[:, 1031:],
        ), dim=1)


def policy_kwargs() -> dict:
    return {"features_extractor_class": DerivedHitLidarExtractor,
            "share_features_extractor": False,
            "net_arch": {"pi": [256, 256], "qf": [256, 256]}}
