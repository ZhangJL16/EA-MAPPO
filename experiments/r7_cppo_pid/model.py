from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import chain
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from experiments.jacobian_energy_bridge.features import StructuredLidarFeatureExtractor


@dataclass(frozen=True)
class R7ModelConfig:
    horizontal_sectors: int = 128
    vertical_sectors: int = 8
    hidden_dim: int = 256
    action_dim: int = 3
    initial_action_std: float = 0.5

    def __post_init__(self) -> None:
        if min(
            self.horizontal_sectors,
            self.vertical_sectors,
            self.hidden_dim,
            self.action_dim,
        ) <= 0:
            raise ValueError("all R7 model dimensions must be positive")
        if not 0.0 < self.initial_action_std <= 2.0:
            raise ValueError("initial_action_std must lie in (0, 2]")

    @property
    def lidar_sectors(self) -> int:
        return self.horizontal_sectors * self.vertical_sectors

    @property
    def observation_dim(self) -> int:
        return 7 + 2 * self.lidar_sectors

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _orthogonal_initialize(module: nn.Module, *, output_gain: float | None = None) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for layer in linear_layers:
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.zeros_(layer.bias)
    if output_gain is not None and linear_layers:
        nn.init.orthogonal_(linear_layers[-1].weight, gain=output_gain)


class R7ActorCritic(nn.Module):
    """R3-structured feed-forward actor with a separate two-head CMDP critic.

    The actor and critic use independent copies of the exact structured LiDAR
    extractor family used by R3.  Only the critic trunk is shared between the
    reward and scalar safety-cost values.  Actor gradients therefore cannot be
    clipped or redirected by value regression.
    """

    def __init__(self, config: R7ModelConfig) -> None:
        super().__init__()
        self.config = config
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(config.observation_dim,),
            dtype=np.float32,
        )
        extractor_kwargs = {
            "horizontal_sectors": config.horizontal_sectors,
            "vertical_sectors": config.vertical_sectors,
        }
        self.actor_extractor = StructuredLidarFeatureExtractor(
            observation_space,
            **extractor_kwargs,
        )
        self.critic_extractor = StructuredLidarFeatureExtractor(
            observation_space,
            **extractor_kwargs,
        )
        feature_dim = int(self.actor_extractor.features_dim)
        self.actor_body = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.action_mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), float(np.log(config.initial_action_std)))
        )
        self.critic_body = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )
        self.reward_value = nn.Linear(config.hidden_dim, 1)
        self.cost_value = nn.Linear(config.hidden_dim, 1)
        _orthogonal_initialize(self.actor_body)
        _orthogonal_initialize(self.action_mean, output_gain=0.01)
        _orthogonal_initialize(self.critic_body)
        _orthogonal_initialize(self.reward_value, output_gain=1.0)
        _orthogonal_initialize(self.cost_value, output_gain=1.0)

    @property
    def observation_dim(self) -> int:
        return self.config.observation_dim

    def actor_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.actor_extractor.parameters(),
            self.actor_body.parameters(),
            self.action_mean.parameters(),
            (self.log_std,),
        )

    def critic_parameters(self) -> Iterable[nn.Parameter]:
        return chain(
            self.critic_extractor.parameters(),
            self.critic_body.parameters(),
            self.reward_value.parameters(),
            self.cost_value.parameters(),
        )

    def _normal(self, observations: torch.Tensor) -> Normal:
        features = self.actor_extractor(observations)
        mean = self.action_mean(self.actor_body(features))
        standard_deviation = self.log_std.exp().clamp(0.05, 1.5).expand_as(mean)
        return Normal(mean, standard_deviation)

    @staticmethod
    def _squashed_log_prob(distribution: Normal, latent: torch.Tensor) -> torch.Tensor:
        # Stable log(1 - tanh(z)^2) from the SAC change-of-variables identity.
        log_jacobian = 2.0 * (np.log(2.0) - latent - F.softplus(-2.0 * latent))
        return (distribution.log_prob(latent) - log_jacobian).sum(dim=-1)

    def values(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.critic_extractor(observations)
        hidden = self.critic_body(features)
        return torch.cat((self.reward_value(hidden), self.cost_value(hidden)), dim=-1)

    def act(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._normal(observations)
        latent = distribution.mean if deterministic else distribution.sample()
        actions = torch.tanh(latent)
        log_prob = self._squashed_log_prob(distribution, latent)
        # CPPO-PID's reference PPO uses Gaussian entropy.  The transform is fixed
        # and entropy is a diagnostic/regularizer rather than a likelihood term.
        entropy = distribution.entropy().sum(dim=-1)
        return actions, latent, log_prob, entropy, self.values(observations)

    def evaluate_latents(
        self,
        observations: torch.Tensor,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self._normal(observations)
        return (
            self._squashed_log_prob(distribution, latents),
            distribution.entropy().sum(dim=-1),
        )

    def architecture_audit(self) -> dict[str, object]:
        actor_parameters = sum(parameter.numel() for parameter in self.actor_parameters())
        critic_parameters = sum(parameter.numel() for parameter in self.critic_parameters())
        return {
            "class": type(self).__name__,
            "config": self.config.as_dict(),
            "actor_extractor": self.actor_extractor.architecture_audit(),
            "critic_extractor": self.critic_extractor.architecture_audit(),
            "actor_critic_features_shared": False,
            "critic_heads": ["reward_value", "scalar_cmdp_cost_value"],
            "policy_distribution": "diagonal_gaussian_tanh_squashed",
            "route_planner": None,
            "waypoint_controller": None,
            "actor_parameters": actor_parameters,
            "critic_parameters": critic_parameters,
            "trainable_parameters": actor_parameters + critic_parameters,
        }
