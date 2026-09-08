from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta


@dataclass(frozen=True)
class R6ModelConfig:
    horizontal_sectors: int = 128
    vertical_sectors: int = 8
    horizontal_fov_degrees: float = 360.0
    vertical_fov_degrees: float = 60.0
    core_dim: int = 64
    ray_dim: int = 64
    attention_heads: int = 4
    recurrent_dim: int = 128
    feedback_dim: int = 5
    action_dim: int = 3
    beta_floor: float = 1.05

    def __post_init__(self) -> None:
        positive = (
            self.horizontal_sectors,
            self.vertical_sectors,
            self.core_dim,
            self.ray_dim,
            self.attention_heads,
            self.recurrent_dim,
            self.feedback_dim,
            self.action_dim,
        )
        if any(int(value) <= 0 for value in positive):
            raise ValueError("all R6 model dimensions must be positive")
        if self.ray_dim % self.attention_heads != 0:
            raise ValueError("ray_dim must be divisible by attention_heads")
        if not 0.0 < self.horizontal_fov_degrees <= 360.0:
            raise ValueError("horizontal FOV must lie in (0, 360]")
        if not 0.0 < self.vertical_fov_degrees < 180.0:
            raise ValueError("vertical FOV must lie in (0, 180)")
        if self.beta_floor <= 1.0:
            raise ValueError("beta_floor must exceed one for a finite interior mode")

    @property
    def lidar_sectors(self) -> int:
        return self.horizontal_sectors * self.vertical_sectors

    @property
    def observation_dim(self) -> int:
        return 7 + 2 * self.lidar_sectors

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def lidar_direction_table(config: R6ModelConfig) -> torch.Tensor:
    horizontal = np.deg2rad(config.horizontal_fov_degrees)
    vertical = np.deg2rad(config.vertical_fov_degrees)
    if np.isclose(horizontal, 2.0 * np.pi):
        azimuth = np.arange(config.horizontal_sectors) * (
            2.0 * np.pi / config.horizontal_sectors
        )
    else:
        azimuth = np.linspace(
            -0.5 * horizontal,
            0.5 * horizontal,
            config.horizontal_sectors,
        )
    elevation = (
        np.zeros(1, dtype=np.float32)
        if config.vertical_sectors == 1
        else np.linspace(
            -0.5 * vertical,
            0.5 * vertical,
            config.vertical_sectors,
        )
    )
    rows: list[list[float]] = []
    for angle_z in elevation:
        cosine = np.cos(angle_z)
        for angle_xy in azimuth:
            rows.append(
                [
                    float(cosine * np.cos(angle_xy)),
                    float(cosine * np.sin(angle_xy)),
                    float(np.sin(angle_z)),
                ]
            )
    return torch.as_tensor(rows, dtype=torch.float32)


class RecurrentSetActorCritic(nn.Module):
    """Goal-conditioned recurrent policy with learned LiDAR ray attention.

    The network receives no planned route, waypoint, deadlock flag, or heuristic
    action. Each ray is represented by range, validity, and its physical
    direction; a goal-conditioned query pools the unordered ray tokens. A GRU
    integrates the current embedding with executed-action feedback from the
    previous closed-loop transition.
    """

    def __init__(self, config: R6ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.core_encoder = nn.Sequential(
            nn.Linear(7, config.core_dim),
            nn.LayerNorm(config.core_dim),
            nn.SiLU(),
            nn.Linear(config.core_dim, config.ray_dim),
            nn.SiLU(),
        )
        self.ray_encoder = nn.Sequential(
            nn.Linear(5, config.ray_dim),
            nn.SiLU(),
            nn.Linear(config.ray_dim, config.ray_dim),
            nn.LayerNorm(config.ray_dim),
            nn.SiLU(),
        )
        self.ray_attention = nn.MultiheadAttention(
            embed_dim=config.ray_dim,
            num_heads=config.attention_heads,
            batch_first=True,
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(config.feedback_dim, 32),
            nn.SiLU(),
        )
        self.recurrent = nn.GRUCell(config.ray_dim * 2 + 32, config.recurrent_dim)
        self.actor = nn.Sequential(
            nn.Linear(config.recurrent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 2 * config.action_dim),
        )
        self.reward_value = nn.Linear(config.recurrent_dim, 1)
        self.collision_value = nn.Linear(config.recurrent_dim, 1)
        self.intervention_value = nn.Linear(config.recurrent_dim, 1)
        self.register_buffer("ray_directions", lidar_direction_table(config))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        for value_head in (
            self.reward_value,
            self.collision_value,
            self.intervention_value,
        ):
            nn.init.orthogonal_(value_head.weight, gain=1.0)

    @property
    def observation_dim(self) -> int:
        return self.config.observation_dim

    @property
    def hidden_dim(self) -> int:
        return self.config.recurrent_dim

    def initial_hidden(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def encode_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(
                f"observation must have shape (batch, {self.observation_dim})"
            )
        sectors = self.config.lidar_sectors
        core = observation[:, :7]
        ranges = observation[:, 7 : 7 + sectors]
        valid = observation[:, 7 + sectors :]
        directions = self.ray_directions.unsqueeze(0).expand(observation.shape[0], -1, -1)
        rays = torch.cat((ranges.unsqueeze(-1), valid.unsqueeze(-1), directions), dim=-1)
        core_embedding = self.core_encoder(core)
        ray_embedding = self.ray_encoder(rays)
        pooled, _ = self.ray_attention(
            core_embedding.unsqueeze(1),
            ray_embedding,
            ray_embedding,
            need_weights=False,
        )
        return torch.cat((core_embedding, pooled.squeeze(1)), dim=-1)

    def _distribution(self, hidden: torch.Tensor) -> Beta:
        parameters = self.actor(hidden)
        alpha_raw, beta_raw = parameters.chunk(2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha_raw) + self.config.beta_floor
        beta = torch.nn.functional.softplus(beta_raw) + self.config.beta_floor
        return Beta(alpha, beta)

    @staticmethod
    def _action_from_unit(unit_action: torch.Tensor) -> torch.Tensor:
        return 2.0 * unit_action - 1.0

    @staticmethod
    def _unit_from_action(action: torch.Tensor) -> torch.Tensor:
        return ((action + 1.0) * 0.5).clamp(1e-6, 1.0 - 1e-6)

    def _outputs(
        self,
        hidden: torch.Tensor,
    ) -> tuple[Beta, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(hidden)
        values = torch.cat(
            (
                self.reward_value(hidden),
                self.collision_value(hidden),
                self.intervention_value(hidden),
            ),
            dim=-1,
        )
        mean_action = self._action_from_unit(distribution.mean)
        return distribution, values, mean_action

    def step(
        self,
        observation: torch.Tensor,
        feedback: torch.Tensor,
        hidden: torch.Tensor,
        episode_start: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if feedback.shape != (observation.shape[0], self.config.feedback_dim):
            raise ValueError("feedback batch shape does not match model configuration")
        if hidden.shape != (observation.shape[0], self.hidden_dim):
            raise ValueError("hidden batch shape does not match model configuration")
        reset = episode_start.to(dtype=hidden.dtype).reshape(-1, 1)
        hidden = hidden * (1.0 - reset)
        encoded = self.encode_observation(observation)
        recurrent_input = torch.cat((encoded, self.feedback_encoder(feedback)), dim=-1)
        next_hidden = self.recurrent(recurrent_input, hidden)
        distribution, values, mean_action = self._outputs(next_hidden)
        unit_action = distribution.mean if deterministic else distribution.sample()
        action = self._action_from_unit(unit_action)
        log_prob = (
            distribution.log_prob(unit_action) - np.log(2.0)
        ).sum(dim=-1)
        entropy = (distribution.entropy() + np.log(2.0)).sum(dim=-1)
        return action, log_prob, entropy, values, next_hidden

    def evaluate_sequence(
        self,
        observations: torch.Tensor,
        feedback: torch.Tensor,
        initial_hidden: torch.Tensor,
        episode_starts: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if observations.ndim != 3:
            raise ValueError("sequence observations must have shape (time, batch, obs)")
        length, batch, _ = observations.shape
        flat_encoded = self.encode_observation(observations.reshape(length * batch, -1))
        encoded = flat_encoded.reshape(length, batch, -1)
        encoded_feedback = self.feedback_encoder(
            feedback.reshape(length * batch, -1)
        ).reshape(length, batch, -1)
        hidden = initial_hidden
        log_probs: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        for index in range(length):
            reset = episode_starts[index].to(dtype=hidden.dtype).reshape(-1, 1)
            hidden = hidden * (1.0 - reset)
            recurrent_input = torch.cat((encoded[index], encoded_feedback[index]), dim=-1)
            hidden = self.recurrent(recurrent_input, hidden)
            distribution, step_values, mean_action = self._outputs(hidden)
            unit_action = self._unit_from_action(actions[index])
            log_probs.append(
                (distribution.log_prob(unit_action) - np.log(2.0)).sum(dim=-1)
            )
            entropies.append((distribution.entropy() + np.log(2.0)).sum(dim=-1))
            values.append(step_values)
            means.append(mean_action)
        return (
            torch.stack(log_probs),
            torch.stack(entropies),
            torch.stack(values),
            torch.stack(means),
        )

    def architecture_audit(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "config": self.config.as_dict(),
            "observation_semantics": {
                "core": [0, 7],
                "lidar_range": [7, 7 + self.config.lidar_sectors],
                "lidar_valid": [
                    7 + self.config.lidar_sectors,
                    7 + 2 * self.config.lidar_sectors,
                ],
                "route_planner": None,
                "waypoint_input": None,
                "heuristic_action_input": None,
            },
            "feedback_semantics": [
                "previous_executed_action_x",
                "previous_executed_action_y",
                "previous_executed_action_z",
                "previous_hocbf_intervention_norm",
                "previous_normalized_progress",
            ],
            "policy_distribution": "independent_beta_mapped_to_minus_one_plus_one",
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters()),
        }
