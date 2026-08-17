from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np


class DeterministicBatchPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


@dataclass(frozen=True)
class ShortRolloutConfig:
    horizon: int
    map_extent: tuple[float, float, float] = (4000.0, 4000.0, 400.0)
    policy_dt: float = 0.2
    physics_dt: float = 0.05
    horizontal_v_max: float = 20.0
    vertical_v_max: float = 5.0
    horizontal_a_max: float = 5.0
    vertical_a_max: float = 3.0
    safe_radius: float = 0.5
    goal_radius: float = 5.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("short-rollout horizon must be positive")
        if self.policy_dt <= 0.0 or self.physics_dt <= 0.0:
            raise ValueError("rollout time steps must be positive")
        ratio = self.policy_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("policy_dt must be an integer multiple of physics_dt")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ShortFrozenSACRolloutContext:
    feature_names = (
        "rollout_mean_speed_fraction",
        "rollout_max_speed_fraction",
        "rollout_mean_acceleration_fraction",
        "rollout_max_acceleration_fraction",
        "rollout_min_boundary_fraction",
        "rollout_mean_direction_change",
        "rollout_progress_fraction",
        "rollout_path_efficiency_fraction",
        "rollout_boundary_projection_fraction",
    )

    def __init__(self, policy: DeterministicBatchPolicy, config: ShortRolloutConfig) -> None:
        self.policy = policy
        self.config = config
        self.extent = np.asarray(config.map_extent, dtype=np.float64)
        self.d_max = float(np.linalg.norm(self.extent))
        self.velocity_scale = np.asarray(
            [config.horizontal_v_max, config.horizontal_v_max, config.vertical_v_max],
            dtype=np.float64,
        )
        self.acceleration_scale = np.asarray(
            [config.horizontal_a_max, config.horizontal_a_max, config.vertical_a_max],
            dtype=np.float64,
        )
        self.substeps = int(round(config.policy_dt / config.physics_dt))

    @property
    def input_dim(self) -> int:
        return len(self.feature_names)

    def _sac_observation(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        goal: np.ndarray,
    ) -> np.ndarray:
        delta = goal - position
        distance = np.linalg.norm(delta, axis=1)
        direction = np.zeros_like(delta)
        nonzero = distance > 1e-8
        direction[nonzero] = delta[nonzero] / distance[nonzero, None]
        velocity_feature = np.clip(velocity / self.velocity_scale[None, :], -1.0, 1.0)
        distance_feature = np.log1p(distance) / np.log1p(self.d_max)
        return np.concatenate(
            [velocity_feature, direction, np.clip(distance_feature[:, None], 0.0, 1.0)],
            axis=1,
        ).astype(np.float32)

    def build(
        self,
        energy_states: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        state = np.asarray(energy_states, dtype=np.float64)
        initial_position = np.asarray(positions, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != 7:
            raise ValueError("short-rollout energy states must have shape (N, 7)")
        if initial_position.shape != (state.shape[0], 3):
            raise ValueError("short-rollout positions must align with states")
        position = initial_position.copy()
        velocity = state[:, :3] * self.velocity_scale[None, :]
        distance = state[:, 6] * self.d_max
        goal = position + state[:, 3:6] * distance[:, None]
        initial_distance = np.linalg.norm(goal - position, axis=1)
        active = initial_distance > self.config.goal_radius
        path_length = np.zeros(state.shape[0], dtype=np.float64)
        speed_sum = np.zeros_like(path_length)
        speed_max = np.zeros_like(path_length)
        acceleration_sum = np.zeros_like(path_length)
        acceleration_max = np.zeros_like(path_length)
        direction_change_sum = np.zeros_like(path_length)
        direction_change_count = np.zeros_like(path_length)
        boundary_projection_count = np.zeros_like(path_length)
        executed_steps = np.zeros_like(path_length)
        minimum_boundary = np.full(state.shape[0], np.inf, dtype=np.float64)
        previous_direction = np.zeros_like(position)

        for _ in range(self.config.horizon):
            if not np.any(active):
                break
            observation = self._sac_observation(position, velocity, goal)
            actions, _ = self.policy.predict(observation, deterministic=True)
            action = np.asarray(actions, dtype=np.float64)
            if action.ndim == 1:
                action = action[None, :]
            if action.shape != (state.shape[0], 3):
                raise ValueError("short-rollout policy returned the wrong action shape")
            action = np.clip(action, -1.0, 1.0)
            horizontal_norm = np.linalg.norm(action[:, :2], axis=1)
            over = horizontal_norm > 1.0
            action[over, :2] /= horizontal_norm[over, None]
            commanded = action * self.acceleration_scale[None, :]
            position_before = position.copy()
            velocity_before = velocity.copy()
            projected_this_step = np.zeros(state.shape[0], dtype=bool)
            for _ in range(self.substeps):
                velocity_after = velocity + commanded * self.config.physics_dt
                horizontal_speed = np.linalg.norm(velocity_after[:, :2], axis=1)
                saturated = horizontal_speed > self.config.horizontal_v_max
                velocity_after[saturated, :2] *= (
                    self.config.horizontal_v_max / horizontal_speed[saturated]
                )[:, None]
                velocity_after[:, 2] = np.clip(
                    velocity_after[:, 2],
                    -self.config.vertical_v_max,
                    self.config.vertical_v_max,
                )
                next_position = position + velocity_after * self.config.physics_dt
                lower = np.full_like(next_position, self.config.safe_radius)
                upper = self.extent[None, :] - self.config.safe_radius
                clipped = np.clip(next_position, lower, upper)
                projected = np.any(np.abs(clipped - next_position) > 1e-10, axis=1) & active
                projected_this_step |= projected
                velocity_after[projected] = np.where(
                    np.abs(clipped[projected] - next_position[projected]) > 1e-10,
                    0.0,
                    velocity_after[projected],
                )
                position[active] = clipped[active]
                velocity[active] = velocity_after[active]
            displacement = position - position_before
            path_length += np.linalg.norm(displacement, axis=1) * active
            speed_fraction = np.linalg.norm(velocity / self.velocity_scale[None, :], axis=1)
            realized_acceleration = (velocity - velocity_before) / self.config.policy_dt
            acceleration_fraction = np.linalg.norm(
                realized_acceleration / self.acceleration_scale[None, :], axis=1
            )
            delta = goal - position
            remaining = np.linalg.norm(delta, axis=1)
            current_direction = np.zeros_like(delta)
            nonzero = remaining > 1e-8
            current_direction[nonzero] = delta[nonzero] / remaining[nonzero, None]
            valid_direction = active & (executed_steps > 0)
            cosine = np.sum(previous_direction * current_direction, axis=1)
            direction_change_sum[valid_direction] += np.arccos(
                np.clip(cosine[valid_direction], -1.0, 1.0)
            )
            direction_change_count[valid_direction] += 1.0
            previous_direction = current_direction
            normalized_position = position / self.extent[None, :]
            boundary = np.minimum(normalized_position, 1.0 - normalized_position).min(axis=1)
            minimum_boundary[active] = np.minimum(minimum_boundary[active], boundary[active])
            speed_sum += speed_fraction * active
            speed_max = np.maximum(speed_max, speed_fraction * active)
            acceleration_sum += acceleration_fraction * active
            acceleration_max = np.maximum(acceleration_max, acceleration_fraction * active)
            boundary_projection_count += projected_this_step.astype(np.float64)
            executed_steps += active.astype(np.float64)
            active = active & (remaining > self.config.goal_radius)

        denominator = np.maximum(executed_steps, 1.0)
        final_distance = np.linalg.norm(goal - position, axis=1)
        progress = np.maximum(initial_distance - final_distance, 0.0)
        features = np.column_stack(
            [
                speed_sum / denominator,
                speed_max,
                acceleration_sum / denominator,
                acceleration_max,
                np.where(np.isfinite(minimum_boundary), minimum_boundary, 0.0),
                direction_change_sum / np.maximum(direction_change_count, 1.0),
                progress / np.maximum(initial_distance, 1e-8),
                progress / np.maximum(path_length, 1e-8),
                boundary_projection_count / denominator,
            ]
        )
        if not np.all(np.isfinite(features)):
            raise RuntimeError("short-rollout context contains non-finite values")
        return features.astype(np.float32)

    def build_chunked(
        self,
        energy_states: np.ndarray,
        positions: np.ndarray,
        *,
        chunk_size: int = 8192,
    ) -> np.ndarray:
        if chunk_size <= 0:
            raise ValueError("short-rollout chunk size must be positive")
        states = np.asarray(energy_states)
        absolute = np.asarray(positions)
        if states.shape[0] != absolute.shape[0]:
            raise ValueError("short-rollout chunked inputs must align")
        return np.concatenate(
            [
                self.build(states[offset : offset + chunk_size], absolute[offset : offset + chunk_size])
                for offset in range(0, states.shape[0], chunk_size)
            ],
            axis=0,
        )


__all__ = [
    "DeterministicBatchPolicy",
    "ShortFrozenSACRolloutContext",
    "ShortRolloutConfig",
]
