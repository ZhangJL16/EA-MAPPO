from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel


@dataclass(frozen=True)
class PhysicsRolloutConfig:
    policy_dt: float = 0.2
    physics_dt: float = 0.05
    horizontal_speed_limit: float = 20.0
    vertical_speed_limit: float = 5.0
    horizontal_acceleration_limit: float = 5.0
    vertical_acceleration_limit: float = 3.0
    world_min: np.ndarray | None = None
    world_max: np.ndarray | None = None
    body_radius: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.policy_dt,
            self.physics_dt,
            self.horizontal_speed_limit,
            self.vertical_speed_limit,
            self.horizontal_acceleration_limit,
            self.vertical_acceleration_limit,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("rollout timing and limits must be finite and positive")
        ratio = self.policy_dt / self.physics_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("policy_dt must be an integer multiple of physics_dt")
        if self.body_radius < 0.0:
            raise ValueError("body_radius must be nonnegative")
        if (self.world_min is None) != (self.world_max is None):
            raise ValueError("world_min and world_max must be supplied together")
        if self.world_min is not None:
            lower = np.asarray(self.world_min, dtype=np.float64)
            upper = np.asarray(self.world_max, dtype=np.float64)
            if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
                raise ValueError("world bounds must be aligned finite (3,) vectors")
            if np.any(lower + self.body_radius >= upper - self.body_radius):
                raise ValueError("body radius leaves no valid world interior")
            object.__setattr__(self, "world_min", lower.copy())
            object.__setattr__(self, "world_max", upper.copy())

    @property
    def physics_substeps(self) -> int:
        return int(round(self.policy_dt / self.physics_dt))


@dataclass(frozen=True)
class PhysicsRolloutResult:
    positions: np.ndarray
    velocities: np.ndarray
    commanded_accelerations: np.ndarray
    realized_accelerations: np.ndarray
    step_energy: np.ndarray
    boundary_contacts: np.ndarray
    substep_start_positions: np.ndarray
    substep_start_velocities: np.ndarray
    substep_realized_accelerations: np.ndarray
    substep_end_positions: np.ndarray
    substep_end_velocities: np.ndarray
    substep_energy: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.commanded_accelerations.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.commanded_accelerations.shape[1])


def _batch_vec3(value: np.ndarray, batch: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (3,):
        array = np.broadcast_to(array, (batch, 3)).copy()
    if array.shape != (batch, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (3,) or ({batch}, 3)")
    return array.copy()


def _clip_acceleration(actions: np.ndarray, config: PhysicsRolloutConfig) -> np.ndarray:
    clipped = np.asarray(actions, dtype=np.float64).copy()
    horizontal = np.linalg.norm(clipped[..., :2], axis=-1)
    scale = np.minimum(1.0, config.horizontal_acceleration_limit / np.maximum(horizontal, 1e-15))
    clipped[..., :2] *= scale[..., None]
    clipped[..., 2] = np.clip(
        clipped[..., 2], -config.vertical_acceleration_limit, config.vertical_acceleration_limit
    )
    return clipped


def _clip_speed(velocity: np.ndarray, config: PhysicsRolloutConfig) -> np.ndarray:
    clipped = velocity.copy()
    horizontal = np.linalg.norm(clipped[:, :2], axis=1)
    scale = np.minimum(1.0, config.horizontal_speed_limit / np.maximum(horizontal, 1e-15))
    clipped[:, :2] *= scale[:, None]
    clipped[:, 2] = np.clip(clipped[:, 2], -config.vertical_speed_limit, config.vertical_speed_limit)
    return clipped


def _batch_energy_numpy(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    duration: float,
    cost_model: TelemetryCostModel,
) -> np.ndarray:
    cost = cost_model.config
    power = (
        cost.base_power
        + np.sum(cost.velocity_coefficients[None, :] * np.abs(velocity), axis=1)
        + np.sum(cost.acceleration_coefficients[None, :] * acceleration**2, axis=1)
        + cost.compute_power
        + cost.communication_power
    )
    return cost.flight_energy_multiplier * (duration * power + cost.simulation_error)


def rollout_action_sequences(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    action_sequences: np.ndarray,
    config: PhysicsRolloutConfig,
    *,
    telemetry_cost_model: TelemetryCostModel | None = None,
) -> PhysicsRolloutResult:
    actions = np.asarray(action_sequences, dtype=np.float64)
    if actions.ndim != 3 or actions.shape[2] != 3 or not np.all(np.isfinite(actions)):
        raise ValueError("action_sequences must be finite with shape (batch, horizon, 3)")
    batch, horizon, _ = actions.shape
    position = _batch_vec3(initial_position, batch, "initial_position")
    velocity = _batch_vec3(initial_velocity, batch, "initial_velocity")
    commanded = _clip_acceleration(actions, config)
    positions = np.empty((batch, horizon + 1, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    positions[:, 0] = position
    velocities[:, 0] = velocity
    realized = np.zeros_like(commanded)
    step_energy = np.zeros((batch, horizon), dtype=np.float64)
    contacts = np.zeros((batch, horizon), dtype=bool)
    substeps = config.physics_substeps
    substep_start_positions = np.empty((batch, horizon, substeps, 3), dtype=np.float64)
    substep_start_velocities = np.empty_like(substep_start_positions)
    substep_acceleration = np.empty_like(substep_start_positions)
    substep_end_positions = np.empty_like(substep_start_positions)
    substep_end_velocities = np.empty_like(substep_start_positions)
    substep_energy = np.zeros((batch, horizon, substeps), dtype=np.float64)
    cost_model = TelemetryCostModel() if telemetry_cost_model is None else telemetry_cost_model

    for step in range(horizon):
        policy_velocity_start = velocity.copy()
        for substep in range(substeps):
            substep_start_positions[:, step, substep] = position
            substep_start_velocities[:, step, substep] = velocity
            before = velocity.copy()
            after_propulsion = _clip_speed(
                before + commanded[:, step] * config.physics_dt,
                config,
            )
            acceleration = (after_propulsion - before) / config.physics_dt
            proposed_position = (
                position
                + before * config.physics_dt
                + 0.5 * acceleration * config.physics_dt**2
            )
            substep_energy[:, step, substep] = _batch_energy_numpy(
                after_propulsion,
                acceleration,
                config.physics_dt,
                cost_model,
            )
            position = proposed_position
            velocity = after_propulsion
            if config.world_min is not None:
                lower = config.world_min + config.body_radius
                upper = config.world_max - config.body_radius
                projected = np.clip(position, lower, upper)
                hit = np.any(np.abs(projected - position) > 1e-12, axis=1)
                contacts[:, step] |= hit
                for axis in range(3):
                    axis_hit = np.abs(projected[:, axis] - position[:, axis]) > 1e-12
                    velocity[axis_hit, axis] = 0.0
                position = projected
            substep_acceleration[:, step, substep] = acceleration
            substep_end_positions[:, step, substep] = position
            substep_end_velocities[:, step, substep] = velocity
        positions[:, step + 1] = position
        velocities[:, step + 1] = velocity
        realized[:, step] = (velocity - policy_velocity_start) / config.policy_dt
        step_energy[:, step] = np.sum(substep_energy[:, step], axis=1)
    return PhysicsRolloutResult(
        positions=positions,
        velocities=velocities,
        commanded_accelerations=commanded,
        realized_accelerations=realized,
        step_energy=step_energy,
        boundary_contacts=contacts,
        substep_start_positions=substep_start_positions,
        substep_start_velocities=substep_start_velocities,
        substep_realized_accelerations=substep_acceleration,
        substep_end_positions=substep_end_positions,
        substep_end_velocities=substep_end_velocities,
        substep_energy=substep_energy,
    )


def rollout_action_sequences_torch(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    action_sequences: np.ndarray,
    config: PhysicsRolloutConfig,
    *,
    telemetry_cost_model: TelemetryCostModel | None = None,
    device: str | torch.device = "cpu",
) -> PhysicsRolloutResult:
    actions_np = np.asarray(action_sequences, dtype=np.float64)
    if actions_np.ndim != 3 or actions_np.shape[2] != 3 or not np.all(np.isfinite(actions_np)):
        raise ValueError("action_sequences must be finite with shape (batch, horizon, 3)")
    target_device = torch.device(device)
    dtype = torch.float64
    actions = torch.as_tensor(actions_np, dtype=dtype, device=target_device)
    batch, horizon, _ = actions.shape
    position = torch.as_tensor(
        _batch_vec3(initial_position, batch, "initial_position"), dtype=dtype, device=target_device
    )
    velocity = torch.as_tensor(
        _batch_vec3(initial_velocity, batch, "initial_velocity"), dtype=dtype, device=target_device
    )
    horizontal = torch.linalg.vector_norm(actions[..., :2], dim=-1)
    scale = torch.minimum(
        torch.ones_like(horizontal),
        torch.tensor(config.horizontal_acceleration_limit, dtype=dtype, device=target_device)
        / torch.clamp(horizontal, min=1e-15),
    )
    commanded = actions.clone()
    commanded[..., :2] *= scale[..., None]
    commanded[..., 2] = torch.clamp(
        commanded[..., 2],
        -config.vertical_acceleration_limit,
        config.vertical_acceleration_limit,
    )
    positions = torch.empty((batch, horizon + 1, 3), dtype=dtype, device=target_device)
    velocities = torch.empty_like(positions)
    positions[:, 0] = position
    velocities[:, 0] = velocity
    realized = torch.zeros_like(commanded)
    step_energy = torch.zeros((batch, horizon), dtype=dtype, device=target_device)
    contacts = torch.zeros((batch, horizon), dtype=torch.bool, device=target_device)
    substeps = config.physics_substeps
    shape = (batch, horizon, substeps, 3)
    substep_start_positions = torch.empty(shape, dtype=dtype, device=target_device)
    substep_start_velocities = torch.empty(shape, dtype=dtype, device=target_device)
    substep_acceleration = torch.empty(shape, dtype=dtype, device=target_device)
    substep_end_positions = torch.empty(shape, dtype=dtype, device=target_device)
    substep_end_velocities = torch.empty(shape, dtype=dtype, device=target_device)
    substep_energy = torch.zeros((batch, horizon, substeps), dtype=dtype, device=target_device)
    cost_model = TelemetryCostModel() if telemetry_cost_model is None else telemetry_cost_model
    cost = cost_model.config
    velocity_coefficients = torch.as_tensor(cost.velocity_coefficients, dtype=dtype, device=target_device)
    acceleration_coefficients = torch.as_tensor(
        cost.acceleration_coefficients, dtype=dtype, device=target_device
    )
    for step in range(horizon):
        policy_velocity_start = velocity.clone()
        for substep in range(substeps):
            substep_start_positions[:, step, substep] = position
            substep_start_velocities[:, step, substep] = velocity
            before = velocity.clone()
            after = before + commanded[:, step] * config.physics_dt
            speed_xy = torch.linalg.vector_norm(after[:, :2], dim=1)
            speed_scale = torch.minimum(
                torch.ones_like(speed_xy),
                torch.tensor(config.horizontal_speed_limit, dtype=dtype, device=target_device)
                / torch.clamp(speed_xy, min=1e-15),
            )
            after[:, :2] *= speed_scale[:, None]
            after[:, 2] = torch.clamp(
                after[:, 2], -config.vertical_speed_limit, config.vertical_speed_limit
            )
            acceleration = (after - before) / config.physics_dt
            proposed_position = (
                position + before * config.physics_dt + 0.5 * acceleration * config.physics_dt**2
            )
            power = (
                cost.base_power
                + torch.sum(velocity_coefficients[None, :] * torch.abs(after), dim=1)
                + torch.sum(acceleration_coefficients[None, :] * acceleration**2, dim=1)
                + cost.compute_power
                + cost.communication_power
            )
            substep_energy[:, step, substep] = cost.flight_energy_multiplier * (
                config.physics_dt * power + cost.simulation_error
            )
            position = proposed_position
            velocity = after
            if config.world_min is not None:
                lower = torch.as_tensor(
                    config.world_min + config.body_radius, dtype=dtype, device=target_device
                )
                upper = torch.as_tensor(
                    config.world_max - config.body_radius, dtype=dtype, device=target_device
                )
                projected = torch.maximum(torch.minimum(position, upper), lower)
                axis_hit = torch.abs(projected - position) > 1e-12
                contacts[:, step] |= torch.any(axis_hit, dim=1)
                velocity = torch.where(axis_hit, torch.zeros_like(velocity), velocity)
                position = projected
            substep_acceleration[:, step, substep] = acceleration
            substep_end_positions[:, step, substep] = position
            substep_end_velocities[:, step, substep] = velocity
        positions[:, step + 1] = position
        velocities[:, step + 1] = velocity
        realized[:, step] = (velocity - policy_velocity_start) / config.policy_dt
        step_energy[:, step] = torch.sum(substep_energy[:, step], dim=1)

    def numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    return PhysicsRolloutResult(
        positions=numpy(positions),
        velocities=numpy(velocities),
        commanded_accelerations=numpy(commanded),
        realized_accelerations=numpy(realized),
        step_energy=numpy(step_energy),
        boundary_contacts=numpy(contacts),
        substep_start_positions=numpy(substep_start_positions),
        substep_start_velocities=numpy(substep_start_velocities),
        substep_realized_accelerations=numpy(substep_acceleration),
        substep_end_positions=numpy(substep_end_positions),
        substep_end_velocities=numpy(substep_end_velocities),
        substep_energy=numpy(substep_energy),
    )


def rollout_single_sequence(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    action_sequence: np.ndarray,
    config: PhysicsRolloutConfig,
    *,
    telemetry_cost_model: TelemetryCostModel | None = None,
) -> PhysicsRolloutResult:
    sequence = np.asarray(action_sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[1] != 3:
        raise ValueError("action_sequence must have shape (horizon, 3)")
    return rollout_action_sequences(
        initial_position,
        initial_velocity,
        sequence[None, ...],
        config,
        telemetry_cost_model=telemetry_cost_model,
    )
