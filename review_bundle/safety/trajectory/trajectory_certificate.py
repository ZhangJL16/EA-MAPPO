from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from review_bundle.safety.collision.feasibility import exact_zoh_sphere_clearance
from review_bundle.safety.collision.hocbf import SphericalObstacle

from .physics_rollout import PhysicsRolloutConfig, PhysicsRolloutResult


@dataclass(frozen=True)
class TrajectoryCertificateResult:
    certified: np.ndarray
    collision_safe: np.ndarray
    input_admissible: np.ndarray
    boundary_safe: np.ndarray
    terminal_backup_feasible: np.ndarray
    minimum_clearance: np.ndarray
    intersample_violations: np.ndarray


TerminalBackupVerifier = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _input_admissible(rollout: PhysicsRolloutResult, config: PhysicsRolloutConfig) -> np.ndarray:
    horizontal = np.linalg.norm(rollout.commanded_accelerations[..., :2], axis=2)
    vertical = np.abs(rollout.commanded_accelerations[..., 2])
    return np.all(
        (horizontal <= config.horizontal_acceleration_limit + 1e-10)
        & (vertical <= config.vertical_acceleration_limit + 1e-10),
        axis=1,
    )


def _continuous_boundary_safe(
    rollout: PhysicsRolloutResult,
    config: PhysicsRolloutConfig,
) -> np.ndarray:
    safe = ~np.any(rollout.boundary_contacts, axis=1)
    if config.world_min is None:
        return safe
    lower = config.world_min + config.body_radius
    upper = config.world_max - config.body_radius
    for candidate in range(rollout.batch_size):
        if not safe[candidate]:
            continue
        for step in range(rollout.horizon):
            for substep in range(config.physics_substeps):
                position = rollout.substep_start_positions[candidate, step, substep]
                velocity = rollout.substep_start_velocities[candidate, step, substep]
                acceleration = rollout.substep_realized_accelerations[candidate, step, substep]
                for axis in range(3):
                    times = [0.0, config.physics_dt]
                    if abs(acceleration[axis]) > 1e-15:
                        critical = -velocity[axis] / acceleration[axis]
                        if 0.0 < critical < config.physics_dt:
                            times.append(float(critical))
                    values = [
                        position[axis]
                        + velocity[axis] * time
                        + 0.5 * acceleration[axis] * time**2
                        for time in times
                    ]
                    if min(values) < lower[axis] - 1e-10 or max(values) > upper[axis] + 1e-10:
                        safe[candidate] = False
                        break
                if not safe[candidate]:
                    break
            if not safe[candidate]:
                break
    return safe


def certify_trajectory(
    rollout: PhysicsRolloutResult,
    obstacles: tuple[SphericalObstacle, ...] | list[SphericalObstacle],
    config: PhysicsRolloutConfig,
    *,
    uav_radius: float,
    uncertainty_margin: float = 0.0,
    terminal_backup_verifier: TerminalBackupVerifier | None = None,
) -> TrajectoryCertificateResult:
    if uav_radius < 0.0 or uncertainty_margin < 0.0:
        raise ValueError("safety radii must be nonnegative")
    batch = rollout.batch_size
    minimum_clearance = np.full(batch, np.inf, dtype=np.float64)
    violations = np.zeros(batch, dtype=np.int64)
    for candidate in range(batch):
        for step in range(rollout.horizon):
            for substep in range(config.physics_substeps):
                time = (step * config.physics_substeps + substep) * config.physics_dt
                uav_position = rollout.substep_start_positions[candidate, step, substep]
                uav_velocity = rollout.substep_start_velocities[candidate, step, substep]
                uav_acceleration = rollout.substep_realized_accelerations[candidate, step, substep]
                for obstacle in obstacles:
                    obstacle_position = (
                        obstacle.center
                        + obstacle.velocity * time
                        + 0.5 * obstacle.acceleration * time**2
                    )
                    obstacle_velocity = obstacle.velocity + obstacle.acceleration * time
                    result = exact_zoh_sphere_clearance(
                        uav_position - obstacle_position,
                        uav_velocity - obstacle_velocity,
                        uav_acceleration - obstacle.acceleration,
                        obstacle.radius + uav_radius + uncertainty_margin,
                        config.physics_dt,
                    )
                    minimum_clearance[candidate] = min(
                        minimum_clearance[candidate], result.minimum_clearance
                    )
                    if not result.safe:
                        violations[candidate] += 1
    if not obstacles:
        minimum_clearance.fill(np.inf)
    collision_safe = violations == 0
    input_admissible = _input_admissible(rollout, config)
    boundary_safe = _continuous_boundary_safe(rollout, config)
    if terminal_backup_verifier is None:
        terminal_backup = np.ones(batch, dtype=bool)
    else:
        terminal_backup = np.asarray(
            terminal_backup_verifier(rollout.positions[:, -1], rollout.velocities[:, -1]),
            dtype=bool,
        )
        if terminal_backup.shape != (batch,):
            raise ValueError("terminal backup verifier must return one boolean per candidate")
    certified = collision_safe & input_admissible & boundary_safe & terminal_backup
    return TrajectoryCertificateResult(
        certified=certified,
        collision_safe=collision_safe,
        input_admissible=input_admissible,
        boundary_safe=boundary_safe,
        terminal_backup_feasible=terminal_backup,
        minimum_clearance=minimum_clearance,
        intersample_violations=violations,
    )
