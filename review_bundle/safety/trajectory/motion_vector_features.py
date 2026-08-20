from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FeatureProvenance(str, Enum):
    DIRECTLY_MEASURED = "DIRECTLY_MEASURED"
    PHYSICALLY_DERIVED = "PHYSICALLY_DERIVED"
    LEARNED = "LEARNED"
    FUTURE_INFORMATION = "FUTURE_INFORMATION"


@dataclass(frozen=True)
class MotionVectorFeatures:
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    relative_acceleration: np.ndarray
    distance: float
    closing_speed: float
    time_to_collision: float
    stopping_distance: float
    sampling_delay_distance: float
    hocbf_slack: float | None
    feasibility_margin: float | None
    provenance: dict[str, FeatureProvenance]


@dataclass(frozen=True)
class LidarTemporalFlow:
    raw_range_delta: np.ndarray
    ego_range_delta: np.ndarray
    motion_compensated_range_delta: np.ndarray
    valid: np.ndarray
    delta_time: float
    provenance: dict[str, FeatureProvenance]


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


def _sphere_ttc(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    contact_distance: float,
) -> float:
    a = float(relative_velocity @ relative_velocity)
    b = 2.0 * float(relative_position @ relative_velocity)
    c = float(relative_position @ relative_position - contact_distance**2)
    if c <= 0.0:
        return 0.0
    if a <= 1e-15:
        return float("inf")
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return float("inf")
    root = np.sqrt(discriminant)
    times = [value for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)) if value >= 0.0]
    return float(min(times)) if times else float("inf")


def relative_motion_features(
    uav_position: np.ndarray,
    uav_velocity: np.ndarray,
    proposed_acceleration: np.ndarray,
    obstacle_position: np.ndarray,
    obstacle_velocity: np.ndarray | None = None,
    obstacle_acceleration: np.ndarray | None = None,
    *,
    contact_distance: float,
    braking_acceleration: float,
    sampling_period: float,
    hocbf_slack: float | None = None,
    feasibility_margin: float | None = None,
) -> MotionVectorFeatures:
    if contact_distance < 0.0 or braking_acceleration <= 0.0 or sampling_period <= 0.0:
        raise ValueError("contact distance must be nonnegative and timing/authority positive")
    position = _vec3(uav_position, "uav_position")
    velocity = _vec3(uav_velocity, "uav_velocity")
    acceleration = _vec3(proposed_acceleration, "proposed_acceleration")
    obstacle_position = _vec3(obstacle_position, "obstacle_position")
    obstacle_velocity = np.zeros(3) if obstacle_velocity is None else _vec3(
        obstacle_velocity, "obstacle_velocity"
    )
    obstacle_acceleration = np.zeros(3) if obstacle_acceleration is None else _vec3(
        obstacle_acceleration, "obstacle_acceleration"
    )
    relative_position = position - obstacle_position
    relative_velocity = velocity - obstacle_velocity
    relative_acceleration = acceleration - obstacle_acceleration
    distance = float(np.linalg.norm(relative_position))
    closing_speed = 0.0 if distance <= 1e-12 else max(
        0.0, -float(relative_position @ relative_velocity) / distance
    )
    return MotionVectorFeatures(
        relative_position=relative_position,
        relative_velocity=relative_velocity,
        relative_acceleration=relative_acceleration,
        distance=distance,
        closing_speed=closing_speed,
        time_to_collision=_sphere_ttc(relative_position, relative_velocity, contact_distance),
        stopping_distance=float(closing_speed**2 / (2.0 * braking_acceleration)),
        sampling_delay_distance=float(closing_speed * sampling_period),
        hocbf_slack=None if hocbf_slack is None else float(hocbf_slack),
        feasibility_margin=None if feasibility_margin is None else float(feasibility_margin),
        provenance={
            "relative_position": FeatureProvenance.PHYSICALLY_DERIVED,
            "relative_velocity": FeatureProvenance.PHYSICALLY_DERIVED,
            "relative_acceleration": FeatureProvenance.PHYSICALLY_DERIVED,
            "distance": FeatureProvenance.PHYSICALLY_DERIVED,
            "closing_speed": FeatureProvenance.PHYSICALLY_DERIVED,
            "time_to_collision": FeatureProvenance.PHYSICALLY_DERIVED,
            "stopping_distance": FeatureProvenance.PHYSICALLY_DERIVED,
            "sampling_delay_distance": FeatureProvenance.PHYSICALLY_DERIVED,
            "hocbf_slack": FeatureProvenance.PHYSICALLY_DERIVED,
            "feasibility_margin": FeatureProvenance.PHYSICALLY_DERIVED,
        },
    )


def lidar_temporal_flow(
    previous_ranges: np.ndarray,
    current_ranges: np.ndarray,
    previous_valid: np.ndarray,
    current_valid: np.ndarray,
    ray_directions: np.ndarray,
    previous_uav_position: np.ndarray,
    current_uav_position: np.ndarray,
    *,
    delta_time: float,
) -> LidarTemporalFlow:
    previous = np.asarray(previous_ranges, dtype=np.float64)
    current = np.asarray(current_ranges, dtype=np.float64)
    previous_mask = np.asarray(previous_valid, dtype=bool)
    current_mask = np.asarray(current_valid, dtype=bool)
    directions = np.asarray(ray_directions, dtype=np.float64)
    if previous.ndim != 1 or current.shape != previous.shape:
        raise ValueError("LiDAR ranges must be aligned vectors")
    if previous_mask.shape != previous.shape or current_mask.shape != previous.shape:
        raise ValueError("LiDAR masks must align with ranges")
    if directions.shape != (previous.size, 3) or not np.all(np.isfinite(directions)):
        raise ValueError("ray_directions must have shape (num_rays, 3)")
    if delta_time <= 0.0 or not np.isfinite(delta_time):
        raise ValueError("delta_time must be finite and positive")
    displacement = _vec3(current_uav_position, "current_uav_position") - _vec3(
        previous_uav_position, "previous_uav_position"
    )
    valid = previous_mask & current_mask & np.isfinite(previous) & np.isfinite(current)
    raw = np.zeros_like(previous)
    raw[valid] = current[valid] - previous[valid]
    ego = directions @ displacement
    compensated = np.zeros_like(previous)
    compensated[valid] = raw[valid] + ego[valid]
    return LidarTemporalFlow(
        raw_range_delta=raw,
        ego_range_delta=ego,
        motion_compensated_range_delta=compensated,
        valid=valid,
        delta_time=float(delta_time),
        provenance={
            "raw_range_delta": FeatureProvenance.DIRECTLY_MEASURED,
            "ego_range_delta": FeatureProvenance.PHYSICALLY_DERIVED,
            "motion_compensated_range_delta": FeatureProvenance.PHYSICALLY_DERIVED,
        },
    )
