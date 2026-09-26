"""Small, explicitly synthetic 2D mechanism map and energy primitives.

This module is not an M100 simulator.  The 3D geometry object is reused only
as a collision engine at a fixed height; no overflight action is exposed.
Full-battery capacity is calibrated separately from policy-independent
round-trip references before any method comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

from nav3d.geometry import CylinderObstacle, World3D


@dataclass(frozen=True)
class SyntheticConfig:
    side_m: float = 200.0
    flight_height_m: float = 2.0
    obstacle_count: int = 8
    obstacle_radius_min_m: float = 8.0
    obstacle_radius_max_m: float = 15.0
    obstacle_separation_m: float = 3.0
    obstacle_edge_margin_m: float = 5.0
    station_edge_margin_m: float = 20.0
    station_obstacle_margin_m: float = 12.0
    body_radius_m: float = 0.4
    planning_margin_m: float = 0.2
    max_speed_mps: float = 7.0
    max_acceleration_mps2: float = 4.0
    physics_dt_s: float = 0.05
    policy_dt_s: float = 0.5
    horizon_s: float = 600.0

    def __post_init__(self) -> None:
        values = (
            self.side_m, self.flight_height_m, self.obstacle_radius_min_m,
            self.obstacle_radius_max_m, self.obstacle_separation_m,
            self.obstacle_edge_margin_m, self.station_edge_margin_m,
            self.station_obstacle_margin_m, self.body_radius_m,
            self.planning_margin_m, self.max_speed_mps,
            self.max_acceleration_mps2, self.physics_dt_s,
            self.policy_dt_s, self.horizon_s,
        )
        if not all(isfinite(x) for x in values) or any(x <= 0 for x in values):
            raise ValueError("synthetic configuration must be positive and finite")
        if self.obstacle_count < 0 or self.obstacle_radius_max_m < self.obstacle_radius_min_m:
            raise ValueError("invalid synthetic obstacle settings")
        if self.station_edge_margin_m * 2 >= self.side_m:
            raise ValueError("station margin leaves no interior")
        if self.obstacle_radius_max_m + self.obstacle_edge_margin_m >= self.side_m / 2:
            raise ValueError("obstacles do not fit in the map")
        if abs(self.policy_dt_s / self.physics_dt_s - round(self.policy_dt_s / self.physics_dt_s)) > 1e-9:
            raise ValueError("policy period must contain whole physical steps")

    @property
    def planning_inflation_m(self) -> float:
        return self.body_radius_m + self.planning_margin_m

    @property
    def policy_hold_steps(self) -> int:
        return round(self.policy_dt_s / self.physics_dt_s)


SYNTHETIC_V0 = SyntheticConfig()


@dataclass(frozen=True)
class SyntheticMap:
    map_id: int
    world: World3D
    station_xy: tuple[float, float]
    config: SyntheticConfig

    def xyz(self, xy: object) -> np.ndarray:
        point = np.asarray(xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("planar point must be a finite 2-vector")
        return np.array((point[0], point[1], self.config.flight_height_m), dtype=float)

    def is_clear(self, xy: object) -> bool:
        return self.world.clearance(self.xyz(xy), self.config.planning_inflation_m) > 0.0

    def segment_clear(self, start_xy: object, end_xy: object) -> bool:
        return self.world.segment_clear(
            self.xyz(start_xy), self.xyz(end_xy), self.config.planning_inflation_m
        )


def make_synthetic_map(map_id: int, config: SyntheticConfig = SYNTHETIC_V0) -> SyntheticMap:
    if not isinstance(map_id, int) or map_id < 0:
        raise ValueError("map ID must be a nonnegative integer")
    rng = np.random.default_rng(4_000_000_007 + 1009 * map_id)
    station = tuple(float(x) for x in rng.uniform(
        config.station_edge_margin_m,
        config.side_m - config.station_edge_margin_m,
        size=2,
    ))
    obstacles: list[CylinderObstacle] = []
    for _ in range(max(1000, 1000 * config.obstacle_count)):
        if len(obstacles) == config.obstacle_count:
            break
        radius = float(rng.uniform(config.obstacle_radius_min_m, config.obstacle_radius_max_m))
        margin = radius + config.obstacle_edge_margin_m
        xy = rng.uniform(margin, config.side_m - margin, size=2)
        if hypot(xy[0] - station[0], xy[1] - station[1]) < radius + config.station_obstacle_margin_m:
            continue
        if any(
            np.linalg.norm(xy - existing.center_xy)
            < radius + existing.radius + config.obstacle_separation_m
            for existing in obstacles
        ):
            continue
        obstacles.append(CylinderObstacle(xy, radius, 0.0, 10.0, f"synthetic_disc_{len(obstacles):02d}"))
    if len(obstacles) != config.obstacle_count:
        raise RuntimeError("could not place all synthetic obstacles")
    case = SyntheticMap(
        map_id,
        World3D((config.side_m, config.side_m, 20.0), tuple(obstacles)),
        station,
        config,
    )
    if not case.is_clear(station):
        raise RuntimeError("generated station lacks planning clearance")
    return case


@dataclass(frozen=True)
class SyntheticEnergy:
    idle_rate: float = 1.0
    speed_squared_rate: float = 0.01
    acceleration_squared_rate: float = 0.02

    def __post_init__(self) -> None:
        if not all(isfinite(x) and x >= 0 for x in (
            self.idle_rate, self.speed_squared_rate, self.acceleration_squared_rate
        )) or self.idle_rate <= 0:
            raise ValueError("invalid synthetic energy coefficients")

    def flight_cost(self, velocity_xy: object, acceleration_xy: object, duration_s: float) -> float:
        velocity = np.asarray(velocity_xy, dtype=float)
        acceleration = np.asarray(acceleration_xy, dtype=float)
        if velocity.shape != (2,) or acceleration.shape != (2,) or not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
            raise ValueError("velocity and acceleration must be finite 2-vectors")
        if not isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration must be finite and nonnegative")
        return duration_s * (
            self.idle_rate
            + self.speed_squared_rate * float(velocity @ velocity)
            + self.acceleration_squared_rate * float(acceleration @ acceleration)
        )


@dataclass(frozen=True)
class SyntheticChargeState:
    energy: float
    elapsed_s: float = 0.0


@dataclass(frozen=True)
class SyntheticCharger:
    capacity: float
    full_charge_seconds: float

    def __post_init__(self) -> None:
        if not all(isfinite(x) and x > 0 for x in (self.capacity, self.full_charge_seconds)):
            raise ValueError("capacity and full-charge time must be positive and finite")

    def charge_to_fraction(self, state: SyntheticChargeState, target_fraction: float) -> SyntheticChargeState:
        if not isfinite(state.energy) or not 0 <= state.energy <= self.capacity:
            raise ValueError("invalid current energy")
        if not isfinite(state.elapsed_s) or state.elapsed_s < 0:
            raise ValueError("invalid current time")
        if not isfinite(target_fraction) or not state.energy / self.capacity <= target_fraction <= 1:
            raise ValueError("target fraction must be between current fraction and one")
        target_energy = target_fraction * self.capacity
        duration_s = (target_energy - state.energy) / self.capacity * self.full_charge_seconds
        return SyntheticChargeState(target_energy, state.elapsed_s + duration_s)
