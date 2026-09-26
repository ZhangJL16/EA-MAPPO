"""Auditable candidate platform and map settings for the future delivery simulator.

The M100 mass/battery/max-takeoff values are manufacturer specifications. Cargo,
onboard-equipment mass, control limits and map scaling are research settings.
No flight-power or charging-time model is implied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

import numpy as np

from nav3d.controller import CBFConfig
from nav3d.geometry import CylinderObstacle, World3D
from nav3d_v2.controller import PLANNING_CLEARANCE_MARGIN


@dataclass(frozen=True)
class DeliveryScenario:
    name: str = "m100_1km_candidate_v0"
    size_m: tuple[float, float, float] = (1000.0, 1000.0, 100.0)
    station_xy_margin_m: float = 100.0
    # Flight handoff above a ground dock; landing and charging are separate.
    station_handoff_height_m: float = 2.0
    base_mass_kg: float = 2.355  # M100 with one TB47D; DJI specification.
    max_takeoff_mass_kg: float = 3.6
    nominal_battery_wh: float = 99.9  # Nameplate only, not usable SOC.
    equipment_mass_kg: float = 0.50  # LiDAR + compute + installation allowance.
    cargo_masses_kg: tuple[float, ...] = (0.0, 0.25, 0.50)
    horizontal_speed_limit_mps: float = 7.0
    vertical_speed_limit_mps: float = 4.0
    horizontal_acceleration_limit_mps2: float = 4.0
    vertical_acceleration_limit_mps2: float = 3.0
    physics_dt_s: float = 0.05
    body_radius_m: float = 0.4
    clearance_margin_m: float = 0.2
    obstacle_count: int = 24
    obstacle_radius_range_m: tuple[float, float] = (12.5, 30.0)
    obstacle_height_range_m: tuple[float, float] = (15.0, 75.0)
    obstacle_xy_margin_m: float = 25.0
    obstacle_separation_m: float = 5.0
    station_obstacle_clearance_m: float = 35.0
    task_altitude_range_m: tuple[float, float] = (5.0, 95.0)
    minimum_task_distance_m: float = 25.0
    goal_radius_m: float = 5.0  # Retains the old service tolerance; review later.
    navigation_guard_s: float = 800.0  # Provisional timeout, not an order SLA.

    def __post_init__(self) -> None:
        scalars = (
            *self.size_m, self.station_xy_margin_m,
            self.station_handoff_height_m, self.base_mass_kg,
            self.max_takeoff_mass_kg, self.nominal_battery_wh,
            self.equipment_mass_kg, *self.cargo_masses_kg,
            self.horizontal_speed_limit_mps, self.vertical_speed_limit_mps,
            self.horizontal_acceleration_limit_mps2,
            self.vertical_acceleration_limit_mps2, self.physics_dt_s,
            self.body_radius_m, self.clearance_margin_m,
            *self.obstacle_radius_range_m, *self.obstacle_height_range_m,
            self.obstacle_xy_margin_m,
            self.obstacle_separation_m, self.station_obstacle_clearance_m,
            *self.task_altitude_range_m, self.minimum_task_distance_m,
            self.goal_radius_m, self.navigation_guard_s,
        )
        if not all(isfinite(value) for value in scalars):
            raise ValueError("scenario values must be finite")
        if any(value <= 0 for value in self.size_m):
            raise ValueError("world dimensions must be positive")
        if self.base_mass_kg <= 0 or self.nominal_battery_wh <= 0 or self.equipment_mass_kg < 0:
            raise ValueError("invalid platform mass or nominal battery")
        if not self.cargo_masses_kg or any(value < 0 for value in self.cargo_masses_kg):
            raise ValueError("cargo masses must be nonempty and nonnegative")
        if self.maximum_loaded_mass_kg > self.max_takeoff_mass_kg:
            raise ValueError("candidate loading exceeds manufacturer takeoff limit")
        if not (0 < self.task_altitude_range_m[0] < self.task_altitude_range_m[1] < self.size_m[2]):
            raise ValueError("task altitude range must be inside the world")
        if not (0 < 2 * self.station_xy_margin_m < min(self.size_m[:2])):
            raise ValueError("station sampling margin leaves no interior")
        if not (self.body_radius_m + PLANNING_CLEARANCE_MARGIN < self.station_handoff_height_m < self.size_m[2]):
            raise ValueError("station flight handoff lacks route-planning floor clearance")
        if self.obstacle_count < 0 or not (0 < self.obstacle_radius_range_m[0] <= self.obstacle_radius_range_m[1]):
            raise ValueError("invalid obstacle count or radii")
        if not (0 < self.obstacle_height_range_m[0] < self.obstacle_height_range_m[1] < self.size_m[2] - self.body_radius_m - PLANNING_CLEARANCE_MARGIN):
            raise ValueError("finite obstacle heights must leave an overflight corridor")
        if self.obstacle_separation_m < 0 or self.station_obstacle_clearance_m < 0:
            raise ValueError("obstacle clearances must be nonnegative")
        if self.obstacle_xy_margin_m <= self.body_radius_m + self.clearance_margin_m:
            raise ValueError("obstacle margin must exceed physical inflation")
        if self.goal_radius_m <= self.body_radius_m or self.minimum_task_distance_m <= self.goal_radius_m:
            raise ValueError("invalid delivery goal or task distance")
        if self.navigation_guard_s / self.physics_dt_s < 1:
            raise ValueError("navigation guard must cover at least one physics step")
        self.navigation_config()  # Delegate control-limit validation to the navigator.

    @property
    def maximum_loaded_mass_kg(self) -> float:
        return self.base_mass_kg + self.equipment_mass_kg + max(self.cargo_masses_kg)

    @property
    def navigation_guard_steps(self) -> int:
        return round(self.navigation_guard_s / self.physics_dt_s)

    def navigation_config(self) -> CBFConfig:
        return CBFConfig(
            dt=self.physics_dt_s,
            body_radius=self.body_radius_m,
            clearance_margin=self.clearance_margin_m,
            max_horizontal_speed=self.horizontal_speed_limit_mps,
            max_vertical_speed=self.vertical_speed_limit_mps,
            max_horizontal_acceleration=self.horizontal_acceleration_limit_mps2,
            max_vertical_acceleration=self.vertical_acceleration_limit_mps2,
            activation_distance=15.0,
        )


M100_1KM = DeliveryScenario()


@dataclass(frozen=True)
class DeliveryMap:
    map_id: int
    world: World3D
    station_m: tuple[float, float, float]


def station_for_map(map_id: int, scenario: DeliveryScenario = M100_1KM) -> tuple[float, float, float]:
    """One policy-independent ground charger per map, with a 2 m flight handoff."""
    if not isinstance(map_id, int) or map_id < 0:
        raise ValueError("map_id must be a nonnegative integer")
    rng = np.random.default_rng(2_000_000_007 + 1013 * map_id)
    x = float(rng.uniform(scenario.station_xy_margin_m, scenario.size_m[0] - scenario.station_xy_margin_m))
    y = float(rng.uniform(scenario.station_xy_margin_m, scenario.size_m[1] - scenario.station_xy_margin_m))
    return x, y, scenario.station_handoff_height_m


def make_map(map_id: int, scenario: DeliveryScenario = M100_1KM) -> DeliveryMap:
    """Make a deterministic radius-scaled, finite-height cylinder map.

    XY coordinates and cylinder radii scale from the old 4 km world; obstacle
    heights do not. Three height bands ensure low, medium, and tall obstacles.
    UAV body size, LiDAR properties, and kinematic limits remain physical.
    A new seed family and non-overlap rule make this a new map distribution,
    never a replacement for legacy or frozen v2 qualification maps.
    """
    if not isinstance(map_id, int) or map_id < 0:
        raise ValueError("map_id must be a nonnegative integer")
    rng = np.random.default_rng(1_000_000_007 + 1009 * map_id)
    station_m = station_for_map(map_id, scenario)
    obstacles: list[CylinderObstacle] = []
    x_size, y_size, _height = scenario.size_m
    r_min, r_max = scenario.obstacle_radius_range_m
    h_min, h_max = scenario.obstacle_height_range_m
    height_band_width = (h_max - h_min) / 3.0
    for _ in range(max(1000, 1000 * scenario.obstacle_count)):
        if len(obstacles) == scenario.obstacle_count:
            break
        radius = float(rng.uniform(r_min, r_max))
        lower = radius + scenario.obstacle_xy_margin_m
        x = float(rng.uniform(lower, x_size - lower))
        y = float(rng.uniform(lower, y_size - lower))
        station_distance = hypot(x - station_m[0], y - station_m[1])
        if station_distance < radius + scenario.station_obstacle_clearance_m:
            continue
        if any(
            hypot(x - existing.center_xy[0], y - existing.center_xy[1])
            < radius + existing.radius + scenario.obstacle_separation_m
            for existing in obstacles
        ):
            continue
        band = len(obstacles) % 3
        obstacle_height = float(rng.uniform(h_min + band * height_band_width, h_min + (band + 1) * height_band_width))
        obstacles.append(CylinderObstacle(np.array([x, y]), radius, 0.0, obstacle_height, f"cylinder_{len(obstacles):02d}"))
    if len(obstacles) != scenario.obstacle_count:
        raise RuntimeError("could not place the requested obstacle count")
    return DeliveryMap(map_id, World3D(scenario.size_m, tuple(obstacles)), station_m)


def make_world(map_id: int, scenario: DeliveryScenario = M100_1KM) -> World3D:
    """Compatibility helper for callers that only need obstacle geometry."""
    return make_map(map_id, scenario).world
