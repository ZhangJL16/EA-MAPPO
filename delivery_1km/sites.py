"""Policy-independent service sites on the ground or above accessible roofs."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Literal

import numpy as np

from nav3d.planner import NoRouteError
from nav3d_v2.controller import PLANNING_CLEARANCE_MARGIN

from .navigation import DeliveryRoutePlanner
from .scenario import DeliveryMap, DeliveryScenario, M100_1KM


@dataclass(frozen=True)
class ServiceSite:
    position_m: tuple[float, float, float]
    kind: Literal["ground", "roof"]
    support_name: str | None = None


def sample_site(
    case: DeliveryMap,
    site_id: int,
    kind: Literal["ground", "roof"],
    scenario: DeliveryScenario = M100_1KM,
) -> ServiceSite:
    """Draw a reproducible site with a static-map route from the station.

    A service point is 2 m above ground or a cylinder roof. This models
    hovering at the pickup/drop point; package handoff dynamics are not yet
    modeled. The route check is generation-only and must not be passed to a
    deployed high-level policy as full-map information.
    """
    if not isinstance(site_id, int) or site_id < 0:
        raise ValueError("site_id must be a nonnegative integer")
    if kind not in ("ground", "roof"):
        raise ValueError("kind must be ground or roof")
    if kind == "roof" and not case.world.obstacles:
        raise ValueError("roof sites require at least one rooftop obstacle")
    rng = np.random.default_rng(3_000_000_007 + 7919 * case.map_id + 23 * site_id + (0 if kind == "ground" else 1))
    planner = DeliveryRoutePlanner(
        case.world, body_radius=scenario.body_radius_m,
        clearance_margin=PLANNING_CLEARANCE_MARGIN,
    )
    inflation = scenario.body_radius_m + PLANNING_CLEARANCE_MARGIN
    for _ in range(2000):
        if kind == "ground":
            margin = scenario.obstacle_xy_margin_m
            point = (
                float(rng.uniform(margin, scenario.size_m[0] - margin)),
                float(rng.uniform(margin, scenario.size_m[1] - margin)),
                scenario.station_handoff_height_m,
            )
            support_name = None
        else:
            obstacle = case.world.obstacles[int(rng.integers(len(case.world.obstacles)))]
            usable_radius = obstacle.radius - 2.0 * inflation
            if usable_radius <= 0:
                continue
            radius = sqrt(float(rng.random())) * usable_radius
            angle = 2.0 * pi * float(rng.random())
            point = (
                float(obstacle.center_xy[0] + radius * cos(angle)),
                float(obstacle.center_xy[1] + radius * sin(angle)),
                float(obstacle.z_high + 2.0),
            )
            support_name = obstacle.name
        if case.world.clearance(point, inflation) <= 0:
            continue
        try:
            planner.plan(case.station_m, point)
        except NoRouteError:
            continue
        return ServiceSite(point, kind, support_name)
    raise RuntimeError("could not sample a reachable service site")
