"""Full-map, nonlearning route and recharge controls for paired diagnostics.

Both variants use only the current target, map, battery and station.  They do
not inspect future targets.  The environment applies the same joint safety
filter to their actions as it will to learned policies.
"""

from __future__ import annotations

from math import isfinite

import numpy as np

from .environment import EnvironmentAction
from .reference import simulate_reference_leg
from .routes import PlanarRoute, PlanarRouter
from .synthetic import SyntheticMap


class FullMapRouteBaseline:
    """Shortest-route waypoint tracker with full or reference-sized charging."""

    def __init__(self, case: SyntheticMap, *, charge_mode: str = "full") -> None:
        if charge_mode not in ("full", "reference_minimum"):
            raise ValueError("charge mode must be full or reference_minimum")
        self.case = case
        self.charge_mode = charge_mode
        self.router = PlanarRouter(case)
        self._route: PlanarRoute | None = None
        self._route_goal: tuple[float, float] | None = None
        self._waypoint_index = 0
        self._last_mode: str | None = None
        self._reference_energy: dict[tuple[float, float], float] = {}

    def _current_target_reference_energy(self, target: tuple[float, float]) -> float:
        if target not in self._reference_energy:
            outbound = simulate_reference_leg(
                self.case, self.case.station_xy, target, router=self.router
            )
            if not outbound.arrived or outbound.collision_count or outbound.infeasible_steps:
                raise RuntimeError("current target lacks the shared outbound reference witness")
            inbound = simulate_reference_leg(
                self.case, outbound.final_position_xy, self.case.station_xy,
                router=self.router,
            )
            if not inbound.arrived or inbound.collision_count or inbound.infeasible_steps:
                raise RuntimeError("current target lacks the shared return reference witness")
            self._reference_energy[target] = outbound.energy + inbound.energy
        return self._reference_energy[target]

    def _desired_velocity(
        self, position: np.ndarray, target: tuple[float, float], mode: str
    ) -> tuple[float, float]:
        if self._route_goal != target or self._route is None or mode != self._last_mode:
            self._route = self.router.plan(position, target)
            self._route_goal = target
            self._waypoint_index = min(1, len(self._route.waypoints_xy) - 1)
        else:
            waypoint = self._route.waypoints_xy[self._waypoint_index]
            if not self.case.segment_clear(position, waypoint):
                self._route = self.router.plan(position, target)
                self._waypoint_index = min(1, len(self._route.waypoints_xy) - 1)
        assert self._route is not None
        while (
            self._waypoint_index < len(self._route.waypoints_xy) - 1
            and np.linalg.norm(position - self._route.waypoints_xy[self._waypoint_index]) <= 0.8
        ):
            self._waypoint_index += 1
        delta = self._route.waypoints_xy[self._waypoint_index] - position
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return (0.0, 0.0)
        desired = delta / distance * min(5.0, 0.8 * distance)
        return float(desired[0]), float(desired[1])

    def act(self, observation: dict) -> EnvironmentAction | None:
        mode = observation["mode"]
        if mode == "return":
            self._last_mode = mode
            return None
        if mode not in ("docked", "flight"):
            raise ValueError(f"unknown environment mode: {mode}")
        energy = float(observation["energy"])
        capacity = float(observation["capacity"])
        target = tuple(float(x) for x in observation["target_xy"])
        position = np.asarray(observation["position_xy"], dtype=float)
        if not all(isfinite(x) for x in (energy, capacity)) or capacity <= 0:
            raise ValueError("invalid battery observation")
        if mode == "docked":
            if self.charge_mode == "full":
                charge_target = capacity
            else:
                # Five percent of capacity is a declared, fixed model-mismatch
                # margin, not a value fitted to this baseline's outcomes.
                charge_target = min(
                    capacity, self._current_target_reference_energy(target) + 0.05 * capacity
                )
            if energy + 1e-8 < charge_target:
                self._last_mode = mode
                return EnvironmentAction(charge_target_fraction=charge_target / capacity)
        desired = self._desired_velocity(position, target, mode)
        self._last_mode = mode
        return EnvironmentAction(desired_velocity_xy=desired)
