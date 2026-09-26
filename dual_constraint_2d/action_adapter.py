"""Shared discrete high-level action semantics and full-map features.

Actions 0-8: route-relative heading {-30, 0, +30} degrees at speeds
{3.5, 5.0, 7.0} m/s.  Action 9: head toward the station in flight.
Actions 10-12: charge to 50/75/100% when docked.  In flight they are
central-heading speed actions so there are no invalid simulator actions.
At the dock, 0-9 depart toward the current target.
"""

from __future__ import annotations

from math import cos, pi, sin

import numpy as np

from .environment import EnvironmentAction
from .routes import NoPlanarRoute, PlanarRouter
from .synthetic import SyntheticMap


ACTION_COUNT = 13
CHARGE_FRACTIONS = (0.5, 0.75, 1.0)
HEADING_OFFSETS_RAD = (-pi / 6, 0.0, pi / 6)
SPEEDS_MPS = (3.5, 5.0, 7.0)
FEATURE_COUNT = 40


class ActionAdapter:
    def __init__(self, case: SyntheticMap) -> None:
        self.case = case
        self.router = PlanarRouter(case)

    def route_direction(self, position_xy: object, destination_xy: object) -> tuple[np.ndarray, float]:
        position = np.asarray(position_xy, dtype=float)
        target = np.asarray(destination_xy, dtype=float)
        try:
            route = self.router.plan(position, target)
        except NoPlanarRoute:
            # Raw training can enter a state that is collision-free at body
            # radius but violates the planner's extra clearance margin.  A
            # finite direct bearing keeps trial-and-error training defined;
            # it is not reported as a certified obstacle-free route.
            delta = target - position
            distance = float(np.linalg.norm(delta))
            return (np.zeros(2), 0.0) if distance <= 1e-12 else (delta / distance, distance)
        waypoint = route.waypoints_xy[min(1, len(route.waypoints_xy) - 1)]
        delta = waypoint - position
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            return np.zeros(2), route.length_m
        return delta / length, route.length_m

    def decode(self, action_id: int, observation: dict) -> EnvironmentAction:
        if not isinstance(action_id, (int, np.integer)) or not 0 <= action_id < ACTION_COUNT:
            raise ValueError("action ID must be in the common discrete action set")
        mode = observation["mode"]
        if mode == "return":
            raise ValueError("return takeover ignores policy actions")
        if mode == "docked" and action_id >= 10:
            fraction = CHARGE_FRACTIONS[action_id - 10]
            if observation["energy"] / observation["capacity"] + 1e-8 < fraction:
                return EnvironmentAction(charge_target_fraction=fraction)
            action_id = 4
        if mode == "flight" and action_id == 9:
            destination = observation["station_xy"]
            offset = 0.0
            speed = 5.0
        else:
            destination = observation["target_xy"]
            if action_id >= 10 or action_id == 9:
                action_id = 4
            offset = HEADING_OFFSETS_RAD[action_id // 3]
            speed = SPEEDS_MPS[action_id % 3]
        direction, _ = self.route_direction(observation["position_xy"], destination)
        rotation = np.array(((cos(offset), -sin(offset)), (sin(offset), cos(offset))))
        desired = rotation @ direction * speed
        # Taper near the last destination so that arrival includes low speed.
        distance = float(np.linalg.norm(
            np.asarray(observation["position_xy"]) - np.asarray(destination)
        ))
        desired *= min(1.0, 0.8 * distance / max(speed, 1e-12))
        return EnvironmentAction((float(desired[0]), float(desired[1])))

    def features(self, observation: dict) -> np.ndarray:
        position = np.asarray(observation["position_xy"], dtype=float)
        target = np.asarray(observation["target_xy"], dtype=float)
        station = np.asarray(observation["station_xy"], dtype=float)
        velocity = np.asarray(observation["velocity_xy"], dtype=float)
        direction, target_length = self.route_direction(position, target)
        _, station_length = self.route_direction(position, station)
        mode = observation["mode"]
        basic = [
            *(position / self.case.config.side_m),
            *(velocity / self.case.config.max_speed_mps),
            *((target - position) / self.case.config.side_m),
            *((station - position) / self.case.config.side_m),
            observation["energy"] / observation["capacity"],
            max(0.0, 1.0 - observation["time_s"] / self.case.config.horizon_s),
            float(mode == "docked"),
            float(mode == "flight"),
            *direction,
            target_length / (2 * self.case.config.side_m),
            station_length / (2 * self.case.config.side_m),
        ]
        obstacles = []
        for x, y, radius in observation["obstacles"]:
            obstacles.extend((
                x / self.case.config.side_m,
                y / self.case.config.side_m,
                radius / self.case.config.side_m,
            ))
        feature = np.asarray((*basic, *obstacles), dtype=np.float32)
        if feature.shape != (FEATURE_COUNT,) or not np.isfinite(feature).all():
            raise RuntimeError("invalid fixed-size full-map feature vector")
        return feature
