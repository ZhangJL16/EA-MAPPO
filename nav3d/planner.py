"""Small-map 3D visibility-graph planner with exact inflated edge checks.

This is a deterministic first baseline. Candidate vertices around boxes and
cylinders allow horizontal detours and flight above or below finite obstacles.
It intentionally reports no route when the finite candidate graph fails.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from math import cos, sin, tau
from time import perf_counter

import numpy as np

from .geometry import BoxObstacle, CylinderObstacle, World3D, vec3


class NoRouteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Route:
    waypoints: tuple[np.ndarray, ...]
    geometric_length: float
    weighted_cost: float
    planning_seconds: float
    candidate_nodes: int
    checked_edges: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints", tuple(vec3(p, "waypoint") for p in self.waypoints))


class VisibilityPlanner:
    def __init__(
        self,
        world: World3D,
        *,
        body_radius: float = 0.5,
        clearance_margin: float = 0.25,
        vertical_weight: float = 1.0,
        cylinder_samples: int = 12,
        max_nodes: int = 650,
    ) -> None:
        if body_radius < 0 or clearance_margin < 0 or vertical_weight < 1 or cylinder_samples < 6 or max_nodes < 2:
            raise ValueError("invalid planner configuration")
        self.world = world
        self.inflation = float(body_radius + clearance_margin)
        self.vertical_weight = float(vertical_weight)
        self.cylinder_samples = int(cylinder_samples)
        self.max_nodes = int(max_nodes)

    def _cost(self, a: np.ndarray, b: np.ndarray) -> float:
        delta = b - a
        return float(np.linalg.norm(delta * np.array([1.0, 1.0, self.vertical_weight])))

    def _height_levels(self, start: np.ndarray, goal: np.ndarray, low: float, high: float) -> list[float]:
        pad = max(0.12, 0.08 * self.inflation)
        levels = (start[2], goal[2], low - self.inflation - pad, high + self.inflation + pad)
        return sorted({round(float(z), 7) for z in levels if self.inflation + 1e-8 < z < self.world.size[2] - self.inflation - 1e-8})

    def _candidate_points(self, start: np.ndarray, goal: np.ndarray) -> list[np.ndarray]:
        result = [start, goal]
        pad = max(0.12, 0.08 * self.inflation)
        for obstacle in self.world.obstacles:
            if isinstance(obstacle, BoxObstacle):
                x_values = (obstacle.low[0] - self.inflation - pad, obstacle.high[0] + self.inflation + pad)
                y_values = (obstacle.low[1] - self.inflation - pad, obstacle.high[1] + self.inflation + pad)
                for z in self._height_levels(start, goal, obstacle.low[2], obstacle.high[2]):
                    for x in x_values:
                        for y in y_values:
                            result.append(np.array([x, y, z], dtype=np.float64))
                    y_middle = float(np.clip((start[1] + goal[1]) / 2, obstacle.low[1], obstacle.high[1]))
                    x_middle = float(np.clip((start[0] + goal[0]) / 2, obstacle.low[0], obstacle.high[0]))
                    for x in x_values:
                        result.append(np.array([x, y_middle, z], dtype=np.float64))
                    for y in y_values:
                        result.append(np.array([x_middle, y, z], dtype=np.float64))
                result.append(np.array([(obstacle.low[0] + obstacle.high[0]) / 2, (obstacle.low[1] + obstacle.high[1]) / 2, obstacle.high[2] + self.inflation + pad]))
            else:
                ring_radius = obstacle.radius + self.inflation + pad
                for z in self._height_levels(start, goal, obstacle.z_low, obstacle.z_high):
                    for index in range(self.cylinder_samples):
                        angle = tau * index / self.cylinder_samples
                        result.append(np.array([obstacle.center_xy[0] + ring_radius * cos(angle), obstacle.center_xy[1] + ring_radius * sin(angle), z]))
                result.append(np.array([obstacle.center_xy[0], obstacle.center_xy[1], obstacle.z_high + self.inflation + pad]))
        unique: dict[tuple[float, float, float], np.ndarray] = {}
        for point in result:
            if self.world.clearance(point, self.inflation) > 1e-8:
                unique[tuple(np.round(point, 7))] = point
        return list(unique.values())

    def plan(self, start: object, goal: object) -> Route:
        begun = perf_counter()
        origin, target = vec3(start, "start"), vec3(goal, "goal")
        if self.world.clearance(origin, self.inflation) <= 0 or self.world.clearance(target, self.inflation) <= 0:
            raise NoRouteError("start or goal violates map clearance")
        if np.linalg.norm(origin - target) < 1e-10:
            return Route((origin,), 0.0, 0.0, perf_counter() - begun, 1, 0)
        if self.world.segment_clear(origin, target, self.inflation):
            length = float(np.linalg.norm(target - origin))
            return Route((origin, target), length, self._cost(origin, target), perf_counter() - begun, 2, 1)
        nodes = self._candidate_points(origin, target)
        if len(nodes) > self.max_nodes:
            raise NoRouteError(f"candidate graph exceeds {self.max_nodes} nodes")
        # The start and goal are inserted first and survive candidate filtering.
        if len(nodes) < 2 or not np.allclose(nodes[0], origin) or not np.allclose(nodes[1], target):
            raise NoRouteError("invalid candidate graph endpoints")
        adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
        checks = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                checks += 1
                if self.world.segment_clear(nodes[i], nodes[j], self.inflation):
                    cost = self._cost(nodes[i], nodes[j])
                    adjacency[i].append((j, cost))
                    adjacency[j].append((i, cost))
        distance = [float("inf")] * len(nodes)
        predecessor = [-1] * len(nodes)
        distance[0] = 0.0
        queue = [(0.0, 0)]
        while queue:
            value, node = heapq.heappop(queue)
            if value > distance[node] + 1e-10:
                continue
            if node == 1:
                break
            for neighbor, edge_cost in adjacency[node]:
                candidate = value + edge_cost
                if candidate + 1e-10 < distance[neighbor]:
                    distance[neighbor] = candidate
                    predecessor[neighbor] = node
                    heapq.heappush(queue, (candidate, neighbor))
        if not np.isfinite(distance[1]):
            raise NoRouteError("visibility graph found no valid route")
        indices = [1]
        while indices[-1] != 0:
            indices.append(predecessor[indices[-1]])
        indices.reverse()
        # Remove intermediate candidates whenever the exact inflated segment is clear.
        smooth = [indices[0]]
        while smooth[-1] != indices[-1]:
            current_index = indices.index(smooth[-1])
            for next_index in range(len(indices) - 1, current_index, -1):
                checks += 1
                if self.world.segment_clear(nodes[smooth[-1]], nodes[indices[next_index]], self.inflation):
                    smooth.append(indices[next_index])
                    break
        waypoints = tuple(nodes[index] for index in smooth)
        geometric = sum(float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:]))
        weighted = sum(self._cost(a, b) for a, b in zip(waypoints, waypoints[1:]))
        return Route(waypoints, geometric, weighted, perf_counter() - begun, len(nodes), checks)
