"""Planar visibility graph for the synthetic static-disc maps.

Every vertex and checked edge remains at the fixed flight height.  The graph
is a conservative finite candidate set; failure to find a route is not a
mathematical proof that no collision-free path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import cos, pi, sin, tau

import numpy as np

from .synthetic import SyntheticMap


class NoPlanarRoute(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanarRoute:
    waypoints_xy: tuple[np.ndarray, ...]
    length_m: float
    edge_checks: int


class PlanarRouter:
    def __init__(self, case: SyntheticMap, *, samples_per_disc: int = 16, ring_pad_m: float = 0.5) -> None:
        if samples_per_disc < 12 or ring_pad_m <= 0:
            raise ValueError("invalid planar ring sampling")
        self.case = case
        self.samples_per_disc = samples_per_disc
        self.ring_pad_m = ring_pad_m
        nodes: list[np.ndarray] = []
        for obstacle in case.world.obstacles:
            radius = obstacle.radius + case.config.planning_inflation_m + ring_pad_m
            for i in range(samples_per_disc):
                angle = tau * i / samples_per_disc
                point = obstacle.center_xy + radius * np.array((cos(angle), sin(angle)))
                if case.is_clear(point):
                    nodes.append(point)
        self.nodes = tuple(nodes)
        self.static_adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
        checks = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                checks += 1
                if case.segment_clear(nodes[i], nodes[j]):
                    cost = float(np.linalg.norm(nodes[i] - nodes[j]))
                    self.static_adjacency[i].append((j, cost))
                    self.static_adjacency[j].append((i, cost))
        self.static_edge_checks = checks

    def plan(self, start_xy: object, goal_xy: object) -> PlanarRoute:
        start = np.asarray(start_xy, dtype=float)
        goal = np.asarray(goal_xy, dtype=float)
        if start.shape != (2,) or goal.shape != (2,) or not np.isfinite(start).all() or not np.isfinite(goal).all():
            raise ValueError("planar endpoints must be finite 2-vectors")
        if not self.case.is_clear(start) or not self.case.is_clear(goal):
            raise NoPlanarRoute("endpoint violates planar clearance")
        if np.linalg.norm(start - goal) < 1e-10:
            return PlanarRoute((start.copy(),), 0.0, 0)
        if self.case.segment_clear(start, goal):
            return PlanarRoute((start.copy(), goal.copy()), float(np.linalg.norm(goal - start)), 1)
        nodes = (start, goal, *self.nodes)
        adjacency: list[list[tuple[int, float]]] = [[], [], *[row.copy() for row in self.static_adjacency]]
        # Static node indices shift by two when endpoints are inserted.
        for i in range(2, len(nodes)):
            adjacency[i] = [(j + 2, weight) for j, weight in adjacency[i]]
        checks = 1
        for endpoint in (0, 1):
            for i in range(2, len(nodes)):
                checks += 1
                if self.case.segment_clear(nodes[endpoint], nodes[i]):
                    cost = float(np.linalg.norm(nodes[endpoint] - nodes[i]))
                    adjacency[endpoint].append((i, cost))
                    adjacency[i].append((endpoint, cost))
        distances = [float("inf")] * len(nodes)
        predecessors = [-1] * len(nodes)
        distances[0] = 0.0
        queue = [(0.0, 0)]
        while queue:
            distance, index = heappop(queue)
            if distance > distances[index] + 1e-10:
                continue
            if index == 1:
                break
            for neighbor, weight in adjacency[index]:
                candidate = distance + weight
                if candidate + 1e-10 < distances[neighbor]:
                    distances[neighbor] = candidate
                    predecessors[neighbor] = index
                    heappush(queue, (candidate, neighbor))
        if not np.isfinite(distances[1]):
            raise NoPlanarRoute("finite planar graph has no route")
        indices = [1]
        while indices[-1] != 0:
            indices.append(predecessors[indices[-1]])
        indices.reverse()
        # Keep only line-of-sight waypoints; validate each smoothed edge.
        selected = [indices[0]]
        while selected[-1] != 1:
            current = indices.index(selected[-1])
            for next_index in range(len(indices) - 1, current, -1):
                checks += 1
                if self.case.segment_clear(nodes[selected[-1]], nodes[indices[next_index]]):
                    selected.append(indices[next_index])
                    break
            else:
                raise NoPlanarRoute("route smoothing lost connectivity")
        waypoints = tuple(nodes[i].copy() for i in selected)
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(waypoints, waypoints[1:]))
        return PlanarRoute(waypoints, length, checks)
