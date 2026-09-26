"""Static 3D map geometry used by the analytic navigator.

Boxes and finite vertical cylinders are inflated conservatively for a spherical
UAV body.  The map is an explicit navigation input; it is never constructed by
reading hidden simulator obstacle fields at policy inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np


def vec3(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return array.copy()


@dataclass(frozen=True)
class BoxObstacle:
    low: np.ndarray
    high: np.ndarray
    name: str = "box"

    def __post_init__(self) -> None:
        low, high = vec3(self.low, "box low"), vec3(self.high, "box high")
        if np.any(high <= low):
            raise ValueError("box high must exceed low on every axis")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def signed_distance_normal(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        nearest = np.clip(point, self.low, self.high)
        delta = point - nearest
        distance = float(np.linalg.norm(delta))
        if distance > 1e-12:
            return distance, delta / distance
        face_distances = np.stack((point - self.low, self.high - point))
        side, axis = np.unravel_index(np.argmin(face_distances), face_distances.shape)
        normal = np.zeros(3)
        normal[axis] = -1.0 if side == 0 else 1.0
        return -float(face_distances[side, axis]), normal


@dataclass(frozen=True)
class CylinderObstacle:
    center_xy: np.ndarray
    radius: float
    z_low: float
    z_high: float
    name: str = "cylinder"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("center_xy must be a finite length-2 vector")
        if not np.isfinite(self.radius) or self.radius <= 0 or not np.isfinite(self.z_low) or not np.isfinite(self.z_high) or self.z_high <= self.z_low:
            raise ValueError("invalid finite-cylinder dimensions")
        object.__setattr__(self, "center_xy", center.copy())

    def signed_distance_normal(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        radial = point[:2] - self.center_xy
        radial_norm = float(np.linalg.norm(radial))
        radial_unit = radial / radial_norm if radial_norm > 1e-12 else np.array([1.0, 0.0])
        closest_xy = self.center_xy + min(radial_norm, self.radius) * radial_unit
        closest_z = float(np.clip(point[2], self.z_low, self.z_high))
        delta = point - np.array([closest_xy[0], closest_xy[1], closest_z])
        distance = float(np.linalg.norm(delta))
        if distance > 1e-12:
            return distance, delta / distance
        gaps = np.array([self.radius - radial_norm, point[2] - self.z_low, self.z_high - point[2]])
        face = int(np.argmin(gaps))
        normal = np.array([radial_unit[0], radial_unit[1], 0.0]) if face == 0 else np.array([0.0, 0.0, -1.0 if face == 1 else 1.0])
        return -float(gaps[face]), normal


Obstacle: TypeAlias = BoxObstacle | CylinderObstacle


def _segment_box(a: np.ndarray, b: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    direction = b - a
    enter, exit_ = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-14:
            if a[axis] < low[axis] or a[axis] > high[axis]:
                return False
            continue
        first = (low[axis] - a[axis]) / direction[axis]
        second = (high[axis] - a[axis]) / direction[axis]
        enter = max(enter, min(first, second))
        exit_ = min(exit_, max(first, second))
        if enter > exit_:
            return False
    return True


def _segment_cylinder(a: np.ndarray, b: np.ndarray, obstacle: CylinderObstacle, inflate: float) -> bool:
    direction = b - a
    rel = a[:2] - obstacle.center_xy
    radius = obstacle.radius + inflate
    quadratic = float(direction[:2] @ direction[:2])
    linear = float(2.0 * (rel @ direction[:2]))
    constant = float(rel @ rel - radius * radius)
    if quadratic < 1e-14:
        if constant > 0.0:
            return False
        radial_lo, radial_hi = 0.0, 1.0
    else:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant < 0.0:
            return False
        root = float(np.sqrt(max(0.0, discriminant)))
        radial_lo = max(0.0, (-linear - root) / (2.0 * quadratic))
        radial_hi = min(1.0, (-linear + root) / (2.0 * quadratic))
        if radial_lo > radial_hi:
            return False
    z_low, z_high = obstacle.z_low - inflate, obstacle.z_high + inflate
    if abs(direction[2]) < 1e-14:
        return z_low <= a[2] <= z_high
    first = (z_low - a[2]) / direction[2]
    second = (z_high - a[2]) / direction[2]
    return max(radial_lo, min(first, second), 0.0) <= min(radial_hi, max(first, second), 1.0)


@dataclass(frozen=True)
class Barrier:
    name: str
    h: float
    normal: np.ndarray


class World3D:
    def __init__(self, size: object, obstacles: tuple[Obstacle, ...] = ()) -> None:
        self.size = vec3(size, "world size")
        if np.any(self.size <= 0.0):
            raise ValueError("world size must be positive")
        self.obstacles = tuple(obstacles)
        if not all(isinstance(item, (BoxObstacle, CylinderObstacle)) for item in self.obstacles):
            raise TypeError("only BoxObstacle and CylinderObstacle are supported")

    def barriers(self, point: object, inflation: float) -> list[Barrier]:
        p = vec3(point, "point")
        if not np.isfinite(inflation) or inflation < 0.0:
            raise ValueError("inflation must be nonnegative")
        result: list[Barrier] = []
        for axis in range(3):
            inward_low = np.eye(3)[axis]
            result.append(Barrier(f"boundary_low_{axis}", float(p[axis] - inflation), inward_low))
            result.append(Barrier(f"boundary_high_{axis}", float(self.size[axis] - p[axis] - inflation), -inward_low))
        for i, obstacle in enumerate(self.obstacles):
            distance, normal = obstacle.signed_distance_normal(p)
            result.append(Barrier(f"{obstacle.name}_{i}", distance - inflation, normal))
        return result

    def clearance(self, point: object, inflation: float) -> float:
        return min(item.h for item in self.barriers(point, inflation))

    def segment_contacts(self, start: object, end: object, inflation: float) -> tuple[bool, bool]:
        a, b = vec3(start, "segment start"), vec3(end, "segment end")
        if not np.isfinite(inflation) or inflation < 0.0:
            raise ValueError("inflation must be nonnegative")
        lower, upper = np.full(3, inflation), self.size - inflation
        boundary = bool(np.any(a < lower) or np.any(a > upper) or np.any(b < lower) or np.any(b > upper))
        obstacle_contact = False
        for obstacle in self.obstacles:
            if isinstance(obstacle, BoxObstacle):
                obstacle_contact |= _segment_box(a, b, obstacle.low - inflation, obstacle.high + inflation)
            else:
                obstacle_contact |= _segment_cylinder(a, b, obstacle, inflation)
        return boundary, obstacle_contact

    def segment_clear(self, start: object, end: object, inflation: float) -> bool:
        return not any(self.segment_contacts(start, end, inflation))
