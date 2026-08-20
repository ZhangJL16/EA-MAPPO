from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hocbf import SphericalObstacle


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


@dataclass(frozen=True)
class SphereGeometry:
    center: np.ndarray
    radius: float
    identifier: str = "sphere"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "center"))
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("sphere radius must be finite and positive")

    def ray_distance(self, origin: np.ndarray, direction: np.ndarray) -> float:
        relative = _vec3(origin, "origin") - self.center
        ray = _vec3(direction, "direction")
        linear = float(relative @ ray)
        constant = float(relative @ relative - self.radius**2)
        discriminant = linear**2 - constant
        if discriminant < 0.0:
            return np.inf
        root = float(np.sqrt(discriminant))
        candidates = (-linear - root, -linear + root)
        positive = [value for value in candidates if value >= 0.0]
        return min(positive) if positive else np.inf

    def clearance(self, point: np.ndarray) -> float:
        return float(np.linalg.norm(_vec3(point, "point") - self.center) - self.radius)


@dataclass(frozen=True)
class VerticalCylinderGeometry:
    center_xy: np.ndarray
    radius: float
    z_low: float
    z_high: float
    identifier: str = "cylinder"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("cylinder center_xy must be a finite (2,) vector")
        if self.radius <= 0.0 or self.z_high <= self.z_low:
            raise ValueError("invalid vertical cylinder geometry")
        object.__setattr__(self, "center_xy", center.copy())

    def ray_distance(self, origin: np.ndarray, direction: np.ndarray) -> float:
        point = _vec3(origin, "origin")
        ray = _vec3(direction, "direction")
        candidates: list[float] = []
        relative_xy = point[:2] - self.center_xy
        quadratic = float(ray[:2] @ ray[:2])
        if quadratic > 1e-15:
            linear = 2.0 * float(relative_xy @ ray[:2])
            constant = float(relative_xy @ relative_xy - self.radius**2)
            discriminant = linear**2 - 4.0 * quadratic * constant
            if discriminant >= 0.0:
                root = float(np.sqrt(discriminant))
                for value in (
                    (-linear - root) / (2.0 * quadratic),
                    (-linear + root) / (2.0 * quadratic),
                ):
                    if value >= 0.0:
                        z = point[2] + value * ray[2]
                        if self.z_low <= z <= self.z_high:
                            candidates.append(float(value))
        if abs(ray[2]) > 1e-15:
            for z in (self.z_low, self.z_high):
                value = (z - point[2]) / ray[2]
                if value >= 0.0:
                    xy = point[:2] + value * ray[:2]
                    if np.linalg.norm(xy - self.center_xy) <= self.radius:
                        candidates.append(float(value))
        return min(candidates) if candidates else np.inf

    def clearance(self, point: np.ndarray) -> float:
        value = _vec3(point, "point")
        radial = float(np.linalg.norm(value[:2] - self.center_xy) - self.radius)
        if self.z_low <= value[2] <= self.z_high:
            return radial
        vertical = min(abs(value[2] - self.z_low), abs(value[2] - self.z_high))
        return float(np.hypot(max(radial, 0.0), vertical))


@dataclass(frozen=True)
class BoxGeometry:
    low: np.ndarray
    high: np.ndarray
    identifier: str = "box"

    def __post_init__(self) -> None:
        low = _vec3(self.low, "low")
        high = _vec3(self.high, "high")
        if np.any(high <= low):
            raise ValueError("box high must exceed low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def ray_distance(self, origin: np.ndarray, direction: np.ndarray) -> float:
        point = _vec3(origin, "origin")
        ray = _vec3(direction, "direction")
        entry = 0.0
        exit_ = np.inf
        for axis in range(3):
            if abs(ray[axis]) <= 1e-15:
                if point[axis] < self.low[axis] or point[axis] > self.high[axis]:
                    return np.inf
                continue
            first = (self.low[axis] - point[axis]) / ray[axis]
            second = (self.high[axis] - point[axis]) / ray[axis]
            entry = max(entry, min(first, second))
            exit_ = min(exit_, max(first, second))
            if entry > exit_:
                return np.inf
        return float(entry) if exit_ >= max(entry, 0.0) else np.inf

    def clearance(self, point: np.ndarray) -> float:
        value = _vec3(point, "point")
        delta = np.maximum(np.maximum(self.low - value, 0.0), value - self.high)
        if np.any(delta > 0.0):
            return float(np.linalg.norm(delta))
        distance_to_faces = np.minimum(value - self.low, self.high - value)
        return float(-np.min(distance_to_faces))


ObstacleGeometry = SphereGeometry | VerticalCylinderGeometry | BoxGeometry


@dataclass(frozen=True)
class Lidar3DConfig:
    horizontal_sectors: int = 128
    vertical_sectors: int = 8
    max_range: float = 100.0
    horizontal_fov_degrees: float = 360.0
    vertical_fov_degrees: float = 60.0

    def __post_init__(self) -> None:
        if self.horizontal_sectors <= 0 or self.vertical_sectors <= 0:
            raise ValueError("LiDAR sector counts must be positive")
        if self.max_range <= 0.0:
            raise ValueError("LiDAR range must be positive")
        if not 0.0 < self.horizontal_fov_degrees <= 360.0:
            raise ValueError("horizontal FOV must lie in (0, 360]")
        if not 0.0 < self.vertical_fov_degrees < 180.0:
            raise ValueError("vertical FOV must lie in (0, 180)")

    @property
    def num_sectors(self) -> int:
        return self.horizontal_sectors * self.vertical_sectors


@dataclass(frozen=True)
class Lidar3DPacket:
    distances: np.ndarray
    hit: np.ndarray
    directions: np.ndarray
    obstacle_indices: np.ndarray
    timestamp: float
    origin: np.ndarray

    def __post_init__(self) -> None:
        distances = np.asarray(self.distances, dtype=np.float64)
        hit = np.asarray(self.hit, dtype=bool)
        directions = np.asarray(self.directions, dtype=np.float64)
        indices = np.asarray(self.obstacle_indices, dtype=np.int64)
        if distances.ndim != 1 or hit.shape != distances.shape or indices.shape != distances.shape:
            raise ValueError("LiDAR packet vectors must be aligned")
        if directions.shape != (distances.size, 3):
            raise ValueError("LiDAR directions must have shape (num_sectors, 3)")
        if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(directions)):
            raise ValueError("LiDAR packet must be finite")
        if np.any(hit & (indices < 0)) or np.any(~hit & (indices >= 0)):
            raise ValueError("LiDAR hit and obstacle index fields disagree")
        object.__setattr__(self, "distances", distances.copy())
        object.__setattr__(self, "hit", hit.copy())
        object.__setattr__(self, "directions", directions.copy())
        object.__setattr__(self, "obstacle_indices", indices.copy())
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))
        if not np.isfinite(self.timestamp):
            raise ValueError("LiDAR timestamp must be finite")

    @property
    def points(self) -> np.ndarray:
        return self.origin + self.distances[:, None] * self.directions


class Lidar3DModel:
    def __init__(self, config: Lidar3DConfig | None = None) -> None:
        self.config = Lidar3DConfig() if config is None else config
        horizontal = np.deg2rad(self.config.horizontal_fov_degrees)
        vertical = np.deg2rad(self.config.vertical_fov_degrees)
        if np.isclose(horizontal, 2.0 * np.pi):
            azimuth = np.arange(self.config.horizontal_sectors) * (
                2.0 * np.pi / self.config.horizontal_sectors
            )
        else:
            azimuth = np.linspace(
                -0.5 * horizontal,
                0.5 * horizontal,
                self.config.horizontal_sectors,
            )
        elevation = np.linspace(
            -0.5 * vertical,
            0.5 * vertical,
            self.config.vertical_sectors,
        )
        directions = []
        for angle_z in elevation:
            cosine = np.cos(angle_z)
            for angle_xy in azimuth:
                directions.append(
                    [
                        cosine * np.cos(angle_xy),
                        cosine * np.sin(angle_xy),
                        np.sin(angle_z),
                    ]
                )
        self.directions = np.asarray(directions, dtype=np.float64)

    def measure(
        self,
        origin: np.ndarray,
        obstacles: list[ObstacleGeometry] | tuple[ObstacleGeometry, ...],
        *,
        timestamp: float,
    ) -> Lidar3DPacket:
        point = _vec3(origin, "origin")
        distances = np.full(self.config.num_sectors, self.config.max_range, dtype=np.float64)
        indices = np.full(self.config.num_sectors, -1, dtype=np.int64)
        for obstacle_index, obstacle in enumerate(obstacles):
            if isinstance(obstacle, SphereGeometry):
                relative = point - obstacle.center
                linear = self.directions @ relative
                constant = float(relative @ relative - obstacle.radius**2)
                discriminant = linear * linear - constant
                valid = discriminant >= 0.0
                roots = np.sqrt(np.maximum(discriminant, 0.0))
                near = -linear - roots
                far = -linear + roots
                candidate = np.where(near >= 0.0, near, far)
                valid &= candidate >= 0.0
                valid &= candidate <= self.config.max_range
                update = valid & (candidate < distances)
                distances[update] = candidate[update]
                indices[update] = obstacle_index
                continue
            for ray_index, direction in enumerate(self.directions):
                distance = obstacle.ray_distance(point, direction)
                if 0.0 <= distance <= self.config.max_range and distance < distances[ray_index]:
                    distances[ray_index] = distance
                    indices[ray_index] = obstacle_index
        hit = indices >= 0
        return Lidar3DPacket(
            distances=distances,
            hit=hit,
            directions=self.directions,
            obstacle_indices=indices,
            timestamp=float(timestamp),
            origin=point,
        )


def raw_lidar_point_obstacles(
    packet: Lidar3DPacket,
    *,
    point_radius: float = 0.0,
) -> list[SphericalObstacle]:
    if point_radius < 0.0:
        raise ValueError("point_radius must be nonnegative")
    points = packet.points
    return [
        SphericalObstacle(
            center=points[index],
            radius=max(float(point_radius), 1e-9),
            identifier=f"lidar_point_{index}",
        )
        for index in np.flatnonzero(packet.hit)
    ]
