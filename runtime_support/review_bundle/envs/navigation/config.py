from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def _positive_vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a finite positive (3,) array")
    return array


@dataclass(frozen=True)
class NavigationConfig:
    world_size: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 2.0]))
    dt: float = 0.2
    v_max: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.12, 0.08]))
    a_max: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.08, 0.05]))
    body_radius: float = 0.05
    lidar_range: float = 6.0
    num_lasers: int = 32
    lidar_range_noise: float = 0.0
    lidar_pose_noise: float = 0.0
    lidar_heading_noise: float = 0.0
    lidar_invalid_probability: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_size", _positive_vec3(self.world_size, "world_size"))
        object.__setattr__(self, "v_max", _positive_vec3(self.v_max, "v_max"))
        object.__setattr__(self, "a_max", _positive_vec3(self.a_max, "a_max"))
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not np.isfinite(self.body_radius) or self.body_radius <= 0.0:
            raise ValueError("body_radius must be finite and positive")
        if not np.isfinite(self.lidar_range) or self.lidar_range <= 0.0:
            raise ValueError("lidar_range must be finite and positive")
        if self.num_lasers != 32:
            raise ValueError("the phase-1 observation contract requires 32 lidar rays")
        if not 0.0 <= self.lidar_invalid_probability <= 1.0:
            raise ValueError("lidar_invalid_probability must lie in [0, 1]")


def apply_configuration_overrides(
    config: NavigationConfig,
    overrides: dict[str, object],
) -> NavigationConfig:
    allowed = {field_name for field_name in config.__dataclass_fields__}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported navigation configuration overrides: {sorted(unknown)}")
    converted = {
        key: np.asarray(value, dtype=np.float64) if key in {"world_size", "v_max", "a_max"} else value
        for key, value in overrides.items()
    }
    return replace(config, **converted)
