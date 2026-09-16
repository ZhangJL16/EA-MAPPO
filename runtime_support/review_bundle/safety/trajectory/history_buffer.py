from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


def _optional_vector(value: np.ndarray | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector")
    return vector.copy()


@dataclass(frozen=True)
class HistoryFrame:
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    realized_acceleration: np.ndarray
    executed_control: np.ndarray | None = None
    lidar_ranges: np.ndarray | None = None
    lidar_valid: np.ndarray | None = None
    battery_context: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        object.__setattr__(self, "position", _vec3(self.position, "position"))
        object.__setattr__(self, "velocity", _vec3(self.velocity, "velocity"))
        object.__setattr__(
            self,
            "realized_acceleration",
            _vec3(self.realized_acceleration, "realized_acceleration"),
        )
        control = None if self.executed_control is None else _vec3(
            self.executed_control, "executed_control"
        )
        ranges = _optional_vector(self.lidar_ranges, "lidar_ranges")
        valid = None
        if self.lidar_valid is not None:
            valid = np.asarray(self.lidar_valid, dtype=bool)
            if valid.ndim != 1:
                raise ValueError("lidar_valid must be a vector")
            valid = valid.copy()
        if ranges is not None and (valid is None or valid.shape != ranges.shape):
            raise ValueError("lidar_ranges and lidar_valid must be aligned")
        if ranges is None and valid is not None:
            raise ValueError("lidar_valid requires lidar_ranges")
        object.__setattr__(self, "executed_control", control)
        object.__setattr__(self, "lidar_ranges", ranges)
        object.__setattr__(self, "lidar_valid", valid)
        object.__setattr__(
            self,
            "battery_context",
            _optional_vector(self.battery_context, "battery_context"),
        )


class HistoryBuffer:
    SUPPORTED_LENGTHS = (2, 4, 8, 16)

    def __init__(self, length: int) -> None:
        if length not in self.SUPPORTED_LENGTHS:
            raise ValueError(f"length must be one of {self.SUPPORTED_LENGTHS}")
        self.length = int(length)
        self._frames: deque[HistoryFrame] = deque(maxlen=self.length)

    def append(self, frame: HistoryFrame) -> None:
        if self._frames and frame.timestamp <= self._frames[-1].timestamp:
            raise ValueError("history timestamps must be strictly increasing")
        self._frames.append(frame)

    def clear(self) -> None:
        self._frames.clear()

    def snapshot(self) -> tuple[HistoryFrame, ...]:
        return tuple(self._frames)

    @property
    def full(self) -> bool:
        return len(self._frames) == self.length

    def __len__(self) -> int:
        return len(self._frames)
