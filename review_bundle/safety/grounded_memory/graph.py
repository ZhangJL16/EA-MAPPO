from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from review_bundle.safety.collision.hocbf import (
    ProjectionResult,
    project_polyhedral_qp,
)


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return array.copy()


@dataclass(frozen=True)
class CertifiedSafetyMemory:
    constraint_rows: np.ndarray
    nominal_lower_bounds: np.ndarray
    uncertainty_support: np.ndarray
    provenance: str

    def __post_init__(self) -> None:
        rows = np.asarray(self.constraint_rows, dtype=np.float64)
        nominal = np.asarray(self.nominal_lower_bounds, dtype=np.float64)
        support = np.asarray(self.uncertainty_support, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError("constraint_rows must have shape (m, 3)")
        if nominal.shape != (rows.shape[0],) or support.shape != nominal.shape:
            raise ValueError("constraint bounds must align with constraint_rows")
        if not np.all(np.isfinite(rows)) or not np.all(np.isfinite(nominal)):
            raise ValueError("certified constraints must be finite")
        if not np.all(np.isfinite(support)) or np.any(support < 0.0):
            raise ValueError("uncertainty support must be finite and nonnegative")
        if not self.provenance:
            raise ValueError("certified memory requires provenance")
        object.__setattr__(self, "constraint_rows", rows.copy())
        object.__setattr__(self, "nominal_lower_bounds", nominal.copy())
        object.__setattr__(self, "uncertainty_support", support.copy())

    @property
    def robust_lower_bounds(self) -> np.ndarray:
        return self.nominal_lower_bounds + self.uncertainty_support


@dataclass(frozen=True)
class LearnedModuleMessages:
    object_message: np.ndarray
    route_message: np.ndarray
    energy_message: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_message",
            _vector(self.object_message, 3, "object_message"),
        )
        object.__setattr__(
            self,
            "route_message",
            _vector(self.route_message, 3, "route_message"),
        )
        object.__setattr__(
            self,
            "energy_message",
            _vector(self.energy_message, 3, "energy_message"),
        )


@dataclass(frozen=True)
class RoutedControlResult:
    nominal_action: np.ndarray
    projection: ProjectionResult
    certified_constraint_digest: tuple[bytes, bytes]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nominal_action",
            _vector(self.nominal_action, 3, "nominal_action"),
        )


class GroundedMemoryGraph:
    def __init__(
        self,
        *,
        route_gain: float = 0.35,
        energy_gain: float = 0.10,
    ) -> None:
        if not np.isfinite(route_gain) or route_gain < 0.0:
            raise ValueError("route_gain must be finite and nonnegative")
        if not np.isfinite(energy_gain) or energy_gain < 0.0:
            raise ValueError("energy_gain must be finite and nonnegative")
        self.route_gain = float(route_gain)
        self.energy_gain = float(energy_gain)

    def nominal_action(
        self,
        base_action: np.ndarray,
        messages: LearnedModuleMessages,
    ) -> np.ndarray:
        base = _vector(base_action, 3, "base_action")
        return (
            base
            + self.route_gain * np.tanh(messages.route_message)
            - self.energy_gain * np.tanh(messages.energy_message)
        )

    def execute(
        self,
        base_action: np.ndarray,
        certified: CertifiedSafetyMemory,
        messages: LearnedModuleMessages,
    ) -> RoutedControlResult:
        nominal = self.nominal_action(base_action, messages)
        projection = project_polyhedral_qp(
            center=nominal,
            hessian=np.eye(3),
            rows=certified.constraint_rows,
            lower_bounds=certified.robust_lower_bounds,
        )
        return RoutedControlResult(
            nominal_action=nominal,
            projection=projection,
            certified_constraint_digest=(
                certified.constraint_rows.tobytes(),
                certified.robust_lower_bounds.tobytes(),
            ),
        )
