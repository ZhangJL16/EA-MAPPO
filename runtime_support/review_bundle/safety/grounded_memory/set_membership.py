from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


def _vec3(value: np.ndarray | float, name: str, *, nonnegative: bool = False) -> np.ndarray:
    vector = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(vector < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return vector


@dataclass(frozen=True)
class SetMembershipConfig:
    sensor_error_bound: np.ndarray | float = 0.10
    velocity_bound: np.ndarray | float = 30.0
    acceleration_bound: np.ndarray | float = 8.0
    jerk_bound: np.ndarray | float = 12.0
    maximum_history_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.maximum_history_seconds) or self.maximum_history_seconds <= 0.0:
            raise ValueError("maximum_history_seconds must be finite and positive")
        for name in (
            "sensor_error_bound",
            "velocity_bound",
            "acceleration_bound",
            "jerk_bound",
        ):
            object.__setattr__(
                self,
                name,
                _vec3(getattr(self, name), name, nonnegative=True),
            )


@dataclass(frozen=True)
class HistoricalPositionConstraint:
    measurement: np.ndarray
    age_seconds: float
    track_id: str
    epoch: int
    sensor_error_bound: np.ndarray | float
    association_verified: bool
    provenance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "measurement", _vec3(self.measurement, "measurement"))
        object.__setattr__(
            self,
            "sensor_error_bound",
            _vec3(self.sensor_error_bound, "sensor_error_bound", nonnegative=True),
        )
        if not np.isfinite(self.age_seconds) or self.age_seconds < 0.0:
            raise ValueError("age_seconds must be finite and nonnegative")
        if not self.track_id or self.epoch < 0 or not self.provenance:
            raise ValueError("history constraints require track, epoch, and provenance")


@dataclass(frozen=True)
class AxisAlignedStateBox:
    lower: np.ndarray
    upper: np.ndarray
    accepted_epochs: tuple[int, ...]
    solve_seconds: float

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (9,) or upper.shape != (9,):
            raise ValueError("state boxes must have nine-dimensional bounds")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("state box bounds must be finite")
        if np.any(lower > upper + 1e-10):
            raise ValueError("state box is empty")
        if not np.isfinite(self.solve_seconds) or self.solve_seconds < 0.0:
            raise ValueError("solve_seconds must be finite and nonnegative")
        object.__setattr__(self, "lower", lower.copy())
        object.__setattr__(self, "upper", upper.copy())

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def radius(self) -> np.ndarray:
        return 0.5 * (self.upper - self.lower)

    @property
    def widths(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def log_volume(self) -> float:
        return float(np.sum(np.log(np.maximum(self.widths, 1e-12))))

    def contains(self, state: np.ndarray, *, tolerance: float = 1e-8) -> bool:
        value = np.asarray(state, dtype=np.float64)
        if value.shape != (9,):
            raise ValueError("state must have shape (9,)")
        return bool(
            np.all(value >= self.lower - tolerance)
            and np.all(value <= self.upper + tolerance)
        )

    def directional_support_radius(self, direction: np.ndarray) -> float:
        normal = _vec3(direction, "direction")
        return float(np.abs(normal) @ self.radius[:3])


@dataclass(frozen=True)
class VerifiedSetUpdate:
    state_box: AxisAlignedStateBox
    proposed_epochs: tuple[int, ...]
    accepted_epochs: tuple[int, ...]
    rejected_epochs: tuple[int, ...]
    rejection_reasons: tuple[str, ...]


def _axis_constraints(
    constraints: list[HistoricalPositionConstraint],
    config: SetMembershipConfig,
    axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    bounds = []
    for item in constraints:
        tau = float(item.age_seconds)
        row = np.array([1.0, -tau, 0.5 * tau * tau], dtype=np.float64)
        residual = (
            item.sensor_error_bound[axis]
            + config.jerk_bound[axis] * tau**3 / 6.0
        )
        upper = item.measurement[axis] + residual
        lower = item.measurement[axis] - residual
        rows.extend((row, -row))
        bounds.extend((upper, -lower))
    return np.asarray(rows, dtype=np.float64), np.asarray(bounds, dtype=np.float64)


def state_box_from_constraints(
    constraints: list[HistoricalPositionConstraint],
    config: SetMembershipConfig,
) -> AxisAlignedStateBox | None:
    if not constraints:
        raise ValueError("at least one position constraint is required")
    started = perf_counter()
    lower = np.empty(9, dtype=np.float64)
    upper = np.empty(9, dtype=np.float64)
    for axis in range(3):
        rows, bounds = _axis_constraints(constraints, config, axis)
        variable_bounds = (
            (None, None),
            (-config.velocity_bound[axis], config.velocity_bound[axis]),
            (-config.acceleration_bound[axis], config.acceleration_bound[axis]),
        )
        for component in range(3):
            objective = np.zeros(3, dtype=np.float64)
            objective[component] = 1.0
            minimum = linprog(
                objective,
                A_ub=rows,
                b_ub=bounds,
                bounds=variable_bounds,
                method="highs",
            )
            maximum = linprog(
                -objective,
                A_ub=rows,
                b_ub=bounds,
                bounds=variable_bounds,
                method="highs",
            )
            if not minimum.success or not maximum.success:
                return None
            index = component * 3 + axis
            lower[index] = minimum.fun
            upper[index] = -maximum.fun
    return AxisAlignedStateBox(
        lower=lower,
        upper=upper,
        accepted_epochs=tuple(sorted(item.epoch for item in constraints)),
        solve_seconds=perf_counter() - started,
    )


class VerifiedHistorySetUpdater:
    def __init__(self, config: SetMembershipConfig) -> None:
        self.config = config

    def _structurally_valid(
        self,
        constraint: HistoricalPositionConstraint,
        *,
        track_id: str,
    ) -> str | None:
        if not constraint.association_verified:
            return "association_unverified"
        if constraint.track_id != track_id:
            return "track_id_mismatch"
        if constraint.age_seconds > self.config.maximum_history_seconds + 1e-12:
            return "history_too_old"
        if np.any(constraint.sensor_error_bound > self.config.sensor_error_bound + 1e-12):
            return "sensor_bound_exceeds_certificate"
        return None

    def update(
        self,
        base_constraint: HistoricalPositionConstraint,
        proposed_constraints: list[HistoricalPositionConstraint],
    ) -> VerifiedSetUpdate:
        base_reason = self._structurally_valid(base_constraint, track_id=base_constraint.track_id)
        if base_reason is not None:
            raise ValueError(f"invalid base constraint: {base_reason}")
        accepted = [base_constraint]
        rejected_epochs: list[int] = []
        rejection_reasons: list[str] = []
        proposed_epochs: list[int] = []
        for candidate in sorted(proposed_constraints, key=lambda item: item.age_seconds):
            proposed_epochs.append(candidate.epoch)
            reason = self._structurally_valid(candidate, track_id=base_constraint.track_id)
            if reason is None and candidate.epoch in {item.epoch for item in accepted}:
                reason = "duplicate_epoch"
            if reason is None:
                tentative = state_box_from_constraints(accepted + [candidate], self.config)
                if tentative is None:
                    reason = "set_membership_infeasible"
                else:
                    accepted.append(candidate)
                    continue
            rejected_epochs.append(candidate.epoch)
            rejection_reasons.append(reason)
        final_box = state_box_from_constraints(accepted, self.config)
        if final_box is None:
            raise RuntimeError("accepted verified constraints became infeasible")
        return VerifiedSetUpdate(
            state_box=final_box,
            proposed_epochs=tuple(proposed_epochs),
            accepted_epochs=tuple(sorted(item.epoch for item in accepted)),
            rejected_epochs=tuple(rejected_epochs),
            rejection_reasons=tuple(rejection_reasons),
        )

    def update_batch(
        self,
        base_constraint: HistoricalPositionConstraint,
        proposed_constraints: list[HistoricalPositionConstraint],
    ) -> VerifiedSetUpdate:
        """Verify a proposal set once, with sequential fallback on inconsistency."""

        base_reason = self._structurally_valid(
            base_constraint,
            track_id=base_constraint.track_id,
        )
        if base_reason is not None:
            raise ValueError(f"invalid base constraint: {base_reason}")
        accepted = [base_constraint]
        proposed_epochs: list[int] = []
        rejected_epochs: list[int] = []
        rejection_reasons: list[str] = []
        seen_epochs = {base_constraint.epoch}
        for candidate in sorted(proposed_constraints, key=lambda item: item.age_seconds):
            proposed_epochs.append(candidate.epoch)
            reason = self._structurally_valid(candidate, track_id=base_constraint.track_id)
            if reason is None and candidate.epoch in seen_epochs:
                reason = "duplicate_epoch"
            if reason is None:
                accepted.append(candidate)
                seen_epochs.add(candidate.epoch)
            else:
                rejected_epochs.append(candidate.epoch)
                rejection_reasons.append(reason)
        final_box = state_box_from_constraints(accepted, self.config)
        if final_box is None:
            return self.update(base_constraint, proposed_constraints)
        return VerifiedSetUpdate(
            state_box=final_box,
            proposed_epochs=tuple(proposed_epochs),
            accepted_epochs=tuple(sorted(item.epoch for item in accepted)),
            rejected_epochs=tuple(rejected_epochs),
            rejection_reasons=tuple(rejection_reasons),
        )
