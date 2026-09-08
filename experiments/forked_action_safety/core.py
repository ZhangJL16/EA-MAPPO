from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionCandidate:
    name: str
    action: np.ndarray

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        if not self.name or action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("candidate requires a name and finite three-vector")
        object.__setattr__(self, "action", np.clip(action, -1.0, 1.0))


def sanitize_snapshot_position(
    position: np.ndarray,
    *,
    world_extent: np.ndarray,
    safe_radius: float,
) -> np.ndarray:
    """Project only float32 reconstruction noise outside the legal interior."""

    value = np.asarray(position, dtype=np.float32).copy()
    extent = np.asarray(world_extent, dtype=np.float32)
    if value.shape != (3,) or extent.shape != (3,):
        raise ValueError("snapshot position and world extent must be three-vectors")
    if not np.all(np.isfinite(value)) or not np.all(np.isfinite(extent)):
        raise ValueError("snapshot position and world extent must be finite")
    if safe_radius < 0.0 or np.any(extent <= 2.0 * safe_radius):
        raise ValueError("position bounds must define a nonempty interior")
    lower = np.full(3, float(safe_radius), dtype=np.float32)
    upper = extent - float(safe_radius)
    # Float32 trajectories at kilometer-scale coordinates can accumulate
    # sub-millimeter endpoint error.  Eight ulps at the world scale covers that
    # numerical mechanism while rejecting centimeter-scale physical violations.
    tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(extent)))
    if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
        raise ValueError("snapshot position has a physical boundary violation")
    return np.clip(value, lower, upper).astype(np.float32)


def sanitize_snapshot_velocity(
    velocity: np.ndarray,
    *,
    horizontal_limit: float,
    vertical_limit: float,
) -> np.ndarray:
    """Project only float32-scale integration excess at a physical limit."""

    value = np.asarray(velocity, dtype=np.float32).copy()
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("snapshot velocity must be a finite three-vector")
    if min(horizontal_limit, vertical_limit) <= 0.0:
        raise ValueError("velocity limits must be positive")
    tolerance = 64.0 * np.finfo(np.float32).eps * max(
        1.0, float(horizontal_limit), float(vertical_limit)
    )
    horizontal_speed = float(np.linalg.norm(value[:2].astype(np.float64)))
    if horizontal_speed > float(horizontal_limit) + tolerance:
        raise ValueError("snapshot horizontal velocity has a physical limit violation")
    if abs(float(value[2])) > float(vertical_limit) + tolerance:
        raise ValueError("snapshot vertical velocity has a physical limit violation")
    if horizontal_speed > float(horizontal_limit):
        # Project in float64, then make the stored float32 vector feasible.
        # A direct in-place float32 scale can round a point back outside the
        # sphere by a fraction of an ulp.  A second call would then change it
        # again, breaking exact fork hashes.  Moving the largest component one
        # float32 representable value toward zero is the smallest deterministic
        # correction and makes this projection bitwise idempotent.
        value[:2] = (
            value[:2].astype(np.float64)
            * (float(horizontal_limit) / horizontal_speed)
        ).astype(np.float32)
        for _ in range(8):
            if float(np.linalg.norm(value[:2].astype(np.float64))) <= float(
                horizontal_limit
            ):
                break
            index = int(np.argmax(np.abs(value[:2])))
            value[index] = np.nextafter(
                value[index], np.float32(0.0), dtype=np.float32
            )
        else:
            raise RuntimeError("float32 velocity projection did not converge")
    value[2] = np.clip(value[2], -float(vertical_limit), float(vertical_limit))
    return value


def analytic_clearance(
    positions: np.ndarray,
    obstacles: list[dict[str, object]],
    *,
    safe_radius: float,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if safe_radius < 0.0:
        raise ValueError("safe radius must be nonnegative")
    if not obstacles:
        return np.full(positions.shape[0], np.inf, dtype=np.float64)
    clearance = np.full(positions.shape[0], np.inf, dtype=np.float64)
    for obstacle in obstacles:
        center = np.asarray(obstacle["position"], dtype=np.float64)
        radius = float(obstacle["radius"])
        if center.shape != (2,) or radius <= 0.0:
            raise ValueError("invalid obstacle specification")
        value = np.linalg.norm(positions[:, :2] - center[None, :], axis=1)
        clearance = np.minimum(clearance, value - radius - safe_radius)
    return clearance


def nearest_obstacle_direction(
    position: np.ndarray,
    obstacles: list[dict[str, object]],
) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
        raise ValueError("position must be a three-vector")
    if not obstacles:
        return np.zeros(3, dtype=np.float32)
    centers = np.asarray([item["position"] for item in obstacles], dtype=np.float64)
    index = int(np.argmin(np.linalg.norm(centers - position[None, :2], axis=1)))
    delta = centers[index] - position[:2]
    norm = float(np.linalg.norm(delta))
    if norm <= np.finfo(np.float32).eps:
        return np.zeros(3, dtype=np.float32)
    return np.asarray([delta[0] / norm, delta[1] / norm, 0.0], dtype=np.float32)


def candidate_actions(
    nominal_action: np.ndarray,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: list[dict[str, object]],
    perturbation: float,
    hold_steps: int,
    policy_dt: float,
    horizontal_a_max: float,
    vertical_a_max: float,
) -> tuple[ActionCandidate, ...]:
    nominal = np.asarray(nominal_action, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    if nominal.shape != (3,) or velocity.shape != (3,):
        raise ValueError("nominal action and velocity must be three-vectors")
    if not 0.0 < perturbation <= 1.0 or hold_steps <= 0 or policy_dt <= 0.0:
        raise ValueError("invalid action intervention scale")
    if min(horizontal_a_max, vertical_a_max) <= 0.0:
        raise ValueError("acceleration limits must be positive")
    duration = hold_steps * policy_dt
    brake = np.asarray(
        [
            -velocity[0] / (duration * horizontal_a_max),
            -velocity[1] / (duration * horizontal_a_max),
            -velocity[2] / (duration * vertical_a_max),
        ],
        dtype=np.float32,
    )
    horizontal_norm = float(np.linalg.norm(brake[:2]))
    if horizontal_norm > 1.0:
        brake[:2] /= horizontal_norm
    toward = nearest_obstacle_direction(position, obstacles)
    candidates = [
        ActionCandidate("nominal", nominal),
        ActionCandidate("brake", brake),
    ]
    for axis, label in enumerate(("x", "y", "z")):
        delta = np.zeros(3, dtype=np.float32)
        delta[axis] = perturbation
        candidates.append(ActionCandidate(f"plus_{label}", nominal + delta))
        candidates.append(ActionCandidate(f"minus_{label}", nominal - delta))
    candidates.extend(
        [
            ActionCandidate("toward_obstacle", toward),
            ActionCandidate("away_from_obstacle", -toward),
        ]
    )
    return tuple(candidates)


def _interior_indices(indices: np.ndarray) -> np.ndarray:
    values = np.asarray(indices, dtype=np.int64)
    if values.size <= 2:
        return values[:1]
    return values[1:-1]


def select_anchor_indices(
    legs: np.ndarray,
    clearances: np.ndarray,
    *,
    anchors_per_scene: int,
) -> np.ndarray:
    legs = np.asarray(legs, dtype=np.int8)
    clearances = np.asarray(clearances, dtype=np.float64)
    if legs.ndim != 1 or clearances.shape != legs.shape:
        raise ValueError("legs and clearances must be aligned vectors")
    if anchors_per_scene <= 0 or legs.size == 0:
        raise ValueError("anchor request must be positive and nonempty")
    selected: list[int] = []
    present_legs = [leg for leg in (0, 1) if np.any(legs == leg)]
    allocation = {leg: anchors_per_scene // len(present_legs) for leg in present_legs}
    for leg in present_legs[: anchors_per_scene % len(present_legs)]:
        allocation[leg] += 1
    for leg in present_legs:
        candidates = _interior_indices(np.flatnonzero(legs == leg))
        if candidates.size == 0:
            continue
        count = min(allocation[leg], candidates.size)
        nearest = int(candidates[np.argmin(clearances[candidates])])
        chosen = [nearest]
        quantiles = np.linspace(0.2, 0.8, max(count, 2))
        for quantile in quantiles:
            candidate = int(candidates[int(round(quantile * (candidates.size - 1)))])
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == count:
                break
        if len(chosen) < count:
            for candidate in candidates:
                if int(candidate) not in chosen:
                    chosen.append(int(candidate))
                if len(chosen) == count:
                    break
        selected.extend(chosen)
    if len(selected) < anchors_per_scene:
        candidates = _interior_indices(np.arange(legs.size))
        for candidate in candidates:
            if int(candidate) not in selected:
                selected.append(int(candidate))
            if len(selected) == anchors_per_scene:
                break
    return np.asarray(sorted(selected[:anchors_per_scene]), dtype=np.int64)


def summarize_paired_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("paired Gate requires branch rows")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["anchor_id"]), []).append(row)
    diverse = []
    varied = []
    safe_unsafe = []
    relative_energy_ranges = []
    observation_digests_match = []
    obstacle_digests_match = []
    for anchor_rows in grouped.values():
        actions = np.asarray([row["first_executed_action"] for row in anchor_rows], dtype=np.float64)
        representatives: list[np.ndarray] = []
        for action in actions:
            if all(np.linalg.norm(action - prior) >= 0.05 for prior in representatives):
                representatives.append(action)
        diverse.append(len(representatives) >= 5)
        safe = np.asarray([bool(row["safe_recharge"]) for row in anchor_rows])
        safe_unsafe.append(bool(np.any(safe) and np.any(~safe)))
        energies = np.asarray(
            [float(row["total_realized_energy"]) for row in anchor_rows if bool(row["safe_recharge"])],
            dtype=np.float64,
        )
        energy_range = 0.0
        if energies.size >= 2:
            energy_range = float((energies.max() - energies.min()) / max(np.median(energies), 1e-12))
        relative_energy_ranges.append(energy_range)
        varied.append(bool(safe_unsafe[-1] or energy_range >= 0.05))
        observation_digests_match.append(
            len({str(row["anchor_observation_sha256"]) for row in anchor_rows}) == 1
        )
        obstacle_digests_match.append(
            len({str(row["obstacle_layout_sha256"]) for row in anchor_rows}) == 1
        )
    unsafe = np.asarray([not bool(row["safe_recharge"]) for row in rows])
    unsafe_prevalence = float(np.mean(unsafe))
    checks = {
        "all_anchors_have_registered_branches": all(len(value) == 10 for value in grouped.values()),
        "executed_action_diversity_at_80_percent": float(np.mean(diverse)) >= 0.80,
        "unsafe_prevalence_between_2_and_40_percent": 0.02 <= unsafe_prevalence <= 0.40,
        "paired_variation_at_20_percent": float(np.mean(varied)) >= 0.20,
        "anchor_observation_exactly_matched": all(observation_digests_match),
        "obstacle_layout_exactly_matched": all(obstacle_digests_match),
    }
    return {
        "checks": checks,
        "promotable": all(checks.values()),
        "num_anchors": len(grouped),
        "num_branches": len(rows),
        "unsafe_prevalence": unsafe_prevalence,
        "collision_prevalence": float(np.mean([bool(row["collision"]) for row in rows])),
        "boundary_prevalence": float(np.mean([bool(row["boundary_contact"]) for row in rows])),
        "fraction_action_diverse_anchors": float(np.mean(diverse)),
        "fraction_safe_unsafe_anchors": float(np.mean(safe_unsafe)),
        "fraction_paired_varied_anchors": float(np.mean(varied)),
        "median_successful_energy_relative_range": float(np.median(relative_energy_ranges)),
    }
