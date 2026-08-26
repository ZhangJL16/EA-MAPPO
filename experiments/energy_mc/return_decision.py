from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class CloneRolloutVarianceAudit:
    num_rollouts: int
    mean: float
    variance: float
    standard_deviation: float
    minimum: float
    maximum: float
    q50: float
    q90: float
    q95: float
    unique_values_at_tolerance: int
    nondegenerate_conditional_distribution: bool
    distributional_headline_supported: bool
    interpretation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReturnDecisionOutcome:
    method: str
    parameter: float
    stranding_rate: float
    tasks_per_simulated_hour: float
    tasks_per_battery_cycle: float
    return_success_rate: float
    mean_unused_energy_fraction_at_charger: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.parameter,
                self.stranding_rate,
                self.tasks_per_simulated_hour,
                self.tasks_per_battery_cycle,
                self.return_success_rate,
                self.mean_unused_energy_fraction_at_charger,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("return-decision outcome values must be finite")
        if not 0.0 <= self.stranding_rate <= 1.0:
            raise ValueError("stranding_rate must lie in [0, 1]")
        if not 0.0 <= self.return_success_rate <= 1.0:
            raise ValueError("return_success_rate must lie in [0, 1]")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OracleHeadroomGate:
    status: str
    evaluable: bool
    passed: bool | None
    minimum_cycles_per_point: int
    stranding_ceiling: float
    minimum_throughput_gain_fraction: float
    oracle_best_tasks_per_hour: float | None
    heuristic_best_tasks_per_hour: float | None
    throughput_gain_fraction: float | None
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def clone_rollout_variance_audit(
    rollout: Callable[[int], float],
    *,
    num_rollouts: int,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> CloneRolloutVarianceAudit:
    if num_rollouts < 2:
        raise ValueError("clone-rollout audit requires at least two rollouts")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("variance-audit tolerances must be nonnegative")
    samples = np.asarray([float(rollout(index)) for index in range(num_rollouts)])
    if samples.shape != (num_rollouts,) or not np.all(np.isfinite(samples)):
        raise ValueError("clone rollouts must return finite scalar resource costs")
    mean = float(np.mean(samples))
    variance = float(np.var(samples, ddof=1))
    scale = max(abs(mean) * relative_tolerance, absolute_tolerance)
    rounded = np.round(samples / max(scale, np.finfo(np.float64).eps)).astype(np.int64)
    unique = int(np.unique(rounded).size)
    nondegenerate = bool(variance > scale**2 and unique > 1)
    interpretation = (
        "nondegenerate conditional return distribution under fixed deployment information"
        if nondegenerate
        else "conditional return is effectively deterministic; use point ETG plus epistemic reliability"
    )
    return CloneRolloutVarianceAudit(
        num_rollouts=int(num_rollouts),
        mean=mean,
        variance=variance,
        standard_deviation=float(np.sqrt(max(variance, 0.0))),
        minimum=float(np.min(samples)),
        maximum=float(np.max(samples)),
        q50=float(np.quantile(samples, 0.50)),
        q90=float(np.quantile(samples, 0.90)),
        q95=float(np.quantile(samples, 0.95)),
        unique_values_at_tolerance=unique,
        nondegenerate_conditional_distribution=nondegenerate,
        distributional_headline_supported=nondegenerate,
        interpretation=interpretation,
    )


def pareto_frontier(
    outcomes: Iterable[ReturnDecisionOutcome],
) -> list[ReturnDecisionOutcome]:
    values = list(outcomes)
    if not values:
        return []
    frontier: list[ReturnDecisionOutcome] = []
    best_throughput = -float("inf")
    for outcome in sorted(
        values,
        key=lambda item: (
            item.stranding_rate,
            -item.tasks_per_simulated_hour,
            item.method,
            item.parameter,
        ),
    ):
        if outcome.tasks_per_simulated_hour > best_throughput:
            frontier.append(outcome)
            best_throughput = outcome.tasks_per_simulated_hour
    return frontier


def wilson_interval(
    count: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0 or count < 0 or count > total:
        raise ValueError("Wilson interval requires 0 <= count <= total and total > 0")
    if not np.isfinite(z) or z <= 0.0:
        raise ValueError("Wilson z value must be finite and positive")
    proportion = count / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    radius = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def oracle_headroom_gate(
    outcomes: Iterable[ReturnDecisionOutcome],
    *,
    cycles_per_point: dict[tuple[str, float], int],
    stranding_upper_bounds: dict[tuple[str, float], float] | None = None,
    minimum_cycles_per_point: int = 100,
    stranding_ceiling: float = 0.05,
    minimum_throughput_gain_fraction: float = 0.05,
) -> OracleHeadroomGate:
    values = list(outcomes)
    if minimum_cycles_per_point <= 0:
        raise ValueError("minimum_cycles_per_point must be positive")
    if not 0.0 <= stranding_ceiling <= 1.0:
        raise ValueError("stranding_ceiling must lie in [0, 1]")
    if minimum_throughput_gain_fraction < 0.0:
        raise ValueError("minimum throughput gain must be nonnegative")
    relevant = [
        value
        for value in values
        if value.method in {"oracle", "soc", "distance"}
    ]
    gate_stranding = {
        (value.method, value.parameter): (
            value.stranding_rate
            if stranding_upper_bounds is None
            else float(
                stranding_upper_bounds.get(
                    (value.method, value.parameter),
                    value.stranding_rate,
                )
            )
        )
        for value in relevant
    }
    if any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in gate_stranding.values()
    ):
        raise ValueError("stranding upper bounds must lie in [0, 1]")
    underpowered = [
        (value.method, value.parameter)
        for value in relevant
        if cycles_per_point.get((value.method, value.parameter), 0)
        < minimum_cycles_per_point
    ]
    if underpowered:
        return OracleHeadroomGate(
            "PENDING_INSUFFICIENT_CYCLES",
            False,
            None,
            minimum_cycles_per_point,
            stranding_ceiling,
            minimum_throughput_gain_fraction,
            None,
            None,
            None,
            f"underpowered points: {underpowered}",
        )
    oracle = [
        value
        for value in relevant
        if value.method == "oracle"
        and gate_stranding[(value.method, value.parameter)] <= stranding_ceiling
    ]
    heuristics = [
        value
        for value in relevant
        if value.method in {"soc", "distance"}
        and gate_stranding[(value.method, value.parameter)] <= stranding_ceiling
    ]
    if not oracle or not heuristics:
        return OracleHeadroomGate(
            "FAIL_NO_COMPARABLE_SAFE_FRONTIER",
            True,
            False,
            minimum_cycles_per_point,
            stranding_ceiling,
            minimum_throughput_gain_fraction,
            None if not oracle else max(item.tasks_per_simulated_hour for item in oracle),
            None
            if not heuristics
            else max(item.tasks_per_simulated_hour for item in heuristics),
            None,
            "Oracle and heuristic methods must both have a point whose preregistered stranding statistic is under the common ceiling",
        )
    oracle_best = max(item.tasks_per_simulated_hour for item in oracle)
    heuristic_best = max(item.tasks_per_simulated_hour for item in heuristics)
    gain = (oracle_best - heuristic_best) / max(
        heuristic_best,
        np.finfo(np.float64).eps,
    )
    passed = bool(gain >= minimum_throughput_gain_fraction)
    return OracleHeadroomGate(
        "PASS" if passed else "FAIL_INSUFFICIENT_ORACLE_HEADROOM",
        True,
        passed,
        minimum_cycles_per_point,
        stranding_ceiling,
        minimum_throughput_gain_fraction,
        float(oracle_best),
        float(heuristic_best),
        float(gain),
        "Oracle must improve throughput at the same preregistered stranding ceiling",
    )


def boundary_weighted_underestimation(
    *,
    remaining_energy: np.ndarray,
    true_required_energy: np.ndarray,
    predicted_required_energy: np.ndarray,
    bandwidth: float,
) -> float:
    remaining = np.asarray(remaining_energy, dtype=np.float64)
    truth = np.asarray(true_required_energy, dtype=np.float64)
    predicted = np.asarray(predicted_required_energy, dtype=np.float64)
    if not (remaining.shape == truth.shape == predicted.shape) or remaining.ndim != 1:
        raise ValueError("boundary-error arrays must be aligned vectors")
    if remaining.size == 0 or not np.all(
        np.isfinite(np.concatenate([remaining, truth, predicted]))
    ):
        raise ValueError("boundary-error arrays must be finite and nonempty")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    weights = np.exp(-np.abs(remaining - truth) / float(bandwidth))
    underestimation = np.maximum(truth - predicted, 0.0)
    return float(np.sum(weights * underestimation) / np.sum(weights))


__all__ = [
    "CloneRolloutVarianceAudit",
    "ReturnDecisionOutcome",
    "OracleHeadroomGate",
    "boundary_weighted_underestimation",
    "clone_rollout_variance_audit",
    "oracle_headroom_gate",
    "pareto_frontier",
    "wilson_interval",
]
