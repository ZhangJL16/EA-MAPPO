from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, log
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.stats import beta, binom


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
    throughput_gain_lower_confidence_bound: float | None = None
    throughput_gain_upper_confidence_bound: float | None = None
    throughput_confidence_level: float | None = None
    throughput_inference: str | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PairedFrontierThroughputInterval:
    confidence_level: float
    bootstrap_replicates: int
    bootstrap_seed: int
    num_paired_schedules: int
    oracle_points: tuple[str, ...]
    heuristic_points: tuple[str, ...]
    eligible_oracle_points: tuple[str, ...]
    eligible_heuristic_points: tuple[str, ...]
    selected_oracle_point: str
    selected_heuristic_point: str
    oracle_best_tasks_per_hour: float
    heuristic_best_tasks_per_hour: float
    throughput_gain_fraction: float
    throughput_gain_lower_confidence_bound: float
    throughput_gain_upper_confidence_bound: float | None
    max_t_critical_value: float
    upper_max_t_critical_value: float
    oracle_rate_lower_bounds: tuple[tuple[str, float], ...]
    oracle_rate_upper_bounds: tuple[tuple[str, float], ...]
    heuristic_rate_lower_bounds: tuple[tuple[str, float], ...]
    heuristic_rate_upper_bounds: tuple[tuple[str, float], ...]
    descriptive_percentile_gain_lower_bound: float
    descriptive_percentile_gain_upper_bound: float
    inference: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OracleShadowCouplingAudit:
    status: str
    passed: bool
    absolute_tolerance: float
    num_method_schedule_pairs: int
    num_compared_decision_events: int
    num_first_disagreements: int
    maximum_position_error: float
    maximum_velocity_error: float
    maximum_task_goal_error: float
    maximum_remaining_energy_error: float
    maximum_exact_requirement_error: float
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimultaneousStrandingPoint:
    method: str
    parameter: float
    stranding_count: int
    independent_cycles: int
    stranding_rate: float
    upper_confidence_bound: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimultaneousStrandingAudit:
    confidence_level: float
    familywise_error_probability: float
    num_candidate_points: int
    correction: str
    per_point_error_probability: float
    points: tuple[SimultaneousStrandingPoint, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["points"] = [point.as_dict() for point in self.points]
        return payload


@dataclass(frozen=True)
class StrandingCertificationPowerAudit:
    status: str
    passed: bool
    confidence_level: float
    familywise_error_probability: float
    num_candidate_points: int
    per_point_error_probability: float
    stranding_ceiling: float
    design_stranding_rate: float
    num_required_safe_families: int
    target_joint_certification_power: float
    target_certification_power: float
    planned_independent_cycles_per_point: int
    maximum_certifiable_stranding_count: int
    planned_certification_power: float
    planned_joint_certification_power_lower_bound: float
    minimum_cycles_for_zero_event_evaluability: int
    minimum_cycles_for_target_power: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequentialBoundaryCertificate:
    estimation_failure_probability: float
    stopped_boundary_occupation: float
    first_disagreement_probability_bound: float
    stranding_deviation_bound: float
    throughput_deviation_bound: float
    excess_loss_bound: float
    learned_stranding_upper: float | None
    learned_throughput_lower: float | None
    population_safety_ceiling_preserved: bool | None
    population_headroom_preserved: bool | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CertifiedEffectiveRequirement:
    branch_lower_requirements: tuple[float, ...]
    branch_upper_requirements: tuple[float, ...]
    effective_lower_requirement: float
    learned_effective_upper_requirement: float
    effective_error_bound: float
    exact_branch_requirements: tuple[float, ...] | None
    exact_effective_requirement: float | None
    exact_mgf_inside_certificate: bool | None
    learned_requirement_is_conservative: bool | None
    commitment_branch_index: int
    effective_requirement_semantics: str

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


def probability_semantics_gate(
    audit: CloneRolloutVarianceAudit,
    *,
    physically_specified_future_disturbance: bool,
) -> dict[str, object]:
    """Classify the Resource-to-Go random object before decision evaluation.

    A non-degenerate repeated clone is not by itself permission to publish
    aleatoric quantiles. Without a declared physical disturbance law it is an
    unexpected nondeterminism failure. With such a law, a broader calibration
    audit is still required before stochastic coverage claims are authorized.
    """

    has_disturbance = bool(physically_specified_future_disturbance)
    nondegenerate = bool(audit.nondegenerate_conditional_distribution)
    if not has_disturbance and nondegenerate:
        return {
            "status": "FAIL_UNEXPECTED_CONDITIONAL_NONDETERMINISM",
            "passed": False,
            "probability_object": "unresolved_conditional_resource_object",
            "aleatoric_q90_q95_claim_authorized": False,
            "reason": (
                "fixed-snapshot Oracle repetitions differ even though no "
                "physical future-disturbance law is declared"
            ),
        }
    if not has_disturbance:
        return {
            "status": "PASS_DETERMINISTIC_POINT_ETG_SEMANTICS",
            "passed": True,
            "probability_object": (
                "deterministic_point_resource_to_go_plus_epistemic_reliability"
            ),
            "aleatoric_q90_q95_claim_authorized": False,
            "reason": (
                "fixed-snapshot Oracle repetitions are effectively identical "
                "and no physical future-disturbance law is declared"
            ),
        }
    return {
        "status": "PENDING_STOCHASTIC_PROCESS_CALIBRATION",
        "passed": False,
        "probability_object": "conditional_resource_distribution_candidate",
        "aleatoric_q90_q95_claim_authorized": False,
        "reason": (
            "a physical future-disturbance law is declared, but clone "
            "variance alone does not establish quantile calibration"
        ),
    }


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


def simultaneous_stranding_upper_bounds(
    counts: Mapping[tuple[str, float], tuple[int, int]],
    *,
    confidence_level: float = 0.95,
) -> SimultaneousStrandingAudit:
    """Construct exact one-sided stranding bounds valid after grid selection.

    Each candidate point receives a Clopper--Pearson upper bound with error
    probability ``(1 - confidence_level) / K``, where ``K`` is the complete
    candidate family. Bonferroni therefore guarantees simultaneous coverage
    of every point with probability at least ``confidence_level`` without any
    independence assumption between candidate points.
    """

    level = float(confidence_level)
    if not 0.0 < level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if not counts:
        raise ValueError("at least one candidate point is required")
    family_alpha = 1.0 - level
    point_alpha = family_alpha / len(counts)
    points: list[SimultaneousStrandingPoint] = []
    for (method, parameter), (count, total) in sorted(counts.items()):
        failures = int(count)
        cycles = int(total)
        if cycles <= 0 or failures < 0 or failures > cycles:
            raise ValueError(
                "simultaneous stranding bounds require "
                "0 <= count <= total and total > 0"
            )
        upper = (
            1.0
            if failures == cycles
            else float(beta.ppf(1.0 - point_alpha, failures + 1, cycles - failures))
        )
        if not np.isfinite(upper):
            raise RuntimeError("Clopper-Pearson upper bound was not finite")
        points.append(
            SimultaneousStrandingPoint(
                method=str(method),
                parameter=float(parameter),
                stranding_count=failures,
                independent_cycles=cycles,
                stranding_rate=float(failures / cycles),
                upper_confidence_bound=float(min(1.0, max(0.0, upper))),
            )
        )
    return SimultaneousStrandingAudit(
        confidence_level=level,
        familywise_error_probability=family_alpha,
        num_candidate_points=len(points),
        correction="bonferroni_one_sided_clopper_pearson",
        per_point_error_probability=point_alpha,
        points=tuple(points),
    )


def stranding_certification_power_audit(
    *,
    num_candidate_points: int,
    planned_independent_cycles_per_point: int,
    stranding_ceiling: float = 0.05,
    design_stranding_rate: float = 0.01,
    confidence_level: float = 0.95,
    num_required_safe_families: int = 2,
    target_joint_certification_power: float = 0.90,
    maximum_search_cycles: int = 100_000,
) -> StrandingCertificationPowerAudit:
    """Audit power to certify a safe point with a Bonferroni exact bound.

    For count ``X`` and cycle count ``n``, the one-sided Clopper--Pearson upper
    bound is below the ceiling ``c`` exactly when
    ``P_{Binomial(n, c)}(Y <= X) <= alpha / K``. This identity avoids nested
    beta-quantile searches and yields both the largest certifiable count and the
    exact certification probability under the preregistered design rate.
    """

    points = int(num_candidate_points)
    cycles = int(planned_independent_cycles_per_point)
    level = float(confidence_level)
    ceiling = float(stranding_ceiling)
    design_rate = float(design_stranding_rate)
    required_families = int(num_required_safe_families)
    target_joint_power = float(target_joint_certification_power)
    search_limit = int(maximum_search_cycles)
    if points <= 0 or cycles <= 0 or search_limit <= 0 or required_families <= 0:
        raise ValueError("candidate points, cycles, and search limit must be positive")
    if not 0.5 < level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if not 0.0 < ceiling < 1.0:
        raise ValueError("stranding_ceiling must lie in (0, 1)")
    if not 0.0 <= design_rate < ceiling:
        raise ValueError("design_stranding_rate must lie in [0, ceiling)")
    if not 0.5 < target_joint_power < 1.0:
        raise ValueError("target_joint_certification_power must lie in (0.5, 1)")

    family_alpha = 1.0 - level
    point_alpha = family_alpha / points
    target_power = 1.0 - (1.0 - target_joint_power) / required_families

    def certifiable_count(total: int) -> int:
        count = int(binom.ppf(point_alpha, total, ceiling))
        while count >= 0 and float(binom.cdf(count, total, ceiling)) > point_alpha:
            count -= 1
        while (
            count + 1 <= total
            and float(binom.cdf(count + 1, total, ceiling)) <= point_alpha
        ):
            count += 1
        return count

    def certification_power(total: int) -> float:
        count = certifiable_count(total)
        return (
            0.0
            if count < 0
            else float(binom.cdf(count, total, design_rate))
        )

    zero_event_minimum = int(ceil(log(point_alpha) / log(1.0 - ceiling)))
    minimum_for_power = None
    for total in range(zero_event_minimum, search_limit + 1):
        if certification_power(total) >= target_power:
            minimum_for_power = total
            break
    if minimum_for_power is None:
        raise RuntimeError(
            "target certification power was not reached within maximum_search_cycles"
        )
    planned_power = certification_power(cycles)
    planned_joint_lower = max(
        0.0,
        1.0 - required_families * (1.0 - planned_power),
    )
    passed = bool(planned_joint_lower >= target_joint_power)
    return StrandingCertificationPowerAudit(
        status="PASS_DESIGN_POWER" if passed else "FAIL_UNDERPOWERED_DESIGN",
        passed=passed,
        confidence_level=level,
        familywise_error_probability=family_alpha,
        num_candidate_points=points,
        per_point_error_probability=point_alpha,
        stranding_ceiling=ceiling,
        design_stranding_rate=design_rate,
        num_required_safe_families=required_families,
        target_joint_certification_power=target_joint_power,
        target_certification_power=target_power,
        planned_independent_cycles_per_point=cycles,
        maximum_certifiable_stranding_count=certifiable_count(cycles),
        planned_certification_power=planned_power,
        planned_joint_certification_power_lower_bound=planned_joint_lower,
        minimum_cycles_for_zero_event_evaluability=zero_event_minimum,
        minimum_cycles_for_target_power=minimum_for_power,
    )


def paired_frontier_throughput_interval(
    cycle_records: Iterable[Mapping[str, object]],
    *,
    oracle_points: Iterable[tuple[str, float]],
    heuristic_points: Iterable[tuple[str, float]],
    eligible_oracle_points: Iterable[tuple[str, float]] | None = None,
    eligible_heuristic_points: Iterable[tuple[str, float]] | None = None,
    confidence_level: float = 0.95,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_830,
) -> PairedFrontierThroughputInterval:
    """Bootstrap simultaneous one-sided selected-frontier throughput bounds.

    Every method/parameter point must contain exactly the same schedule IDs.
    Every replicate resamples those IDs once. A studentized maximum-deviation
    statistics span the complete candidate family. The Oracle-lower and
    heuristic-upper bands support a lower gain bound; the Oracle-upper and
    heuristic-lower bands support an upper gain bound. Both remain valid after
    a data-dependent safety set and fastest-point selection. The old
    selected-max percentile interval is retained only as a descriptive
    diagnostic.
    """

    level = float(confidence_level)
    replicates = int(bootstrap_replicates)
    if not 0.5 < level < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    if replicates < 1000:
        raise ValueError("paired frontier bootstrap requires at least 1000 replicates")
    oracle_keys = tuple(dict.fromkeys(oracle_points))
    heuristic_keys = tuple(dict.fromkeys(heuristic_points))
    if not oracle_keys or not heuristic_keys:
        raise ValueError("paired frontier inference requires both point families")
    eligible_oracle_keys = (
        oracle_keys
        if eligible_oracle_points is None
        else tuple(dict.fromkeys(eligible_oracle_points))
    )
    eligible_heuristic_keys = (
        heuristic_keys
        if eligible_heuristic_points is None
        else tuple(dict.fromkeys(eligible_heuristic_points))
    )
    if not eligible_oracle_keys or not eligible_heuristic_keys:
        raise ValueError("paired frontier inference requires both eligible families")
    if not set(eligible_oracle_keys).issubset(oracle_keys) or not set(
        eligible_heuristic_keys
    ).issubset(heuristic_keys):
        raise ValueError("eligible points must be subsets of the complete families")

    selected_keys = oracle_keys + heuristic_keys
    grouped: dict[
        tuple[str, float],
        dict[str, tuple[float, float]],
    ] = {key: {} for key in selected_keys}
    for record in cycle_records:
        key = (str(record.get("method")), float(record.get("parameter")))
        if key not in grouped:
            continue
        schedule_id = str(record.get("paired_schedule_id"))
        if not schedule_id or schedule_id == "None":
            raise ValueError("paired cycle is missing paired_schedule_id")
        if schedule_id in grouped[key]:
            raise ValueError(f"duplicate cycle for point={key}, schedule={schedule_id}")
        tasks = float(record.get("tasks_completed"))
        seconds = float(record.get("simulated_seconds"))
        if not np.isfinite(tasks) or tasks < 0.0:
            raise ValueError("cycle task counts must be finite and nonnegative")
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError("cycle simulated seconds must be finite and positive")
        grouped[key][schedule_id] = (tasks, seconds)

    schedule_sets = [set(grouped[key]) for key in selected_keys]
    if not schedule_sets[0]:
        raise ValueError("paired frontier inference has no cycle records")
    if any(values != schedule_sets[0] for values in schedule_sets[1:]):
        raise ValueError(
            "all eligible Oracle and heuristic points must share exactly the same schedules"
        )
    schedule_ids = tuple(sorted(schedule_sets[0]))
    num_schedules = len(schedule_ids)
    arrays: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for key in selected_keys:
        arrays[key] = (
            np.asarray([grouped[key][schedule][0] for schedule in schedule_ids]),
            np.asarray([grouped[key][schedule][1] for schedule in schedule_ids]),
        )

    def pooled_rate(key: tuple[str, float]) -> float:
        tasks, seconds = arrays[key]
        return float(3600.0 * np.sum(tasks) / np.sum(seconds))

    oracle_rates = {key: pooled_rate(key) for key in oracle_keys}
    heuristic_rates = {key: pooled_rate(key) for key in heuristic_keys}
    best_oracle_key = max(eligible_oracle_keys, key=oracle_rates.get)
    best_heuristic_key = max(eligible_heuristic_keys, key=heuristic_rates.get)
    oracle_best = oracle_rates[best_oracle_key]
    heuristic_best = heuristic_rates[best_heuristic_key]
    if not np.isfinite(heuristic_best) or heuristic_best <= 0.0:
        raise ValueError("best heuristic throughput must be finite and positive")
    point_gain = float(oracle_best / heuristic_best - 1.0)

    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(
        0,
        num_schedules,
        size=(replicates, num_schedules),
    )

    def bootstrap_rate_matrix(keys: tuple[tuple[str, float], ...]) -> np.ndarray:
        rates: list[np.ndarray] = []
        for key in keys:
            tasks, seconds = arrays[key]
            task_sums = np.sum(tasks[indices], axis=1)
            second_sums = np.sum(seconds[indices], axis=1)
            rates.append(3600.0 * task_sums / second_sums)
        return np.stack(rates, axis=1)

    oracle_bootstrap = bootstrap_rate_matrix(oracle_keys)
    heuristic_bootstrap = bootstrap_rate_matrix(heuristic_keys)
    if np.any(~np.isfinite(heuristic_bootstrap)) or np.any(
        heuristic_bootstrap <= 0.0
    ):
        raise ValueError(
            "paired bootstrap produced a nonpositive best heuristic throughput"
        )
    oracle_index = {key: index for index, key in enumerate(oracle_keys)}
    heuristic_index = {key: index for index, key in enumerate(heuristic_keys)}
    eligible_oracle_bootstrap = oracle_bootstrap[
        :, [oracle_index[key] for key in eligible_oracle_keys]
    ]
    eligible_heuristic_bootstrap = heuristic_bootstrap[
        :, [heuristic_index[key] for key in eligible_heuristic_keys]
    ]
    gains = (
        np.max(eligible_oracle_bootstrap, axis=1)
        / np.max(eligible_heuristic_bootstrap, axis=1)
        - 1.0
    )
    if not np.all(np.isfinite(gains)):
        raise ValueError("paired bootstrap gain must be finite")
    alpha = 1.0 - level

    oracle_point_rates = np.asarray([oracle_rates[key] for key in oracle_keys])
    heuristic_point_rates = np.asarray(
        [heuristic_rates[key] for key in heuristic_keys]
    )
    bootstrap_ddof = 1 if num_schedules > 1 else 0
    oracle_standard_errors = np.std(
        oracle_bootstrap,
        axis=0,
        ddof=bootstrap_ddof,
    )
    heuristic_standard_errors = np.std(
        heuristic_bootstrap,
        axis=0,
        ddof=bootstrap_ddof,
    )
    oracle_scales = np.where(
        oracle_standard_errors > np.finfo(np.float64).eps,
        oracle_standard_errors,
        1.0,
    )
    heuristic_scales = np.where(
        heuristic_standard_errors > np.finfo(np.float64).eps,
        heuristic_standard_errors,
        1.0,
    )
    lower_direction_statistics = np.max(
        np.concatenate(
            [
                (oracle_point_rates[None, :] - oracle_bootstrap)
                / oracle_scales[None, :],
                (heuristic_bootstrap - heuristic_point_rates[None, :])
                / heuristic_scales[None, :],
            ],
            axis=1,
        ),
        axis=1,
    )
    max_t_critical = float(
        max(0.0, np.quantile(lower_direction_statistics, level))
    )
    oracle_lower = np.maximum(
        0.0,
        oracle_point_rates - max_t_critical * oracle_standard_errors,
    )
    heuristic_upper = (
        heuristic_point_rates + max_t_critical * heuristic_standard_errors
    )
    selected_oracle_lower = max(
        oracle_lower[oracle_index[key]] for key in eligible_oracle_keys
    )
    selected_heuristic_upper = max(
        heuristic_upper[heuristic_index[key]] for key in eligible_heuristic_keys
    )
    if not np.isfinite(selected_heuristic_upper) or selected_heuristic_upper <= 0.0:
        raise ValueError("simultaneous heuristic throughput upper bound must be positive")
    simultaneous_lower_gain = float(
        selected_oracle_lower / selected_heuristic_upper - 1.0
    )

    upper_direction_statistics = np.max(
        np.concatenate(
            [
                (oracle_bootstrap - oracle_point_rates[None, :])
                / oracle_scales[None, :],
                (heuristic_point_rates[None, :] - heuristic_bootstrap)
                / heuristic_scales[None, :],
            ],
            axis=1,
        ),
        axis=1,
    )
    upper_max_t_critical = float(
        max(0.0, np.quantile(upper_direction_statistics, level))
    )
    oracle_upper = (
        oracle_point_rates + upper_max_t_critical * oracle_standard_errors
    )
    heuristic_lower = np.maximum(
        0.0,
        heuristic_point_rates
        - upper_max_t_critical * heuristic_standard_errors,
    )
    selected_oracle_upper = max(
        oracle_upper[oracle_index[key]] for key in eligible_oracle_keys
    )
    selected_heuristic_lower = max(
        heuristic_lower[heuristic_index[key]] for key in eligible_heuristic_keys
    )
    simultaneous_upper_gain = (
        None
        if not np.isfinite(selected_heuristic_lower)
        or selected_heuristic_lower <= 0.0
        else float(selected_oracle_upper / selected_heuristic_lower - 1.0)
    )

    def label(key: tuple[str, float]) -> str:
        return f"{key[0]}|{key[1]:.12g}"

    return PairedFrontierThroughputInterval(
        confidence_level=level,
        bootstrap_replicates=replicates,
        bootstrap_seed=int(bootstrap_seed),
        num_paired_schedules=num_schedules,
        oracle_points=tuple(label(key) for key in oracle_keys),
        heuristic_points=tuple(label(key) for key in heuristic_keys),
        eligible_oracle_points=tuple(label(key) for key in eligible_oracle_keys),
        eligible_heuristic_points=tuple(
            label(key) for key in eligible_heuristic_keys
        ),
        selected_oracle_point=label(best_oracle_key),
        selected_heuristic_point=label(best_heuristic_key),
        oracle_best_tasks_per_hour=oracle_best,
        heuristic_best_tasks_per_hour=heuristic_best,
        throughput_gain_fraction=point_gain,
        throughput_gain_lower_confidence_bound=simultaneous_lower_gain,
        throughput_gain_upper_confidence_bound=simultaneous_upper_gain,
        max_t_critical_value=max_t_critical,
        upper_max_t_critical_value=upper_max_t_critical,
        oracle_rate_lower_bounds=tuple(
            (label(key), float(oracle_lower[index]))
            for index, key in enumerate(oracle_keys)
        ),
        oracle_rate_upper_bounds=tuple(
            (label(key), float(oracle_upper[index]))
            for index, key in enumerate(oracle_keys)
        ),
        heuristic_rate_lower_bounds=tuple(
            (label(key), float(heuristic_lower[index]))
            for index, key in enumerate(heuristic_keys)
        ),
        heuristic_rate_upper_bounds=tuple(
            (label(key), float(heuristic_upper[index]))
            for index, key in enumerate(heuristic_keys)
        ),
        descriptive_percentile_gain_lower_bound=float(np.quantile(gains, alpha)),
        descriptive_percentile_gain_upper_bound=float(np.quantile(gains, level)),
        inference=(
            "paired_schedule_studentized_max_t_rate_band_over_full_candidate_family"
        ),
    )


def audit_oracle_shadow_coupling(
    decision_events: Iterable[Mapping[str, object]],
    *,
    absolute_tolerance: float = 1e-6,
) -> OracleShadowCouplingAudit:
    """Verify pathwise coupling up to and including first disagreement.

    A method event is compared with the independently executed Oracle point at
    the same keyed schedule, reserve, global step, information time, and task
    index.  Events after the stopped first disagreement are intentionally out of
    scope because the two commitment policies then follow different paths.
    """

    tolerance = float(absolute_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("absolute_tolerance must be finite and nonnegative")
    records = list(decision_events)
    required = (
        "method",
        "parameter",
        "paired_schedule_id",
        "reserve_fraction",
        "decision_index",
        "global_step",
        "information_time",
        "task_index_in_cycle",
        "position",
        "velocity",
        "task_goal",
        "remaining_energy",
        "exact_effective_requirement",
        "exact_effective_requirement_is_infinite",
        "exact_return_now_deadline_feasible",
        "exact_task_then_return_deadline_feasible",
        "oracle_commit",
        "method_commit",
        "first_disagreement",
        "first_disagreement_direction",
    )
    for index, record in enumerate(records):
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"decision_events[{index}] missing fields: {missing}")

    oracle_groups: dict[
        tuple[str, float],
        dict[tuple[int, str, int], Mapping[str, object]],
    ] = {}
    method_groups: dict[
        tuple[str, float, str, float],
        list[Mapping[str, object]],
    ] = {}
    for record in records:
        method = str(record["method"])
        parameter = float(record["parameter"])
        schedule = str(record["paired_schedule_id"])
        reserve = round(float(record["reserve_fraction"]), 12)
        event_key = (
            int(record["global_step"]),
            str(record["information_time"]),
            int(record["task_index_in_cycle"]),
        )
        if method == "oracle":
            group = oracle_groups.setdefault((schedule, reserve), {})
            if event_key in group:
                raise ValueError(
                    "duplicate Oracle event for schedule/reserve/event key: "
                    f"{schedule}, {reserve}, {event_key}"
                )
            group[event_key] = record
        else:
            method_groups.setdefault(
                (method, parameter, schedule, reserve),
                [],
            ).append(record)

    failures: list[str] = []
    compared = 0
    first_disagreements = 0
    max_position = 0.0
    max_velocity = 0.0
    max_task_goal = 0.0
    max_remaining = 0.0
    max_requirement = 0.0

    def vector_error(left: object, right: object, name: str) -> float:
        first = np.asarray(left, dtype=np.float64)
        second = np.asarray(right, dtype=np.float64)
        if first.shape != (3,) or second.shape != (3,):
            raise ValueError(f"{name} must be aligned three-vectors")
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise ValueError(f"{name} must be finite")
        return float(np.max(np.abs(first - second)))

    for group_key, method_records in sorted(method_groups.items()):
        method, parameter, schedule, reserve = group_key
        ordered = sorted(method_records, key=lambda item: int(item["decision_index"]))
        markers = [
            index
            for index, item in enumerate(ordered)
            if bool(item["first_disagreement"])
        ]
        if len(markers) > 1:
            failures.append(f"{group_key}: multiple first-disagreement markers")
        marker = None if not markers else markers[0]
        if marker is not None:
            first_disagreements += 1
        compared_records = ordered if marker is None else ordered[: marker + 1]
        oracle_records = oracle_groups.get((schedule, reserve))
        if oracle_records is None:
            failures.append(f"{group_key}: missing same-schedule/same-reserve Oracle run")
            continue
        for local_index, method_event in enumerate(compared_records):
            event_key = (
                int(method_event["global_step"]),
                str(method_event["information_time"]),
                int(method_event["task_index_in_cycle"]),
            )
            oracle_event = oracle_records.get(event_key)
            if oracle_event is None:
                failures.append(f"{group_key}: missing Oracle event {event_key}")
                continue
            compared += 1
            position_error = vector_error(
                method_event["position"],
                oracle_event["position"],
                "position",
            )
            velocity_error = vector_error(
                method_event["velocity"],
                oracle_event["velocity"],
                "velocity",
            )
            task_goal_error = vector_error(
                method_event["task_goal"],
                oracle_event["task_goal"],
                "task_goal",
            )
            remaining_error = abs(
                float(method_event["remaining_energy"])
                - float(oracle_event["remaining_energy"])
            )
            method_infinite = bool(
                method_event["exact_effective_requirement_is_infinite"]
            )
            oracle_infinite = bool(
                oracle_event["exact_effective_requirement_is_infinite"]
            )
            if method_infinite != oracle_infinite:
                failures.append(
                    f"{group_key}, event={event_key}: exact deadline-feasibility mismatch"
                )
                requirement_error = 0.0
            elif method_infinite:
                requirement_error = 0.0
            else:
                requirement_error = abs(
                    float(method_event["exact_effective_requirement"])
                    - float(oracle_event["exact_effective_requirement"])
                )
            for field in (
                "exact_return_now_deadline_feasible",
                "exact_task_then_return_deadline_feasible",
            ):
                if bool(method_event[field]) != bool(oracle_event[field]):
                    failures.append(
                        f"{group_key}, event={event_key}: {field} mismatch"
                    )
            max_position = max(max_position, position_error)
            max_velocity = max(max_velocity, velocity_error)
            max_task_goal = max(max_task_goal, task_goal_error)
            max_remaining = max(max_remaining, remaining_error)
            max_requirement = max(max_requirement, requirement_error)
            errors = {
                "position": position_error,
                "velocity": velocity_error,
                "task_goal": task_goal_error,
                "remaining_energy": remaining_error,
                "exact_requirement": requirement_error,
            }
            violated = {
                name: value for name, value in errors.items() if value > tolerance
            }
            if violated:
                failures.append(
                    f"{group_key}, event={event_key}: coupling errors {violated}"
                )
            shadow_commit = bool(method_event["oracle_commit"])
            executed_oracle_commit = bool(oracle_event["method_commit"])
            if shadow_commit != executed_oracle_commit:
                failures.append(
                    f"{group_key}, event={event_key}: Oracle-shadow decision mismatch"
                )
            is_marker = marker is not None and local_index == marker
            method_commit = bool(method_event["method_commit"])
            if is_marker:
                if method_commit == executed_oracle_commit:
                    failures.append(
                        f"{group_key}, event={event_key}: marked disagreement has equal decisions"
                    )
                expected_direction = (
                    "method_early_commit_oracle_continue"
                    if method_commit
                    else "method_late_continue_oracle_commit"
                )
                if method_event["first_disagreement_direction"] != expected_direction:
                    failures.append(
                        f"{group_key}, event={event_key}: disagreement direction mismatch"
                    )
            elif method_commit != executed_oracle_commit:
                failures.append(
                    f"{group_key}, event={event_key}: unmarked earlier disagreement"
                )

    if not method_groups:
        failures.append("no non-Oracle method/schedule pairs were available")
    passed = not failures
    return OracleShadowCouplingAudit(
        status="PASS" if passed else "FAIL_ORACLE_SHADOW_COUPLING",
        passed=passed,
        absolute_tolerance=tolerance,
        num_method_schedule_pairs=len(method_groups),
        num_compared_decision_events=compared,
        num_first_disagreements=first_disagreements,
        maximum_position_error=max_position,
        maximum_velocity_error=max_velocity,
        maximum_task_goal_error=max_task_goal,
        maximum_remaining_energy_error=max_remaining,
        maximum_exact_requirement_error=max_requirement,
        failures=tuple(failures),
    )


def oracle_headroom_gate(
    outcomes: Iterable[ReturnDecisionOutcome],
    *,
    cycles_per_point: dict[tuple[str, float], int],
    stranding_upper_bounds: dict[tuple[str, float], float] | None = None,
    minimum_cycles_per_point: int = 300,
    stranding_ceiling: float = 0.05,
    minimum_throughput_gain_fraction: float = 0.05,
    throughput_gain_lower_confidence_bound: float | None = None,
    throughput_gain_upper_confidence_bound: float | None = None,
    throughput_confidence_level: float | None = None,
) -> OracleHeadroomGate:
    values = list(outcomes)
    if minimum_cycles_per_point <= 0:
        raise ValueError("minimum_cycles_per_point must be positive")
    if not 0.0 <= stranding_ceiling <= 1.0:
        raise ValueError("stranding_ceiling must lie in [0, 1]")
    if minimum_throughput_gain_fraction < 0.0:
        raise ValueError("minimum throughput gain must be nonnegative")
    if (
        throughput_gain_lower_confidence_bound is None
        and throughput_gain_upper_confidence_bound is None
    ):
        if throughput_confidence_level is not None:
            raise ValueError(
                "throughput confidence level requires a gain confidence bound"
            )
    else:
        if throughput_confidence_level is None or not (
            0.5 < float(throughput_confidence_level) < 1.0
        ):
            raise ValueError(
                "throughput confidence level must lie in (0.5, 1)"
            )
    lower_gain = None
    if throughput_gain_lower_confidence_bound is not None:
        lower_gain = float(throughput_gain_lower_confidence_bound)
        if not np.isfinite(lower_gain):
            raise ValueError("throughput gain lower confidence bound must be finite")
    upper_gain = None
    if throughput_gain_upper_confidence_bound is not None:
        upper_gain = float(throughput_gain_upper_confidence_bound)
        if not np.isfinite(upper_gain):
            raise ValueError("throughput gain upper confidence bound must be finite")
    if lower_gain is not None and upper_gain is not None and lower_gain > upper_gain:
        raise ValueError("throughput gain lower bound cannot exceed upper bound")
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
    point_passed = bool(gain >= minimum_throughput_gain_fraction)
    if lower_gain is None and upper_gain is None:
        status = "PENDING_THROUGHPUT_INFERENCE"
        evaluable = False
        passed = None
    elif point_passed and lower_gain is not None and (
        lower_gain >= minimum_throughput_gain_fraction
    ):
        status = "PASS"
        evaluable = True
        passed = True
    elif upper_gain is not None and upper_gain < minimum_throughput_gain_fraction:
        status = "FAIL_INSUFFICIENT_ORACLE_HEADROOM"
        evaluable = True
        passed = False
    else:
        status = "INCONCLUSIVE_ORACLE_HEADROOM"
        evaluable = True
        passed = False
    return OracleHeadroomGate(
        status,
        evaluable,
        passed,
        minimum_cycles_per_point,
        stranding_ceiling,
        minimum_throughput_gain_fraction,
        float(oracle_best),
        float(heuristic_best),
        float(gain),
        (
            "Oracle must improve throughput at the same preregistered stranding "
            "ceiling. PASS requires the point estimate and a full-family "
            "one-sided lower bound to clear the gain threshold; scientific "
            "FAIL requires the corresponding upper bound to lie below it; "
            "a bound crossing is inconclusive"
        ),
        lower_gain,
        upper_gain,
        (
            None
            if throughput_confidence_level is None
            else float(throughput_confidence_level)
        ),
        (
            None
            if lower_gain is None and upper_gain is None
            else "paired_frontier_full_family_directional_max_t_bootstrap"
        ),
    )


def sequential_boundary_certificate(
    *,
    estimation_failure_probability: float,
    boundary_probabilities: np.ndarray,
    boundary_widths: np.ndarray,
    throughput_upper_bound: float,
    loss_lipschitz: float = 0.0,
    loss_upper_bound: float = 0.0,
    oracle_stranding_rate: float | None = None,
    oracle_throughput: float | None = None,
    heuristic_throughput: float | None = None,
    stranding_ceiling: float | None = None,
    minimum_throughput_gain_fraction: float | None = None,
) -> SequentialBoundaryCertificate:
    """Evaluate the population consequences of a stopped boundary-mass bound.

    ``boundary_probabilities[t]`` is the oracle-shadow probability of reaching
    decision epoch ``t`` with the exact effective return score inside the
    declared error band.  The function does not estimate these probabilities;
    it only evaluates the theorem once they have been bounded independently.
    """

    alpha = float(estimation_failure_probability)
    probabilities = np.asarray(boundary_probabilities, dtype=np.float64)
    widths = np.asarray(boundary_widths, dtype=np.float64)
    throughput_cap = float(throughput_upper_bound)
    loss_slope = float(loss_lipschitz)
    loss_cap = float(loss_upper_bound)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("estimation failure probability must lie in [0, 1]")
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("boundary probabilities must be a nonempty vector")
    if widths.shape != probabilities.shape:
        raise ValueError("boundary widths must align with boundary probabilities")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("boundary probabilities must be finite and nonnegative")
    if np.any(probabilities > 1.0):
        raise ValueError("each boundary probability must not exceed one")
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError("boundary widths must be finite and nonnegative")
    if not np.isfinite(throughput_cap) or throughput_cap < 0.0:
        raise ValueError("throughput upper bound must be finite and nonnegative")
    if not np.isfinite(loss_slope) or loss_slope < 0.0:
        raise ValueError("loss Lipschitz constant must be finite and nonnegative")
    if not np.isfinite(loss_cap) or loss_cap < 0.0:
        raise ValueError("loss upper bound must be finite and nonnegative")

    boundary_occupation = float(np.sum(probabilities))
    disagreement = float(min(1.0, alpha + boundary_occupation))
    throughput_deviation = throughput_cap * disagreement
    excess_loss = float(
        loss_cap * alpha + loss_slope * np.sum(widths * probabilities)
    )

    optional_values = (
        oracle_stranding_rate,
        oracle_throughput,
        heuristic_throughput,
        stranding_ceiling,
        minimum_throughput_gain_fraction,
    )
    if all(value is None for value in optional_values):
        learned_stranding_upper = None
        learned_throughput_lower = None
        safety_preserved = None
        headroom_preserved = None
    elif any(value is None for value in optional_values):
        raise ValueError("all population headroom inputs must be supplied together")
    else:
        oracle_stranding = float(oracle_stranding_rate)
        oracle_rate = float(oracle_throughput)
        heuristic_rate = float(heuristic_throughput)
        ceiling = float(stranding_ceiling)
        gain = float(minimum_throughput_gain_fraction)
        if not 0.0 <= oracle_stranding <= 1.0:
            raise ValueError("oracle stranding rate must lie in [0, 1]")
        if not 0.0 <= ceiling <= 1.0:
            raise ValueError("stranding ceiling must lie in [0, 1]")
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (oracle_rate, heuristic_rate, gain)
        ):
            raise ValueError("throughputs and minimum gain must be finite and nonnegative")
        learned_stranding_upper = float(min(1.0, oracle_stranding + disagreement))
        learned_throughput_lower = float(max(0.0, oracle_rate - throughput_deviation))
        safety_preserved = bool(learned_stranding_upper <= ceiling)
        headroom_preserved = bool(
            safety_preserved
            and learned_throughput_lower
            >= (1.0 + gain) * heuristic_rate
        )

    return SequentialBoundaryCertificate(
        estimation_failure_probability=alpha,
        stopped_boundary_occupation=boundary_occupation,
        first_disagreement_probability_bound=disagreement,
        stranding_deviation_bound=disagreement,
        throughput_deviation_bound=float(throughput_deviation),
        excess_loss_bound=excess_loss,
        learned_stranding_upper=learned_stranding_upper,
        learned_throughput_lower=learned_throughput_lower,
        population_safety_ceiling_preserved=safety_preserved,
        population_headroom_preserved=headroom_preserved,
    )


def certified_effective_evar_requirement(
    *,
    critic_mgf_values: np.ndarray,
    absolute_certificate_radii: np.ndarray,
    risk_parameters: np.ndarray,
    tail_probabilities: float | np.ndarray,
    exact_mgf_values: np.ndarray | None = None,
    commitment_branch_index: int = -1,
) -> CertifiedEffectiveRequirement:
    """Propagate simultaneous MGF intervals to the commitment branch.

    Rows index the ReturnManager requirements (normally return-now and
    task-then-return); columns index a common positive risk-parameter grid.
    The last row is the task-then-return commitment branch by default.  Other
    rows remain separately certified but do not enter the binary stopping
    boundary in the non-nested hybrid system.
    The certificate is useful only when every lower MGF endpoint is positive.
    Optional exact values are accepted solely for theorem verification/audits;
    deployment does not have access to them.
    """

    critic = np.asarray(critic_mgf_values, dtype=np.float64)
    radii = np.asarray(absolute_certificate_radii, dtype=np.float64)
    lambdas = np.asarray(risk_parameters, dtype=np.float64)
    if critic.ndim != 2 or critic.shape[0] == 0 or critic.shape[1] == 0:
        raise ValueError("critic MGF values must be a nonempty branch-by-risk matrix")
    if radii.shape != critic.shape:
        raise ValueError("certificate radii must align with critic MGF values")
    if lambdas.shape != (critic.shape[1],):
        raise ValueError("risk parameters must align with the matrix columns")
    if not np.all(np.isfinite(critic)) or not np.all(np.isfinite(radii)):
        raise ValueError("critic values and certificate radii must be finite")
    if not np.all(np.isfinite(lambdas)) or np.any(lambdas <= 0.0):
        raise ValueError("risk parameters must be finite and strictly positive")
    if np.any(radii < 0.0):
        raise ValueError("certificate radii must be nonnegative")
    selected_branch = int(commitment_branch_index)
    if selected_branch < 0:
        selected_branch += critic.shape[0]
    if not 0 <= selected_branch < critic.shape[0]:
        raise ValueError("commitment_branch_index is outside the branch matrix")

    lower_mgf = critic - radii
    upper_mgf = critic + radii
    if np.any(lower_mgf <= 0.0):
        raise ValueError("every lower certified MGF endpoint must be positive")

    tails = np.asarray(tail_probabilities, dtype=np.float64)
    if tails.ndim == 0:
        tails = np.full(critic.shape[0], float(tails), dtype=np.float64)
    if tails.shape != (critic.shape[0],):
        raise ValueError("tail probabilities must be scalar or one per branch")
    if not np.all(np.isfinite(tails)) or np.any(tails <= 0.0) or np.any(tails >= 1.0):
        raise ValueError("tail probabilities must lie strictly between zero and one")

    log_tail = np.log(1.0 / tails)[:, None]
    branch_lower = np.min((np.log(lower_mgf) + log_tail) / lambdas, axis=1)
    branch_upper = np.min((np.log(upper_mgf) + log_tail) / lambdas, axis=1)
    effective_lower = float(branch_lower[selected_branch])
    effective_upper = float(branch_upper[selected_branch])
    error_bound = float(max(0.0, effective_upper - effective_lower))

    exact_branch: tuple[float, ...] | None = None
    exact_effective: float | None = None
    inside: bool | None = None
    conservative: bool | None = None
    if exact_mgf_values is not None:
        exact = np.asarray(exact_mgf_values, dtype=np.float64)
        if exact.shape != critic.shape or not np.all(np.isfinite(exact)):
            raise ValueError("exact MGF values must be a finite aligned matrix")
        if np.any(exact <= 0.0):
            raise ValueError("exact MGF values must be strictly positive")
        exact_values = np.min((np.log(exact) + log_tail) / lambdas, axis=1)
        exact_branch = tuple(float(value) for value in exact_values)
        exact_effective = float(exact_values[selected_branch])
        inside = bool(np.all((lower_mgf <= exact) & (exact <= upper_mgf)))
        conservative = bool(inside and effective_upper >= exact_effective)

    return CertifiedEffectiveRequirement(
        branch_lower_requirements=tuple(float(value) for value in branch_lower),
        branch_upper_requirements=tuple(float(value) for value in branch_upper),
        effective_lower_requirement=effective_lower,
        learned_effective_upper_requirement=effective_upper,
        effective_error_bound=error_bound,
        exact_branch_requirements=exact_branch,
        exact_effective_requirement=exact_effective,
        exact_mgf_inside_certificate=inside,
        learned_requirement_is_conservative=conservative,
        commitment_branch_index=selected_branch,
        effective_requirement_semantics="task_then_return_stopping_boundary",
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
    "CertifiedEffectiveRequirement",
    "CloneRolloutVarianceAudit",
    "SequentialBoundaryCertificate",
    "ReturnDecisionOutcome",
    "OracleHeadroomGate",
    "OracleShadowCouplingAudit",
    "PairedFrontierThroughputInterval",
    "SimultaneousStrandingAudit",
    "SimultaneousStrandingPoint",
    "StrandingCertificationPowerAudit",
    "boundary_weighted_underestimation",
    "audit_oracle_shadow_coupling",
    "certified_effective_evar_requirement",
    "clone_rollout_variance_audit",
    "oracle_headroom_gate",
    "paired_frontier_throughput_interval",
    "pareto_frontier",
    "probability_semantics_gate",
    "sequential_boundary_certificate",
    "simultaneous_stranding_upper_bounds",
    "stranding_certification_power_audit",
    "wilson_interval",
]
