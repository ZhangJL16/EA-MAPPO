from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from scipy.optimize import brentq
from scipy.stats import binom, binomtest


@dataclass(frozen=True)
class BinomialCoverageEvidence:
    successes: int
    count: int
    target: float
    empirical_coverage: float
    standard_error: float
    exact_interval_low: float
    exact_interval_high: float
    wilson_interval_low: float
    wilson_interval_high: float
    lower_tail_pvalue_at_target: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def wilson_interval(
    successes: int,
    count: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("successes must lie in [0, count] with count > 0")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def binomial_coverage_evidence(
    successes: int,
    count: int,
    *,
    target: float = 0.95,
    confidence: float = 0.95,
) -> BinomialCoverageEvidence:
    if count <= 0 or not 0 <= successes <= count:
        raise ValueError("successes must lie in [0, count] with count > 0")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie in (0, 1)")
    empirical = successes / count
    exact = binomtest(successes, count).proportion_ci(confidence, method="exact")
    wilson_low, wilson_high = wilson_interval(successes, count, confidence=confidence)
    return BinomialCoverageEvidence(
        successes=int(successes),
        count=int(count),
        target=float(target),
        empirical_coverage=float(empirical),
        standard_error=float(math.sqrt(empirical * (1.0 - empirical) / count)),
        exact_interval_low=float(exact.low),
        exact_interval_high=float(exact.high),
        wilson_interval_low=float(wilson_low),
        wilson_interval_high=float(wilson_high),
        lower_tail_pvalue_at_target=float(binom.cdf(successes, count, target)),
    )


def probability_empirical_meets_target(
    count: int,
    *,
    true_coverage: float,
    empirical_target: float = 0.95,
) -> float:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 < true_coverage < 1.0 or not 0.0 < empirical_target < 1.0:
        raise ValueError("coverage values must lie in (0, 1)")
    threshold = int(math.ceil(empirical_target * count))
    return float(binom.sf(threshold - 1, count, true_coverage))


def underlying_coverage_for_pass_probability(
    count: int,
    *,
    pass_probability: float,
    empirical_target: float = 0.95,
) -> float:
    if not 0.5 < pass_probability < 1.0:
        raise ValueError("pass_probability must lie in (0.5, 1)")
    if not 0.0 < empirical_target < 1.0:
        raise ValueError("empirical_target must lie in (0, 1)")
    return float(
        brentq(
            lambda coverage: probability_empirical_meets_target(
                count,
                true_coverage=coverage,
                empirical_target=empirical_target,
            )
            - pass_probability,
            empirical_target,
            1.0 - 1e-12,
        )
    )


def holm_undercoverage_audit(
    groups: dict[str, BinomialCoverageEvidence],
    *,
    family_alpha: float = 0.05,
) -> dict[str, object]:
    if not groups:
        raise ValueError("at least one group is required")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family_alpha must lie in (0, 1)")
    ordered = sorted(groups.items(), key=lambda row: row[1].lower_tail_pvalue_at_target)
    rejected: list[str] = []
    decisions: dict[str, dict[str, float | bool | int]] = {}
    continue_rejecting = True
    total = len(ordered)
    for index, (name, evidence) in enumerate(ordered):
        threshold = family_alpha / (total - index)
        reject = bool(
            continue_rejecting and evidence.lower_tail_pvalue_at_target <= threshold
        )
        if reject:
            rejected.append(name)
        else:
            continue_rejecting = False
        decisions[name] = {
            "rank": index + 1,
            "pvalue": evidence.lower_tail_pvalue_at_target,
            "holm_threshold": threshold,
            "significant_undercoverage": reject,
        }
    return {
        "family_alpha": float(family_alpha),
        "num_groups": total,
        "rejected_groups": rejected,
        "passed": not rejected,
        "decisions": decisions,
        "interpretation": (
            "failure to reject undercoverage is not proof of arbitrary conditional coverage"
        ),
    }


def sample_size_power_table(
    counts: Iterable[int] = (100, 200, 250, 500, 1000),
    *,
    empirical_target: float = 0.95,
) -> list[dict[str, float | int]]:
    rows = []
    for count in counts:
        rows.append(
            {
                "count": int(count),
                "success_threshold": int(math.ceil(empirical_target * count)),
                "pass_probability_if_true_equals_target": probability_empirical_meets_target(
                    int(count),
                    true_coverage=empirical_target,
                    empirical_target=empirical_target,
                ),
                "true_coverage_for_90_percent_pass_probability": (
                    underlying_coverage_for_pass_probability(
                        int(count),
                        pass_probability=0.90,
                        empirical_target=empirical_target,
                    )
                ),
                "true_coverage_for_95_percent_pass_probability": (
                    underlying_coverage_for_pass_probability(
                        int(count),
                        pass_probability=0.95,
                        empirical_target=empirical_target,
                    )
                ),
            }
        )
    return rows


__all__ = [
    "BinomialCoverageEvidence",
    "binomial_coverage_evidence",
    "holm_undercoverage_audit",
    "probability_empirical_meets_target",
    "sample_size_power_table",
    "underlying_coverage_for_pass_probability",
    "wilson_interval",
]
