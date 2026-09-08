from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


SCHEMA_VERSION = "return-to-charge-oracle-pareto-v7"

GATE_NAMES = (
    "navigation",
    "probability_semantics",
    "battery_calibration",
    "battery_validation",
    "oracle_decision_headroom",
)

DECISION_EVENT_FIELDS = (
    "record_id",
    "method",
    "parameter",
    "reserve_fraction",
    "evaluation_seed",
    "paired_schedule_id",
    "cycle_index",
    "decision_index",
    "global_step",
    "information_time",
    "remaining_energy",
    "reserve_energy",
    "exact_return_now_requirement",
    "exact_task_then_return_requirement",
    "exact_effective_requirement",
    "exact_effective_requirement_semantics",
    "exact_effective_requirement_is_infinite",
    "exact_return_now_deadline_feasible",
    "exact_task_then_return_deadline_feasible",
    "estimated_return_now_requirement",
    "estimated_task_then_return_requirement",
    "estimated_effective_requirement",
    "estimated_effective_requirement_semantics",
    "estimated_effective_requirement_is_infinite",
    "estimated_return_now_deadline_feasible",
    "estimated_task_then_return_deadline_feasible",
    "oracle_commit",
    "oracle_decision_reason",
    "method_commit",
    "method_decision_reason",
    "commitment_state_before",
    "commitment_state_after",
    "oracle_shadow_active",
    "oracle_shadow_stopping_rule",
    "oracle_score_margin",
    "method_score_margin",
    "method_margin_unit",
    "decisions_disagree",
    "first_disagreement",
    "first_disagreement_direction",
    "decision_check_interval_steps",
    "threshold_crossing_overshoot",
    "observed_interval_requirement_drift",
    "feature_provenance",
)

CYCLE_FIELDS = (
    "record_id",
    "method",
    "parameter",
    "evaluation_seed",
    "paired_schedule_id",
    "cycle_index",
    "end_reason",
    "return_success",
    "stranded",
    "energy_exhausted",
    "tasks_completed",
    "simulated_seconds",
    "tasks_per_simulated_hour",
    "unused_energy_fraction_at_charger",
    "first_disagreement_step",
    "first_disagreement_direction",
    "paired_oracle_return_success",
    "paired_oracle_stranded",
    "paired_oracle_tasks_completed",
)

METHOD_SUMMARY_FIELDS = (
    "method",
    "parameter",
    "independent_cycles",
    "stranding_count",
    "stranding_rate",
    "stranding_rate_wilson95_upper",
    "tasks_per_simulated_hour",
    "tasks_per_battery_cycle",
    "mean_unused_energy_fraction_at_charger",
)


def empty_oracle_pareto_bundle() -> dict[str, object]:
    """Return a fail-closed template; it contains no empirical result."""

    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_mode": "EXPLORATORY",
        "claim_status": "NOT_EVALUATED",
        "formal_eligible": False,
        "provenance": {},
        "protocol": {},
        "gates": {
            name: {"status": "NOT_EVALUATED", "passed": None, "artifact": None}
            for name in GATE_NAMES
        },
        "oracle_shadow_coupling_audit": {
            "status": "NOT_EVALUATED",
            "passed": False,
        },
        "simultaneous_stranding_audit": {
            "status": "NOT_EVALUATED",
            "correction": None,
            "num_candidate_points": 0,
            "points": [],
        },
        "stranding_certification_power_audit": None,
        "paired_throughput_interval": None,
        "decision_events": [],
        "cycles": [],
        "method_summaries": [],
        "theory_diagnostics": [],
        "pareto_frontier": [],
    }


def _missing(record: Mapping[str, object], fields: Sequence[str]) -> list[str]:
    return [field for field in fields if field not in record]


def _records(
    bundle: Mapping[str, object],
    name: str,
    fields: Sequence[str],
) -> list[str]:
    value = bundle.get(name)
    if not isinstance(value, list):
        return [f"{name} must be a list"]
    errors: list[str] = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            errors.append(f"{name}[{index}] must be an object")
            continue
        missing = _missing(record, fields)
        if missing:
            errors.append(f"{name}[{index}] missing fields: {missing}")
    return errors


def _is_sha256(value: object) -> bool:
    text = str(value).lower() if value is not None else ""
    return bool(
        len(text) == 64
        and all(character in "0123456789abcdef" for character in text)
    )


def validate_oracle_pareto_bundle(bundle: Mapping[str, object]) -> list[str]:
    """Validate the Oracle/Pareto evidence contract without promoting claims.

    The validator deliberately permits an empty exploratory template.  Formal
    eligibility, however, requires the upstream Gate chain, paired decision
    events, independent cycle records, and aggregate method summaries.
    """

    errors: list[str] = []
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must equal {SCHEMA_VERSION!r}")
    if bundle.get("evidence_mode") not in {"EXPLORATORY", "FORMAL"}:
        errors.append("evidence_mode must be EXPLORATORY or FORMAL")
    for name in (
        "provenance",
        "protocol",
        "gates",
        "oracle_shadow_coupling_audit",
        "simultaneous_stranding_audit",
    ):
        if not isinstance(bundle.get(name), Mapping):
            errors.append(f"{name} must be an object")
    throughput = bundle.get("paired_throughput_interval")
    if throughput is not None and not isinstance(throughput, Mapping):
        errors.append("paired_throughput_interval must be an object or null")
    power = bundle.get("stranding_certification_power_audit")
    if power is not None and not isinstance(power, Mapping):
        errors.append(
            "stranding_certification_power_audit must be an object or null"
        )

    gates = bundle.get("gates")
    if isinstance(gates, Mapping):
        for gate_name in GATE_NAMES:
            gate = gates.get(gate_name)
            if not isinstance(gate, Mapping):
                errors.append(f"gates.{gate_name} must be an object")
                continue
            missing = _missing(gate, ("status", "passed", "artifact"))
            if missing:
                errors.append(f"gates.{gate_name} missing fields: {missing}")

    errors.extend(_records(bundle, "decision_events", DECISION_EVENT_FIELDS))
    decision_events = bundle.get("decision_events")
    if isinstance(decision_events, list):
        for index, record in enumerate(decision_events):
            if not isinstance(record, Mapping):
                continue
            for field in (
                "exact_effective_requirement_semantics",
                "estimated_effective_requirement_semantics",
            ):
                if record.get(field) != "task_then_return_stopping_boundary":
                    errors.append(
                        f"decision_events[{index}].{field} must declare the "
                        "task-then-return stopping boundary"
                    )
    errors.extend(_records(bundle, "cycles", CYCLE_FIELDS))
    errors.extend(_records(bundle, "method_summaries", METHOD_SUMMARY_FIELDS))
    for name in ("theory_diagnostics", "pareto_frontier"):
        if not isinstance(bundle.get(name), list):
            errors.append(f"{name} must be a list")

    claim_status = bundle.get("claim_status")
    formal_eligible = bundle.get("formal_eligible")
    if not isinstance(formal_eligible, bool):
        errors.append("formal_eligible must be boolean")
        formal_eligible = False
    formal_claim = claim_status in {
        "ORACLE_HEADROOM_GATE_PASS",
        "PARETO_EVIDENCE_READY",
    }
    if formal_claim and not formal_eligible:
        errors.append("a formal claim requires formal_eligible=true")
    if formal_claim:
        if (
            not isinstance(throughput, Mapping)
            or throughput.get("inference")
            != "paired_schedule_studentized_max_t_rate_band_over_full_candidate_family"
        ):
            errors.append(
                "a formal claim requires a full-family simultaneous throughput audit"
            )
        else:
            lower = throughput.get("throughput_gain_lower_confidence_bound")
            upper = throughput.get("throughput_gain_upper_confidence_bound")
            lower_critical = throughput.get("max_t_critical_value")
            upper_critical = throughput.get("upper_max_t_critical_value")
            if (
                not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or not math.isfinite(float(lower))
                or not math.isfinite(float(upper))
                or float(lower) > float(upper)
            ):
                errors.append(
                    "a formal claim requires ordered numeric full-family "
                    "throughput gain lower and upper bounds"
                )
            if (
                not isinstance(lower_critical, (int, float))
                or not isinstance(upper_critical, (int, float))
                or not math.isfinite(float(lower_critical))
                or not math.isfinite(float(upper_critical))
                or float(lower_critical) < 0.0
                or float(upper_critical) < 0.0
            ):
                errors.append(
                    "a formal claim requires finite nonnegative directional "
                    "max-t critical values"
                )
            for field in (
                "oracle_rate_lower_bounds",
                "oracle_rate_upper_bounds",
                "heuristic_rate_lower_bounds",
                "heuristic_rate_upper_bounds",
            ):
                values = throughput.get(field)
                if (
                    not isinstance(values, Sequence)
                    or isinstance(values, (str, bytes))
                    or not values
                ):
                    errors.append(
                        f"a formal claim requires nonempty {field}"
                    )

    if formal_eligible:
        if bundle.get("evidence_mode") != "FORMAL":
            errors.append("formal_eligible=true requires evidence_mode=FORMAL")
        if not isinstance(power, Mapping) or power.get("passed") is not True:
            errors.append(
                "formal eligibility requires a passed stranding-certification "
                "power audit"
            )
        provenance = bundle.get("provenance")
        if isinstance(provenance, Mapping):
            if provenance.get("navigation_artifact_schema") not in {
                "navigation_repair_completion_wrapper",
                "checkpoint_evaluator_completion_wrapper",
            }:
                errors.append(
                    "formal eligibility requires an authorized navigation "
                    "completion-wrapper schema"
                )
            for field in (
                "navigation_artifact_config_sha256",
                "navigation_checkpoint_sha256",
                "navigation_wrapper_checkpoint_sha256",
                "navigation_evaluation_sha256",
                "battery_calibration_sha256",
                "battery_validation_sha256",
            ):
                if not _is_sha256(provenance.get(field)):
                    errors.append(
                        f"formal eligibility requires provenance.{field}"
                    )
            environment_contract = provenance.get(
                "navigation_environment_contract"
            )
            if (
                not isinstance(environment_contract, Mapping)
                or environment_contract.get("config_sha256")
                != provenance.get("navigation_artifact_config_sha256")
                or not isinstance(
                    environment_contract.get("environment_kwargs"), Mapping
                )
                or not isinstance(
                    environment_contract.get("stage_b_runtime_overrides"),
                    Mapping,
                )
            ):
                errors.append(
                    "formal eligibility requires a matching reconstructed "
                    "navigation environment contract"
                )
            if provenance.get("navigation_checkpoint_sha256") != provenance.get(
                "navigation_wrapper_checkpoint_sha256"
            ):
                errors.append(
                    "formal eligibility requires matching loaded and wrapper "
                    "navigation checkpoint hashes"
                )
        if isinstance(gates, Mapping):
            upstream = GATE_NAMES[:-1]
            for gate_name in upstream:
                gate = gates.get(gate_name)
                if not isinstance(gate, Mapping) or gate.get("passed") is not True:
                    errors.append(
                        f"formal eligibility requires gates.{gate_name}.passed=true"
                    )
        for name in ("decision_events", "cycles", "method_summaries"):
            if not bundle.get(name):
                errors.append(f"formal eligibility requires nonempty {name}")
        coupling = bundle.get("oracle_shadow_coupling_audit")
        if not isinstance(coupling, Mapping) or coupling.get("passed") is not True:
            errors.append(
                "formal eligibility requires oracle_shadow_coupling_audit.passed=true"
            )
        safety = bundle.get("simultaneous_stranding_audit")
        if not isinstance(safety, Mapping):
            errors.append(
                "formal eligibility requires simultaneous_stranding_audit"
            )
        else:
            points = safety.get("points")
            if (
                safety.get("correction")
                != "bonferroni_one_sided_clopper_pearson"
                or not isinstance(points, list)
                or not points
                or len(points) != safety.get("num_candidate_points")
            ):
                errors.append(
                    "formal eligibility requires a complete selection-valid "
                    "simultaneous stranding audit"
                )
        decision_events = bundle.get("decision_events")
        if isinstance(decision_events, list):
            for index, record in enumerate(decision_events):
                if not isinstance(record, Mapping):
                    continue
                shadow_active = record.get("oracle_shadow_active")
                if not isinstance(shadow_active, bool):
                    errors.append(
                        f"formal decision_events[{index}]."
                        "oracle_shadow_active must be boolean"
                    )
                exact_infinite = record.get(
                    "exact_effective_requirement_is_infinite"
                )
                exact_effective = record.get("exact_effective_requirement")
                exact_fields = (
                    "exact_return_now_requirement",
                    "exact_task_then_return_requirement",
                    "exact_effective_requirement",
                    "exact_effective_requirement_is_infinite",
                    "exact_return_now_deadline_feasible",
                    "exact_task_then_return_deadline_feasible",
                    "oracle_commit",
                )
                if shadow_active is False:
                    for field in exact_fields:
                        if record.get(field) is not None:
                            errors.append(
                                f"formal decision_events[{index}].{field} must "
                                "be null when oracle_shadow_active=false"
                            )
                    continue
                for field in (
                    "exact_return_now_requirement",
                    "exact_task_then_return_requirement",
                ):
                    value = record.get(field)
                    if (
                        not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        errors.append(
                            f"formal decision_events[{index}].{field} must be "
                            "finite while the Oracle shadow is active"
                        )
                for field in (
                    "exact_return_now_deadline_feasible",
                    "exact_task_then_return_deadline_feasible",
                    "oracle_commit",
                ):
                    if not isinstance(record.get(field), bool):
                        errors.append(
                            f"formal decision_events[{index}].{field} must be boolean"
                        )
                if exact_infinite is True:
                    if exact_effective is not None:
                        errors.append(
                            f"formal decision_events[{index}] infinite exact "
                            "requirement must use null numeric value"
                        )
                elif exact_infinite is False:
                    if (
                        not isinstance(exact_effective, (int, float))
                        or not math.isfinite(float(exact_effective))
                    ):
                        errors.append(
                            f"formal decision_events[{index}] finite exact "
                            "requirement must be a finite number"
                        )
                else:
                    errors.append(
                        f"formal decision_events[{index}]."
                        "exact_effective_requirement_is_infinite must be boolean"
                    )
        cycles = bundle.get("cycles")
        if isinstance(cycles, list):
            for index, record in enumerate(cycles):
                if not isinstance(record, Mapping):
                    continue
                for field in (
                    "paired_oracle_return_success",
                    "paired_oracle_stranded",
                    "paired_oracle_tasks_completed",
                ):
                    if record.get(field) is None:
                        errors.append(
                            f"formal cycles[{index}].{field} must not be null"
                        )

    if claim_status == "ORACLE_HEADROOM_GATE_PASS":
        gate = gates.get("oracle_decision_headroom") if isinstance(gates, Mapping) else None
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append(
                "ORACLE_HEADROOM_GATE_PASS requires oracle_decision_headroom.passed=true"
            )
    if claim_status == "PARETO_EVIDENCE_READY":
        gate = gates.get("oracle_decision_headroom") if isinstance(gates, Mapping) else None
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            errors.append(
                "PARETO_EVIDENCE_READY requires oracle_decision_headroom.passed=true"
            )
        if not bundle.get("pareto_frontier"):
            errors.append("PARETO_EVIDENCE_READY requires a nonempty pareto_frontier")
    return errors
