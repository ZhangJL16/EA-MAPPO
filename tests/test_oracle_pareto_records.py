from __future__ import annotations

from experiments.energy_mc.oracle_pareto_records import (
    CYCLE_FIELDS,
    DECISION_EVENT_FIELDS,
    METHOD_SUMMARY_FIELDS,
    empty_oracle_pareto_bundle,
    validate_oracle_pareto_bundle,
)


def _record(fields: tuple[str, ...]) -> dict[str, object]:
    return {field: None for field in fields}


def test_empty_exploratory_template_is_valid_but_not_formal() -> None:
    bundle = empty_oracle_pareto_bundle()
    assert validate_oracle_pareto_bundle(bundle) == []
    assert bundle["formal_eligible"] is False
    assert bundle["claim_status"] == "NOT_EVALUATED"


def test_formal_claim_cannot_bypass_failed_upstream_gates() -> None:
    bundle = empty_oracle_pareto_bundle()
    bundle["claim_status"] = "ORACLE_HEADROOM_GATE_PASS"
    errors = validate_oracle_pareto_bundle(bundle)
    assert any("formal_eligible=true" in error for error in errors)
    assert any("oracle_decision_headroom.passed=true" in error for error in errors)


def test_formal_eligibility_requires_paired_event_cycle_and_summary_records() -> None:
    bundle = empty_oracle_pareto_bundle()
    bundle["evidence_mode"] = "FORMAL"
    bundle["formal_eligible"] = True
    for gate in bundle["gates"].values():
        gate["passed"] = True
        gate["status"] = "PASS"
    errors = validate_oracle_pareto_bundle(bundle)
    assert any("nonempty decision_events" in error for error in errors)
    assert any("nonempty cycles" in error for error in errors)
    assert any("nonempty method_summaries" in error for error in errors)


def test_complete_record_shapes_are_accepted_for_oracle_headroom() -> None:
    bundle = empty_oracle_pareto_bundle()
    bundle["evidence_mode"] = "FORMAL"
    bundle["formal_eligible"] = True
    bundle["claim_status"] = "ORACLE_HEADROOM_GATE_PASS"
    bundle["provenance"] = {
        "navigation_artifact_schema": "navigation_repair_completion_wrapper",
        "navigation_artifact_config_sha256": "e" * 64,
        "navigation_environment_contract": {
            "config_sha256": "e" * 64,
            "environment_kwargs": {"lidar_enabled": True},
            "stage_b_runtime_overrides": {
                "phase": "ENERGY_MANAGED",
            },
        },
        "navigation_checkpoint_sha256": "a" * 64,
        "navigation_wrapper_checkpoint_sha256": "a" * 64,
        "navigation_evaluation_sha256": "b" * 64,
        "battery_calibration_sha256": "c" * 64,
        "battery_validation_sha256": "d" * 64,
    }
    for gate in bundle["gates"].values():
        gate["passed"] = True
        gate["status"] = "PASS"
    bundle["oracle_shadow_coupling_audit"] = {
        "status": "PASS",
        "passed": True,
    }
    bundle["simultaneous_stranding_audit"] = {
        "confidence_level": 0.95,
        "correction": "bonferroni_one_sided_clopper_pearson",
        "num_candidate_points": 1,
        "points": [{"method": "oracle", "upper_confidence_bound": 0.04}],
    }
    bundle["stranding_certification_power_audit"] = {
        "status": "PASS_DESIGN_POWER",
        "passed": True,
    }
    bundle["paired_throughput_interval"] = {
        "inference": (
            "paired_schedule_studentized_max_t_rate_band_over_full_candidate_family"
        ),
        "throughput_gain_lower_confidence_bound": 0.08,
        "throughput_gain_upper_confidence_bound": 0.12,
        "max_t_critical_value": 1.96,
        "upper_max_t_critical_value": 1.97,
        "oracle_rate_lower_bounds": (("oracle|0", 108.0),),
        "oracle_rate_upper_bounds": (("oracle|0", 112.0),),
        "heuristic_rate_lower_bounds": (("soc|0.2", 98.0),),
        "heuristic_rate_upper_bounds": (("soc|0.2", 102.0),),
    }
    decision = _record(DECISION_EVENT_FIELDS)
    decision.update(
        {
            "exact_return_now_requirement": 1.0,
            "exact_task_then_return_requirement": 2.0,
            "exact_effective_requirement": 2.0,
            "exact_effective_requirement_semantics": (
                "task_then_return_stopping_boundary"
            ),
            "exact_effective_requirement_is_infinite": False,
            "exact_return_now_deadline_feasible": True,
            "exact_task_then_return_deadline_feasible": True,
            "oracle_commit": False,
            "oracle_shadow_active": True,
            "estimated_effective_requirement_semantics": (
                "task_then_return_stopping_boundary"
            ),
        }
    )
    cycle = _record(CYCLE_FIELDS)
    cycle.update(
        {
            "paired_oracle_return_success": True,
            "paired_oracle_stranded": False,
            "paired_oracle_tasks_completed": 2,
        }
    )
    bundle["decision_events"] = [decision]
    bundle["cycles"] = [cycle]
    bundle["method_summaries"] = [_record(METHOD_SUMMARY_FIELDS)]
    assert validate_oracle_pareto_bundle(bundle) == []


def test_formal_shadow_records_distinguish_infinite_from_not_evaluated() -> None:
    bundle = empty_oracle_pareto_bundle()
    bundle["evidence_mode"] = "FORMAL"
    bundle["formal_eligible"] = True
    bundle["provenance"] = {
        "navigation_artifact_schema": "navigation_repair_completion_wrapper",
        "navigation_artifact_config_sha256": "e" * 64,
        "navigation_environment_contract": {
            "config_sha256": "e" * 64,
            "environment_kwargs": {},
            "stage_b_runtime_overrides": {},
        },
        "navigation_checkpoint_sha256": "a" * 64,
        "navigation_wrapper_checkpoint_sha256": "a" * 64,
        "navigation_evaluation_sha256": "b" * 64,
        "battery_calibration_sha256": "c" * 64,
        "battery_validation_sha256": "d" * 64,
    }
    for gate in bundle["gates"].values():
        gate.update({"passed": True, "status": "PASS"})
    bundle["oracle_shadow_coupling_audit"] = {"status": "PASS", "passed": True}
    bundle["simultaneous_stranding_audit"] = {
        "correction": "bonferroni_one_sided_clopper_pearson",
        "num_candidate_points": 1,
        "points": [{}],
    }
    bundle["stranding_certification_power_audit"] = {"passed": True}
    active_infinite = _record(DECISION_EVENT_FIELDS)
    active_infinite.update(
        {
            "oracle_shadow_active": True,
            "exact_return_now_requirement": 4.0,
            "exact_task_then_return_requirement": 9.0,
            "exact_effective_requirement": None,
            "exact_effective_requirement_semantics": (
                "task_then_return_stopping_boundary"
            ),
            "exact_effective_requirement_is_infinite": True,
            "exact_return_now_deadline_feasible": True,
            "exact_task_then_return_deadline_feasible": False,
            "oracle_commit": True,
            "estimated_effective_requirement_semantics": (
                "task_then_return_stopping_boundary"
            ),
        }
    )
    inactive = _record(DECISION_EVENT_FIELDS)
    inactive["oracle_shadow_active"] = False
    inactive["exact_effective_requirement_semantics"] = (
        "task_then_return_stopping_boundary"
    )
    inactive["estimated_effective_requirement_semantics"] = (
        "task_then_return_stopping_boundary"
    )
    cycle = _record(CYCLE_FIELDS)
    cycle.update(
        {
            "paired_oracle_return_success": True,
            "paired_oracle_stranded": False,
            "paired_oracle_tasks_completed": 1,
        }
    )
    bundle["decision_events"] = [active_infinite, inactive]
    bundle["cycles"] = [cycle]
    bundle["method_summaries"] = [_record(METHOD_SUMMARY_FIELDS)]
    assert validate_oracle_pareto_bundle(bundle) == []
