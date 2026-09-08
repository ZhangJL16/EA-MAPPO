from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.energy_mc.gate_b_prerequisites import (
    CONTINUOUS_ENDURANCE_ESTIMAND,
    FIXED_BASELINE_CONTRACT_SCHEMA,
    FIXED_BASELINE_RECALIBRATION_SCHEMA,
    FIXED_BASELINE_VALIDATION_SCHEMA,
    audit_gate_b_prerequisites,
    navigation_artifact_view,
)
from scripts.prepare_fixed_baseline_energy_contract import (
    attest_fixed_baseline_calibration,
    build_fixed_baseline_contract,
)
from scripts.validate_battery_for_fixed_baseline import (
    aggregate_validation_batches,
    audit_refined_calibration_provenance,
)
from scripts.recalibrate_battery_from_continuous_endurance import (
    build_refined_calibration,
)


def _navigation_metrics() -> dict[str, object]:
    return {
        "num_tasks": 500,
        "global_env_transitions": 500_000,
        "overall_success_rate": 0.96,
        "distance_bucket_success": {
            "100-500": 0.99,
            "500-1500": 0.98,
            "1500-2500": 0.94,
            "2500-4000": 0.97,
            ">4000": 0.92,
        },
        "mean_path_ratio": 1.188,
        "boundary_contact_step_rate": 1.0e-5,
        "boundary_contact_episode_rate": 0.008,
        "obstacle_collision_steps": 2242,
        "obstacle_collision_step_rate": 0.0057,
        "obstacle_collision_episode_rate": 0.006,
        "episodes_with_obstacle_collision": 3,
    }


def _contract() -> dict[str, object]:
    return {
        "schema": FIXED_BASELINE_CONTRACT_SCHEMA,
        "status": "AUTHORIZED",
        "baseline_id": "R3",
        "energy_research_authorized": True,
        "frozen_policy_required": True,
        "performance_metrics_are_descriptive": True,
        "checkpoint_sha256": "a" * 64,
        "navigation_metrics": _navigation_metrics(),
    }


def _calibration() -> dict[str, object]:
    tasks = [
        {
            "task_index": index,
            "distance_bucket": "100-500",
            "success": index < 475,
            "end_reason": "goal_reached" if index < 475 else "task_step_limit",
            "total_realized_energy": 10.0,
            "simulation_flight_time": 50.0,
        }
        for index in range(500)
    ]
    return {
        "num_tasks": 500,
        "num_successful_tasks": 475,
        "calibration_success_rate": 0.95,
        "calibration_distance_bucket_success": {"100-500": 0.95},
        "battery_calibration_navigation_valid": False,
        "fixed_baseline_calibration_evaluable": True,
        "calibration_failure_taxonomy_complete": True,
        "policy_unchanged": True,
        "sac_training": False,
        "td_training": False,
        "energy_source": "TelemetryCostModel.realized_cost",
        "calibrated_battery_capacity": 360.0,
        "target_nominal_endurance_minutes": 30.0,
        "mean_power": 0.2,
        "tasks": tasks,
    }


def _validation() -> dict[str, object]:
    return {
        "schema": FIXED_BASELINE_VALIDATION_SCHEMA,
        "endurance_estimand": CONTINUOUS_ENDURANCE_ESTIMAND,
        "battery_validation_runs": 100,
        "battery_calibration_valid": True,
        "all_runs_depleted": True,
        "continuous_workload_until_depletion": True,
        "censored_run_count": 0,
        "engineering_calibration_tolerance_fraction": 0.20,
        "relative_endurance_error": 0.05,
        "battery_capacity": 360.0,
        "mean_depletion_time": 1890.0,
        "target_nominal_endurance_minutes": 30.0,
    }


def test_r3_contract_authorizes_conditional_energy_without_rewriting_legacy_gate() -> None:
    view = navigation_artifact_view(_contract())
    assert view.authorization_passed is True
    assert view.fixed_baseline_contract is True
    assert view.performance_metrics_descriptive is True
    audit = audit_gate_b_prerequisites(_contract(), _calibration(), _validation())
    assert audit.passed is True
    assert audit.baseline_id == "R3"
    assert audit.navigation_success_rate == pytest.approx(0.96)
    assert audit.calibration_success_rate == pytest.approx(0.95)
    assert audit.navigation_performance_metrics_descriptive is True
    assert audit.battery_validation_schema == FIXED_BASELINE_VALIDATION_SCHEMA
    assert audit.endurance_estimand == CONTINUOUS_ENDURANCE_ESTIMAND
    assert audit.continuous_workload_until_depletion is True
    assert audit.censored_run_count == 0
    assert audit.battery_validation_runs == 100


def test_same_r3_metrics_still_fail_the_legacy_deployment_gate() -> None:
    navigation = _navigation_metrics()
    calibration = _calibration()
    calibration.update(
        {
            "td_replay_writes": 0,
            "battery_calibration_navigation_valid": True,
        }
    )
    audit = audit_gate_b_prerequisites(navigation, calibration, _validation())
    assert audit.passed is False
    assert any("navigation success" in failure for failure in audit.failures)
    assert any("navigation mean path ratio" in failure for failure in audit.failures)
    assert any("obstacle collisions" in failure for failure in audit.failures)


def test_fixed_baseline_requires_complete_failure_taxonomy() -> None:
    calibration = _calibration()
    calibration["calibration_failure_taxonomy_complete"] = False
    audit = audit_gate_b_prerequisites(_contract(), calibration, _validation())
    assert audit.passed is False
    assert any("failure taxonomy" in failure for failure in audit.failures)


def test_fixed_baseline_rejects_censored_or_legacy_endurance_semantics() -> None:
    validation = _validation()
    validation.update(
        {
            "schema": "fixed_baseline_battery_validation_v1",
            "all_runs_depleted": False,
            "continuous_workload_until_depletion": False,
            "censored_run_count": 41,
        }
    )
    audit = audit_gate_b_prerequisites(_contract(), _calibration(), validation)
    assert audit.passed is False
    assert audit.battery_validation_schema == "fixed_baseline_battery_validation_v1"
    assert audit.endurance_estimand == CONTINUOUS_ENDURANCE_ESTIMAND
    assert audit.continuous_workload_until_depletion is False
    assert audit.censored_run_count == 41
    assert audit.battery_validation_runs == 100
    assert any("continuous-workload v2 schema" in failure for failure in audit.failures)
    assert any("workload continuity" in failure for failure in audit.failures)
    assert any("censored endurance runs" in failure for failure in audit.failures)


def test_contract_builder_and_calibration_attestation_are_hash_bound(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "R3"
    checkpoint = artifact / "phase1_navigation" / "checkpoint_transition_500000.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"R3-frozen-policy")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_path = artifact / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    source_navigation = {
        "variant": "R3",
        "checkpoint_sha256": checkpoint_sha,
        "navigation_gate_passed": False,
        "final_navigation": _navigation_metrics(),
    }
    source_navigation_path = artifact / "EVALUATION_COMPLETED.json"
    source_navigation_path.write_text(
        json.dumps(source_navigation) + "\n",
        encoding="utf-8",
    )
    contract = build_fixed_baseline_contract(
        source_navigation=source_navigation,
        source_navigation_path=source_navigation_path,
        checkpoint=checkpoint,
        artifact=artifact,
        artifact_config_path=config_path,
    )
    contract_path = tmp_path / "navigation_platform_contract.json"
    contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
    calibration = _calibration()
    source_calibration_path = tmp_path / "source_calibration.json"
    source_calibration_path.write_text(
        json.dumps(calibration) + "\n",
        encoding="utf-8",
    )
    attested = attest_fixed_baseline_calibration(
        source_calibration=calibration,
        source_calibration_path=source_calibration_path,
        contract=contract,
        contract_path=contract_path,
    )
    assert attested["fixed_baseline_calibration_evaluable"] is True
    assert attested["legacy_battery_calibration_navigation_valid"] is False
    assert attested["calibration_end_reason_counts"] == {
        "goal_reached": 475,
        "task_step_limit": 25,
    }
    assert attested["navigation_checkpoint_sha256"] == checkpoint_sha


def test_validation_batches_cover_every_independent_run_once() -> None:
    batches = []
    for start in (0, 2):
        records = [
            {
                "run_index": index,
                "actual_depletion_time": 1800.0,
                "actual_policy_steps_to_depletion": 9000,
                "tasks_before_depletion": 10,
                "distance_before_depletion": 1000.0,
                "task_step_limit_rollovers_before_depletion": 1,
                "episode_guard_exceeded_before_depletion": False,
                "continuous_workload_until_depletion": True,
                "energy_exhausted": True,
            }
            for index in range(start, start + 2)
        ]
        batches.append(
            {
                "batch_start": start,
                "batch_end_exclusive": start + 2,
                "batch_validation_sha256": str(start),
                "battery_validation_env_transitions": 18_000,
                "execution": {"wall_clock_seconds": 1.0},
                "records": records,
            }
        )
    result = aggregate_validation_batches(
        batch_payloads=batches,
        expected_runs=4,
        battery_capacity=360.0,
        target_minutes=30.0,
    )
    assert result["battery_validation_runs"] == 4
    assert result["all_runs_depleted"] is True
    assert result["continuous_workload_until_depletion"] is True
    assert result["censored_run_count"] == 0
    assert result["task_step_limit_rollovers"] == 4
    assert result["battery_calibration_valid"] is True
    assert result["relative_endurance_error"] == pytest.approx(0.0)


def test_validation_batches_reject_missing_run_ids() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        aggregate_validation_batches(
            batch_payloads=[
                {
                    "battery_validation_env_transitions": 1,
                    "records": [
                        {
                            "run_index": 1,
                            "actual_depletion_time": 1800.0,
                            "actual_policy_steps_to_depletion": 9000,
                            "tasks_before_depletion": 10,
                            "distance_before_depletion": 1000.0,
                            "continuous_workload_until_depletion": True,
                            "energy_exhausted": True,
                        }
                    ],
                }
            ],
            expected_runs=1,
            battery_capacity=360.0,
            target_minutes=30.0,
        )


def _write_failed_continuous_validation_pair(
    tmp_path: Path,
) -> tuple[Path, Path]:
    calibration = _calibration()
    calibration.update(
        {
            "schema": "fixed_baseline_battery_calibration_attestation_v1",
            "baseline_id": "R3",
            "td_training": False,
            "td_replay_writes": 0,
        }
    )
    calibration_path = tmp_path / "initial_calibration.json"
    calibration_path.write_text(json.dumps(calibration) + "\n", encoding="utf-8")
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    records = [
        {
            "run_index": index,
            "run_seed": 90_001 + index,
            "actual_depletion_time": 2250.0,
            "continuous_workload_until_depletion": True,
            "energy_exhausted": True,
        }
        for index in range(100)
    ]
    validation = {
        "schema": FIXED_BASELINE_VALIDATION_SCHEMA,
        "baseline_id": "R3",
        "endurance_estimand": CONTINUOUS_ENDURANCE_ESTIMAND,
        "continuous_workload_until_depletion": True,
        "all_runs_depleted": True,
        "censored_run_count": 0,
        "battery_validation_runs": 100,
        "battery_calibration_valid": False,
        "battery_calibration_sha256": calibration_sha,
        "battery_capacity": 360.0,
        "target_nominal_endurance_seconds": 1800.0,
        "target_nominal_endurance_minutes": 30.0,
        "engineering_calibration_tolerance_fraction": 0.20,
        "mean_depletion_time": 2250.0,
        "relative_endurance_error": 0.25,
        "records": records,
    }
    validation_path = tmp_path / "failed_validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    return calibration_path, validation_path


def test_failed_endurance_is_repurposed_with_disjoint_confirmation(
    tmp_path: Path,
) -> None:
    calibration_path, validation_path = _write_failed_continuous_validation_pair(
        tmp_path
    )
    refined = build_refined_calibration(
        initial_calibration_path=calibration_path,
        failed_validation_path=validation_path,
        confirmatory_seed_start=91_001,
    )
    assert refined["schema"] == FIXED_BASELINE_RECALIBRATION_SCHEMA
    assert refined["source_validation_role"] == (
        "capacity_refinement_only_not_confirmation"
    )
    assert refined["calibrated_battery_capacity"] == pytest.approx(288.0)
    assert refined["confirmatory_validation_seed_start"] == 91_001
    assert refined["confirmatory_validation_seed_end_inclusive"] == 91_100
    assert audit_refined_calibration_provenance(
        refined,
        validation_seed_start=91_001,
    ) == []


def test_refined_calibration_rejects_seed_reuse_and_runtime_split_drift(
    tmp_path: Path,
) -> None:
    calibration_path, validation_path = _write_failed_continuous_validation_pair(
        tmp_path
    )
    with pytest.raises(ValueError, match="overlaps"):
        build_refined_calibration(
            initial_calibration_path=calibration_path,
            failed_validation_path=validation_path,
            confirmatory_seed_start=90_050,
        )
    refined = build_refined_calibration(
        initial_calibration_path=calibration_path,
        failed_validation_path=validation_path,
        confirmatory_seed_start=91_001,
    )
    failures = audit_refined_calibration_provenance(
        refined,
        validation_seed_start=92_001,
    )
    assert any("preregistered split" in failure for failure in failures)


def test_refined_calibration_audit_recomputes_capacity_formula(
    tmp_path: Path,
) -> None:
    calibration_path, validation_path = _write_failed_continuous_validation_pair(
        tmp_path
    )
    refined = build_refined_calibration(
        initial_calibration_path=calibration_path,
        failed_validation_path=validation_path,
        confirmatory_seed_start=91_001,
    )
    refined["calibrated_battery_capacity"] = 300.0
    failures = audit_refined_calibration_provenance(
        refined,
        validation_seed_start=91_001,
    )
    assert any("declared formula" in failure for failure in failures)
