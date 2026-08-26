from __future__ import annotations

import numpy as np
import pytest

from experiments.energy_mc.gate_b_prerequisites import audit_gate_b_prerequisites
from experiments.energy_mc.return_decision import (
    ReturnDecisionOutcome,
    boundary_weighted_underestimation,
    clone_rollout_variance_audit,
    oracle_headroom_gate,
    pareto_frontier,
    wilson_interval,
)
from scripts.run_return_decision_stage_b import (
    aggregate_seed_audits,
    derive_successful_energy_per_meter,
    environment_from_args,
    evaluate_parameter_across_seeds,
    load_passed_oracle_headroom_gate,
    load_policy,
    parse_args,
    stage_b_completion_contract,
)
from scripts.evaluate_jseb_checkpoints import (
    compact_result,
    parse_args as parse_checkpoint_evaluator_args,
    select_checkpoints,
)
from scripts.calibrate_battery_for_navigation_checkpoint import (
    navigation_readiness_failures,
)


def test_clone_rollout_audit_rejects_fake_distributional_claim() -> None:
    audit = clone_rollout_variance_audit(lambda seed: 12.5, num_rollouts=20)
    assert audit.variance == pytest.approx(0.0)
    assert audit.q50 == audit.q95 == pytest.approx(12.5)
    assert audit.nondegenerate_conditional_distribution is False
    assert audit.distributional_headline_supported is False


def test_clone_rollout_audit_detects_remaining_conditional_randomness() -> None:
    audit = clone_rollout_variance_audit(
        lambda seed: 10.0 + (-1.0 if seed % 2 else 1.0),
        num_rollouts=20,
    )
    assert audit.variance > 0.0
    assert audit.unique_values_at_tolerance == 2
    assert audit.nondegenerate_conditional_distribution is True


def test_pareto_frontier_keeps_only_left_upper_nondominated_points() -> None:
    def outcome(method: str, stranding: float, throughput: float) -> ReturnDecisionOutcome:
        return ReturnDecisionOutcome(method, 0.1, stranding, throughput, 2.0, 1.0 - stranding, 0.1)

    frontier = pareto_frontier(
        [
            outcome("safe", 0.0, 4.0),
            outcome("dominated", 0.1, 3.0),
            outcome("middle", 0.1, 5.0),
            outcome("fast", 0.2, 6.0),
        ]
    )
    assert [item.method for item in frontier] == ["safe", "middle", "fast"]


def test_boundary_weighting_prioritizes_errors_near_switch_surface() -> None:
    value = boundary_weighted_underestimation(
        remaining_energy=np.asarray([10.0, 100.0]),
        true_required_energy=np.asarray([10.0, 10.0]),
        predicted_required_energy=np.asarray([5.0, 0.0]),
        bandwidth=2.0,
    )
    assert value == pytest.approx(5.0, rel=1e-6)


def test_wilson_interval_does_not_call_zero_failures_proven_safe() -> None:
    lower, upper = wilson_interval(0, 100)
    assert lower == pytest.approx(0.0)
    assert 0.03 < upper < 0.04


def test_oracle_headroom_gate_requires_cycles_and_common_safe_frontier() -> None:
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.01, 100.0, 3.0, 0.99, 0.2),
        ReturnDecisionOutcome("distance", 0.1, 0.10, 110.0, 3.2, 0.90, 0.1),
        ReturnDecisionOutcome("oracle", 0.1, 0.01, 110.0, 3.3, 0.99, 0.1),
    ]
    pending = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 2 for item in outcomes},
    )
    assert pending.status == "PENDING_INSUFFICIENT_CYCLES"
    passed = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 100 for item in outcomes},
        minimum_throughput_gain_fraction=0.05,
    )
    assert passed.status == "PASS"
    assert passed.throughput_gain_fraction == pytest.approx(0.10)


def test_oracle_gate_uses_wilson_upper_bound_not_zero_failure_point_estimate() -> None:
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.0, 100.0, 3.0, 1.0, 0.2),
        ReturnDecisionOutcome("oracle", 0.1, 0.0, 110.0, 3.3, 1.0, 0.1),
    ]
    gate = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 100 for item in outcomes},
        stranding_upper_bounds={
            ("soc", 0.2): 0.06,
            ("oracle", 0.1): 0.04,
        },
    )
    assert gate.status == "FAIL_NO_COMPARABLE_SAFE_FRONTIER"


def test_seed_aggregation_counts_cycles_and_transitions_exactly() -> None:
    common = {
        "method": "oracle",
        "parameter": 0.05,
        "attempted_cycles": 50,
        "successful_returns": 50,
        "energy_exhaustions": 0,
        "emergency_guards": 0,
        "tasks_completed": 150,
        "simulated_seconds": 1800.0,
        "policy_steps": 9000,
        "unused_energy_fraction_sum": 5.0,
        "hocbf_interventions": 100,
        "obstacle_collision_count": 0,
    }
    seed_audits = [
        {**common, "evaluation_seed": 0},
        {**common, "evaluation_seed": 1},
    ]
    outcome, aggregate = aggregate_seed_audits(
        method="oracle",
        parameter=0.05,
        seed_audits=seed_audits,
    )
    assert aggregate["attempted_cycles"] == 100
    assert aggregate["policy_steps"] == 18_000
    assert aggregate["evaluation_seeds"] == [0, 1]
    assert outcome.tasks_per_simulated_hour == pytest.approx(300.0)
    assert outcome.tasks_per_battery_cycle == pytest.approx(3.0)
    assert outcome.mean_unused_energy_fraction_at_charger == pytest.approx(0.1)
    assert 0.03 < aggregate["stranding_rate_wilson95_upper"] < 0.04


def test_formal_defaults_provide_exact_minimum_gate_cycles() -> None:
    args = parse_args(["--output-dir", "/tmp/stage_b_gate_test"])
    assert args.cycles_per_point == 1
    assert args.evaluation_seeds == list(range(110_001, 110_101))
    assert (
        args.cycles_per_point * len(args.evaluation_seeds)
        == args.minimum_cycles_for_gate
        == 100
    )
    assert args.battery_capacity is None


def test_checkpoint_evaluator_defaults_to_parallel_deterministic_workers() -> None:
    args = parse_checkpoint_evaluator_args(["--artifact", "/tmp/artifact"])
    assert args.evaluation_num_envs == 6
    assert args.evaluation_progress_interval_tasks == 25


def test_checkpoint_evaluator_rejects_nonpositive_worker_count() -> None:
    with pytest.raises(SystemExit):
        parse_checkpoint_evaluator_args(
            ["--artifact", "/tmp/artifact", "--evaluation-num-envs", "0"]
        )


def test_checkpoint_evaluator_distinguishes_completion_from_gate_pass(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint_transition_500000.zip"
    checkpoint.write_bytes(b"checkpoint")
    result = {
        "num_tasks": 500,
        "global_env_transitions": 500_000,
        "evaluation_env_transitions": 10_000,
        "overall_success_rate": 0.90,
        "distance_bucket_success": {
            "100-500": 1.0,
            "500-1500": 0.90,
            "1500-2500": 0.90,
            "2500-4000": 0.90,
            ">4000": 0.80,
        },
        "mean_path_ratio": 1.4,
        "obstacle_collision_steps": 0,
        "obstacle_collision_episode_rate": 0.0,
        "boundary_contact_step_rate": 0.0,
        "hocbf_intervention_step_rate": 0.2,
        "hocbf_emergency_brake_step_rate": 0.01,
        "nominal_safe_action_rate": 0.8,
        "projection_valid_step_rate": 1.0,
    }
    row = compact_result(500_000, checkpoint, result, elapsed=1.0)
    assert row["navigation_energy_gate_passed"] is False
    assert row["navigation_safety_gate_passed"] is True
    assert row["navigation_gate_passed"] is False
    assert row["navigation_gate_failures"] == [
        "navigation_energy_readiness_gate_failed"
    ]


def test_formal_stage_b_rejects_clustered_cycles_as_independent_wilson_units(
    tmp_path,
) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path / "clustered"),
                "--cycles-per-point",
                "20",
                "--evaluation-seeds",
                "0",
                "1",
                "2",
                "3",
                "4",
            ]
        )


def test_post_gate_td_comparison_requires_explicit_passed_gate(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path / "invalid"),
                "--methods",
                "td_frozen",
            ]
        )
    passed_path = tmp_path / "oracle_gate.json"
    passed_path.write_text(
        '{"status":"PASS","evaluable":true,"passed":true}\n',
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "valid"),
            "--methods",
            "td_frozen",
            "td_online",
            "--oracle-headroom-json",
            str(passed_path),
        ]
    )
    assert args.methods == ["td_frozen", "td_online"]
    assert load_passed_oracle_headroom_gate(passed_path)["status"] == "PASS"


def test_post_gate_comparison_rejects_nonpassing_gate(tmp_path) -> None:
    failed_path = tmp_path / "oracle_gate.json"
    failed_path.write_text(
        '{"status":"FAIL_INSUFFICIENT_ORACLE_HEADROOM",'
        '"evaluable":true,"passed":false}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="requires an evaluable PASS"):
        load_passed_oracle_headroom_gate(failed_path)


def test_stage_b_parallel_seed_evaluation_preserves_exact_cycle_accounting(tmp_path) -> None:
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path / "parallel_stage_b"),
            "--smoke",
            "--evaluation-num-envs",
            "2",
        ]
    )
    args.evaluation_seeds = [301, 302]
    args.cycles_per_point = 1
    args.minimum_cycles_for_gate = 2
    probe = environment_from_args(args, reserve_fraction=0.0)
    policy = load_policy(args, probe)
    probe.close()
    results = evaluate_parameter_across_seeds(
        args,
        policy,
        method="soc",
        parameter=0.20,
    )
    assert len(results) == 2
    audits = [audit for _, audit, _ in results]
    assert [audit["evaluation_seed"] for audit in audits] == [301, 302]
    assert sum(audit["attempted_cycles"] for audit in audits) == 2
    assert all(audit["attempted_cycles"] == 1 for audit in audits)


def test_formal_oracle_gate_failure_stops_downstream_without_failed_artifact() -> None:
    args = parse_args(["--output-dir", "/tmp/formal_stage_b_contract"])
    sentinel, exit_code, status = stage_b_completion_contract(
        args,
        {
            "status": "FAIL_INSUFFICIENT_ORACLE_HEADROOM",
            "evaluable": True,
            "passed": False,
        },
    )
    assert sentinel == "STOPPED_AFTER_ORACLE_HEADROOM_GATE.json"
    assert exit_code == 4
    assert status == "FAIL_INSUFFICIENT_ORACLE_HEADROOM"


def test_passed_and_post_gate_stage_b_contracts_complete(tmp_path) -> None:
    formal = parse_args(["--output-dir", str(tmp_path / "formal")])
    assert stage_b_completion_contract(
        formal,
        {"status": "PASS", "evaluable": True, "passed": True},
    ) == ("COMPLETED.json", 0, "ORACLE_HEADROOM_GATE_PASS")
    inherited_path = tmp_path / "gate.json"
    post_gate = parse_args(
        [
            "--output-dir",
            str(tmp_path / "post_gate"),
            "--methods",
            "td_frozen",
            "--oracle-headroom-json",
            str(inherited_path),
        ]
    )
    assert stage_b_completion_contract(
        post_gate,
        {"status": "PASS", "evaluable": True, "passed": True},
    ) == ("COMPLETED.json", 0, "POST_ORACLE_HEADROOM_DECISION_COMPARISON")


def valid_gate_b_prerequisites() -> tuple[dict, dict, dict]:
    buckets = {
        "100-500": 0.99,
        "500-1500": 0.99,
        "1500-2500": 0.99,
        "2500-4000": 0.99,
        ">4000": 0.99,
    }
    navigation = {
        "num_tasks": 500,
        "overall_success_rate": 0.99,
        "distance_bucket_success": buckets,
        "mean_path_ratio": 1.05,
        "boundary_contact_step_rate": 0.005,
        "obstacle_collision_steps": 0,
    }
    calibration = {
        "num_tasks": 500,
        "battery_calibration_navigation_valid": True,
        "calibration_success_rate": 0.99,
        "calibration_distance_bucket_success": buckets,
        "policy_unchanged": True,
        "sac_training": False,
        "td_replay_writes": 0,
        "energy_source": "TelemetryCostModel.realized_cost",
        "calibrated_battery_capacity": 350.0,
        "target_nominal_endurance_minutes": 30.0,
    }
    validation = {
        "battery_validation_runs": 100,
        "battery_calibration_valid": True,
        "all_runs_depleted": True,
        "engineering_calibration_tolerance_fraction": 0.20,
        "relative_endurance_error": 0.05,
        "battery_capacity": 350.0,
        "mean_depletion_time": 1890.0,
        "target_nominal_endurance_minutes": 30.0,
    }
    return navigation, calibration, validation


def test_gate_b_prerequisites_pass_only_complete_valid_evidence() -> None:
    audit = audit_gate_b_prerequisites(*valid_gate_b_prerequisites())
    assert audit.passed is True
    assert audit.status == "PASS"
    assert audit.calibrated_battery_capacity == pytest.approx(350.0)


def test_gate_b_prerequisites_reject_failed_navigation_and_censored_endurance() -> None:
    navigation, calibration, validation = valid_gate_b_prerequisites()
    calibration["battery_calibration_navigation_valid"] = False
    calibration["calibration_success_rate"] = 0.80
    validation["battery_calibration_valid"] = False
    validation["all_runs_depleted"] = False
    audit = audit_gate_b_prerequisites(navigation, calibration, validation)
    assert audit.passed is False
    assert any("calibration navigation validity" in item for item in audit.failures)
    assert any("censored/task-limit" in item for item in audit.failures)


def test_checkpoint_selection_can_run_only_formal_500k_policy(tmp_path) -> None:
    checkpoints = [
        (100_000, tmp_path / "checkpoint_transition_100000.zip"),
        (500_000, tmp_path / "checkpoint_transition_500000.zip"),
    ]
    selected = select_checkpoints(checkpoints, [500_000])
    assert selected == [checkpoints[1]]
    with pytest.raises(FileNotFoundError):
        select_checkpoints(checkpoints, [450_000])


def test_battery_calibration_runner_rejects_nonready_navigation() -> None:
    navigation, _, _ = valid_gate_b_prerequisites()
    navigation["global_env_transitions"] = 500_000
    assert navigation_readiness_failures(navigation) == []
    navigation["overall_success_rate"] = 0.80
    navigation["boundary_contact_step_rate"] = 0.02
    failures = navigation_readiness_failures(navigation)
    assert "navigation energy/readiness gate failed" in failures
    assert "navigation collision/boundary gate failed" in failures


def test_distance_baseline_uses_successful_calibration_energy_per_path_meter() -> None:
    calibration = {
        "tasks": [
            {
                "success": True,
                "total_realized_energy": 10.0,
                "actual_path_length": 100.0,
            },
            {
                "success": True,
                "total_realized_energy": 30.0,
                "actual_path_length": 300.0,
            },
            {
                "success": False,
                "total_realized_energy": 1000.0,
                "actual_path_length": 1.0,
            },
        ]
    }
    assert derive_successful_energy_per_meter(calibration) == pytest.approx(0.1)
