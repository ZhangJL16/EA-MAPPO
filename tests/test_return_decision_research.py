from __future__ import annotations

import hashlib
import json
from argparse import Namespace

import numpy as np
import pytest

import scripts.run_return_decision_stage_b as stage_b
from scripts.diagnose_return_decision_stage_b import diagnostic_args
from scripts.analyze_return_rollout_stalls import summarize_results as summarize_stalls

from experiments.energy_mc.gate_b_prerequisites import (
    audit_gate_b_prerequisites,
    navigation_artifact_view,
)
from experiments.energy_mc.return_decision import (
    ReturnDecisionOutcome,
    audit_oracle_shadow_coupling,
    boundary_weighted_underestimation,
    clone_rollout_variance_audit,
    oracle_headroom_gate,
    paired_frontier_throughput_interval,
    pareto_frontier,
    probability_semantics_gate,
    simultaneous_stranding_upper_bounds,
    stranding_certification_power_audit,
    wilson_interval,
)
from scripts.run_return_decision_stage_b import (
    _linux_thread_count,
    aggregate_seed_audits,
    attach_paired_oracle_outcomes,
    derive_successful_energy_per_meter,
    environment_from_args,
    evaluate_parameter_across_seeds,
    formal_design_manifest,
    load_passed_oracle_headroom_gate,
    load_policy,
    load_prerequisite_audit,
    main as run_stage_b_main,
    parse_args,
    stage_b_episode_guard_contract,
    stage_b_completion_contract,
    write_results,
)
from scripts.evaluate_jseb_checkpoints import (
    compact_result,
    normalize_environment_reconstruction_command,
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


def test_probability_gate_fails_unexplained_clone_nondeterminism() -> None:
    audit = clone_rollout_variance_audit(
        lambda repetition: 10.0 + float(repetition % 2),
        num_rollouts=20,
    )
    gate = probability_semantics_gate(
        audit,
        physically_specified_future_disturbance=False,
    )
    assert gate["status"] == "FAIL_UNEXPECTED_CONDITIONAL_NONDETERMINISM"
    assert gate["passed"] is False
    assert gate["aleatoric_q90_q95_claim_authorized"] is False


def test_probability_gate_does_not_let_variance_replace_calibration() -> None:
    audit = clone_rollout_variance_audit(
        lambda repetition: 10.0 + float(repetition % 2),
        num_rollouts=20,
    )
    gate = probability_semantics_gate(
        audit,
        physically_specified_future_disturbance=True,
    )
    assert gate["status"] == "PENDING_STOCHASTIC_PROCESS_CALIBRATION"
    assert gate["passed"] is False
    assert gate["aleatoric_q90_q95_claim_authorized"] is False


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


def test_simultaneous_stranding_bound_exposes_100_cycle_selection_gap() -> None:
    audit_100 = simultaneous_stranding_upper_bounds(
        {(f"method-{index}", 0.0): (0, 100) for index in range(12)}
    )
    audit_110 = simultaneous_stranding_upper_bounds(
        {(f"method-{index}", 0.0): (0, 110) for index in range(12)}
    )
    assert audit_100.num_candidate_points == 12
    assert audit_100.points[0].upper_confidence_bound > 0.05
    assert audit_110.points[0].upper_confidence_bound < 0.05
    assert audit_110.correction == "bonferroni_one_sided_clopper_pearson"


def test_single_point_exact_upper_matches_zero_event_formula() -> None:
    audit = simultaneous_stranding_upper_bounds({("oracle", 0.0): (0, 100)})
    expected = 1.0 - 0.05 ** (1.0 / 100)
    assert audit.points[0].upper_confidence_bound == pytest.approx(expected)


def test_stranding_certification_power_rejects_minimal_zero_event_design() -> None:
    audit_110 = stranding_certification_power_audit(
        num_candidate_points=12,
        planned_independent_cycles_per_point=110,
    )
    audit_320 = stranding_certification_power_audit(
        num_candidate_points=12,
        planned_independent_cycles_per_point=320,
    )
    assert audit_110.minimum_cycles_for_zero_event_evaluability == 107
    assert audit_110.maximum_certifiable_stranding_count == 0
    assert audit_110.planned_certification_power == pytest.approx(0.99**110)
    assert audit_110.passed is False
    assert audit_110.planned_joint_certification_power_lower_bound == 0.0
    assert audit_320.maximum_certifiable_stranding_count == 6
    assert audit_320.planned_certification_power == pytest.approx(
        0.9562346748821128
    )
    assert audit_320.planned_joint_certification_power_lower_bound == pytest.approx(
        0.9124693497642256
    )
    assert audit_320.minimum_cycles_for_target_power == 314
    assert audit_320.target_certification_power == pytest.approx(0.95)
    assert audit_320.passed is True


def test_two_percent_stranding_requires_larger_powered_design() -> None:
    audit = stranding_certification_power_audit(
        num_candidate_points=12,
        planned_independent_cycles_per_point=320,
        design_stranding_rate=0.02,
    )
    assert audit.planned_certification_power == pytest.approx(0.5416900507757205)
    assert audit.minimum_cycles_for_target_power == 694
    assert audit.passed is False


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
    pending_inference = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        minimum_throughput_gain_fraction=0.05,
    )
    assert pending_inference.status == "PENDING_THROUGHPUT_INFERENCE"
    assert pending_inference.evaluable is False
    passed = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        minimum_throughput_gain_fraction=0.05,
        throughput_gain_lower_confidence_bound=0.08,
        throughput_gain_upper_confidence_bound=0.12,
        throughput_confidence_level=0.95,
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
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        stranding_upper_bounds={
            ("soc", 0.2): 0.06,
            ("oracle", 0.1): 0.04,
        },
    )
    assert gate.status == "FAIL_NO_COMPARABLE_SAFE_FRONTIER"


def test_paired_frontier_bootstrap_resamples_schedule_and_selection_together() -> None:
    records = []
    for schedule in range(20):
        common = {
            "paired_schedule_id": f"schedule-{schedule}",
            "simulated_seconds": 3600.0,
        }
        records.extend(
            [
                {
                    **common,
                    "method": "oracle",
                    "parameter": 0.0,
                    "tasks_completed": 11,
                },
                {
                    **common,
                    "method": "oracle",
                    "parameter": 0.1,
                    "tasks_completed": 9,
                },
                {
                    **common,
                    "method": "soc",
                    "parameter": 0.2,
                    "tasks_completed": 10,
                },
            ]
        )
    interval = paired_frontier_throughput_interval(
        records,
        oracle_points=[("oracle", 0.0), ("oracle", 0.1)],
        heuristic_points=[("soc", 0.2)],
        bootstrap_replicates=2000,
        bootstrap_seed=7,
    )
    assert interval.num_paired_schedules == 20
    assert interval.selected_oracle_point == "oracle|0"
    assert interval.throughput_gain_fraction == pytest.approx(0.10)
    assert interval.throughput_gain_lower_confidence_bound == pytest.approx(0.10)
    assert interval.throughput_gain_upper_confidence_bound == pytest.approx(0.10)
    assert interval.max_t_critical_value == pytest.approx(0.0)
    assert interval.upper_max_t_critical_value == pytest.approx(0.0)
    assert "max_t_rate_band" in interval.inference


def test_full_family_max_t_band_survives_safety_set_selection() -> None:
    records = []
    for schedule in range(60):
        values = {
            ("oracle", 0.0): 11 + schedule % 2,
            ("oracle", 0.1): 11 + (schedule * 7) % 5,
            ("soc", 0.2): 10 + (schedule * 3) % 4,
            ("distance", 0.0): 10 + schedule % 3,
        }
        for (method, parameter), tasks in values.items():
            records.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "paired_schedule_id": f"schedule-{schedule}",
                    "tasks_completed": tasks,
                    "simulated_seconds": 3600.0,
                }
            )
    interval = paired_frontier_throughput_interval(
        records,
        oracle_points=[("oracle", 0.0), ("oracle", 0.1)],
        heuristic_points=[("soc", 0.2), ("distance", 0.0)],
        eligible_oracle_points=[("oracle", 0.0)],
        eligible_heuristic_points=[("distance", 0.0)],
        bootstrap_replicates=5000,
        bootstrap_seed=7,
    )
    assert interval.oracle_points == ("oracle|0", "oracle|0.1")
    assert interval.eligible_oracle_points == ("oracle|0",)
    assert interval.throughput_gain_fraction == pytest.approx(0.04545454545)
    assert interval.throughput_gain_lower_confidence_bound == pytest.approx(
        0.01175598318
    )
    assert (
        interval.throughput_gain_lower_confidence_bound
        < interval.descriptive_percentile_gain_lower_bound
    )
    assert interval.throughput_gain_upper_confidence_bound is not None
    assert (
        interval.throughput_gain_upper_confidence_bound
        > interval.throughput_gain_fraction
    )


def test_paired_frontier_bootstrap_rejects_unmatched_schedule_sets() -> None:
    records = [
        {
            "method": "oracle",
            "parameter": 0.0,
            "paired_schedule_id": "a",
            "tasks_completed": 2,
            "simulated_seconds": 1.0,
        },
        {
            "method": "soc",
            "parameter": 0.2,
            "paired_schedule_id": "b",
            "tasks_completed": 1,
            "simulated_seconds": 1.0,
        },
    ]
    with pytest.raises(ValueError, match="share exactly the same schedules"):
        paired_frontier_throughput_interval(
            records,
            oracle_points=[("oracle", 0.0)],
            heuristic_points=[("soc", 0.2)],
            bootstrap_replicates=1000,
        )


def test_oracle_shadow_coupling_is_exact_through_first_disagreement() -> None:
    def event(
        *,
        method: str,
        parameter: float,
        decision_index: int,
        step: int,
        method_commit: bool,
        oracle_commit: bool,
        first: bool,
    ) -> dict[str, object]:
        return {
            "method": method,
            "parameter": parameter,
            "paired_schedule_id": "schedule-0",
            "reserve_fraction": 0.0,
            "decision_index": decision_index,
            "global_step": step,
            "information_time": "step_start_pre_action",
            "task_index_in_cycle": 0,
            "position": [float(step), 0.0, 0.0],
            "velocity": [1.0, 0.0, 0.0],
            "task_goal": [10.0, 0.0, 0.0],
            "remaining_energy": 10.0 - step,
            "exact_effective_requirement": 2.0 + step,
            "exact_effective_requirement_is_infinite": False,
            "exact_return_now_deadline_feasible": True,
            "exact_task_then_return_deadline_feasible": True,
            "oracle_commit": oracle_commit,
            "method_commit": method_commit,
            "first_disagreement": first,
            "first_disagreement_direction": (
                "method_late_continue_oracle_commit" if first else None
            ),
        }

    records = [
        event(
            method="soc",
            parameter=0.2,
            decision_index=0,
            step=1,
            method_commit=False,
            oracle_commit=False,
            first=False,
        ),
        event(
            method="soc",
            parameter=0.2,
            decision_index=1,
            step=2,
            method_commit=False,
            oracle_commit=True,
            first=True,
        ),
        event(
            method="oracle",
            parameter=0.0,
            decision_index=0,
            step=1,
            method_commit=False,
            oracle_commit=False,
            first=False,
        ),
        event(
            method="oracle",
            parameter=0.0,
            decision_index=1,
            step=2,
            method_commit=True,
            oracle_commit=True,
            first=False,
        ),
    ]
    audit = audit_oracle_shadow_coupling(records)
    assert audit.status == "PASS"
    assert audit.num_method_schedule_pairs == 1
    assert audit.num_compared_decision_events == 2
    assert audit.num_first_disagreements == 1
    assert audit.maximum_position_error == pytest.approx(0.0)

    perturbed = [dict(record) for record in records]
    perturbed[0]["position"] = [1.1, 0.0, 0.0]
    failed = audit_oracle_shadow_coupling(perturbed)
    assert failed.status == "FAIL_ORACLE_SHADOW_COUPLING"
    assert failed.maximum_position_error == pytest.approx(0.1)


def test_oracle_gate_rejects_uncertain_point_estimate_headroom() -> None:
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.01, 100.0, 3.0, 0.99, 0.2),
        ReturnDecisionOutcome("oracle", 0.0, 0.01, 110.0, 3.3, 0.99, 0.1),
    ]
    gate = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        throughput_gain_lower_confidence_bound=0.01,
        throughput_gain_upper_confidence_bound=0.15,
        throughput_confidence_level=0.95,
    )
    assert gate.throughput_gain_fraction == pytest.approx(0.10)
    assert gate.throughput_gain_lower_confidence_bound == pytest.approx(0.01)
    assert gate.throughput_gain_upper_confidence_bound == pytest.approx(0.15)
    assert gate.status == "INCONCLUSIVE_ORACLE_HEADROOM"
    assert gate.passed is False


def test_oracle_gate_does_not_turn_low_point_estimate_into_scientific_fail() -> None:
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.01, 100.0, 3.0, 0.99, 0.2),
        ReturnDecisionOutcome("oracle", 0.0, 0.01, 103.0, 3.1, 0.99, 0.1),
    ]
    gate = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        throughput_gain_lower_confidence_bound=-0.01,
        throughput_gain_upper_confidence_bound=0.08,
        throughput_confidence_level=0.95,
    )
    assert gate.throughput_gain_fraction == pytest.approx(0.03)
    assert gate.status == "INCONCLUSIVE_ORACLE_HEADROOM"
    assert gate.passed is False


def test_oracle_gate_scientific_fail_requires_upper_bound_below_threshold() -> None:
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.01, 100.0, 3.0, 0.99, 0.2),
        ReturnDecisionOutcome("oracle", 0.0, 0.01, 103.0, 3.1, 0.99, 0.1),
    ]
    gate = oracle_headroom_gate(
        outcomes,
        cycles_per_point={(item.method, item.parameter): 300 for item in outcomes},
        throughput_gain_lower_confidence_bound=-0.01,
        throughput_gain_upper_confidence_bound=0.04,
        throughput_confidence_level=0.95,
    )
    assert gate.status == "FAIL_INSUFFICIENT_ORACLE_HEADROOM"
    assert gate.passed is False


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
        {
            **common,
            "evaluation_seed": 0,
            "exact_return_now_deadline_infeasible_event_count": 2,
            "exact_task_then_return_deadline_infeasible_event_count": 3,
            "estimated_return_now_deadline_infeasible_event_count": 1,
            "estimated_task_then_return_deadline_infeasible_event_count": 4,
        },
        {
            **common,
            "evaluation_seed": 1,
            "exact_return_now_deadline_infeasible_event_count": 5,
            "exact_task_then_return_deadline_infeasible_event_count": 7,
            "estimated_return_now_deadline_infeasible_event_count": 6,
            "estimated_task_then_return_deadline_infeasible_event_count": 8,
        },
    ]
    outcome, aggregate = aggregate_seed_audits(
        method="oracle",
        parameter=0.05,
        seed_audits=seed_audits,
    )
    assert aggregate["attempted_cycles"] == 100
    assert aggregate["policy_steps"] == 18_000
    assert aggregate["evaluation_seeds"] == [0, 1]
    assert aggregate["exact_return_now_deadline_infeasible_event_count"] == 7
    assert (
        aggregate["exact_task_then_return_deadline_infeasible_event_count"]
        == 10
    )
    assert (
        aggregate["estimated_return_now_deadline_infeasible_event_count"]
        == 7
    )
    assert (
        aggregate["estimated_task_then_return_deadline_infeasible_event_count"]
        == 12
    )
    assert outcome.tasks_per_simulated_hour == pytest.approx(300.0)
    assert outcome.tasks_per_battery_cycle == pytest.approx(3.0)
    assert outcome.mean_unused_energy_fraction_at_charger == pytest.approx(0.1)
    assert 0.03 < aggregate["stranding_rate_wilson95_upper"] < 0.04


def test_cycle_pairing_requires_same_keyed_schedule_and_reserve() -> None:
    common = {
        "parameter": 0.05,
        "evaluation_seed": 11,
        "paired_schedule_id": "keyed-task-v1:11:cycle:0",
        "cycle_index": 0,
        "return_success": True,
        "stranded": False,
        "energy_exhausted": False,
        "tasks_completed": 3,
        "simulation_seconds": 100.0,
        "policy_steps": 500,
        "unused_energy_fraction_at_charger": 0.1,
        "reserve_fraction": 0.05,
        "paired_oracle_return_success": None,
        "paired_oracle_stranded": None,
        "paired_oracle_tasks_completed": None,
    }
    records = attach_paired_oracle_outcomes(
        [
            {**common, "record_id": "distance", "method": "distance"},
            {
                **common,
                "record_id": "oracle",
                "method": "oracle",
                "tasks_completed": 5,
            },
            {
                **common,
                "record_id": "unpaired",
                "method": "soc",
                "reserve_fraction": 0.0,
            },
        ]
    )
    by_id = {record["record_id"]: record for record in records}
    assert by_id["distance"]["paired_oracle_available"] is True
    assert by_id["distance"]["paired_oracle_tasks_completed"] == 5
    assert by_id["oracle"]["paired_oracle_tasks_completed"] == 5
    assert by_id["unpaired"]["paired_oracle_available"] is False


def test_formal_defaults_provide_exact_minimum_gate_cycles() -> None:
    args = parse_args(["--output-dir", "/tmp/stage_b_gate_test"])
    assert args.cycles_per_point == 1
    assert args.evaluation_num_envs == 12
    assert args.torch_threads == 1
    assert args.evaluation_seeds == list(range(110_001, 110_321))
    assert (
        args.cycles_per_point * len(args.evaluation_seeds)
        == args.minimum_cycles_for_gate
        == 320
    )
    assert args.oracle_familywise_safety_confidence_level == pytest.approx(0.95)
    assert args.oracle_design_stranding_rate == pytest.approx(0.01)
    assert args.oracle_min_joint_safety_certification_power == pytest.approx(0.90)
    assert args.battery_capacity is None
    assert args.oracle_throughput_confidence_level == pytest.approx(0.95)
    assert args.oracle_throughput_bootstrap_replicates == 10_000
    assert args.oracle_throughput_bootstrap_seed == 20_260_830
    manifest = formal_design_manifest(args)
    assert manifest["num_candidate_points"] == 12
    assert manifest["independent_cycles_per_point"] == 320
    assert manifest["total_battery_cycle_jobs"] == 3840
    assert manifest["parallel_worker_processes"] == 12
    assert manifest["maximum_policy_worker_initializations"] == 12
    assert manifest["oracle_horizon_semantics"] == (
        "finite_deadline_completion_resource_as_tagged_extended_real"
    )
    assert "infinite_horizon_unreachability" in manifest["oracle_horizon_non_claim"]
    assert "failed_TASK_goal" in manifest["task_timeout_semantics"]
    assert "CHARGER_COMMITTED" in manifest["committed_return_timeout_semantics"]
    assert "certified_physical_energy_exhaustion" in manifest[
        "episode_guard_semantics"
    ]
    assert manifest["stranding_certification_power_audit"]["passed"] is True
    assert (
        manifest["throughput_design_status"]
        == "VALID_SIMULTANEOUS_THREE_WAY_GATE_WITHOUT_ASSUMED_EFFECT_POWER"
    )


def test_formal_design_rejects_missing_same_reserve_oracle_pair() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "/tmp/stage_b_missing_oracle_pair",
                "--methods",
                "soc",
                "oracle",
                "--reserve-fractions",
                "0.05",
            ]
        )


def test_formal_cli_rejects_minimally_evaluable_but_underpowered_design() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "/tmp/stage_b_underpowered_safety",
                "--minimum-cycles-for-gate",
                "110",
                "--evaluation-seeds",
                *[str(seed) for seed in range(110_001, 110_111)],
            ]
        )


def test_oracle_throughput_inference_rejects_underresolved_bootstrap() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                "/tmp/stage_b_bad_bootstrap",
                "--oracle-throughput-bootstrap-replicates",
                "999",
            ]
        )


def test_stage_b_result_writer_embeds_paired_frontier_interval(tmp_path) -> None:
    args = parse_args(["--output-dir", str(tmp_path), "--smoke"])
    args.minimum_cycles_for_gate = 20
    args.oracle_stranding_ceiling = 1.0
    args.oracle_throughput_bootstrap_replicates = 2000
    outcomes = [
        ReturnDecisionOutcome("soc", 0.2, 0.0, 10.0, 10.0, 1.0, 0.1),
        ReturnDecisionOutcome("oracle", 0.0, 0.0, 11.0, 11.0, 1.0, 0.1),
    ]
    aggregates = []
    cycles = []
    for outcome in outcomes:
        aggregates.append(
            {
                **outcome.as_dict(),
                "evaluation_seeds": list(range(20)),
                "attempted_cycles": 20,
                "independent_cycles": 20,
                "successful_returns": 20,
                "energy_exhaustions": 0,
                "stranding_count": 0,
                "emergency_guards": 0,
                "tasks_completed": int(20 * outcome.tasks_per_battery_cycle),
                "simulated_seconds": 20 * 3600.0,
                "policy_steps": 200,
                "stranding_rate_wilson95_lower": 0.0,
                "stranding_rate_wilson95_upper": 0.0,
                "hocbf_interventions": 0,
                "obstacle_collision_count": 0,
            }
        )
        reserve = 0.0
        for schedule in range(20):
            cycles.append(
                {
                    "record_id": f"{outcome.method}-{schedule}",
                    "method": outcome.method,
                    "parameter": outcome.parameter,
                    "evaluation_seed": schedule,
                    "paired_schedule_id": f"schedule-{schedule}",
                    "cycle_index": 0,
                    "end_reason": "charger_reached",
                    "return_success": True,
                    "stranded": False,
                    "energy_exhausted": False,
                    "tasks_completed": int(outcome.tasks_per_battery_cycle),
                    "simulated_seconds": 3600.0,
                    "tasks_per_simulated_hour": outcome.tasks_per_simulated_hour,
                    "unused_energy_fraction_at_charger": 0.1,
                    "first_disagreement_step": None,
                    "first_disagreement_direction": None,
                    "reserve_fraction": reserve,
                    "paired_oracle_return_success": None,
                    "paired_oracle_stranded": None,
                    "paired_oracle_tasks_completed": None,
                    "battery_cycle_record": {},
                }
            )
    gate = write_results(
        tmp_path,
        args=args,
        p0={"distributional_headline_supported": False},
        prerequisite_audit={"passed": True, "status": "SMOKE_BYPASS"},
        outcomes=outcomes,
        aggregate_audits=aggregates,
        seed_audits=[{"method": "soc"}, {"method": "oracle"}],
        cycle_records=cycles,
        inherited_oracle_gate=None,
    )
    assert gate["status"] == "PASS"
    assert gate["throughput_gain_lower_confidence_bound"] == pytest.approx(0.1)
    assert gate["throughput_gain_upper_confidence_bound"] == pytest.approx(0.1)
    assert gate["paired_throughput_interval"]["num_paired_schedules"] == 20
    assert (tmp_path / "oracle_pareto_evidence.json").is_file()
    evidence = (tmp_path / "oracle_pareto_evidence.json").read_text(
        encoding="utf-8"
    )
    assert '"schema_version": "return-to-charge-oracle-pareto-v7"' in evidence
    assert '"simultaneous_stranding_audit"' in evidence


def test_checkpoint_evaluator_defaults_to_parallel_deterministic_workers() -> None:
    args = parse_checkpoint_evaluator_args(["--artifact", "/tmp/artifact"])
    assert args.evaluation_num_envs == 6
    assert args.evaluation_progress_interval_tasks == 25


def test_r5_environment_reconstruction_preserves_robust_hocbf_contract() -> None:
    normalized, audit = normalize_environment_reconstruction_command(
        [
            "--output-dir",
            "/tmp/r5",
            "--navigation-repair-variant",
            "R5",
            "--hocbf-sampled-data-robust",
            "--training-vec-env",
            "subproc",
            "--training-vec-start-method",
            "forkserver",
        ]
    )
    assert "R5" not in normalized
    assert "R4" in normalized
    assert "--hocbf-sampled-data-robust" not in normalized
    assert "--training-vec-env" not in normalized
    assert audit["source_navigation_repair_variant"] == "R5"
    assert audit["sampled_data_robust_hocbf_preserved"] is True


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
        '{"status":"PASS","evaluable":true,"passed":true,'
        '"evidence_integrity_passed":true,'
        '"stranding_statistic":'
        '"familywise_one_sided_clopper_pearson_upper_bound_with_bonferroni",'
        '"simultaneous_stranding_audit":{'
        '"correction":"bonferroni_one_sided_clopper_pearson",'
        '"num_candidate_points":1,"points":[{}]},'
        '"stranding_certification_power_audit":{'
        '"status":"PASS_DESIGN_POWER","passed":true},'
        '"throughput_statistic":'
        '"paired_schedule_simultaneous_max_t_rate_band_over_full_candidate_family",'
        '"paired_throughput_interval":{'
        '"inference":'
        '"paired_schedule_studentized_max_t_rate_band_over_full_candidate_family",'
        '"throughput_gain_lower_confidence_bound":0.08,'
        '"throughput_gain_upper_confidence_bound":0.12,'
        '"max_t_critical_value":1.96,'
        '"upper_max_t_critical_value":1.97,'
        '"oracle_rate_lower_bounds":[["oracle|0",108.0]],'
        '"oracle_rate_upper_bounds":[["oracle|0",112.0]],'
        '"heuristic_rate_lower_bounds":[["soc|0.2",98.0]],'
        '"heuristic_rate_upper_bounds":[["soc|0.2",102.0]],'
        '"oracle_points":["oracle|0"],"heuristic_points":["soc|0.2"],'
        '"eligible_oracle_points":["oracle|0"],'
        '"eligible_heuristic_points":["soc|0.2"]}}\n',
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


def test_post_gate_comparison_rejects_legacy_pass_without_coupling_audit(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "legacy_oracle_gate.json"
    legacy_path.write_text(
        '{"status":"PASS","evaluable":true,"passed":true}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="evidence integrity"):
        load_passed_oracle_headroom_gate(legacy_path)


def test_post_gate_comparison_rejects_pointwise_only_safety_pass(tmp_path) -> None:
    legacy_path = tmp_path / "pointwise_only_oracle_gate.json"
    legacy_path.write_text(
        '{"status":"PASS","evaluable":true,"passed":true,'
        '"evidence_integrity_passed":true,'
        '"stranding_statistic":"two_sided_wilson_95_upper_bound"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="simultaneous stranding safety"):
        load_passed_oracle_headroom_gate(legacy_path)


def test_post_gate_comparison_rejects_percentile_only_throughput_pass(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "percentile_only_oracle_gate.json"
    legacy_path.write_text(
        '{"status":"PASS","evaluable":true,"passed":true,'
        '"evidence_integrity_passed":true,'
        '"stranding_statistic":'
        '"familywise_one_sided_clopper_pearson_upper_bound_with_bonferroni",'
        '"simultaneous_stranding_audit":{'
        '"correction":"bonferroni_one_sided_clopper_pearson",'
        '"num_candidate_points":1,"points":[{}]},'
        '"stranding_certification_power_audit":{'
        '"status":"PASS_DESIGN_POWER","passed":true},'
        '"throughput_statistic":'
        '"paired_schedule_pooled_rate_ratio_with_frontier_selection_inside_bootstrap",'
        '"paired_throughput_interval":{'
        '"inference":'
        '"paired_schedule_percentile_bootstrap_with_frontier_selection_inside_each_replicate"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="simultaneous throughput"):
        load_passed_oracle_headroom_gate(legacy_path)


def test_post_gate_comparison_rejects_lower_bound_only_legacy_pass(
    tmp_path,
) -> None:
    legacy_path = tmp_path / "lower_bound_only_oracle_gate.json"
    legacy_path.write_text(
        '{"status":"PASS","evaluable":true,"passed":true,'
        '"evidence_integrity_passed":true,'
        '"stranding_statistic":'
        '"familywise_one_sided_clopper_pearson_upper_bound_with_bonferroni",'
        '"simultaneous_stranding_audit":{'
        '"correction":"bonferroni_one_sided_clopper_pearson",'
        '"num_candidate_points":1,"points":[{}]},'
        '"stranding_certification_power_audit":{'
        '"status":"PASS_DESIGN_POWER","passed":true},'
        '"throughput_statistic":'
        '"paired_schedule_simultaneous_max_t_rate_band_over_full_candidate_family",'
        '"paired_throughput_interval":{'
        '"inference":'
        '"paired_schedule_studentized_max_t_rate_band_over_full_candidate_family",'
        '"throughput_gain_lower_confidence_bound":0.08,'
        '"oracle_points":["oracle|0"],"heuristic_points":["soc|0.2"],'
        '"eligible_oracle_points":["oracle|0"],'
        '"eligible_heuristic_points":["soc|0.2"]}}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="lower/upper-bound audit"):
        load_passed_oracle_headroom_gate(legacy_path)


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
            "evidence_integrity_passed": True,
        },
    )
    assert sentinel == "STOPPED_AFTER_ORACLE_HEADROOM_GATE.json"
    assert exit_code == 4
    assert status == "FAIL_INSUFFICIENT_ORACLE_HEADROOM"


def test_passed_and_post_gate_stage_b_contracts_complete(tmp_path) -> None:
    formal = parse_args(["--output-dir", str(tmp_path / "formal")])
    assert stage_b_completion_contract(
        formal,
        {
            "status": "PASS",
            "evaluable": True,
            "passed": True,
            "evidence_integrity_passed": True,
        },
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
        {
            "status": "PASS",
            "evaluable": True,
            "passed": True,
            "evidence_integrity_passed": True,
        },
    ) == ("COMPLETED.json", 0, "POST_ORACLE_HEADROOM_DECISION_COMPARISON")


def test_formal_stage_b_stops_when_oracle_shadow_coupling_fails() -> None:
    args = parse_args(["--output-dir", "/tmp/formal_stage_b_bad_coupling"])
    assert stage_b_completion_contract(
        args,
        {
            "status": "PASS",
            "evaluable": True,
            "passed": True,
            "evidence_integrity_passed": False,
        },
    ) == (
        "STOPPED_AFTER_EVIDENCE_INTEGRITY.json",
        4,
        "FAIL_ORACLE_SHADOW_EVIDENCE_INTEGRITY",
    )


def valid_gate_b_prerequisites() -> tuple[dict, dict, dict]:
    buckets = {
        "100-500": 0.99,
        "500-1500": 0.99,
        "1500-2500": 0.99,
        "2500-4000": 0.99,
        ">4000": 0.99,
    }
    navigation = {
        "global_env_transitions": 500_000,
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
        "tasks": [
            {
                "success": True,
                "total_realized_energy": 10.0,
                "actual_path_length": 100.0,
            },
            {
                "success": True,
                "total_realized_energy": 20.0,
                "actual_path_length": 200.0,
            },
        ],
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
    assert audit.status == "FAIL_GATE_B_PREREQUISITES"
    assert any("calibration navigation validity" in item for item in audit.failures)
    assert any("censored/task-limit" in item for item in audit.failures)


def test_navigation_completion_wrapper_preserves_metrics_and_authorization() -> None:
    navigation, calibration, validation = valid_gate_b_prerequisites()
    wrapped = {
        "checkpoint_sha256": "a" * 64,
        "navigation_gate_passed": False,
        "downstream_navigation_ready": False,
        "downstream_stages_authorized": False,
        "final_navigation": navigation,
    }
    view = navigation_artifact_view(wrapped)
    assert view.schema == "navigation_repair_completion_wrapper"
    assert view.metrics["overall_success_rate"] == pytest.approx(0.99)
    assert view.checkpoint_sha256 == "a" * 64
    assert view.authorization_passed is False
    audit = audit_gate_b_prerequisites(wrapped, calibration, validation)
    assert audit.navigation_success_rate == pytest.approx(0.99)
    assert any("does not authorize" in failure for failure in audit.failures)


def test_formal_stage_b_requires_one_checkpoint_identity_chain(tmp_path) -> None:
    navigation, calibration, validation = valid_gate_b_prerequisites()
    artifact = tmp_path / "navigation_artifact"
    checkpoint = artifact / "phase1_navigation" / "checkpoint_transition_500000.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"frozen-navigation-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    artifact_config = {
        "exact_command": [
            "scripts/run_jacobian_safety_energy_1m.py",
            "--output-dir",
            str(artifact),
            "--navigation-repair-variant",
            "R1",
            "--phase1-transitions",
            "500000",
            "--phase-end-eval-only",
            "--eval-navigation-tasks",
            "500",
            "--repair-source-navigation-evaluation",
            "source.json",
            "--repair-evaluation-tasks",
            "tasks.json",
        ]
    }
    config_path = artifact / "config.json"
    config_path.write_text(json.dumps(artifact_config) + "\n", encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    navigation_wrapper = {
        "status": "COMPLETED",
        "checkpoint_sha256": checkpoint_sha,
        "navigation_gate_passed": True,
        "downstream_navigation_ready": True,
        "downstream_stages_authorized": True,
        "final_navigation": navigation,
    }
    navigation_path = tmp_path / "navigation.json"
    navigation_path.write_text(
        json.dumps(navigation_wrapper) + "\n",
        encoding="utf-8",
    )
    navigation_sha = hashlib.sha256(navigation_path.read_bytes()).hexdigest()
    calibration.update(
        {
            "navigation_checkpoint_sha256": checkpoint_sha,
            "navigation_evaluation_sha256": navigation_sha,
            "navigation_artifact_config_sha256": config_sha,
        }
    )
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration) + "\n", encoding="utf-8")
    calibration_sha = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    validation.update(
        {
            "navigation_checkpoint_sha256": checkpoint_sha,
            "navigation_evaluation_sha256": navigation_sha,
            "navigation_artifact_config_sha256": config_sha,
            "battery_calibration_sha256": calibration_sha,
        }
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")

    def stage_b_args():
        return parse_args(
            [
                "--output-dir",
                str(tmp_path / "unused"),
                "--navigation-checkpoint",
                str(checkpoint),
                "--navigation-artifact",
                str(artifact),
                "--navigation-evaluation-json",
                str(navigation_path),
                "--battery-calibration-json",
                str(calibration_path),
                "--battery-validation-json",
                str(validation_path),
            ]
        )

    args = stage_b_args()
    audit = load_prerequisite_audit(args)
    assert audit.passed is True
    assert args.battery_capacity == pytest.approx(350.0)
    assert args.energy_per_meter == pytest.approx(0.1)
    contract = args.navigation_environment_contract
    assert contract["environment_kwargs"]["lidar_enabled"] is True
    assert contract["environment_kwargs"]["num_obstacles"] == 24
    assert contract["stage_b_runtime_overrides"][
        "operational_energy_capacity"
    ] == pytest.approx(350.0)
    runtime = contract["stage_b_runtime_overrides"]
    assert runtime["continuous_task_workload"] is True
    guard = runtime["episode_guard_contract"]
    assert guard["minimum_energy_per_policy_step"] == pytest.approx(0.012)
    assert guard["maximum_steps_to_exhaustion_per_cycle"] == 29_167
    assert guard["effective_phase2_episode_limit"] == 29_168
    assert guard["guard_cannot_preempt_energy_exhaustion"] is True

    validation["navigation_checkpoint_sha256"] = "0" * 64
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    mismatched = load_prerequisite_audit(stage_b_args())
    assert mismatched.passed is False
    assert any(
        "battery validation provenance mismatch" in failure
        for failure in mismatched.failures
    )


def test_stage_b_episode_guard_is_derived_from_physical_energy_lower_bound() -> None:
    args = parse_args(["--output-dir", "/tmp/stage_b_guard", "--smoke"])
    args.battery_capacity = 304.953884
    contract = stage_b_episode_guard_contract(args)
    assert contract["source_phase2_episode_limit"] == 20_000
    assert contract["minimum_energy_per_policy_step"] == pytest.approx(0.012)
    assert contract["maximum_steps_to_exhaustion_per_cycle"] == 25_413
    assert contract["effective_phase2_episode_limit"] == 25_414
    assert contract["guard_cannot_preempt_energy_exhaustion"] is True


def test_formal_stage_b_records_scientific_prerequisite_stop_not_crash(
    tmp_path,
) -> None:
    navigation, calibration, validation = valid_gate_b_prerequisites()
    navigation["overall_success_rate"] = 0.90
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = []
    for name, payload in (
        ("navigation.json", navigation),
        ("calibration.json", calibration),
        ("validation.json", validation),
    ):
        path = inputs / name
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        paths.append(path)
    output = tmp_path / "stage_b"
    with pytest.raises(SystemExit) as stopped:
        run_stage_b_main(
            [
                "--output-dir",
                str(output),
                "--navigation-evaluation-json",
                str(paths[0]),
                "--battery-calibration-json",
                str(paths[1]),
                "--battery-validation-json",
                str(paths[2]),
            ]
        )
    assert stopped.value.code == 4
    stop_path = output / "STOPPED_PREREQUISITES_NOT_READY.json"
    assert stop_path.is_file()
    assert not (output / "FAILED.json").exists()
    stop = json.loads(stop_path.read_text(encoding="utf-8"))
    assert stop["status"] == "FAIL_GATE_B_PREREQUISITES"
    assert stop["downstream_authorized"] is False
    assert stop["prerequisite_audit"]["passed"] is False


def test_seed_family_worker_persists_candidate_and_rollout_failure_context(
    tmp_path,
    monkeypatch,
) -> None:
    args = Namespace(output_dir=tmp_path)
    monkeypatch.setattr(stage_b, "_STAGE_B_WORKER_ARGS", args)
    monkeypatch.setattr(stage_b, "_STAGE_B_WORKER_POLICY", object())
    monkeypatch.setattr(
        stage_b,
        "parameter_grid",
        lambda unused: iter((("soc", 0.2), ("oracle", 0.0))),
    )

    def fail_evaluation(*unused_args, **unused_kwargs):
        error = RuntimeError("diagnostic rollout failure")
        error.stage_b_context = {
            "evaluation_seed": 110007,
            "method": "soc",
            "cycle_index": 0,
        }
        error.rollout_diagnostics = {
            "rollout_label": "return_after_task",
            "final_distance_to_goal": 123.0,
        }
        raise error

    monkeypatch.setattr(stage_b, "evaluate_method", fail_evaluation)
    with pytest.raises(RuntimeError, match="diagnostic rollout failure"):
        stage_b._evaluate_stage_b_seed_family_job(110007)
    failure_path = tmp_path / "seed_failures" / "seed_110007.json"
    assert failure_path.is_file()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["protocol"] == "stage_b_seed_family_failure_v1"
    assert failure["evaluation_seed"] == 110007
    assert failure["candidate_index"] == 0
    assert failure["method"] == "soc"
    assert failure["parameter"] == pytest.approx(0.2)
    assert failure["completed_candidate_count"] == 0
    assert failure["stage_b_context"]["cycle_index"] == 0
    assert failure["rollout_diagnostics"]["rollout_label"] == (
        "return_after_task"
    )
    assert "fail_evaluation" in failure["traceback"]


def test_stage_b_diagnostic_can_isolate_native_worker_crashes_with_retries(
    tmp_path,
) -> None:
    source_launch = tmp_path / "launch.json"
    source_launch.write_text(
        json.dumps(
            {
                "arguments": {
                    "evaluation_num_envs": 12,
                    "formal_seed_retries": 2,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    args = diagnostic_args(
        source_launch,
        tmp_path / "diagnostic",
        [110004, 110012, 110013],
        evaluation_num_envs=1,
        formal_seed_retries=2,
    )
    assert args.evaluation_num_envs == 1
    assert args.formal_seed_retries == 2
    assert args.evaluation_seeds == [110004, 110012, 110013]
    assert args.resume is False


def test_stage_b_import_contract_caps_native_thread_environments() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert stage_b.os.environ[name] == "1"
    count = _linux_thread_count()
    assert count is None or count >= 1


def test_stall_ablation_summary_preserves_paired_success_patterns() -> None:
    results = [
        {
            "conditions": {
                "observed": {"success": False},
                "policy_no_filter": {"success": True, "rollout_steps": 40},
                "heuristic_filter": {"success": False},
            }
        },
        {
            "conditions": {
                "observed": {"success": False},
                "policy_no_filter": {"success": True, "rollout_steps": 60},
                "heuristic_filter": {"success": True, "rollout_steps": 50},
            }
        },
    ]
    summary = summarize_stalls(results)
    assert summary["condition_summaries"]["observed"]["success_count"] == 0
    policy = summary["condition_summaries"]["policy_no_filter"]
    assert policy["success_count"] == 2
    assert policy["successful_rollout_steps_mean"] == pytest.approx(50.0)
    assert policy["successful_rollout_steps_max"] == 60
    assert sum(summary["paired_success_patterns"].values()) == 2


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
