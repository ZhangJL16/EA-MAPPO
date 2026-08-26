from __future__ import annotations

from scripts.train_quantile_td_for_navigation_checkpoint import (
    FORMAL_ENERGY_EVAL_FREQUENCY,
    FORMAL_ENERGY_EVAL_TASKS,
    FORMAL_TD_TRANSITIONS,
    parse_args,
    prerequisite_failures,
)
from scripts.train_uav_energy_delivery_sac import td_readiness_audit


def passing_navigation() -> dict[str, object]:
    return {
        "num_tasks": 500,
        "global_env_transitions": 500_000,
        "overall_success_rate": 0.99,
        "distance_bucket_success": {
            "100-500": 0.99,
            "500-1500": 0.99,
            "1500-2500": 0.99,
            "2500-4000": 0.99,
            ">4000": 0.99,
        },
        "mean_path_ratio": 1.05,
        "boundary_contact_step_rate": 0.0,
        "obstacle_collision_steps": 0,
    }


def test_formal_quantile_td_protocol_is_fixed() -> None:
    args = parse_args(
        [
            "--artifact",
            "/tmp/artifact",
            "--checkpoint",
            "/tmp/checkpoint_transition_500000.zip",
            "--navigation-evaluation-json",
            "/tmp/navigation.json",
            "--oracle-headroom-json",
            "/tmp/oracle.json",
            "--output-dir",
            "/tmp/output",
        ]
    )
    assert args.transition_budget == FORMAL_TD_TRANSITIONS
    assert args.energy_eval_tasks == FORMAL_ENERGY_EVAL_TASKS
    assert args.energy_eval_freq_transitions == FORMAL_ENERGY_EVAL_FREQUENCY
    assert args.evaluation_num_envs == 6
    assert args.td_collection_seed != args.energy_eval_seed


def test_quantile_td_requires_oracle_headroom_pass() -> None:
    navigation = passing_navigation()
    passing_gate = {"evaluable": True, "passed": True, "status": "PASS"}
    assert prerequisite_failures(navigation, passing_gate) == []
    failures = prerequisite_failures(
        navigation,
        {"evaluable": True, "passed": False, "status": "FAIL"},
    )
    assert "Oracle headroom gate did not pass" in failures
    assert "Oracle headroom gate status is not PASS" in failures


def test_quantile_td_rejects_navigation_not_ready() -> None:
    navigation = passing_navigation()
    navigation["overall_success_rate"] = 0.8
    failures = prerequisite_failures(
        navigation,
        {"evaluable": True, "passed": True, "status": "PASS"},
    )
    assert "navigation energy/readiness gate failed" in failures


def heldout_td_summary(*, completed_per_bucket: int = 100) -> dict[str, object]:
    bucket = {
        "num_completed_goals": completed_per_bucket,
        "finite_predictions": True,
        "quantile_ordering_valid": True,
        "mean_true_total_energy": 10.0,
        "td_mae": 2.0,
        "td_rmse": 3.0,
        "td_bias": 0.1,
        "td_underestimation_rate": 0.4,
        "q95_coverage": 0.6,
    }
    return {
        "num_tasks": 500,
        "metrics": {
            "overall": {**bucket, "num_completed_goals": completed_per_bucket * 5},
            "by_initial_goal_distance": {
                name: dict(bucket)
                for name in [
                    "100-500",
                    "500-1500",
                    "1500-2500",
                    "2500-4000",
                    ">4000",
                ]
            },
        },
    }


def test_td_readiness_requires_complete_stratified_heldout_evidence() -> None:
    audit = td_readiness_audit(heldout_td_summary())
    assert audit["td_energy_ready"] is True
    assert audit["minimum_completed_goals_per_distance_bucket"] == 95
    assert audit["failures"] == []
    selected = heldout_td_summary(completed_per_bucket=94)
    audit = td_readiness_audit(selected)
    assert audit["td_energy_ready"] is False
    assert audit["heldout_navigation_coverage_valid"] is False


def test_td_readiness_rejects_zero_predictor_scale_and_dominant_underestimation() -> None:
    selected = heldout_td_summary()
    selected["metrics"]["overall"]["td_mae"] = 11.0
    selected["metrics"]["overall"]["td_underestimation_rate"] = 0.8
    selected["metrics"]["by_initial_goal_distance"][">4000"]["td_mae"] = 11.0
    audit = td_readiness_audit(selected)
    assert audit["td_energy_ready"] is False
    assert audit["overall_error_better_than_zero_predictor"] is False
    assert audit["catastrophic_far_distance_error"] is True
    assert audit["underestimation_not_dominant"] is False
