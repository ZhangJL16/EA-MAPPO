from __future__ import annotations

from scripts.train_quantile_td_for_navigation_checkpoint import (
    FORMAL_ENERGY_EVAL_FREQUENCY,
    FORMAL_ENERGY_EVAL_TASKS,
    FORMAL_TD_TRANSITIONS,
    parse_args,
    prerequisite_failures,
)


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
