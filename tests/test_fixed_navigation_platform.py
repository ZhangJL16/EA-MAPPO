from __future__ import annotations

from scripts.evaluate_fixed_navigation_platform import (
    FORMAL_SEED,
    FORMAL_TASKS,
    PLATFORM_NAME,
    _gate_summary,
    parse_args,
)


def _passing_result() -> dict[str, object]:
    return {
        "num_tasks": 500,
        "global_env_transitions": 0,
        "evaluation_env_transitions": 1000,
        "overall_success_rate": 0.99,
        "distance_bucket_success": {"a": 0.98, "b": 0.99},
        "mean_path_ratio": 1.05,
        "obstacle_collision_steps": 0,
        "obstacle_collision_episode_rate": 0.0,
        "boundary_contact_step_rate": 0.0,
        "hocbf_intervention_step_rate": 0.1,
        "hocbf_emergency_brake_step_rate": 0.01,
        "nominal_safe_action_rate": 0.9,
        "projection_valid_step_rate": 1.0,
    }


def test_formal_defaults_are_frozen() -> None:
    args = parse_args(
        ["--source-artifact", "source", "--output-dir", "output"]
    )
    assert args.num_tasks == FORMAL_TASKS == 500
    assert args.selection_seed == FORMAL_SEED == 170_002
    assert PLATFORM_NAME == "deterministic_go_to_goal_v1"


def test_smoke_cannot_authorize_downstream_gate() -> None:
    args = parse_args(
        [
            "--source-artifact",
            "source",
            "--output-dir",
            "output",
            "--smoke",
        ]
    )
    assert args.num_tasks == 10


def test_gate_summary_preserves_formal_thresholds() -> None:
    passing = _gate_summary(_passing_result())
    assert passing["navigation_gate_passed"] is True
    failed_result = _passing_result()
    failed_result["obstacle_collision_steps"] = 1
    failed = _gate_summary(failed_result)
    assert failed["navigation_gate_passed"] is False
    assert "navigation_collision_boundary_gate_failed" in failed[
        "navigation_gate_failures"
    ]
