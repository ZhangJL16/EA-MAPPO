from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.energy_mc.final_risk import MondrianGoalMissionRiskEstimator
import scripts.run_energy_phase2_final_500k as final_runner
from scripts.run_energy_phase2_final_500k import (
    DEFAULT_PREREGISTERED_RUN,
    FORMAL_BATTERY_CAPACITY,
    FORMAL_RESERVE_FRACTION,
    FORMAL_TRANSITION_BUDGET,
    formal_summary,
    load_frozen_estimator,
    validate_preregistered_method,
    write_energy_metrics,
)


def test_frozen_v5_formal_loader_uses_preregistered_method() -> None:
    preregistration = validate_preregistered_method(DEFAULT_PREREGISTERED_RUN)
    estimator = load_frozen_estimator(preregistration, device="cpu")
    assert isinstance(estimator, MondrianGoalMissionRiskEstimator)
    assert estimator.estimator_type == "mc_mondrian_goal_mission_risk_v5"
    assert estimator.goal_coverage == pytest.approx(0.975)
    assert estimator.mission_coverage == pytest.approx(0.99)
    assert estimator.goal_feature_builder.mode == "compact_decision_context"
    assert estimator.goal_feature_builder.input_dim == 12
    assert estimator.update_count == 0
    assert not estimator.replay
    assert not estimator.trainable_replay


def test_formal_protocol_constants_are_frozen() -> None:
    assert FORMAL_TRANSITION_BUDGET == 500_000
    assert FORMAL_RESERVE_FRACTION == pytest.approx(0.10)
    assert FORMAL_BATTERY_CAPACITY == pytest.approx(378.72626091628933)


def test_energy_metrics_add_explicit_risk_corrections(tmp_path: Path) -> None:
    row = {
        "task_energy_prediction": 10.0,
        "task_energy_upper95": 12.5,
        "return_now_energy_prediction": 8.0,
        "return_now_energy_upper95": 9.0,
        "return_after_task_energy_prediction": 7.0,
        "return_after_task_energy_upper95": 8.25,
        "mission_energy_prediction": 17.0,
        "mission_energy_upper95": 20.0,
    }
    (tmp_path / "training_curve.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    write_energy_metrics(tmp_path)
    result = json.loads(
        (tmp_path / "energy_metrics.jsonl").read_text(encoding="utf-8")
    )
    assert result["task_risk_correction"] == pytest.approx(2.5)
    assert result["return_now_risk_correction"] == pytest.approx(1.0)
    assert result["return_after_task_risk_correction"] == pytest.approx(1.25)
    assert result["mission_risk_correction"] == pytest.approx(3.0)


def test_formal_summary_exposes_required_metric_names() -> None:
    summary = {
        "total_delivery_tasks_completed": 100,
        "charger_returns_attempted": 8,
        "tasks_per_completed_recharge_cycle": 12.5,
        "mean_remaining_energy_fraction_at_charger": 0.14,
        "mean_energy_utilization_per_completed_recharge_cycle": 0.86,
    }
    attribution = {
        "cause_counts": {
            "point_estimate": 1,
            "uncertainty_margin": 2,
            "reserve": 4,
            "both": 1,
        },
        "mean_reserve_to_uncertainty_ratio": 20.0,
    }
    result = formal_summary(summary, attribution)
    assert result["tasks_completed"] == 100
    assert result["autonomous_return_attempts"] == 8
    assert result["tasks_per_completed_battery_cycle"] == pytest.approx(12.5)
    assert result["mean_charger_arrival_SOC"] == pytest.approx(0.14)
    assert result["energy_utilization_per_completed_cycle"] == pytest.approx(0.86)
    assert result["switch_attribution"] == {
        "POINT_ESTIMATE": 1,
        "UNCERTAINTY": 2,
        "RESERVE": 4,
        "COMBINED": 1,
        "switches_involving_reserve": 5,
        "pure_uncertainty_triggered_switches": 2,
        "pure_point_estimate_triggered_switches": 1,
        "mean_reserve_to_uncertainty_ratio": 20.0,
    }
    assert result["unnecessary_return_metric_type"] == (
        "offline_counterfactual_proxy"
    )


def test_formal_runner_writes_required_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "formal"
    preregistration = validate_preregistered_method(DEFAULT_PREREGISTERED_RUN)

    monkeypatch.setattr(final_runner, "assert_clean_worktree", lambda: None)
    monkeypatch.setattr(
        final_runner,
        "validate_preregistered_method",
        lambda _: preregistration,
    )
    monkeypatch.setattr(final_runner, "assert_file_hash", lambda *args: None)
    monkeypatch.setattr(final_runner, "load_frozen_estimator", lambda *args, **kwargs: object())
    monkeypatch.setattr(final_runner, "load_frozen_sac", lambda *args, **kwargs: object())

    def fake_phase2(*args, **kwargs):
        destination = kwargs["output"]
        row = {
            "task_energy_prediction": 1.0,
            "task_energy_upper95": 1.2,
            "return_now_energy_prediction": 2.0,
            "return_now_energy_upper95": 2.3,
            "return_after_task_energy_prediction": 3.0,
            "return_after_task_energy_upper95": 3.4,
            "mission_energy_prediction": 4.0,
            "mission_energy_upper95": 4.6,
        }
        (destination / "training_curve.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )
        (destination / "battery_cycles.jsonl").write_text("", encoding="utf-8")
        (destination / "switching_events.jsonl").write_text("", encoding="utf-8")
        return {
            "total_delivery_tasks_completed": 10,
            "charger_returns_attempted": 1,
            "tasks_per_completed_recharge_cycle": 10.0,
            "mean_remaining_energy_fraction_at_charger": 0.1,
            "mean_energy_utilization_per_completed_recharge_cycle": 0.9,
        }

    monkeypatch.setattr(final_runner, "run_phase2", fake_phase2)
    monkeypatch.setattr(
        final_runner,
        "phase2_counterfactual_audit",
        lambda *args, **kwargs: {
            "audited_switches": 1,
            "unnecessary_returns": 0,
            "unnecessary_return_rate": 0.0,
        },
    )
    monkeypatch.setattr(
        final_runner,
        "phase2_switch_attribution",
        lambda _: {
            "cause_counts": {
                "point_estimate": 0,
                "uncertainty_margin": 0,
                "reserve": 1,
                "both": 0,
            },
            "mean_reserve_to_uncertainty_ratio": 20.0,
        },
    )
    args = SimpleNamespace(
        output_dir=str(output),
        preregistered_run=str(DEFAULT_PREREGISTERED_RUN),
        seed=0,
        device="cpu",
        navigation_device="cpu",
        phase2_transition_budget=500_000,
        phase2_episode_max_steps=500_001,
        energy_reserve_fraction=0.10,
        log_frequency=1000,
    )
    result = final_runner.run(args)
    assert result["status"] == "COMPLETED"
    for filename in (
        "CONFIG.json",
        "RUN_METADATA.json",
        "phase2_summary.json",
        "battery_cycles.jsonl",
        "switching_events.jsonl",
        "energy_metrics.jsonl",
        "COMPLETED.json",
    ):
        assert (output / filename).is_file()
