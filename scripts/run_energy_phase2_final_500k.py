from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.energy_mc.adaptive_uncertainty import HeteroscedasticResidualModel
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    PositiveResidualQuantileModel,
)
from experiments.energy_mc.final_risk import (
    MondrianGoalMissionRiskEstimator,
    MondrianMissionCalibration,
    MondrianTrajectoryCalibration,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.explore_adaptive_energy_uncertainty import environment_args
from scripts.run_energy_risk_v5 import phase2_switch_attribution
from scripts.run_energy_uncertainty_v4 import phase2_counterfactual_audit
from scripts.train_uav_energy_mc import load_frozen_sac, run_phase2


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTERED_RUN = (
    ROOT / "artifacts/uav_energy_risk_v5_20260818_0330"
)
FORMAL_TRANSITION_BUDGET = 500_000
FORMAL_RESERVE_FRACTION = 0.10
FORMAL_BATTERY_CAPACITY = 378.72626091628933


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def assert_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError("formal Phase2 requires a clean git worktree")


def assert_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"frozen {label} hash mismatch: expected {expected}, got {actual}"
        )


def validate_preregistered_method(
    preregistered_run: Path,
) -> dict[str, object]:
    completed = load_json(preregistered_run / "COMPLETED.json")
    readiness = load_json(preregistered_run / "FRESH_V5_READINESS.json")
    preregistration = load_json(
        preregistered_run / "PREREGISTERED_ENERGY_RISK_V5.json"
    )
    if completed.get("status") != "COMPLETED":
        raise RuntimeError("source Energy Risk v5 run is not complete")
    if readiness.get("fresh_v5_status") != "PASS":
        raise RuntimeError("source Energy Risk v5 offline gate did not pass")
    if preregistration["goal"]["method"] != "k2_suffix":
        raise RuntimeError("formal Goal method must remain K2 suffix risk")
    if float(preregistration["goal"]["conformal_construction_coverage"]) != 0.975:
        raise RuntimeError("formal Goal Mondrian construction must remain 97.5%")
    if preregistration["goal"]["features"]["mode"] != "compact_decision_context":
        raise RuntimeError("formal Goal risk must use compact decision context")
    if float(preregistration["mission"]["conformal_construction_coverage"]) != 0.99:
        raise RuntimeError("formal Mission Mondrian construction must remain 99%")
    if preregistration["mission"]["method"] != (
        "frozen_heteroscedastic_laplace_plus_distance_mondrian"
    ):
        raise RuntimeError("formal Mission method must remain frozen Laplace")
    if not bool(preregistration["sac"]["frozen"]):
        raise RuntimeError("formal navigation policy must be frozen")
    if bool(preregistration["sac"]["retrained"]):
        raise RuntimeError("formal navigation policy cannot be retrained")
    if bool(preregistration["point_model"]["retrained"]):
        raise RuntimeError("formal point estimator cannot be retrained")
    return preregistration


def load_frozen_estimator(
    preregistration: dict[str, object],
    *,
    device: str,
) -> MondrianGoalMissionRiskEstimator:
    point_path = Path(preregistration["point_model"]["path"])
    goal_path = Path(preregistration["goal"]["model_path"])
    mission_path = Path(preregistration["mission"]["model_path"])
    assert_file_hash(
        point_path,
        str(preregistration["point_model"]["sha256"]),
        "MC point estimator",
    )
    assert_file_hash(
        goal_path,
        str(preregistration["goal"]["model_sha256"]),
        "Goal K2 risk model",
    )
    assert_file_hash(
        mission_path,
        str(preregistration["mission"]["model_sha256"]),
        "Mission Laplace model",
    )
    point = EnergyToGoRegressor.load(point_path, device=device)
    goal_risk = PositiveResidualQuantileModel.load(goal_path, device=device)
    mission_risk = HeteroscedasticResidualModel.load(mission_path, device=device)
    feature_payload = preregistration["goal"]["features"]
    feature_builder = GoalRiskFeatureBuilder(
        mode=str(feature_payload["mode"]),
        map_extent=tuple(float(value) for value in feature_payload["map_extent"]),
    )
    goal_calibration = MondrianTrajectoryCalibration(
        **preregistration["goal"]["calibration"]
    )
    mission_calibration = MondrianMissionCalibration(
        **preregistration["mission"]["calibration"]
    )
    estimator = MondrianGoalMissionRiskEstimator(
        point,
        feature_builder,
        goal_risk,
        goal_calibration,
        mission_risk,
        mission_calibration,
        goal_coverage=0.975,
        mission_coverage=0.99,
    )
    if estimator.update_count != 0 or estimator.replay or estimator.trainable_replay:
        raise RuntimeError("formal frozen estimator unexpectedly contains training state")
    return estimator


def write_energy_metrics(output: Path) -> None:
    source = output / "training_curve.jsonl"
    destination = output / "energy_metrics.jsonl"
    with source.open("r", encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            row["task_risk_correction"] = (
                float(row["task_energy_upper95"])
                - float(row["task_energy_prediction"])
            )
            row["return_now_risk_correction"] = (
                float(row["return_now_energy_upper95"])
                - float(row["return_now_energy_prediction"])
            )
            row["return_after_task_risk_correction"] = (
                float(row["return_after_task_energy_upper95"])
                - float(row["return_after_task_energy_prediction"])
            )
            row["mission_risk_correction"] = (
                float(row["mission_energy_upper95"])
                - float(row["mission_energy_prediction"])
            )
            writer.write(json.dumps(json_value(row), sort_keys=True) + "\n")


def formal_summary(
    summary: dict[str, object],
    switch_attribution: dict[str, object],
) -> dict[str, object]:
    cause_counts = switch_attribution["cause_counts"]
    return {
        **summary,
        "tasks_completed": summary["total_delivery_tasks_completed"],
        "autonomous_return_attempts": summary["charger_returns_attempted"],
        "tasks_per_completed_battery_cycle": summary[
            "tasks_per_completed_recharge_cycle"
        ],
        "mean_charger_arrival_SOC": summary[
            "mean_remaining_energy_fraction_at_charger"
        ],
        "energy_utilization_per_completed_cycle": summary[
            "mean_energy_utilization_per_completed_recharge_cycle"
        ],
        "switch_attribution": {
            "POINT_ESTIMATE": int(cause_counts["point_estimate"]),
            "UNCERTAINTY": int(cause_counts["uncertainty_margin"]),
            "RESERVE": int(cause_counts["reserve"]),
            "COMBINED": int(cause_counts["both"]),
            "switches_involving_reserve": int(cause_counts["reserve"])
            + int(cause_counts["both"]),
            "pure_uncertainty_triggered_switches": int(
                cause_counts["uncertainty_margin"]
            ),
            "pure_point_estimate_triggered_switches": int(
                cause_counts["point_estimate"]
            ),
            "mean_reserve_to_uncertainty_ratio": switch_attribution[
                "mean_reserve_to_uncertainty_ratio"
            ],
        },
        "unnecessary_return_metric_type": "offline_counterfactual_proxy",
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    assert_clean_worktree()
    if args.phase2_transition_budget != FORMAL_TRANSITION_BUDGET:
        raise ValueError("formal Phase2 transition budget must be exactly 500000")
    if args.energy_reserve_fraction != FORMAL_RESERVE_FRACTION:
        raise ValueError("formal Phase2 reserve fraction must be exactly 0.10")
    if args.phase2_episode_max_steps <= args.phase2_transition_budget:
        raise ValueError("emergency guard must exceed the full Phase2 budget")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "RUNNING.json",
        {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()},
    )
    try:
        preregistered_run = Path(args.preregistered_run).resolve()
        preregistration = validate_preregistered_method(preregistered_run)
        sac_path = Path(preregistration["sac"]["path"])
        assert_file_hash(
            sac_path,
            str(preregistration["sac"]["sha256"]),
            "frozen SAC checkpoint",
        )
        estimator = load_frozen_estimator(preregistration, device=args.device)
        policy = load_frozen_sac(sac_path, args.navigation_device)
        phase_args = environment_args(
            seed=args.seed,
            phase2_budget=args.phase2_transition_budget,
        )
        phase_args.phase2_episode_max_steps = args.phase2_episode_max_steps
        phase_args.energy_reserve_fraction = args.energy_reserve_fraction
        phase_args.phase2_log_frequency = args.log_frequency
        config = {
            "phase2_transition_budget": args.phase2_transition_budget,
            "phase2_episode_max_steps": args.phase2_episode_max_steps,
            "seed": args.seed,
            "deterministic_navigation": True,
            "sac_training": False,
            "td_enabled": False,
            "obstacles": False,
            "lidar_policy_input": False,
            "cbf": False,
            "battery_capacity": FORMAL_BATTERY_CAPACITY,
            "battery_capacity_source": "formal_30min_frozen_policy_calibration",
            "energy_reserve_fraction": args.energy_reserve_fraction,
            "energy_reserve_absolute": (
                FORMAL_BATTERY_CAPACITY * args.energy_reserve_fraction
            ),
            "goal_method": "k2_suffix_max_positive_residual",
            "goal_feature_mode": "compact_decision_context_12d",
            "goal_mondrian_construction_coverage": 0.975,
            "mission_method": "frozen_heteroscedastic_laplace",
            "mission_mondrian_construction_coverage": 0.99,
            "switching_logic": [
                "remaining_energy <= return_now_upper + reserve",
                "remaining_energy <= mission_upper + reserve",
            ],
            "source_preregistered_run": str(preregistered_run),
        }
        write_json(output / "CONFIG.json", config)
        metadata = {
            "git_sha": git_sha(),
            "pid": os.getpid(),
            "started_at": utc_now(),
            "command": [sys.executable, *sys.argv],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": args.device,
            "navigation_device": args.navigation_device,
            "sac_checkpoint": str(sac_path),
            "sac_sha256": preregistration["sac"]["sha256"],
            "point_checkpoint": preregistration["point_model"]["path"],
            "point_sha256": preregistration["point_model"]["sha256"],
            "goal_risk_checkpoint": preregistration["goal"]["model_path"],
            "goal_risk_sha256": preregistration["goal"]["model_sha256"],
            "mission_risk_checkpoint": preregistration["mission"]["model_path"],
            "mission_risk_sha256": preregistration["mission"]["model_sha256"],
            "fresh_v5_used_for_tuning": False,
        }
        write_json(output / "RUN_METADATA.json", metadata)
        summary = run_phase2(
            policy,
            estimator,
            phase_args,
            capacity=FORMAL_BATTERY_CAPACITY,
            output=output,
        )
        summary.update(
            phase2_counterfactual_audit(
                policy,
                phase_args,
                output,
                capacity=FORMAL_BATTERY_CAPACITY,
            )
        )
        attribution = phase2_switch_attribution(output)
        final = formal_summary(summary, attribution)
        write_energy_metrics(output)
        write_json(output / "phase2_summary.json", final)
        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "git_sha": metadata["git_sha"],
            "formal_phase2_500k": True,
            "phase2_summary": final,
        }
        write_json(output / "COMPLETED.json", completed)
        return completed
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Energy Risk v5 formal 500k Phase2 deployment"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--preregistered-run",
        default=str(DEFAULT_PREREGISTERED_RUN),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--navigation-device", default="cuda")
    parser.add_argument(
        "--phase2-transition-budget",
        type=int,
        default=FORMAL_TRANSITION_BUDGET,
    )
    parser.add_argument(
        "--phase2-episode-max-steps",
        type=int,
        default=FORMAL_TRANSITION_BUDGET + 1,
    )
    parser.add_argument(
        "--energy-reserve-fraction",
        type=float,
        default=FORMAL_RESERVE_FRACTION,
    )
    parser.add_argument("--log-frequency", type=int, default=1000)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
