from __future__ import annotations

import argparse
import copy
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.energy_mc.conformal import (
    PackedMissionDataset,
    collect_mission_trajectory,
    evaluate_group_conformal,
    evaluate_mission_conformal,
    mission_calibration_from_dataset,
)
from experiments.energy_mc.core import (
    PackedEnergyDataset,
    collect_energy_trajectory,
    generate_intersection_stratified_energy_goal_specs,
    generate_stratified_task_specs,
)
from review_bundle.safety.energy.mc_regression import (
    GROUP_CONFORMAL_ESTIMATOR_TYPE,
    EnergyToGoRegressor,
    HierarchicalConformalCalibration,
    HierarchicalConformalEnergyEstimator,
    ModelBasedEnergyRolloutEstimator,
)
from scripts.train_uav_energy_delivery_sac import environment_from_args
from scripts.train_uav_energy_mc import (
    file_sha256,
    git_clean,
    git_sha,
    load_frozen_sac,
    run_exhaustion_smoke,
    run_managed_lifecycle,
    run_phase2,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAC = ROOT / (
    "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
    "phase1_navigation/checkpoint_transition_500000.zip"
)
DEFAULT_POINT_ESTIMATOR = ROOT / (
    "artifacts/uav_energy_delivery_mc_formal_20260817_154321/"
    "energy_model/best_validation.pt"
)
DEFAULT_OLD_CALIBRATION_DATA = ROOT / (
    "artifacts/uav_energy_delivery_mc_formal_20260817_154321/"
    "energy_dataset/calibration"
)
DEFAULT_BATTERY_CALIBRATION = ROOT / (
    "artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614/"
    "battery_calibration/battery_calibration.json"
)
EXPECTED_POINT_SHA256 = "86b371ca92d6cdd337dc44a67e84f3242decd5661921be0f2fe1d5e4fce8f92b"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, object] | list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment_factory(args: argparse.Namespace):
    return lambda: environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)


def _mission_environment_factory(args: argparse.Namespace):
    mission_args = copy.copy(args)
    mission_args.minimum_task_distance = 5.0
    return lambda: environment_from_args(
        mission_args,
        phase=SACTrainingPhase.TD_PRETRAINING,
    )


def collect_intersection_split(
    policy,
    args: argparse.Namespace,
    *,
    count: int,
    seed: int,
    trajectory_id_offset: int,
    output: Path,
    label: str,
) -> PackedEnergyDataset:
    probe = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
    charger = probe.charger_position.copy()
    probe.close()
    specs = generate_intersection_stratified_energy_goal_specs(
        num_trajectories=count,
        seed=seed,
        charger_position=charger,
        trajectory_id_offset=trajectory_id_offset,
    )
    trajectories = []
    for index, spec in enumerate(specs):
        trajectories.append(
            collect_energy_trajectory(
                policy,
                _environment_factory(args),
                spec,
                seed=seed + index,
            )
        )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"[{label}] collected {index + 1}/{count}", flush=True)
    dataset = PackedEnergyDataset.from_trajectories(trajectories)
    dataset.save(output)
    return dataset


def collect_mission_split(
    policy,
    args: argparse.Namespace,
    *,
    count: int,
    seed: int,
    trajectory_id_offset: int,
    output: Path,
    label: str,
) -> PackedMissionDataset:
    probe = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
    charger = probe.charger_position.copy()
    probe.close()
    specs = generate_stratified_task_specs(
        num_trajectories=count,
        seed=seed,
        trajectory_id_offset=trajectory_id_offset,
    )
    trajectories = []
    for index, spec in enumerate(specs):
        trajectories.append(
            collect_mission_trajectory(
                policy,
                _mission_environment_factory(args),
                spec,
                charger_position=charger,
                seed=seed + index,
            )
        )
        if (index + 1) % 100 == 0 or index + 1 == count:
            print(f"[{label}] collected {index + 1}/{count}", flush=True)
    dataset = PackedMissionDataset.from_trajectories(trajectories)
    dataset.save(output)
    return dataset


def coverage_gate(
    goal_evaluation: dict[str, object],
    mission_evaluation: dict[str, object],
) -> dict[str, object]:
    goal_checks: dict[str, bool] = {
        "overall_at_least_95": goal_evaluation["overall"][
            "whole_trajectory_simultaneous_coverage"
        ]
        >= 0.95,
    }
    for goal_type in ("TASK", "CHARGER"):
        goal_checks[f"goal_type_{goal_type}_at_least_95"] = (
            goal_evaluation["by_goal_type"][goal_type][
                "whole_trajectory_simultaneous_coverage"
            ]
            >= 0.95
        )
    for bucket, metrics in goal_evaluation["by_initial_distance_bucket"].items():
        goal_checks[f"distance_{bucket}_at_least_95"] = (
            metrics["whole_trajectory_simultaneous_coverage"] >= 0.95
        )
    for group, metrics in goal_evaluation[
        "by_goal_type_and_initial_distance"
    ].items():
        if int(metrics["num_trajectories"]) >= 100:
            goal_checks[f"intersection_{group}_at_least_95"] = (
                metrics["whole_trajectory_simultaneous_coverage"] >= 0.95
            )
    mission_checks: dict[str, bool] = {
        "mission_overall_at_least_95": mission_evaluation["overall"][
            "whole_trajectory_simultaneous_coverage"
        ]
        >= 0.95,
    }
    for bucket, metrics in mission_evaluation[
        "by_initial_task_distance_bucket"
    ].items():
        mission_checks[f"mission_distance_{bucket}_at_least_95"] = (
            metrics["whole_trajectory_simultaneous_coverage"] >= 0.95
        )
    checks = {**goal_checks, **mission_checks}
    return {
        "nominal_target": 0.95,
        "engineering_minimum": 0.93,
        "all_primary_groups_reach_nominal_95": all(checks.values()),
        "checks": checks,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrected group-conformal UAV energy Phase2 pipeline"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-mc-energy-estimator", default=str(DEFAULT_POINT_ESTIMATOR))
    parser.add_argument("--resume-after-phase1-checkpoint", default=str(DEFAULT_SAC))
    parser.add_argument("--base-calibration-dataset", default=str(DEFAULT_OLD_CALIBRATION_DATA))
    parser.add_argument("--battery-calibration", default=str(DEFAULT_BATTERY_CALIBRATION))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--navigation-device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new-calibration-trajectories", type=int, default=2000)
    parser.add_argument("--conformal-test-trajectories", type=int, default=2000)
    parser.add_argument("--mission-calibration-trajectories", type=int, default=1000)
    parser.add_argument("--mission-test-trajectories", type=int, default=1000)
    parser.add_argument("--new-calibration-seed", type=int, default=510_001)
    parser.add_argument("--conformal-test-seed", type=int, default=520_001)
    parser.add_argument("--mission-calibration-seed", type=int, default=530_001)
    parser.add_argument("--mission-test-seed", type=int, default=540_001)
    parser.add_argument("--energy-test-seed", type=int, default=550_001)
    parser.add_argument("--energy-upper-coverage", type=float, default=0.95)
    parser.add_argument("--energy-reserve-fraction", type=float, default=0.10)
    parser.add_argument("--phase2-transition-budget", type=int, default=500_000)
    parser.add_argument("--phase2-episode-max-steps", type=int, default=600_000)
    parser.add_argument("--phase2-log-frequency", type=int, default=1000)
    parser.add_argument("--run-phase2", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--xy-sampling-margin", type=float, default=100.0)
    parser.add_argument("--task-z-min", type=float, default=20.0)
    parser.add_argument("--task-z-max", type=float, default=380.0)
    parser.add_argument("--phase1-episode-max-steps", type=int, default=4000)
    parser.add_argument("--render-vertical-exaggeration", type=float, default=4.0)
    parser.add_argument("--base-power", type=float, default=0.05)
    parser.add_argument(
        "--velocity-coefficients",
        type=float,
        nargs=3,
        default=[0.005, 0.005, 0.005],
    )
    parser.add_argument(
        "--acceleration-coefficients",
        type=float,
        nargs=3,
        default=[0.005, 0.005, 0.005],
    )
    parser.add_argument("--compute-power", type=float, default=0.005)
    parser.add_argument("--communication-power", type=float, default=0.005)
    parser.add_argument("--simulation-error", type=float, default=0.0)
    parser.add_argument("--flight-energy-multiplier", type=float, default=1.0)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    args = parser.parse_args(argv)
    seeds = {
        args.seed,
        args.new_calibration_seed,
        args.conformal_test_seed,
        args.mission_calibration_seed,
        args.mission_test_seed,
        args.energy_test_seed,
    }
    if len(seeds) != 6:
        parser.error("all formal seeds must be distinct")
    if args.phase2_episode_max_steps <= args.phase2_transition_budget and not args.smoke:
        parser.error(
            "formal single-env Phase2 guard must exceed the full transition budget"
        )
    if args.smoke:
        args.new_calibration_trajectories = 65
        args.conformal_test_trajectories = 65
        args.mission_calibration_trajectories = 25
        args.mission_test_trajectories = 25
        args.phase2_transition_budget = min(args.phase2_transition_budget, 1000)
        args.phase2_episode_max_steps = max(
            args.phase2_episode_max_steps,
            args.phase2_transition_budget + 1,
        )
    if args.mission_calibration_trajectories % 5 != 0 or args.mission_test_trajectories % 5 != 0:
        parser.error("mission split sizes must be positive multiples of five")
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_num_threads <= 0:
        raise ValueError("torch-num-threads must be positive")
    torch.set_num_threads(args.torch_num_threads)
    if not args.allow_dirty and not args.smoke and not git_clean():
        raise RuntimeError("corrected formal run requires a clean worktree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    for directory in (
        "calibration_v2",
        "final_conformal_test_v2",
        "mission_calibration",
        "mission_test_v2",
        "model",
        "lifecycle",
        "phase2",
    ):
        (output / directory).mkdir()
    write_json(
        output / "RUNNING.json",
        {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()},
    )
    try:
        point_path = Path(args.reuse_mc_energy_estimator).resolve()
        point_sha = file_sha256(point_path)
        if point_sha != EXPECTED_POINT_SHA256:
            raise RuntimeError(
                f"unexpected frozen point-estimator hash {point_sha}"
            )
        policy = load_frozen_sac(Path(args.resume_after_phase1_checkpoint), args.navigation_device)
        point_estimator = EnergyToGoRegressor.load(point_path, device=args.device)
        battery = json.loads(Path(args.battery_calibration).read_text(encoding="utf-8"))
        capacity = float(battery["calibrated_battery_capacity"])
        config = {
            "git_sha": git_sha(),
            "git_clean_at_start": git_clean(),
            "exact_argv": os.sys.argv,
            "source_sac_checkpoint": str(Path(args.resume_after_phase1_checkpoint).resolve()),
            "source_sac_sha256": file_sha256(Path(args.resume_after_phase1_checkpoint)),
            "mc_point_estimator_checkpoint": str(point_path),
            "mc_point_estimator_sha256": point_sha,
            "point_estimator_retrained": False,
            "base_calibration_dataset": str(Path(args.base_calibration_dataset).resolve()),
            "base_calibration_manifest_sha256": file_sha256(
                Path(args.base_calibration_dataset) / "manifest.json"
            ),
            "base_calibration_transitions_sha256": file_sha256(
                Path(args.base_calibration_dataset) / "transitions.npz"
            ),
            "split_seeds": {
                "new_calibration": args.new_calibration_seed,
                "conformal_test_v2": args.conformal_test_seed,
                "mission_calibration": args.mission_calibration_seed,
                "mission_test_v2": args.mission_test_seed,
                "energy_exhaustion": args.energy_test_seed,
            },
            "trajectory_id_ranges": {
                "new_calibration_start": 2_000_000,
                "conformal_test_v2_start": 3_000_000,
                "mission_calibration_start": 4_000_000,
                "mission_test_v2_start": 5_000_000,
            },
            "new_calibration_trajectories": args.new_calibration_trajectories,
            "new_conformal_test_trajectories": args.conformal_test_trajectories,
            "mission_calibration_trajectories": args.mission_calibration_trajectories,
            "mission_test_trajectories": args.mission_test_trajectories,
            "conformal_method": "trajectory_level_hierarchical_global_goal_distance_max",
            "mission_calibration": "trajectory_level_global_and_task_distance",
            "coverage_target": args.energy_upper_coverage,
            "battery_capacity": capacity,
            "energy_reserve_fraction": args.energy_reserve_fraction,
            "phase2_transition_budget": args.phase2_transition_budget,
            "phase2_episode_max_steps": args.phase2_episode_max_steps,
            "formal_guard_cannot_trigger": args.phase2_episode_max_steps
            > args.phase2_transition_budget,
            "single_phase2_environment": True,
            "torch_num_threads": args.torch_num_threads,
        }
        write_json(output / "config.json", config)

        oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=4000)
        oracle_lifecycle = run_managed_lifecycle(
            policy,
            oracle,
            args,
            capacity=capacity,
            seed=args.seed + 600_000,
            max_steps=30_000,
            oracle_decision_interval=500,
        )
        write_json(output / "lifecycle" / "oracle.json", oracle_lifecycle)
        if not oracle_lifecycle["passed"]:
            raise RuntimeError("oracle lifecycle failed")

        old_calibration = PackedEnergyDataset.load(args.base_calibration_dataset)
        new_calibration = collect_intersection_split(
            policy,
            args,
            count=args.new_calibration_trajectories,
            seed=args.new_calibration_seed,
            trajectory_id_offset=2_000_000,
            output=output / "calibration_v2" / "new_trajectories",
            label="calibration_v2",
        )
        combined_calibration = PackedEnergyDataset.concatenate(
            [old_calibration, new_calibration]
        )
        calibration = HierarchicalConformalCalibration.fit(
            point_estimator.predict_batch(combined_calibration.states),
            combined_calibration.targets,
            combined_calibration.trajectory_ids,
            combined_calibration.goal_types,
            combined_calibration.distance_buckets,
            coverage=args.energy_upper_coverage,
        )
        write_json(output / "calibration_v2" / "hierarchical_calibration.json", calibration.as_dict())
        if not args.smoke and min(calibration.intersection_counts.values()) < 150:
            raise RuntimeError("important calibration intersections have fewer than 150 trajectories")

        test_v2 = collect_intersection_split(
            policy,
            args,
            count=args.conformal_test_trajectories,
            seed=args.conformal_test_seed,
            trajectory_id_offset=3_000_000,
            output=output / "final_conformal_test_v2" / "trajectories",
            label="test_v2",
        )
        if combined_calibration.successful_trajectory_ids & test_v2.successful_trajectory_ids:
            raise RuntimeError("conformal test v2 leaked into calibration")

        mission_calibration_data = collect_mission_split(
            policy,
            args,
            count=args.mission_calibration_trajectories,
            seed=args.mission_calibration_seed,
            trajectory_id_offset=4_000_000,
            output=output / "mission_calibration" / "trajectories",
            label="mission_calibration",
        )
        mission_test = collect_mission_split(
            policy,
            args,
            count=args.mission_test_trajectories,
            seed=args.mission_test_seed,
            trajectory_id_offset=5_000_000,
            output=output / "mission_test_v2" / "trajectories",
            label="mission_test_v2",
        )
        if mission_calibration_data.successful_mission_ids & mission_test.successful_mission_ids:
            raise RuntimeError("mission test v2 leaked into mission calibration")
        mission_calibration = mission_calibration_from_dataset(
            point_estimator,
            mission_calibration_data,
            coverage=args.energy_upper_coverage,
        )
        write_json(
            output / "mission_calibration" / "mission_calibration.json",
            mission_calibration.as_dict(),
        )

        estimator = HierarchicalConformalEnergyEstimator(
            point_estimator,
            calibration,
            mission_calibration,
        )
        estimator.save(output / "model" / "group_conformal_energy_estimator.pt")
        goal_evaluation = evaluate_group_conformal(estimator, test_v2)
        mission_evaluation = evaluate_mission_conformal(estimator, mission_test)
        write_json(
            output / "final_conformal_test_v2" / "evaluation.json",
            goal_evaluation,
        )
        write_json(
            output / "mission_test_v2" / "evaluation.json",
            mission_evaluation,
        )
        readiness = coverage_gate(goal_evaluation, mission_evaluation)
        readiness["calibration_intersection_counts"] = calibration.intersection_counts
        write_json(output / "CONFORMAL_READINESS.json", readiness)

        learned_lifecycle = run_managed_lifecycle(
            policy,
            estimator,
            args,
            capacity=capacity,
            seed=args.seed + 700_000,
            max_steps=30_000,
        )
        exhaustion = run_exhaustion_smoke(policy, args)
        write_json(output / "lifecycle" / "learned_group_conformal.json", learned_lifecycle)
        write_json(output / "lifecycle" / "energy_exhaustion.json", exhaustion)
        formal_ready = bool(
            readiness["all_primary_groups_reach_nominal_95"]
            and learned_lifecycle["passed"]
            and exhaustion["passed"]
        )
        if not formal_ready and not args.smoke:
            stopped = {
                "status": "STOPPED_AFTER_CONFORMAL_CALIBRATION",
                "readiness": readiness,
                "oracle_lifecycle": oracle_lifecycle,
                "learned_lifecycle": learned_lifecycle,
                "energy_exhaustion": exhaustion,
            }
            write_json(output / "STOPPED_AFTER_CONFORMAL_CALIBRATION.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return stopped

        phase2 = None
        if args.run_phase2:
            phase2 = run_phase2(
                policy,
                estimator,
                args,
                capacity=capacity,
                output=output / "phase2",
            )
            if int(phase2["guard_truncated_segments"]) != 0:
                raise RuntimeError("corrected formal Phase2 triggered the emergency guard")
        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "energy_estimator_type": GROUP_CONFORMAL_ESTIMATOR_TYPE,
            "point_estimator_retrained": False,
            "conformal_readiness": readiness,
            "oracle_lifecycle": oracle_lifecycle,
            "learned_lifecycle": learned_lifecycle,
            "energy_exhaustion": exhaustion,
            "phase2": phase2,
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return completed
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    run(parse_args())
