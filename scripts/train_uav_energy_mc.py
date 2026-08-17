from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.energy_mc.core import (
    DISTANCE_BUCKETS,
    EnergyGoalSpec,
    PackedEnergyDataset,
    assert_disjoint_trajectory_splits,
    collect_energy_trajectory,
    generate_energy_goal_specs,
)
from review_bundle.safety.energy.mc_regression import (
    ENERGY_ESTIMATOR_TYPE,
    DistanceEnergyEstimator,
    EnergyToGoRegressor,
    ModelBasedEnergyRolloutEstimator,
    energy_regression_metrics,
)
from review_bundle.safety.switching import SortieMode
from scripts.train_uav_energy_delivery_sac import (
    ENERGY_UNIT,
    environment_from_args,
    freeze_navigation_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAC = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip"
)
DEFAULT_CALIBRATION = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614/battery_calibration/battery_calibration.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_clean() -> bool:
    return not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(payload), sort_keys=True) + "\n")


def load_frozen_sac(path: Path, device: str) -> SAC:
    if not path.is_file():
        raise FileNotFoundError(path)
    model = SAC.load(path, device=device)
    if model.observation_space.shape != (7,) or model.action_space.shape != (3,):
        raise ValueError("source SAC must use the 7D relative-goal contract and 3D action")
    freeze_navigation_policy(model)
    if any(parameter.requires_grad for parameter in model.policy.parameters()):
        raise RuntimeError("source SAC policy is not frozen")
    return model


def _environment_factory(args: argparse.Namespace):
    return lambda: environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)


def collect_split(
    policy: SAC,
    args: argparse.Namespace,
    *,
    name: str,
    count: int,
    seed: int,
    trajectory_id_offset: int,
    output: Path,
) -> PackedEnergyDataset:
    probe = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
    charger = probe.charger_position
    probe.close()
    specs = generate_energy_goal_specs(
        num_trajectories=count,
        seed=seed,
        charger_position=charger,
        trajectory_id_offset=trajectory_id_offset,
    )
    trajectories = []
    for index, spec in enumerate(specs):
        trajectory = collect_energy_trajectory(
            policy,
            _environment_factory(args),
            spec,
            seed=seed + index,
        )
        trajectories.append(trajectory)
        if (index + 1) % max(1, min(100, count)) == 0 or index + 1 == count:
            print(
                f"[{name}] collected {index + 1}/{count} trajectories",
                flush=True,
            )
    dataset = PackedEnergyDataset.from_trajectories(trajectories)
    dataset.save(output / name)
    return dataset


def grouped_evaluation(
    estimator: EnergyToGoRegressor,
    dataset: PackedEnergyDataset,
) -> dict[str, object]:
    predictions = estimator.predict_batch(dataset.states)
    upper = predictions + estimator.upper_delta

    def metrics(mask: np.ndarray) -> dict[str, object]:
        if not np.any(mask):
            return {"count": 0}
        return energy_regression_metrics(
            predictions[mask],
            dataset.targets[mask],
            upper_bounds=upper[mask],
        )

    all_mask = np.ones(dataset.targets.shape, dtype=bool)
    return {
        "overall": metrics(all_mask),
        "by_goal_type": {
            goal_type: metrics(dataset.goal_types == goal_type)
            for goal_type in np.unique(dataset.goal_types)
        },
        "by_distance_bucket": {
            bucket_name: metrics(dataset.distance_buckets == bucket_name)
            for bucket_name, _, _ in DISTANCE_BUCKETS
        },
        "by_boundary_contact": {
            "clean": metrics(~dataset.boundary_contact_trajectories),
            "boundary_contact": metrics(dataset.boundary_contact_trajectories),
        },
    }


def readiness_audit(evaluation: dict[str, object]) -> dict[str, object]:
    overall = evaluation["overall"]
    mean_true = float(overall["mean_true_energy"])
    catastrophic_buckets = [
        name
        for name, metrics in evaluation["by_distance_bucket"].items()
        if metrics.get("count", 0) > 0
        and (
            not bool(metrics.get("finite_predictions", False))
            or float(metrics.get("mean_relative_error", np.inf)) > 0.25
        )
    ]
    checks = {
        "finite_predictions": bool(overall["finite_predictions"]),
        "positive_predictions": bool(overall["positive_predictions"]),
        "mae_less_than_half_mean_true_energy": float(overall["MAE"]) < 0.5 * mean_true,
        "mean_relative_error_at_most_10_percent": float(overall["mean_relative_error"]) <= 0.10,
        "absolute_bias_at_most_10_percent_mean_true": abs(float(overall["bias"])) <= 0.10 * mean_true,
        "upper95_coverage_at_least_93_percent": float(overall["upper95_coverage"]) >= 0.93,
        "no_catastrophic_distance_bucket": not catastrophic_buckets,
    }
    return {
        "energy_estimator_ready": all(checks.values()),
        "checks": checks,
        "catastrophic_distance_buckets": catastrophic_buckets,
    }


def run_oracle_baseline(
    oracle: ModelBasedEnergyRolloutEstimator,
    policy: SAC,
    args: argparse.Namespace,
    specs: list[EnergyGoalSpec],
    output: Path,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        environment = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
        environment.reset(
            seed=args.energy_test_seed + index,
            options={
                "start_position": spec.start_position,
                "start_velocity": spec.initial_velocity,
                "task_point": spec.goal_position,
            },
        )
        prediction = oracle.estimate_context(environment, spec.goal_position)
        truth = collect_energy_trajectory(
            policy,
            _environment_factory(args),
            spec,
            seed=args.energy_test_seed + index,
        )
        records.append(
            {
                "trajectory_id": spec.trajectory_id,
                "goal_type": spec.goal_type,
                "true_energy": truth.total_realized_energy,
                "predicted_energy": prediction.prediction,
                "absolute_error": abs(prediction.prediction - truth.total_realized_energy),
                "rollout_simulation_steps": prediction.rollout_steps,
                "wall_clock_inference_seconds": prediction.wall_clock_seconds,
            }
        )
        environment.close()
    summary = {
        "estimator_type": oracle.estimator_type,
        "num_trajectories": len(records),
        "MAE": float(np.mean([row["absolute_error"] for row in records])),
        "mean_rollout_simulation_steps": float(
            np.mean([row["rollout_simulation_steps"] for row in records])
        ),
        "mean_wall_clock_inference_seconds": float(
            np.mean([row["wall_clock_inference_seconds"] for row in records])
        ),
        "records": records,
    }
    write_json(output, summary)
    return summary


def run_managed_lifecycle(
    policy: SAC,
    estimator,
    args: argparse.Namespace,
    *,
    capacity: float,
    seed: int,
    max_steps: int,
    required_cycles: int = 2,
    oracle_decision_interval: int | None = None,
) -> dict[str, object]:
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=capacity,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=lambda observation: np.asarray(
            policy.predict(observation, deterministic=True)[0], dtype=np.float32
        ),
        training_enabled=False,
    )
    environment.enable_phase_two()
    manual_oracle = oracle_decision_interval is not None
    if manual_oracle:
        environment.mission_switching_enabled = False
    observation, _ = environment.reset(seed=seed)
    cycle_records: list[dict[str, object]] = []
    switch_count = 0
    terminated = False
    truncated = False
    end_reason = None
    for step in range(max_steps):
        if (
            manual_oracle
            and environment.mode is SortieMode.TASK
            and step % int(oracle_decision_interval) == 0
        ):
            switched = environment._refresh_mission_decision()
            if switched:
                switch_count += 1
                environment._finalize_goal_trajectory(
                    success=False,
                    censored_reason="task_interrupted_by_charger_commitment",
                )
                environment.steps_in_current_task = 0
                environment._start_goal_trajectory(environment.charger_position)
                observation = environment.sac_observation_for_goal(environment.charger_position)
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        if not manual_oracle:
            switch_count += int(info["switched_now"])
        if info["battery_cycle_record"] is not None:
            cycle_records.append(dict(info["battery_cycle_record"]))
        if terminated or truncated or len(cycle_records) >= required_cycles:
            end_reason = info["end_reason"]
            break
    successful_cycles = [row for row in cycle_records if bool(row["return_success"])]
    tasks_by_cycle = [int(row["tasks_completed_in_cycle"]) for row in successful_cycles]
    consecutive_zero = 0
    max_consecutive_zero = 0
    total_tasks = 0
    for tasks in tasks_by_cycle:
        consecutive_zero = consecutive_zero + 1 if tasks == 0 else 0
        max_consecutive_zero = max(max_consecutive_zero, consecutive_zero)
    passed = bool(
        len(successful_cycles) >= required_cycles
        and all(tasks > 0 for tasks in tasks_by_cycle[:required_cycles])
        and max_consecutive_zero == 0
        and not terminated
        and not truncated
    )
    summary = {
        "estimator_type": estimator.estimator_type,
        "passed": passed,
        "policy_steps": int(environment.current_step),
        "tasks_completed": int(environment.tasks_completed),
        "completed_battery_cycles": len(cycle_records),
        "successful_battery_cycles": len(successful_cycles),
        "tasks_by_successful_cycle": tasks_by_cycle,
        "switch_count": switch_count,
        "max_consecutive_charger_returns_without_task": max_consecutive_zero,
        "charger_loop_detected": max_consecutive_zero > 0,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "end_reason": end_reason,
        "cycle_records": cycle_records,
    }
    environment.close()
    return summary


def run_exhaustion_smoke(
    policy: SAC,
    args: argparse.Namespace,
) -> dict[str, object]:
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=0.02,
    )
    environment.enable_battery_validation()
    observation, _ = environment.reset(seed=args.energy_test_seed + 90_000)
    result: dict[str, object] = {}
    for _ in range(100):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            result = {
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "end_reason": info["end_reason"],
                "remaining_energy": info["remaining_energy"],
                "passed": bool(terminated and info["end_reason"] == "energy_exhausted"),
            }
            break
    environment.close()
    return result


def run_phase2(
    policy: SAC,
    estimator,
    args: argparse.Namespace,
    *,
    capacity: float,
    output: Path,
) -> dict[str, object]:
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=capacity,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=lambda observation: np.asarray(
            policy.predict(observation, deterministic=True)[0], dtype=np.float32
        ),
        training_enabled=False,
    )
    environment.enable_phase_two()
    observation, _ = environment.reset(seed=args.seed + 200_000)
    completed_cycles: list[dict[str, object]] = []
    partial_segments: list[dict[str, object]] = []
    switch_events = 0
    exhaustions = 0
    truncations = 0
    total_tasks = 0
    tasks_after_recharge = 0
    gym_episodes = 1
    has_recharged = False
    consecutive_zero = 0
    max_consecutive_zero = 0
    completed_task_stream: list[list[float]] = []
    for transition in range(1, args.phase2_transition_budget + 1):
        task_goal_before_step = environment.current_task_point.copy()
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        switch_events += int(info["switched_now"])
        total_tasks += int(info["task_completed_now"])
        if info["task_completed_now"]:
            completed_task_stream.append(task_goal_before_step.astype(float).tolist())
            append_jsonl(
                output / "completed_task_stream.jsonl",
                {
                    "task_index": len(completed_task_stream) - 1,
                    "completion_transition": transition,
                    "task_goal": task_goal_before_step,
                },
            )
        if info["task_completed_now"] and has_recharged:
            tasks_after_recharge += 1
        if info["switched_now"] and environment.switching_events:
            append_jsonl(output / "switching_events.jsonl", environment.switching_events[-1])
        if info["battery_cycle_record"] is not None:
            record = dict(info["battery_cycle_record"])
            append_jsonl(output / "battery_cycles.jsonl", record)
            if bool(record["return_success"]):
                completed_cycles.append(record)
                append_jsonl(output / "completed_recharge_cycles.jsonl", record)
                has_recharged = True
                zero = int(record["tasks_completed_in_cycle"]) == 0
                consecutive_zero = consecutive_zero + 1 if zero else 0
                max_consecutive_zero = max(max_consecutive_zero, consecutive_zero)
            else:
                partial_segments.append(record)
                append_jsonl(output / "partial_battery_segments.jsonl", record)
        if transition % args.phase2_log_frequency == 0 or terminated or truncated:
            append_jsonl(
                output / "training_curve.jsonl",
                {
                    "global_env_transitions": transition,
                    "tasks_completed": total_tasks,
                    "tasks_per_1000_transitions": 1000.0
                    * float(total_tasks)
                    / transition,
                    "mode": info["mode"],
                    "remaining_energy": info["remaining_energy"],
                    "remaining_energy_fraction": info["remaining_energy_fraction"],
                    "step_realized_energy": info["realized_energy_cost"],
                    "task_energy_prediction": info["task_energy_prediction"],
                    "task_energy_upper95": info["task_energy_upper95"],
                    "return_now_energy_prediction": info["return_now_energy_prediction"],
                    "return_now_energy_upper95": info["return_now_energy_upper95"],
                    "return_after_task_energy_prediction": info[
                        "return_after_task_energy_prediction"
                    ],
                    "return_after_task_energy_upper95": info[
                        "return_after_task_energy_upper95"
                    ],
                    "mission_energy_upper95": info["mission_energy_upper95"],
                    "mission_energy_prediction": info["mission_energy_prediction"],
                    "mission_component_upper95_sum": info[
                        "mission_component_upper95_sum"
                    ],
                    "energy_reserve": info["energy_reserve"],
                    "battery_cycle_id": info["battery_cycle_id"],
                },
            )
        if terminated or truncated:
            exhaustions += int(terminated and info["end_reason"] == "energy_exhausted")
            truncations += int(truncated)
            observation, _ = environment.reset(seed=args.seed + 200_000 + transition)
            gym_episodes += 1
            has_recharged = False
    if environment.cycle_policy_steps > 0:
        budget_partial = {
            "segment_type": "training_budget_partial_segment",
            "battery_cycle_id": int(environment.battery_cycle_id),
            "cycle_start_transition": int(environment.cycle_start_global_step),
            "cycle_end_transition": int(args.phase2_transition_budget),
            "cycle_policy_steps": int(environment.cycle_policy_steps),
            "cycle_simulation_time": float(
                environment.simulation_time - environment.cycle_start_simulation_time
            ),
            "tasks_completed_in_cycle": int(environment.tasks_in_current_battery_cycle),
            "energy_used": float(environment.cycle_start_energy - environment.agent.energy),
            "remaining_energy_at_segment_end": float(environment.agent.energy),
            "remaining_energy_fraction_at_segment_end": float(
                environment.agent.energy / capacity
            ),
            "mode_at_segment_end": environment.mode.value,
            "return_commit_step": None
            if environment.return_commit_record is None
            else environment.return_commit_record.get("global_step"),
            "return_success": False,
            "charger_reached": False,
            "energy_exhausted": False,
            "emergency_time_limit": False,
        }
        partial_segments.append(budget_partial)
        append_jsonl(output / "partial_battery_segments.jsonl", budget_partial)
    post_recharge_cycles = completed_cycles[1:]
    remaining_at_charger = [
        float(row["remaining_energy_at_cycle_end"]) for row in completed_cycles
    ]
    remaining_fraction_at_charger = [
        float(row["remaining_energy_fraction_at_charger"]) for row in completed_cycles
    ]
    guard_segments = [
        row for row in partial_segments if bool(row.get("emergency_time_limit", False))
    ]
    failed_returns = [
        row
        for row in partial_segments
        if row.get("return_commit_step") is not None and not bool(row.get("return_success"))
    ]
    completed_tasks = sum(
        int(row["tasks_completed_in_cycle"]) for row in completed_cycles
    )
    summary = {
        "energy_estimator_type": getattr(estimator, "estimator_type", ENERGY_ESTIMATOR_TYPE),
        "environment_transitions": args.phase2_transition_budget,
        "requested_transition_budget": args.phase2_transition_budget,
        "actual_training_transitions": args.phase2_transition_budget,
        "exact_budget_match": True,
        "total_delivery_tasks_completed": total_tasks,
        "completed_task_stream_length": len(completed_task_stream),
        "tasks_per_1000_transitions": 1000.0
        * float(total_tasks)
        / args.phase2_transition_budget,
        "completed_recharge_cycles": len(completed_cycles),
        "battery_cycles_completed": len(completed_cycles),
        "partial_battery_segments": len(partial_segments),
        "guard_truncated_segments": len(guard_segments),
        "gym_episodes": gym_episodes,
        "charger_arrivals": len(completed_cycles),
        "successful_recharges": len(completed_cycles),
        "tasks_per_completed_recharge_cycle": None
        if not completed_cycles
        else float(completed_tasks / len(completed_cycles)),
        "charger_returns_attempted": switch_events,
        "successful_autonomous_returns": len(completed_cycles),
        "charger_returns_successful": len(completed_cycles),
        "failed_autonomous_returns": len(failed_returns),
        "mean_remaining_energy_at_real_charger_arrival": None
        if not remaining_at_charger
        else float(np.mean(remaining_at_charger)),
        "mean_remaining_energy_at_charger": None
        if not remaining_at_charger
        else float(np.mean(remaining_at_charger)),
        "mean_remaining_fraction_at_real_charger_arrival": None
        if not remaining_fraction_at_charger
        else float(np.mean(remaining_fraction_at_charger)),
        "mean_remaining_energy_fraction_at_charger": None
        if not remaining_fraction_at_charger
        else float(np.mean(remaining_fraction_at_charger)),
        "mean_energy_utilization_per_completed_recharge_cycle": None
        if not remaining_fraction_at_charger
        else float(1.0 - np.mean(remaining_fraction_at_charger)),
        "mean_energy_utilization": None
        if not remaining_fraction_at_charger
        else float(1.0 - np.mean(remaining_fraction_at_charger)),
        "energy_exhaustions": exhaustions,
        "energy_exhaustion_count": exhaustions,
        "energy_exhaustion_rate": float(exhaustions / max(switch_events, 1)),
        "tasks_completed_after_recharge": tasks_after_recharge,
        "fraction_of_cycles_with_post_recharge_task": None
        if not post_recharge_cycles
        else float(
            np.mean([int(row["tasks_completed_in_cycle"]) > 0 for row in post_recharge_cycles])
        ),
        "consecutive_charger_returns_without_task": max_consecutive_zero,
        "charger_loop_count": sum(
            int(row["tasks_completed_in_cycle"]) == 0 for row in completed_cycles
        ),
        "CHARGER_LOOP_WARNING": max_consecutive_zero > 2,
        "emergency_truncations": truncations,
        "energy_reserve_fraction": args.energy_reserve_fraction,
        "energy_reserve_absolute": capacity * args.energy_reserve_fraction,
        "trajectory_global_margin": getattr(
            getattr(estimator, "trajectory_calibration", None),
            "global_margin",
            getattr(estimator, "upper_delta", None),
        ),
        "mission_global_margin": getattr(
            getattr(estimator, "mission_calibration", None),
            "global_margin",
            None,
        ),
        "sac_frozen": True,
    }
    write_json(output / "summary.json", summary)
    estimator.save(output / "energy_to_go_final.pt")
    environment.close()
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC-supervised UAV Energy-to-Go pipeline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-after-phase1-checkpoint", default=str(DEFAULT_SAC))
    parser.add_argument("--battery-calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--energy-estimator", choices=["mc-regression"], default="mc-regression")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--navigation-device", default="cpu")
    parser.add_argument("--energy-deployment-device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-training-trajectories", type=int, default=4000)
    parser.add_argument("--energy-validation-trajectories", type=int, default=500)
    parser.add_argument("--energy-calibration-trajectories", type=int, default=500)
    parser.add_argument("--energy-test-trajectories", type=int, default=500)
    parser.add_argument("--energy-training-seed", type=int, default=130_001)
    parser.add_argument("--energy-validation-seed", type=int, default=140_001)
    parser.add_argument("--energy-calibration-seed", type=int, default=150_001)
    parser.add_argument("--energy-test-seed", type=int, default=160_001)
    parser.add_argument("--energy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--energy-batch-size", type=int, default=256)
    parser.add_argument("--energy-max-epochs", type=int, default=200)
    parser.add_argument("--energy-patience", type=int, default=20)
    parser.add_argument("--energy-upper-coverage", type=float, default=0.95)
    parser.add_argument("--energy-target-normalization", choices=["battery-capacity"], default="battery-capacity")
    parser.add_argument("--energy-upper-bound", choices=["one-sided-conformal"], default="one-sided-conformal")
    parser.add_argument("--energy-reserve-fraction", type=float, default=0.10)
    parser.add_argument("--phase2-transition-budget", type=int, default=500_000)
    parser.add_argument("--phase2-episode-max-steps", type=int, default=20_000)
    parser.add_argument("--phase2-log-frequency", type=int, default=1000)
    parser.add_argument("--run-phase2", action="store_true")
    parser.add_argument("--oracle-baseline-trajectories", type=int, default=10)
    parser.add_argument("--oracle-lifecycle-max-steps", type=int, default=30_000)
    parser.add_argument("--oracle-decision-interval", type=int, default=500)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--learned-lifecycle-max-steps", type=int, default=30_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--xy-sampling-margin", type=float, default=100.0)
    parser.add_argument("--task-z-min", type=float, default=20.0)
    parser.add_argument("--task-z-max", type=float, default=380.0)
    parser.add_argument("--phase1-episode-max-steps", type=int, default=4000)
    parser.add_argument("--render-vertical-exaggeration", type=float, default=4.0)
    parser.add_argument("--base-power", type=float, default=0.05)
    parser.add_argument("--velocity-coefficients", type=float, nargs=3, default=[0.005, 0.005, 0.005])
    parser.add_argument("--acceleration-coefficients", type=float, nargs=3, default=[0.005, 0.005, 0.005])
    parser.add_argument("--compute-power", type=float, default=0.005)
    parser.add_argument("--communication-power", type=float, default=0.005)
    parser.add_argument("--simulation-error", type=float, default=0.0)
    parser.add_argument("--flight-energy-multiplier", type=float, default=1.0)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    args = parser.parse_args(argv)
    counts = (
        args.energy_training_trajectories,
        args.energy_validation_trajectories,
        args.energy_calibration_trajectories,
        args.energy_test_trajectories,
    )
    if any(count <= 0 or count % 5 != 0 for count in counts):
        parser.error("all energy split trajectory counts must be positive multiples of five")
    seeds = {
        args.seed,
        args.energy_training_seed,
        args.energy_validation_seed,
        args.energy_calibration_seed,
        args.energy_test_seed,
    }
    if len(seeds) != 5:
        parser.error("training and all energy split seeds must be distinct")
    if args.smoke:
        args.energy_training_trajectories = 75
        args.energy_validation_trajectories = 15
        args.energy_calibration_trajectories = 15
        args.energy_test_trajectories = 15
        args.energy_max_epochs = min(args.energy_max_epochs, 80)
        args.energy_patience = min(args.energy_patience, 12)
        args.oracle_baseline_trajectories = min(args.oracle_baseline_trajectories, 5)
        args.phase2_transition_budget = min(args.phase2_transition_budget, 2000)
    return args


def train(args: argparse.Namespace) -> dict[str, object]:
    if not args.allow_dirty and not args.smoke and not git_clean():
        raise RuntimeError("formal MC experiment requires a clean worktree")
    output = Path(args.output_dir)
    torch.set_num_threads(args.torch_num_threads)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("phase1_navigation", "energy_dataset", "energy_model", "oracle", "lifecycle", "phase2"):
        (output / directory).mkdir()
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()})
    sac_path = Path(args.resume_after_phase1_checkpoint).expanduser().resolve()
    calibration_path = Path(args.battery_calibration).expanduser().resolve()
    try:
        policy = load_frozen_sac(sac_path, args.navigation_device)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        capacity = float(calibration["calibrated_battery_capacity"])
        config = {
            "status": "RUNNING",
            "started_at": utc_now(),
            "git_sha": git_sha(),
            "git_clean_at_start": git_clean(),
            "pid": os.getpid(),
            "exact_argv": os.sys.argv,
            "source_sac_checkpoint": str(sac_path),
            "source_sac_sha256": file_sha256(sac_path),
            "navigation_inference_device": args.navigation_device,
            "energy_training_device": args.device,
            "energy_deployment_device": args.energy_deployment_device,
            "source_sac_training_transitions": 500_000,
            "navigation_policy": "frozen_sac_500k",
            "battery_calibration": str(calibration_path),
            "battery_capacity": capacity,
            "energy_estimator_type": ENERGY_ESTIMATOR_TYPE,
            "bootstrapping": False,
            "gamma": None,
            "energy_input": "velocity3_goal_direction3_linear_distance_over_dmax1",
            "target_normalization": "mc_energy_to_go_over_calibrated_battery_capacity",
            "split_unit": "complete_goal_trajectory",
            "split_trajectories": {
                "train": args.energy_training_trajectories,
                "validation": args.energy_validation_trajectories,
                "calibration": args.energy_calibration_trajectories,
                "test": args.energy_test_trajectories,
            },
            "upper_bound": "one_sided_split_conformal_trajectory_max_residual",
            "upper_coverage_target": args.energy_upper_coverage,
            "energy_reserve_fraction": args.energy_reserve_fraction,
            "phase2_transition_budget": args.phase2_transition_budget,
            "run_phase2": args.run_phase2,
        }
        write_json(output / "config.json", config)
        write_json(
            output / "phase1_navigation" / "resume_audit.json",
            {
                "phase1_retrained": False,
                "source_checkpoint": str(sac_path),
                "source_checkpoint_sha256": file_sha256(sac_path),
                "sac_frozen": True,
            },
        )

        oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=args.phase1_episode_max_steps)
        oracle_lifecycle = run_managed_lifecycle(
            policy,
            oracle,
            args,
            capacity=capacity,
            seed=args.seed + 300_000,
            max_steps=args.oracle_lifecycle_max_steps,
            oracle_decision_interval=args.oracle_decision_interval,
        )
        write_json(output / "oracle" / "lifecycle.json", oracle_lifecycle)
        print(f"Oracle lifecycle: {'PASS' if oracle_lifecycle['passed'] else 'FAIL'}", flush=True)
        if not oracle_lifecycle["passed"]:
            stopped = {
                "status": "STOPPED_AFTER_ORACLE_LIFECYCLE",
                "oracle_lifecycle": oracle_lifecycle,
            }
            write_json(output / "STOPPED_AFTER_ORACLE_LIFECYCLE.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return stopped

        split_definitions = (
            ("train", args.energy_training_trajectories, args.energy_training_seed),
            ("validation", args.energy_validation_trajectories, args.energy_validation_seed),
            ("calibration", args.energy_calibration_trajectories, args.energy_calibration_seed),
            ("test", args.energy_test_trajectories, args.energy_test_seed),
        )
        splits: dict[str, PackedEnergyDataset] = {}
        offset = 0
        for name, count, seed in split_definitions:
            splits[name] = collect_split(
                policy,
                args,
                name=name,
                count=count,
                seed=seed,
                trajectory_id_offset=offset,
                output=output / "energy_dataset",
            )
            offset += count
        print("Energy trajectory collection complete", flush=True)
        assert_disjoint_trajectory_splits(splits)
        write_json(
            output / "energy_dataset" / "split_audit.json",
            {
                "trajectory_level_split": True,
                "disjoint": True,
                "trajectory_ids": {
                    name: sorted(dataset.successful_trajectory_ids)
                    for name, dataset in splits.items()
                },
            },
        )

        probe = environment_from_args(args, phase=SACTrainingPhase.TD_PRETRAINING)
        oracle_specs = generate_energy_goal_specs(
            num_trajectories=max(5, int(np.ceil(args.oracle_baseline_trajectories / 5) * 5)),
            seed=args.energy_test_seed + 1,
            charger_position=probe.charger_position,
            trajectory_id_offset=1_000_000,
        )[: args.oracle_baseline_trajectories]
        probe.close()
        run_oracle_baseline(
            oracle,
            policy,
            args,
            oracle_specs,
            output / "oracle" / "baseline.json",
        )

        d_max = float(np.linalg.norm([4000.0, 4000.0, 400.0]))
        successful_train_metadata = [
            row for row in splits["train"].metadata if bool(row["success"])
        ]
        energy_per_meter = float(
            sum(float(row["total_realized_energy"]) for row in successful_train_metadata)
            / max(
                sum(float(row["initial_goal_distance"]) for row in successful_train_metadata),
                1e-12,
            )
        )
        distance_estimator = DistanceEnergyEstimator(
            energy_per_meter=energy_per_meter,
            d_max=d_max,
        )
        distance_predictions = distance_estimator.predict_batch(splits["test"].states)
        distance_baseline = {
            "estimator_type": distance_estimator.estimator_type,
            "energy_per_meter": energy_per_meter,
            "fit_split": "train_trajectories_only",
            "test_metrics": energy_regression_metrics(
                distance_predictions,
                splits["test"].targets,
            ),
        }
        write_json(output / "energy_model" / "distance_baseline.json", distance_baseline)
        write_json(
            output / "energy_model" / "historical_quantile_td_baseline.json",
            {
                "estimator_type": "historical_divergent_goal_conditioned_quantile_td",
                "artifact": str(
                    ROOT
                    / "artifacts/uav_energy_delivery_v3_energy_formal_20260816_175614/phase1_td"
                ),
                "status": "failed_baseline_retained_not_used_for_switching",
            },
        )

        estimator = EnergyToGoRegressor(
            battery_capacity=capacity,
            learning_rate=args.energy_learning_rate,
            batch_size=args.energy_batch_size,
            seed=args.seed,
            device=args.device,
        )
        history = estimator.fit(
            splits["train"].states,
            splits["train"].targets,
            splits["validation"].states,
            splits["validation"].targets,
            max_epochs=args.energy_max_epochs,
            patience=args.energy_patience,
            loss="huber",
        )
        estimator.save(output / "energy_model" / "best_validation.pt")
        print(
            f"Energy regressor trained: best epoch={history.best_epoch} "
            f"validation MAE={history.best_validation_mae:.6f}",
            flush=True,
        )
        write_json(output / "energy_model" / "training_history.json", history.__dict__)
        validation_evaluation = grouped_evaluation(estimator, splits["validation"])
        write_json(output / "energy_model" / "validation_evaluation.json", validation_evaluation)

        calibration_summary = estimator.calibrate_upper_bound(
            splits["calibration"].states,
            splits["calibration"].targets,
            splits["calibration"].trajectory_ids,
            coverage=args.energy_upper_coverage,
        )
        estimator.save(output / "energy_model" / "calibrated_best_validation.pt")
        write_json(output / "energy_model" / "upper95_calibration.json", calibration_summary)
        test_evaluation = grouped_evaluation(estimator, splits["test"])
        write_json(output / "energy_model" / "heldout_test.json", test_evaluation)
        readiness = readiness_audit(test_evaluation)
        print(
            f"Energy readiness: {'PASS' if readiness['energy_estimator_ready'] else 'FAIL'}",
            flush=True,
        )
        write_json(output / "energy_model" / "readiness.json", readiness)

        deployment_estimator = EnergyToGoRegressor.load(
            output / "energy_model" / "calibrated_best_validation.pt",
            device=args.energy_deployment_device,
        )
        learned_lifecycle = run_managed_lifecycle(
            policy,
            deployment_estimator,
            args,
            capacity=capacity,
            seed=args.seed + 400_000,
            max_steps=args.learned_lifecycle_max_steps,
        )
        write_json(output / "lifecycle" / "learned_estimator.json", learned_lifecycle)
        print(
            f"Learned lifecycle: {'PASS' if learned_lifecycle['passed'] else 'FAIL'}",
            flush=True,
        )
        exhaustion = run_exhaustion_smoke(policy, args)
        write_json(output / "lifecycle" / "energy_exhaustion.json", exhaustion)
        smoke_prediction_gate = bool(
            test_evaluation["overall"]["finite_predictions"]
            and test_evaluation["overall"]["positive_predictions"]
        )
        readiness_blocks = bool(
            not readiness["energy_estimator_ready"] and not args.smoke
        )
        if (
            readiness_blocks
            or not smoke_prediction_gate
            or not learned_lifecycle["passed"]
            or not exhaustion["passed"]
        ):
            stopped = {
                "status": "STOPPED_AFTER_ENERGY_ESTIMATOR",
                "energy_readiness": readiness,
                "smoke_prediction_gate": smoke_prediction_gate,
                "learned_lifecycle": learned_lifecycle,
                "energy_exhaustion": exhaustion,
            }
            write_json(output / "STOPPED_AFTER_ENERGY_ESTIMATOR.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return stopped

        phase2_summary = None
        if args.run_phase2:
            phase2_summary = run_phase2(
                policy,
                deployment_estimator,
                args,
                capacity=capacity,
                output=output / "phase2",
            )
        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "energy_estimator_type": ENERGY_ESTIMATOR_TYPE,
            "oracle_lifecycle": oracle_lifecycle,
            "energy_readiness": readiness,
            "learned_lifecycle": learned_lifecycle,
            "energy_exhaustion": exhaustion,
            "heldout_test": test_evaluation,
            "phase2": phase2_summary,
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
    train(parse_args())
