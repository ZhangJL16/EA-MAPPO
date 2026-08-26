from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import CallbackList

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.jacobian_energy_bridge.callbacks import SafetyBridgeCollectionCallback
from experiments.jacobian_energy_bridge.dataset import (
    PackedBridgeDataset,
    SafetyBridgeTrajectoryWriter,
    concatenate_bridge_datasets,
    load_bridge_dataset,
)
from experiments.jacobian_energy_bridge.energy_model import (
    ActionConditionedEnergyCritic,
    CalibratedCompactEnergyEstimator,
    FlexibleEnergyRegressor,
    SafetyBridgeEncoder,
    fit_result_dict,
)
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from experiments.jacobian_energy_bridge.safety_buffer import SafetyBridgeReplay
from review_bundle.safety.energy.mc_regression import (
    HierarchicalConformalCalibration,
    MissionConformalCalibration,
)
from scripts.train_uav_energy_delivery_sac import (
    DISTANCE_BUCKETS,
    HeuristicGoalPolicy,
    NavigationBudgetCallback,
    NavigationTask,
    batch_action_provider,
    environment_from_args,
    evaluate_navigation_tasks,
    freeze_navigation_policy,
    generate_navigation_curves,
    generate_stratified_navigation_tasks,
    make_navigation_vec_env,
    navigation_energy_gate_passed,
    navigation_observation_dim,
    navigation_safety_gate_passed,
    run_battery_calibration,
    run_battery_validation,
    save_navigation_tasks,
    single_action_provider,
    telemetry_config_from_args,
)


FORMAL_PHASE1_TRANSITIONS = 500_000
FORMAL_PHASE2_ENERGY_TRANSITIONS = 500_000
FORMAL_TOTAL_TRANSITIONS = 1_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()


def prepare_output_directory(path: str | Path) -> Path:
    output = Path(path)
    allowed_prelaunch_files = {
        "exact_command.txt",
        "git_sha.txt",
        "git_status.txt",
        "tmux_session.txt",
        "run.log",
    }
    if output.exists():
        unexpected = {
            entry.name
            for entry in output.iterdir()
            if entry.name not in allowed_prelaunch_files
        }
        if unexpected:
            raise FileExistsError(
                f"JSEB output directory is not fresh: {sorted(unexpected)}"
            )
    else:
        output.mkdir(parents=True)
    return output


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_value(payload), sort_keys=True) + "\n")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_hash(model: JacobianBridgeSAC) -> str:
    digest = hashlib.sha256()
    for parameter in model.policy.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jacobian Safety-Energy Bridge static-obstacle protocol")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-phase1-checkpoint")
    parser.add_argument("--source-phase1-artifact")
    parser.add_argument("--source-phase1-transition", type=int)
    parser.add_argument(
        "--intermediate-checkpoint-energy-ablation",
        action="store_true",
        help="reuse an explicitly named intermediate navigation checkpoint and run frozen-policy energy learning",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        default=6,
        help="parallel CPU environments with central batched GPU policy inference",
    )
    parser.add_argument("--evaluation-progress-interval-tasks", type=int, default=10)
    parser.add_argument("--phase1-transitions", type=int, default=FORMAL_PHASE1_TRANSITIONS)
    parser.add_argument(
        "--phase2-energy-transitions",
        type=int,
        default=FORMAL_PHASE2_ENERGY_TRANSITIONS,
        help="frozen-navigation mission transitions used only for Energy Model learning",
    )
    parser.add_argument("--phase1-episode-max-steps", type=int, default=4000)
    parser.add_argument("--phase2-episode-max-steps", type=int, default=20_000)
    parser.add_argument("--eval-freq-transitions", type=int, default=100_000)
    parser.add_argument(
        "--phase-end-eval-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable periodic navigation evaluation and evaluate only after Phase 1",
    )
    parser.add_argument("--checkpoint-freq-transitions", type=int, default=50_000)
    parser.add_argument("--log-freq-transitions", type=int, default=8_000)
    parser.add_argument("--gif-freq-transitions", type=int, default=500_000)
    parser.add_argument("--eval-navigation-tasks", type=int, default=500)
    parser.add_argument("--navigation-eval-max-steps", type=int, default=4000)
    parser.add_argument("--eval-task-seed", type=int, default=70_001)
    parser.add_argument("--battery-calibration-tasks", type=int, default=500)
    parser.add_argument("--battery-calibration-seed", type=int, default=80_001)
    parser.add_argument("--battery-validation-runs", type=int, default=100)
    parser.add_argument("--battery-validation-seed", type=int, default=90_001)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    parser.add_argument("--energy-reserve-fraction", type=float, default=0.10)
    parser.add_argument("--persistent-eval-transitions", type=int, default=100_000)
    parser.add_argument("--minimum-task-distance", type=float, default=100.0)
    parser.add_argument("--xy-sampling-margin", type=float, default=100.0)
    parser.add_argument("--task-z-min", type=float, default=20.0)
    parser.add_argument("--task-z-max", type=float, default=380.0)
    parser.add_argument("--lidar-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lidar-range", type=float, default=100.0)
    parser.add_argument("--lidar-horizontal-sectors", type=int, default=128)
    parser.add_argument("--lidar-vertical-sectors", type=int, default=8)
    parser.add_argument("--num-obstacles", type=int, default=24)
    parser.add_argument("--obstacle-radius-min", type=float, default=50.0)
    parser.add_argument("--obstacle-radius-max", type=float, default=120.0)
    parser.add_argument("--obstacle-sampling-margin", type=float, default=20.0)
    parser.add_argument("--obstacle-collision-penalty", type=float, default=1.2)
    parser.add_argument("--repeat-collision-scale", type=float, default=0.35)
    parser.add_argument("--safety-intervention-penalty", type=float, default=0.05)
    parser.add_argument("--hocbf-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hocbf-k1", type=float, default=1.0)
    parser.add_argument("--hocbf-k2", type=float, default=1.0)
    parser.add_argument("--hocbf-uncertainty-margin", type=float, default=1.0)
    parser.add_argument("--hocbf-top-k", type=int, default=16)
    parser.add_argument(
        "--projection-geometry-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--render-vertical-exaggeration", type=float, default=4.0)
    parser.add_argument("--gif-seed", type=int, default=110_001)
    parser.add_argument("--gif-frame-skip", type=int, default=20)
    parser.add_argument("--base-power", type=float, default=0.05)
    parser.add_argument("--velocity-coefficients", type=float, nargs=3, default=[0.005] * 3)
    parser.add_argument("--acceleration-coefficients", type=float, nargs=3, default=[0.005] * 3)
    parser.add_argument("--compute-power", type=float, default=0.005)
    parser.add_argument("--communication-power", type=float, default=0.005)
    parser.add_argument("--simulation-error", type=float, default=0.0)
    parser.add_argument("--flight-energy-multiplier", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gradient-steps", type=int, default=-1)
    parser.add_argument("--shield-loss-weight", type=float, default=0.10)
    parser.add_argument("--energy-loss-weight", type=float, default=0.02)
    parser.add_argument("--bridge-replay-capacity", type=int, default=50_000)
    parser.add_argument("--bridge-batch-size", type=int, default=256)
    parser.add_argument("--bridge-learning-starts", type=int, default=10_000)
    parser.add_argument("--bridge-trust-region", type=float, default=0.35)
    parser.add_argument("--bridge-gradient-clip", type=float, default=10.0)
    parser.add_argument("--bridge-slack-scale", type=float, default=5.0)
    parser.add_argument("--energy-hidden-dim", type=int, default=128)
    parser.add_argument("--energy-epochs", type=int, default=30)
    parser.add_argument("--energy-batch-size", type=int, default=256)
    parser.add_argument("--energy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--energy-warmup-transitions", type=int, default=25_000)
    parser.add_argument("--energy-ramp-transitions", type=int, default=75_000)
    parser.add_argument("--conformal-coverage", type=float, default=0.95)
    parser.add_argument(
        "--ablation",
        choices=("A", "B", "C", "D"),
        default="D",
        help="A=baseline, B=J-Safety, C=J-Context, D=full JSEB",
    )
    parser.add_argument(
        "--baseline-evaluation",
        default="artifacts/uav_energy_delivery_1024ray_hocbf_radius2x_1m_20260824_030434/eval/eval_transition_500000.json",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-failed-battery-calibration", action="store_true")
    args = parser.parse_args(argv)
    args.lidar_sectors = args.lidar_horizontal_sectors * args.lidar_vertical_sectors
    args.legacy_lidar_alias_used = False
    args.phase1_transition_budget = args.phase1_transitions
    args.phase1b_transition_budget = args.phase2_energy_transitions
    args.phase2_transition_budget = 0
    args.max_steps_per_task = args.phase1_episode_max_steps
    if args.smoke and args.pilot:
        parser.error("--smoke and --pilot are mutually exclusive")
    if args.intermediate_checkpoint_energy_ablation:
        if not args.resume_phase1_checkpoint:
            parser.error("intermediate checkpoint ablation requires --resume-phase1-checkpoint")
        if args.source_phase1_transition is None or args.source_phase1_transition <= 0:
            parser.error("intermediate checkpoint ablation requires a positive --source-phase1-transition")
    elif args.resume_phase1_checkpoint or args.source_phase1_transition is not None:
        parser.error("resume options require --intermediate-checkpoint-energy-ablation")
    if args.smoke:
        args.evaluation_num_envs = 1
        args.phase1_transitions = 2_000
        args.phase2_energy_transitions = 2_000
        args.num_envs = 4
        args.lidar_horizontal_sectors = 8
        args.lidar_vertical_sectors = 2
        args.lidar_sectors = 16
        args.eval_navigation_tasks = 5
        args.navigation_eval_max_steps = 250
        args.battery_calibration_tasks = 5
        args.battery_validation_runs = 1
        args.target_nominal_endurance_minutes = 0.1
        args.persistent_eval_transitions = 500
        args.learning_starts = 128
        args.batch_size = 64
        args.bridge_learning_starts = 64
        args.bridge_batch_size = 32
        args.bridge_replay_capacity = 2_000
        args.buffer_size = 5_000
        args.energy_epochs = 2
        args.energy_batch_size = 64
        args.energy_warmup_transitions = 100
        args.energy_ramp_transitions = 400
        args.eval_freq_transitions = 2_000
        args.checkpoint_freq_transitions = 1_000
        args.log_freq_transitions = 400
        args.gif_freq_transitions = 2_000
    elif args.pilot:
        args.evaluation_num_envs = 1
        args.phase1_transitions = 25_000
        args.phase2_energy_transitions = 25_000
        args.num_envs = 4
        args.eval_navigation_tasks = 10
        args.navigation_eval_max_steps = 1_200
        args.battery_calibration_tasks = 5
        args.battery_validation_runs = 1
        args.target_nominal_endurance_minutes = 0.5
        args.persistent_eval_transitions = 1_000
        args.learning_starts = 1_000
        args.bridge_learning_starts = 1_000
        args.bridge_replay_capacity = 20_000
        args.buffer_size = 50_000
        args.energy_epochs = 5
        args.energy_warmup_transitions = 2_000
        args.energy_ramp_transitions = 10_000
        args.eval_freq_transitions = 25_000
        args.checkpoint_freq_transitions = 5_000
        args.log_freq_transitions = 1_000
        args.gif_freq_transitions = 25_000
    args.phase1_transition_budget = args.phase1_transitions
    args.phase1b_transition_budget = args.phase2_energy_transitions
    args.phase2_transition_budget = 0
    args.energy_eval_tasks = max(5, args.eval_navigation_tasks)
    args.energy_eval_seed = 120_001
    args.energy_eval_freq_transitions = max(args.phase2_energy_transitions, 1)
    args.td_collection_seed = 100_001
    args.run_phase2 = True
    args.run_td_pretraining = False
    args.allow_failed_battery_calibration = bool(
        args.allow_failed_battery_calibration or args.smoke or args.pilot
    )
    if args.ablation == "A":
        args.projection_geometry_enabled = False
        args.shield_loss_weight = 0.0
        args.energy_loss_weight = 0.0
    elif args.ablation in {"B", "C"}:
        args.shield_loss_weight = max(float(args.shield_loss_weight), 0.0)
        args.energy_loss_weight = 0.0
    for name in ("phase1_transitions", "phase2_energy_transitions"):
        if getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.evaluation_num_envs <= 0:
        parser.error("evaluation-num-envs must be positive")
    if args.evaluation_progress_interval_tasks <= 0:
        parser.error("evaluation-progress-interval-tasks must be positive")
    if args.phase1_transitions % args.num_envs != 0:
        parser.error("Phase 1 budget must be divisible by num-envs")
    if not args.lidar_enabled or not args.hocbf_enabled or args.num_obstacles != 24:
        parser.error("JSEB requires static obstacles, LiDAR, and HOCBF")
    if not (args.smoke or args.pilot):
        if not args.phase_end_eval_only:
            parser.error("formal JSEB requires phase-end-only evaluation")
        if args.ablation != "D" or not args.projection_geometry_enabled:
            parser.error("formal JSEB requires full ablation D with projection geometry")
        total = sum(
            [
                args.phase1_transitions,
                args.phase2_energy_transitions,
            ]
        )
        expected_total = (
            args.phase2_energy_transitions
            if args.intermediate_checkpoint_energy_ablation
            else FORMAL_TOTAL_TRANSITIONS
        )
        if args.intermediate_checkpoint_energy_ablation:
            if expected_total != 500_000:
                parser.error("intermediate frozen-policy energy run requires exactly 500000 Phase 2 transitions")
        elif total != FORMAL_TOTAL_TRANSITIONS:
            parser.error("formal JSEB training budget must equal exactly 1,000,000 transitions")
        if (args.lidar_horizontal_sectors, args.lidar_vertical_sectors) != (128, 8):
            parser.error("formal JSEB requires 128 x 8 LiDAR")
        if (args.obstacle_radius_min, args.obstacle_radius_max) != (50.0, 120.0):
            parser.error("formal JSEB requires obstacle radii 50-120 m")
    return args


def compact_summary(summary: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in summary.items() if key != "records"}


def paired_navigation_comparison(
    candidate: dict[str, object],
    baseline_path: str | Path,
) -> dict[str, object]:
    path = Path(baseline_path)
    if not path.exists():
        return {"available": False, "reason": "baseline_evaluation_missing"}
    baseline = json.loads(path.read_text(encoding="utf-8"))
    candidate_records = {int(row["task_index"]): row for row in candidate["records"]}
    baseline_records = {int(row["task_index"]): row for row in baseline["records"]}
    common = sorted(set(candidate_records) & set(baseline_records))
    if not common:
        return {"available": False, "reason": "no_common_task_indices"}
    return {
        "available": True,
        "paired_tasks": len(common),
        "baseline_path": str(path),
        "success_rate_delta": float(
            np.mean([candidate_records[index]["success"] for index in common])
            - np.mean([baseline_records[index]["success"] for index in common])
        ),
        "mean_path_ratio_delta": float(
            np.mean([candidate_records[index]["path_ratio"] for index in common])
            - np.mean([baseline_records[index]["path_ratio"] for index in common])
        ),
        "intervention_rate_delta": float(
            candidate["hocbf_intervention_step_rate"]
            - baseline["hocbf_intervention_step_rate"]
        ),
        "emergency_rate_delta": float(
            candidate["hocbf_emergency_brake_step_rate"]
            - baseline["hocbf_emergency_brake_step_rate"]
        ),
        "collision_step_rate_delta": float(
            candidate["obstacle_collision_step_rate"]
            - baseline["obstacle_collision_step_rate"]
        ),
    }


def make_bridge_replay(args: argparse.Namespace, *, seed_offset: int = 0) -> SafetyBridgeReplay:
    return SafetyBridgeReplay(
        capacity=args.bridge_replay_capacity,
        observation_dim=navigation_observation_dim(args),
        seed=args.seed + seed_offset,
        maximum_barrier_constraints=args.hocbf_top_k,
        slack_scale=args.bridge_slack_scale,
    )


def collect_smoke_bridge_diagnostics(
    args: argparse.Namespace,
    writer: SafetyBridgeTrajectoryWriter,
    replay: SafetyBridgeReplay,
) -> dict[str, object]:
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    policy = HeuristicGoalPolicy()
    center = np.array([2000.0, 2000.0, 200.0], dtype=np.float32)
    offsets = (
        np.array([160.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 160.0, 0.0], dtype=np.float32),
        np.array([-160.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, -160.0, 0.0], dtype=np.float32),
    )
    transitions = 0
    successes = 0
    for index, offset in enumerate(offsets):
        used, success, _, _, _, _ = _run_goal_segment(
            policy,
            environment,
            writer,
            replay,
            start=center,
            goal=center + offset,
            goal_type="TASK",
            reset_seed=args.seed + 910_000 + index,
            remaining_budget=300,
        )
        transitions += used
        successes += int(success)
    environment.close()
    if successes < 3:
        raise RuntimeError(
            "smoke diagnostic collection needs at least three completed trajectories"
        )
    return {
        "policy": "heuristic_smoke_diagnostic_only",
        "evaluation_env_transitions": transitions,
        "training_budget_contribution": 0,
        "successful_trajectories": successes,
    }


def train_phase1(
    args: argparse.Namespace,
    output: Path,
    eval_tasks: list[NavigationTask],
) -> tuple[JacobianBridgeSAC, dict[str, object], PackedBridgeDataset]:
    vector_environment = make_navigation_vec_env(args)
    replay = make_bridge_replay(args)
    writer = SafetyBridgeTrajectoryWriter(
        output / "phase1_safety_bridge" / "trajectories",
        num_envs=args.num_envs,
        maximum_barrier_constraints=args.hocbf_top_k,
        slack_scale=args.bridge_slack_scale,
    )
    model = JacobianBridgeSAC(
        "MlpPolicy",
        vector_environment,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps,
        shield_loss_weight=args.shield_loss_weight,
        energy_loss_weight=args.energy_loss_weight,
        bridge_batch_size=args.bridge_batch_size,
        bridge_learning_starts=args.bridge_learning_starts,
        bridge_trust_region=args.bridge_trust_region,
        bridge_gradient_clip=args.bridge_gradient_clip,
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
    )
    model.set_bridge_replay(replay)
    navigation_callback = NavigationBudgetCallback(
        args=args,
        output=output,
        eval_tasks=eval_tasks,
    )
    bridge_callback = SafetyBridgeCollectionCallback(
        replay=replay,
        trajectory_writer=writer,
        metrics_path=output / "phase1_safety_bridge" / "metrics.jsonl",
        log_frequency_transitions=args.log_freq_transitions,
    )
    start = time.perf_counter()
    model.learn(
        total_timesteps=args.phase1_transitions,
        callback=CallbackList([navigation_callback, bridge_callback]),
        reset_num_timesteps=True,
        progress_bar=False,
    )
    elapsed = time.perf_counter() - start
    if model.num_timesteps != args.phase1_transitions:
        raise RuntimeError("Phase 1 did not match the exact transition budget")
    writer.discard_partials()
    smoke_diagnostic_collection = None
    if args.smoke and not writer.manifest_path.exists():
        smoke_diagnostic_collection = collect_smoke_bridge_diagnostics(
            args,
            writer,
            replay,
        )
    final_evaluation = evaluate_navigation_tasks(
        model,
        args,
        eval_tasks,
        global_env_transitions=args.phase1_transitions,
        output_path=output / "eval" / f"eval_transition_{args.phase1_transitions:06d}.json",
    )
    navigation_callback.final_evaluation = final_evaluation
    navigation_callback.evaluation_env_transitions += int(
        final_evaluation["evaluation_env_transitions"]
    )
    checkpoint = output / "phase1_navigation" / f"checkpoint_transition_{args.phase1_transitions:06d}.zip"
    model.save(checkpoint)
    audit = navigation_callback.audit()
    audit.update(
        {
            "requested_transition_budget": args.phase1_transitions,
            "actual_training_transitions": int(model.num_timesteps),
            "exact_budget_match": model.num_timesteps == args.phase1_transitions,
            "wall_clock_seconds": elapsed,
            "transitions_per_second": args.phase1_transitions / max(elapsed, 1e-9),
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": file_sha256(checkpoint),
            "actual_gradient_updates": int(model._n_updates),
            "gradient_update_to_transition_ratio": float(
                model._n_updates / max(model.num_timesteps, 1)
            ),
            "bridge_metrics": bridge_callback.metrics(),
            "bridge_training": model.bridge_training_metrics(),
            "bridge_replay": replay.metadata(),
            "trajectory_dataset": writer.metadata(),
            "smoke_diagnostic_collection": smoke_diagnostic_collection,
            "paired_radius2x_baseline": paired_navigation_comparison(
                final_evaluation,
                args.baseline_evaluation,
            ),
            "navigation_energy_ready": navigation_energy_gate_passed(final_evaluation),
            "navigation_safety_ready": navigation_safety_gate_passed(final_evaluation),
        }
    )
    write_json(output / "phase1_navigation" / "summary.json", audit)
    write_json(output / "phase1_safety_bridge" / "summary.json", writer.metadata())
    vector_environment.close()
    generate_navigation_curves(output)
    dataset = load_bridge_dataset(output / "phase1_safety_bridge" / "trajectories")
    return model, audit, dataset


def load_intermediate_phase1(
    args: argparse.Namespace,
    output: Path,
) -> tuple[JacobianBridgeSAC, dict[str, object]]:
    checkpoint = Path(args.resume_phase1_checkpoint).expanduser().resolve()
    source_artifact = (
        Path(args.source_phase1_artifact).expanduser().resolve()
        if args.source_phase1_artifact
        else checkpoint.parent.parent
    )
    expected_name = f"checkpoint_transition_{args.source_phase1_transition:06d}.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"JSEB checkpoint does not exist: {checkpoint}")
    if checkpoint.name != expected_name:
        raise ValueError(
            f"JSEB checkpoint must be named {expected_name}, got {checkpoint.name}"
        )
    model = JacobianBridgeSAC.load(checkpoint, device=args.device)
    expected_observation_shape = (navigation_observation_dim(args),)
    if model.observation_space.shape != expected_observation_shape:
        raise ValueError(
            f"JSEB checkpoint observation shape is {model.observation_space.shape}, "
            f"expected {expected_observation_shape}"
        )
    if model.action_space.shape != (3,):
        raise ValueError(
            f"JSEB checkpoint action shape is {model.action_space.shape}, expected (3,)"
        )
    if int(model.num_timesteps) != int(args.source_phase1_transition):
        raise ValueError(
            f"JSEB checkpoint records {model.num_timesteps} transitions, "
            f"expected {args.source_phase1_transition}"
        )
    freeze_navigation_policy(model)
    audit = {
        "requested_transition_budget": int(args.source_phase1_transition),
        "actual_training_transitions": int(args.source_phase1_transition),
        "exact_budget_match": True,
        "source_phase1_artifact": str(source_artifact),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_checkpoint_model_num_timesteps": int(model.num_timesteps),
        "phase1_retrained": False,
        "navigation_readiness_gate": (
            "BYPASSED_BY_EXPLICIT_USER_REQUEST_FOR_100K_EXPLORATORY_ENERGY_ABLATION"
        ),
        "navigation_energy_ready": None,
        "navigation_safety_ready": None,
        "claim_status": "EXPLORATORY_INTERMEDIATE_CHECKPOINT_ENERGY_ABLATION_NOT_FORMAL",
        "phase1_trajectory_dataset_reused": False,
        "data_leakage_control": (
            "existing_500k_phase1_bridge_trajectories_are_not_loaded; "
            "all downstream bridge and energy data are freshly collected"
        ),
    }
    write_json(output / "phase1_navigation" / "resume_audit.json", audit)
    return model, audit


def disjoint_bridge_and_energy_data(
    dataset: PackedBridgeDataset,
    mission_units: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[PackedBridgeDataset, PackedBridgeDataset, list[dict[str, object]], dict[str, object]]:
    if len(mission_units) < 5:
        raise RuntimeError(
            "frozen-navigation energy run needs at least five complete missions to make "
            "disjoint bridge-source and energy-model datasets"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mission_units))
    bridge_count = max(2, int(np.floor(0.20 * len(order))))
    bridge_count = min(bridge_count, len(order) - 3)
    bridge_units = [mission_units[int(index)] for index in order[:bridge_count]]
    energy_units = [mission_units[int(index)] for index in order[bridge_count:]]

    def trajectory_ids(units: list[dict[str, object]]) -> set[int]:
        ids: set[int] = set()
        for unit in units:
            ids.add(int(unit["task_trajectory_id"]))
            ids.add(int(unit["return_trajectory_id"]))
            direct_id = unit.get("direct_charger_trajectory_id")
            if direct_id is not None:
                ids.add(int(direct_id))
        return ids

    bridge_ids = trajectory_ids(bridge_units)
    energy_ids = trajectory_ids(energy_units)
    if bridge_ids & energy_ids:
        raise RuntimeError("bridge-source and energy-model trajectory sets overlap")
    audit = {
        "protocol": "fresh_frozen_policy_disjoint_mission_split",
        "bridge_source_missions": len(bridge_units),
        "energy_model_missions": len(energy_units),
        "bridge_source_trajectory_ids": sorted(bridge_ids),
        "energy_model_trajectory_ids": sorted(energy_ids),
        "trajectory_overlap": False,
        "existing_phase1_500k_data_reused": False,
    }
    return dataset.subset(bridge_ids), dataset.subset(energy_ids), energy_units, audit


def _run_goal_segment(
    policy,
    environment,
    writer: SafetyBridgeTrajectoryWriter,
    replay: SafetyBridgeReplay,
    *,
    start: np.ndarray,
    goal: np.ndarray,
    goal_type: str,
    reset_seed: int,
    remaining_budget: int,
    static_obstacles: list[dict[str, object]] | None = None,
) -> tuple[
    int,
    bool,
    int | None,
    list[dict[str, object]] | None,
    dict[str, int],
    np.ndarray,
]:
    options: dict[str, object] = {
        "start_position": start,
        "start_velocity": np.zeros(3, dtype=np.float32),
        "task_point": goal,
    }
    if static_obstacles is not None:
        options["static_obstacles"] = static_obstacles
    observation, _ = environment.reset(seed=reset_seed, options=options)
    layout = environment.static_obstacle_layout()
    transitions = 0
    counters = {"collisions": 0, "interventions": 0, "emergencies": 0}
    trajectory_id: int | None = None
    while transitions < remaining_budget:
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, raw_info = environment.step(action)
        transitions += 1
        info = dict(raw_info)
        info["goal_type"] = goal_type
        replay.add_from_info(info)
        before = writer.completed_trajectories
        writer.observe(0, info, bool(terminated or truncated))
        if writer.completed_trajectories > before:
            trajectory_id = before
        counters["collisions"] += int(bool(info["obstacle_collision"]))
        counters["interventions"] += int(bool(info.get("hocbf_intervened", False)))
        counters["emergencies"] += int(bool(info.get("hocbf_emergency_brake", False)))
        if terminated or truncated:
            return (
                transitions,
                bool(info["is_success"]),
                trajectory_id,
                layout,
                counters,
                environment.agent.pos.copy(),
            )
    writer.discard_partials()
    return transitions, False, None, layout, counters, environment.agent.pos.copy()


def collect_mission_budget(
    policy,
    args: argparse.Namespace,
    *,
    transition_budget: int,
    output: Path,
    seed: int,
    replay: SafetyBridgeReplay,
) -> tuple[PackedBridgeDataset | None, list[dict[str, object]], dict[str, object]]:
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    writer = SafetyBridgeTrajectoryWriter(
        output / "trajectories",
        num_envs=1,
        maximum_barrier_constraints=args.hocbf_top_k,
        slack_scale=args.bridge_slack_scale,
    )
    task_count = 1000 if not (args.smoke or args.pilot) else (25 if args.pilot else 5)
    if args.smoke:
        charger = environment.charger_position.copy()
        starts_and_goals = (
            ([-120.0, 0.0, 0.0], [120.0, 0.0, 0.0]),
            ([0.0, -120.0, 0.0], [0.0, 120.0, 0.0]),
            ([-120.0, -120.0, 0.0], [0.0, 120.0, 0.0]),
            ([120.0, -120.0, 0.0], [-120.0, 0.0, 0.0]),
            ([0.0, 120.0, 0.0], [120.0, 0.0, 0.0]),
        )
        tasks = []
        for start_offset, goal_offset in starts_and_goals:
            start = charger + np.asarray(start_offset, dtype=np.float32)
            goal = charger + np.asarray(goal_offset, dtype=np.float32)
            distance = float(np.linalg.norm(goal - start))
            tasks.append(
                NavigationTask(
                    start_position=start,
                    goal_position=goal,
                    initial_velocity=np.zeros(3, dtype=np.float32),
                    straight_line_distance=distance,
                    distance_bucket="100-500",
                )
            )
    else:
        tasks = generate_stratified_navigation_tasks(num_tasks=task_count, seed=seed)
    transitions = 0
    mission_units: list[dict[str, object]] = []
    collisions = interventions = emergencies = 0
    task_index = 0
    started = time.perf_counter()
    while transitions < transition_budget:
        task = tasks[task_index % len(tasks)]
        base_seed = seed + task_index * 3
        used, task_success, task_id, layout, counters, task_endpoint = _run_goal_segment(
            policy,
            environment,
            writer,
            replay,
            start=task.start_position,
            goal=task.goal_position,
            goal_type="TASK",
            reset_seed=base_seed,
            remaining_budget=transition_budget - transitions,
        )
        transitions += used
        collisions += counters["collisions"]
        interventions += counters["interventions"]
        emergencies += counters["emergencies"]
        if transitions >= transition_budget:
            break
        return_id = None
        return_success = False
        return_distance = float(np.linalg.norm(task_endpoint - environment.charger_position))
        if task_success and return_distance >= args.minimum_task_distance:
            used, return_success, return_id, _, counters, _ = _run_goal_segment(
                policy,
                environment,
                writer,
                replay,
                start=task_endpoint,
                goal=environment.charger_position.copy(),
                goal_type="CHARGER",
                reset_seed=base_seed + 1,
                remaining_budget=transition_budget - transitions,
                static_obstacles=layout,
            )
            transitions += used
            collisions += counters["collisions"]
            interventions += counters["interventions"]
            emergencies += counters["emergencies"]
        if transitions >= transition_budget:
            break
        direct_id = None
        direct_success = False
        direct_distance = float(
            np.linalg.norm(task.start_position - environment.charger_position)
        )
        if direct_distance >= args.minimum_task_distance:
            used, direct_success, direct_id, _, counters, _ = _run_goal_segment(
                policy,
                environment,
                writer,
                replay,
                start=task.start_position,
                goal=environment.charger_position.copy(),
                goal_type="CHARGER",
                reset_seed=base_seed + 2,
                remaining_budget=transition_budget - transitions,
            )
            transitions += used
            collisions += counters["collisions"]
            interventions += counters["interventions"]
            emergencies += counters["emergencies"]
        if task_success and return_success and task_id is not None and return_id is not None:
            mission_units.append(
                {
                    "mission_id": len(mission_units),
                    "task_trajectory_id": task_id,
                    "return_trajectory_id": return_id,
                    "direct_charger_trajectory_id": direct_id if direct_success else None,
                    "task_distance": task.straight_line_distance,
                    "task_distance_bucket": task.distance_bucket,
                }
            )
        task_index += 1
    if transitions != transition_budget:
        raise RuntimeError("mission collection did not match the exact transition budget")
    partials = writer.discard_partials()
    environment.close()
    write_json(output / "mission_units.json", mission_units)
    metadata = {
        "requested_training_transitions": transition_budget,
        "actual_training_transitions": transitions,
        "exact_budget_match": transitions == transition_budget,
        "completed_missions": len(mission_units),
        "partial_trajectories_at_budget_stop": partials,
        "obstacle_collision_steps": collisions,
        "hocbf_intervention_steps": interventions,
        "hocbf_emergency_steps": emergencies,
        "hocbf_intervention_rate": interventions / max(transitions, 1),
        "wall_clock_seconds": time.perf_counter() - started,
        "trajectory_dataset": writer.metadata(),
        "replay": replay.metadata(),
        "collection_policy": (
            "heuristic_smoke_diagnostic_only"
            if isinstance(policy, HeuristicGoalPolicy)
            else "frozen_jseb_policy"
        ),
    }
    manifest = output / "trajectories" / "manifest.jsonl"
    dataset = (
        load_bridge_dataset(output / "trajectories")
        if manifest.exists() and manifest.stat().st_size > 0
        else None
    )
    metadata["completed_trajectory_dataset_available"] = dataset is not None
    write_json(output / "collection_summary.json", metadata)
    return dataset, mission_units, metadata


def split_by_mission_units(
    dataset: PackedBridgeDataset,
    mission_units: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[dict[str, PackedBridgeDataset], dict[str, list[dict[str, object]]]]:
    if len(mission_units) < 4:
        raise RuntimeError(
            "at least four complete TASK-to-CHARGER missions are required"
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(mission_units))
    test_count = max(1, int(np.floor(0.20 * len(order))))
    calibration_count = max(1, int(np.floor(0.20 * len(order))))
    validation_count = max(1, int(np.floor(0.10 * len(order))))
    train_count = len(order) - test_count - calibration_count - validation_count
    if train_count < 1:
        raise RuntimeError("mission split cannot allocate four nonempty partitions")
    train_end = train_count
    validation_end = train_end + validation_count
    calibration_end = validation_end + calibration_count
    indices = {
        "train": order[:train_end],
        "validation": order[train_end:validation_end],
        "calibration": order[validation_end:calibration_end],
        "test": order[calibration_end:],
    }
    unit_splits = {
        name: [mission_units[int(index)] for index in values]
        for name, values in indices.items()
    }
    dataset_splits: dict[str, PackedBridgeDataset] = {}
    seen: set[int] = set()
    for name, units in unit_splits.items():
        ids: set[int] = set()
        for unit in units:
            ids.add(int(unit["task_trajectory_id"]))
            ids.add(int(unit["return_trajectory_id"]))
            if unit.get("direct_charger_trajectory_id") is not None:
                ids.add(int(unit["direct_charger_trajectory_id"]))
        if seen & ids:
            raise RuntimeError("trajectory leakage across JSEB splits")
        seen |= ids
        dataset_splits[name] = dataset.subset(ids)
    return dataset_splits, unit_splits


def split_goal_trajectories(
    dataset: PackedBridgeDataset,
    *,
    seed: int,
) -> dict[str, PackedBridgeDataset]:
    ids = np.asarray(sorted(dataset.unique_trajectory_ids), dtype=np.int64)
    if ids.size < 4:
        raise RuntimeError("at least four completed goal trajectories are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    test_count = max(1, int(np.floor(0.20 * ids.size)))
    calibration_count = max(1, int(np.floor(0.20 * ids.size)))
    validation_count = max(1, int(np.floor(0.10 * ids.size)))
    train_count = int(ids.size) - test_count - calibration_count - validation_count
    if train_count < 1:
        raise RuntimeError("goal split cannot allocate four nonempty partitions")
    train_end = train_count
    validation_end = train_end + validation_count
    calibration_end = validation_end + calibration_count
    split_ids = {
        "train": set(int(value) for value in ids[:train_end]),
        "validation": set(int(value) for value in ids[train_end:validation_end]),
        "calibration": set(int(value) for value in ids[validation_end:calibration_end]),
        "test": set(int(value) for value in ids[calibration_end:]),
    }
    if any(not values for values in split_ids.values()):
        raise RuntimeError("goal-trajectory split produced an empty partition")
    return {
        name: dataset.subset(values)
        for name, values in split_ids.items()
    }


def simple_split(dataset: PackedBridgeDataset, *, seed: int) -> tuple[PackedBridgeDataset, PackedBridgeDataset]:
    ids = np.asarray(sorted(dataset.unique_trajectory_ids), dtype=np.int64)
    if ids.size < 2:
        raise RuntimeError("SafetyBridge training needs at least two completed trajectories")
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    split = min(max(1, int(0.8 * ids.size)), ids.size - 1)
    return dataset.subset(set(ids[:split])), dataset.subset(set(ids[split:]))


def encode_contexts(
    encoder: SafetyBridgeEncoder,
    contexts: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    encoder.eval()
    with torch.no_grad():
        result = encoder.encode(
            torch.as_tensor(contexts, dtype=torch.float32, device=device)
        )
    return result.cpu().numpy().astype(np.float32)


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    residual = predicted - truth
    under = residual < 0.0
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(residual)),
        "underestimation_rate": float(np.mean(under)),
        "mean_underestimation_magnitude": float(
            np.mean(-residual[under]) if np.any(under) else 0.0
        ),
    }


def fit_energy_models(
    phase1_dataset: PackedBridgeDataset,
    dataset: PackedBridgeDataset | None,
    mission_units: list[dict[str, object]],
    args: argparse.Namespace,
    output: Path,
    *,
    seed: int,
) -> tuple[
    dict[str, FlexibleEnergyRegressor],
    SafetyBridgeEncoder,
    ActionConditionedEnergyCritic,
    dict[str, PackedBridgeDataset],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    bridge_train, bridge_validation = simple_split(phase1_dataset, seed=seed)
    burden_scale = max(float(np.quantile(bridge_train.burden_to_go, 0.95)), 1e-3)
    torch.manual_seed(seed)
    encoder = SafetyBridgeEncoder(latent_dim=8, hidden_dim=32)
    bridge_fit = encoder.fit_dataset(
        bridge_train,
        bridge_validation,
        epochs=args.energy_epochs,
        batch_size=args.energy_batch_size,
        learning_rate=args.energy_learning_rate,
        burden_scale=burden_scale,
        seed=seed,
        device=args.device,
    )
    phase_dataset_trajectory_count = (
        0 if dataset is None else len(dataset.unique_trajectory_ids)
    )
    if len(mission_units) >= 3:
        if dataset is None:
            raise RuntimeError("mission metadata exists without completed trajectories")
        splits, unit_splits = split_by_mission_units(
            dataset,
            mission_units,
            seed=seed + 1,
        )
        split_protocol = "complete_task_to_charger_missions"
    elif args.smoke or args.pilot:
        diagnostic_dataset = (
            phase1_dataset
            if dataset is None
            else concatenate_bridge_datasets(phase1_dataset, dataset)
        )
        splits = split_goal_trajectories(diagnostic_dataset, seed=seed + 1)
        unit_splits = {
            "train": [],
            "validation": [],
            "calibration": [],
            "test": [],
        }
        split_protocol = "diagnostic_real_goal_trajectories_no_mission_claim"
    else:
        raise RuntimeError(
            "formal energy fitting requires at least three complete TASK-to-CHARGER missions"
        )
    energy_scale = max(float(np.quantile(splits["train"].energy_to_go, 0.95)), 1e-3)

    def features(name: str, split: PackedBridgeDataset) -> np.ndarray:
        if name == "E0":
            return split.compact_energy_states
        if name == "E1":
            return np.concatenate(
                [split.compact_energy_states, split.safety_contexts], axis=1
            )
        if name == "E2":
            return np.concatenate(
                [
                    split.compact_energy_states,
                    encode_contexts(encoder, split.safety_contexts, device=args.device),
                ],
                axis=1,
            )
        raise ValueError(name)

    models: dict[str, FlexibleEnergyRegressor] = {}
    model_results: dict[str, object] = {}
    enabled_point_models = (
        ("E0", "E1", "E2") if args.ablation in {"C", "D"} else ("E0",)
    )
    for index, name in enumerate(enabled_point_models):
        input_dim = features(name, splits["train"]).shape[1]
        torch.manual_seed(seed + 10 + index)
        model = FlexibleEnergyRegressor(
            input_dim,
            hidden_dim=args.energy_hidden_dim,
            energy_scale=energy_scale,
        )
        fit = model.fit_arrays(
            features(name, splits["train"]),
            splits["train"].energy_to_go,
            features(name, splits["validation"]),
            splits["validation"].energy_to_go,
            epochs=args.energy_epochs,
            batch_size=args.energy_batch_size,
            learning_rate=args.energy_learning_rate,
            seed=seed + 20 + index,
            device=args.device,
        )
        test_prediction = model.predict(features(name, splits["test"]), device=args.device)
        model_results[name] = {
            "fit": fit_result_dict(fit),
            "test": regression_metrics(test_prediction, splits["test"].energy_to_go),
            "input_dim": input_dim,
        }
        model.save(output / f"{name.lower()}_energy_model.pt")
        models[name] = model

    critic = ActionConditionedEnergyCritic(
        hidden_dim=args.energy_hidden_dim,
        energy_scale=energy_scale,
    )

    def critic_features(split: PackedBridgeDataset) -> np.ndarray:
        return np.concatenate(
            [
                split.compact_energy_states,
                split.safety_contexts,
                split.executed_actions,
            ],
            axis=1,
        )

    critic_fit = critic.fit_arrays(
        critic_features(splits["train"]),
        splits["train"].energy_to_go,
        critic_features(splits["validation"]),
        splits["validation"].energy_to_go,
        epochs=args.energy_epochs,
        batch_size=args.energy_batch_size,
        learning_rate=args.energy_learning_rate,
        seed=seed + 50,
        device=args.device,
    )
    critic_test = critic.predict(critic_features(splits["test"]), device=args.device)
    gradient_actions = torch.as_tensor(
        splits["test"].executed_actions[: min(32, len(splits["test"]))],
        dtype=torch.float32,
        device=args.device,
    ).clone().requires_grad_(True)
    gradient_energy = critic.energy(
        torch.as_tensor(
            splits["test"].compact_energy_states[: gradient_actions.shape[0]],
            dtype=torch.float32,
            device=args.device,
        ),
        torch.as_tensor(
            splits["test"].safety_contexts[: gradient_actions.shape[0]],
            dtype=torch.float32,
            device=args.device,
        ),
        gradient_actions,
    ).mean()
    gradient_energy.backward()
    gradient_norm = float(torch.linalg.vector_norm(gradient_actions.grad).detach().cpu())
    critic.save(output / "action_conditioned_energy_critic.pt")
    critic.freeze()
    summary = {
        "energy_scale": energy_scale,
        "burden_scale": burden_scale,
        "bridge_fit": fit_result_dict(bridge_fit),
        "models": model_results,
        "selected_by_validation": min(
            model_results,
            key=lambda name: model_results[name]["fit"]["best_validation_mae"],
        ),
        "enabled_point_models": list(enabled_point_models),
        "action_conditioned_critic": {
            "fit": fit_result_dict(critic_fit),
            "test": regression_metrics(critic_test, splits["test"].energy_to_go),
            "action_gradient_norm": gradient_norm,
            "trained_on_executed_actions": True,
        },
        "split_trajectory_ids": {
            name: sorted(split.unique_trajectory_ids) for name, split in splits.items()
        },
        "split_protocol": split_protocol,
        "split_roles": {
            "train": "model_parameter_fitting",
            "validation": "model_selection_and_early_stopping",
            "calibration": "conformal_margin_only",
            "test": "held_out_reporting_only",
        },
        "complete_mission_count": len(mission_units),
        "mission_claim_available": split_protocol == "complete_task_to_charger_missions",
        "phase_dataset_trajectory_count": phase_dataset_trajectory_count,
        "phase1_dataset_reused_for_diagnostic_split": split_protocol
        == "diagnostic_real_goal_trajectories_no_mission_claim",
    }
    write_json(output / "model_summary.json", summary)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "latent_dim": encoder.latent_dim,
            "hidden_dim": encoder.hidden_dim,
            "burden_scale": burden_scale,
        },
        output / "safety_bridge_encoder.pt",
    )
    return models, encoder, critic, splits, unit_splits, summary


def trajectory_initial_rows(dataset: PackedBridgeDataset) -> dict[int, int]:
    return {
        int(trajectory_id): int(np.flatnonzero(dataset.trajectory_ids == trajectory_id)[0])
        for trajectory_id in np.unique(dataset.trajectory_ids)
    }


def fit_final_conformal(
    models: dict[str, FlexibleEnergyRegressor],
    splits: dict[str, PackedBridgeDataset],
    unit_splits: dict[str, list[dict[str, object]]],
    args: argparse.Namespace,
    output: Path,
    *,
    deployed_policy_data_available: bool,
) -> tuple[CalibratedCompactEnergyEstimator | None, dict[str, object]]:
    point_model = models["E0"]
    calibration = splits["calibration"]
    test = splits["test"]
    calibration_predictions = point_model.predict(
        calibration.compact_energy_states,
        device=args.device,
    )
    goal_calibration = HierarchicalConformalCalibration.fit(
        calibration_predictions,
        calibration.energy_to_go,
        calibration.trajectory_ids,
        calibration.goal_types,
        calibration.distance_buckets,
        coverage=args.conformal_coverage,
    )
    test_predictions = point_model.predict(test.compact_energy_states, device=args.device)
    test_upper = np.asarray(
        [
            prediction
            + goal_calibration.margin_for(str(goal_type), float(distance))
            for prediction, goal_type, distance in zip(
                test_predictions,
                test.goal_types,
                test.initial_distances,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    whole_coverage = []
    for trajectory_id in np.unique(test.trajectory_ids):
        mask = test.trajectory_ids == trajectory_id
        whole_coverage.append(bool(np.all(test.energy_to_go[mask] <= test_upper[mask])))

    goal_summary = {
        "point": regression_metrics(test_predictions, test.energy_to_go),
        "goal_state_coverage": float(np.mean(test.energy_to_go <= test_upper)),
        "goal_whole_trajectory_coverage": float(np.mean(whole_coverage)),
        "goal_mean_bound_width": float(np.mean(test_upper - test_predictions)),
        "goal_calibration": goal_calibration.as_dict(),
        "trajectory_disjoint": True,
        "calibrated_on_deployed_frozen_policy": bool(
            deployed_policy_data_available
        ),
        "deployed_switching_point_model": "E0_compact_7D",
        "context_models_role": "prediction_ablation_only_because_future_goal_J_is_not_observed",
    }
    if (
        not deployed_policy_data_available
        or not unit_splits["calibration"]
        or not unit_splits["test"]
    ):
        reason = (
            "no_completed_deployed_policy_trajectory"
            if not deployed_policy_data_available
            else "pilot_has_no_complete_trajectory_disjoint_TASK_to_CHARGER_missions"
        )
        summary = {
            **goal_summary,
            "mission_calibration_available": False,
            "mission_coverage": None,
            "mission_calibration": None,
            "persistent_switching_evaluation_available": False,
            "reason": reason,
        }
        write_json(output / "conformal_summary.json", summary)
        return None, summary

    def mission_arrays(
        split: PackedBridgeDataset,
        units: list[dict[str, object]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        initial = trajectory_initial_rows(split)
        predictions = point_model.predict(split.compact_energy_states, device=args.device)
        scores: list[float] = []
        buckets: list[str] = []
        point_values: list[float] = []
        task_distances: list[float] = []
        for unit in units:
            task_id = int(unit["task_trajectory_id"])
            return_id = int(unit["return_trajectory_id"])
            if task_id not in initial or return_id not in initial:
                continue
            task_index = initial[task_id]
            return_index = initial[return_id]
            point = float(predictions[task_index] + predictions[return_index])
            truth = float(
                split.energy_to_go[task_index] + split.energy_to_go[return_index]
            )
            scores.append(truth - point)
            point_values.append(point)
            buckets.append(str(unit["task_distance_bucket"]))
            task_distances.append(float(unit["task_distance"]))
        return (
            np.asarray(scores, dtype=np.float64),
            np.asarray(buckets, dtype="U16"),
            np.asarray(point_values, dtype=np.float64),
            np.asarray(task_distances, dtype=np.float64),
        )

    calibration_scores, calibration_buckets, _, _ = mission_arrays(
        calibration,
        unit_splits["calibration"],
    )
    if calibration_scores.size == 0:
        raise RuntimeError("no complete calibration missions")
    mission_calibration = MissionConformalCalibration.fit(
        calibration_scores,
        calibration_buckets,
        coverage=args.conformal_coverage,
    )
    test_scores, _, _, test_distances = mission_arrays(test, unit_splits["test"])
    mission_coverage = float(
        np.mean(
            [
                score <= mission_calibration.margin_for(distance)
                for score, distance in zip(test_scores, test_distances, strict=True)
            ]
        )
    ) if test_scores.size else None
    estimator = CalibratedCompactEnergyEstimator(
        point_model,
        goal_calibration,
        mission_calibration,
        device=args.device,
    )
    estimator.save(output / "jseb_final_energy_estimator.pt")
    summary = {
        **goal_summary,
        "mission_calibration_available": True,
        "mission_coverage": mission_coverage,
        "mission_calibration": mission_calibration.as_dict(),
        "persistent_switching_evaluation_available": True,
    }
    write_json(output / "conformal_summary.json", summary)
    return estimator, summary


def run_persistent_evaluation(
    model: JacobianBridgeSAC,
    estimator: CalibratedCompactEnergyEstimator,
    args: argparse.Namespace,
    *,
    battery_capacity: float,
    output: Path,
) -> dict[str, object]:
    environment = environment_from_args(
        args,
        phase=SACTrainingPhase.NAVIGATION,
        battery_capacity=battery_capacity,
    )
    environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=single_action_provider(model),
        training_enabled=False,
    )
    environment.enable_phase_two()
    observation, _ = environment.reset(seed=args.seed + 900_000)
    tasks = cycles = returns = exhaustions = collisions = interventions = emergencies = 0
    total_energy = total_distance = total_simulation_time = 0.0
    previous_position = environment.agent.pos.copy()
    for transition in range(1, args.persistent_eval_transitions + 1):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = environment.step(action)
        tasks += int(info["task_completed_now"])
        collisions += int(info["obstacle_collision"])
        interventions += int(info.get("hocbf_intervened", False))
        emergencies += int(info.get("hocbf_emergency_brake", False))
        total_energy += float(info["realized_energy_cost"])
        total_simulation_time += float(info["transition_dt"])
        total_distance += float(np.linalg.norm(environment.agent.pos - previous_position))
        previous_position = environment.agent.pos.copy()
        if info["switched_now"] and environment.switching_events:
            append_jsonl(output / "switching_events.jsonl", environment.switching_events[-1])
        if info["battery_cycle_record"] is not None:
            record = dict(info["battery_cycle_record"])
            append_jsonl(output / "battery_cycles.jsonl", record)
            cycles += 1
            returns += int(bool(record["return_success"]))
        if terminated or truncated:
            exhaustions += int(terminated and info["end_reason"] == "energy_exhausted")
            observation, _ = environment.reset(seed=args.seed + 900_000 + transition)
            previous_position = environment.agent.pos.copy()
    summary = {
        "evaluation_env_transitions": args.persistent_eval_transitions,
        "training_budget_contribution": 0,
        "tasks_completed": tasks,
        "tasks_per_1000_transitions": 1000.0 * tasks / args.persistent_eval_transitions,
        "tasks_per_battery_cycle": None if cycles == 0 else tasks / cycles,
        "tasks_per_simulation_hour": 3600.0 * tasks / max(total_simulation_time, 1e-9),
        "battery_cycles": cycles,
        "charger_return_successes": returns,
        "charger_return_success_rate": returns / max(cycles, 1),
        "energy_exhaustions": exhaustions,
        "obstacle_collision_steps": collisions,
        "hocbf_intervention_rate": interventions / args.persistent_eval_transitions,
        "hocbf_emergency_rate": emergencies / args.persistent_eval_transitions,
        "realized_energy_per_task": total_energy / max(tasks, 1),
        "realized_energy_per_meter": total_energy / max(total_distance, 1e-9),
        "realized_energy_per_simulation_minute": 60.0 * total_energy / max(total_simulation_time, 1e-9),
        "battery_capacity": battery_capacity,
    }
    write_json(output / "summary.json", summary)
    environment.close()
    return summary


def formal_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "protocol": (
            "FROZEN_NAVIGATION_100K_CHECKPOINT_ENERGY_LEARNING"
            if args.intermediate_checkpoint_energy_ablation
            else "FROZEN_NAVIGATION_ENERGY_LEARNING_STATIC_1M"
        ),
        "ablation": {
            "id": args.ablation,
            "name": {
                "A": "Current baseline",
                "B": "J-Safety",
                "C": "J-Context",
                "D": "Full JSEB",
            }[args.ablation],
            "projection_geometry_enabled": args.projection_geometry_enabled,
        },
        "status": "RUNNING",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "git_sha": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
        "exact_command": sys.argv,
        "training_budget": {
            "phase1": (
                args.source_phase1_transition
                if args.intermediate_checkpoint_energy_ablation
                else args.phase1_transitions
            ),
            "phase2_frozen_energy_learning": args.phase2_energy_transitions,
            "total": sum(
                [
                    (
                        args.source_phase1_transition
                        if args.intermediate_checkpoint_energy_ablation
                        else args.phase1_transitions
                    ),
                    args.phase2_energy_transitions,
                ]
            ),
            "evaluation_transitions_excluded": True,
            "current_run_phase2_total": args.phase2_energy_transitions,
            "source_navigation_reused": bool(
                args.intermediate_checkpoint_energy_ablation
            ),
            "navigation_policy_updates": 0,
            "navigation_policy_frozen": True,
        },
        "source_checkpoint": {
            "path": args.resume_phase1_checkpoint,
            "source_artifact": args.source_phase1_artifact,
            "transition": args.source_phase1_transition,
            "intermediate_energy_ablation": bool(
                args.intermediate_checkpoint_energy_ablation
            ),
        },
        "evaluation_protocol": {
            "phase_end_eval_only": bool(args.phase_end_eval_only),
            "periodic_navigation_evaluation_enabled": not bool(
                args.phase_end_eval_only
            ),
            "legacy_eval_frequency_transitions": args.eval_freq_transitions,
            "phase1_final_navigation_tasks": args.eval_navigation_tasks,
            "frozen_energy_evaluation": "phase_end",
            "parallel_environment_workers": args.evaluation_num_envs,
            "policy_inference": (
                "central_batched_gpu"
                if args.evaluation_num_envs > 1
                else "single_observation"
            ),
            "progress_interval_tasks": args.evaluation_progress_interval_tasks,
        },
        "environment": {
            "map_m": [4000.0, 4000.0, 400.0],
            "static_obstacles": args.num_obstacles,
            "obstacle_radius_m": [args.obstacle_radius_min, args.obstacle_radius_max],
            "lidar": [args.lidar_horizontal_sectors, args.lidar_vertical_sectors],
            "lidar_range_m": args.lidar_range,
            "policy_dt_s": 0.2,
            "physics_dt_s": 0.05,
            "v_max_mps": [20.0, 20.0, 5.0],
            "a_max_mps2": [5.0, 5.0, 3.0],
            "hocbf_final_hard_layer": True,
            "emergency_reverse_braking": True,
            "observation_dim": navigation_observation_dim(args),
            "telemetry_cost": telemetry_config_from_args(args).__dict__,
        },
        "policy_adaptation": {
            "enabled": False,
            "removed_stage": "phase2b_joint_finetune",
            "reason": "observed navigation degradation under joint policy fine-tuning",
            "energy_learning_only": True,
        },
        "stage_order": [
            "frozen_navigation_checkpoint",
            "phase2_frozen_energy_learning",
            "energy_model_fit_and_heldout_evaluation",
            "battery_calibration",
            "battery_endurance_validation",
            "persistent_delivery_evaluation",
        ],
        "battery_calibration_protocol": {
            "ordered_after_energy_learning": True,
            "training_budget_contribution": 0,
            "uses_td_predictions": False,
            "purpose": "set synthetic-unit battery capacity for downstream switching evaluation",
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "claim_boundaries": {
            "static_obstacles_only": True,
            "jacobian_not_a_certificate": True,
            "active_set_switches_masked": True,
            "hocbf_remains_final_safety_layer": True,
            "real_uav_safety_claim": False,
            "formal_paper_comparison": not args.intermediate_checkpoint_energy_ablation,
            "intermediate_checkpoint_exploratory_only": bool(
                args.intermediate_checkpoint_energy_ablation
            ),
            "navigation_policy_frozen_during_energy_learning": True,
        },
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    long_run = not (args.smoke or args.pilot)
    formal = long_run and not args.intermediate_checkpoint_energy_ablation
    if long_run and (args.allow_dirty or not git_clean()):
        raise RuntimeError("long JSEB runs require a clean worktree and forbid --allow-dirty")
    output = prepare_output_directory(args.output_dir)
    for directory in (
        "phase1_navigation",
        "phase1_safety_bridge",
        "phase2_frozen_energy_learning",
        "eval",
        "gifs",
        "persistent_evaluation",
        "battery_calibration",
    ):
        (output / directory).mkdir()
    config = formal_config(args)
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": utc_now()})
    try:
        eval_tasks = generate_stratified_navigation_tasks(
            num_tasks=args.eval_navigation_tasks,
            seed=args.eval_task_seed,
        )
        save_navigation_tasks(
            output / "eval_navigation_tasks.json",
            eval_tasks,
            seed=args.eval_task_seed,
            role="fixed_navigation_evaluation",
        )
        if args.intermediate_checkpoint_energy_ablation:
            model, phase1 = load_intermediate_phase1(args, output)
            phase1_dataset = None
            downstream_ready = True
        else:
            model, phase1, phase1_dataset = train_phase1(args, output, eval_tasks)
            downstream_ready = bool(
                phase1["navigation_energy_ready"] and phase1["navigation_safety_ready"]
            )
        if formal and not downstream_ready:
            stopped = {
                "status": "STOPPED_AFTER_PHASE1",
                "navigation_training_completed": True,
                "downstream_navigation_ready": False,
                "phase1": phase1,
            }
            write_json(output / "STOPPED_AFTER_PHASE1.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return stopped

        freeze_navigation_policy(model)
        frozen_policy_hash_before = policy_hash(model)
        energy_replay = make_bridge_replay(args, seed_offset=20_000)
        energy_collection_policy = HeuristicGoalPolicy() if args.smoke else model
        energy_dataset, energy_units, energy_collection = collect_mission_budget(
            energy_collection_policy,
            args,
            transition_budget=args.phase2_energy_transitions,
            output=output / "phase2_frozen_energy_learning",
            seed=args.seed + 400_000,
            replay=energy_replay,
        )
        if energy_dataset is None:
            raise RuntimeError(
                "frozen-policy energy collection produced no complete trajectory dataset"
            )
        frozen_policy_hash_after_collection = policy_hash(model)
        if frozen_policy_hash_before != frozen_policy_hash_after_collection:
            raise RuntimeError("navigation policy changed during frozen energy collection")
        if args.intermediate_checkpoint_energy_ablation:
            bridge_source_dataset, energy_model_dataset, energy_model_units, split_audit = (
                disjoint_bridge_and_energy_data(
                    energy_dataset,
                    energy_units,
                    seed=args.seed + 450_000,
                )
            )
            write_json(
                output
                / "phase2_frozen_energy_learning"
                / "fresh_data_split_audit.json",
                split_audit,
            )
        else:
            bridge_source_dataset = phase1_dataset
            energy_model_dataset = energy_dataset
            energy_model_units = energy_units
        final_models, final_encoder, final_critic, final_splits, final_unit_splits, final_models_summary = fit_energy_models(
            bridge_source_dataset,
            energy_model_dataset,
            energy_model_units,
            args,
            output / "phase2_frozen_energy_learning" / "models",
            seed=args.seed + 500_000,
        )
        estimator, conformal = fit_final_conformal(
            final_models,
            final_splits,
            final_unit_splits,
            args,
            output / "phase2_frozen_energy_learning",
            deployed_policy_data_available=bool(
                final_models_summary["phase_dataset_trajectory_count"] > 0
            ),
        )
        if estimator is None and long_run:
            raise RuntimeError(
                "formal JSEB requires complete mission calibration before switching"
            )

        calibration_policy = model if long_run else HeuristicGoalPolicy()
        calibration_tasks = generate_stratified_navigation_tasks(
            num_tasks=args.battery_calibration_tasks,
            seed=args.battery_calibration_seed,
        )
        calibration = run_battery_calibration(
            calibration_policy,
            args,
            calibration_tasks,
            output / "battery_calibration",
        )
        capacity = float(calibration["calibrated_battery_capacity"])
        validation = run_battery_validation(
            calibration_policy,
            args,
            battery_capacity=capacity,
            output=output / "battery_calibration",
        )
        if long_run and not args.allow_failed_battery_calibration and (
            not calibration["battery_calibration_navigation_valid"]
            or not validation["battery_calibration_valid"]
        ):
            stopped = {
                "status": "STOPPED_AFTER_BATTERY_CALIBRATION",
                "energy_learning_completed": True,
                "battery_calibration": compact_summary(calibration),
                "battery_validation": compact_summary(validation),
            }
            write_json(output / "STOPPED_AFTER_BATTERY_CALIBRATION.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return stopped

        final_navigation = evaluate_navigation_tasks(
            model,
            args,
            eval_tasks,
            global_env_transitions=(
                FORMAL_TOTAL_TRANSITIONS
                if formal
                else sum(
                    [
                        (
                            args.source_phase1_transition
                            if args.intermediate_checkpoint_energy_ablation
                            else args.phase1_transitions
                        ),
                        args.phase2_energy_transitions,
                    ]
                )
            ),
            output_path=output / "eval" / "final_policy_navigation.json",
        )
        if estimator is None:
            persistent = {
                "status": "NOT_RUN_NO_COMPLETE_MISSION_CALIBRATION",
                "evaluation_env_transitions": 0,
                "training_budget_contribution": 0,
                "claim_available": False,
            }
            write_json(output / "persistent_evaluation" / "summary.json", persistent)
        else:
            persistent = run_persistent_evaluation(
                model,
                estimator,
                args,
                battery_capacity=capacity,
                output=output / "persistent_evaluation",
            )
        source_navigation_transitions = (
            args.source_phase1_transition
            if args.intermediate_checkpoint_energy_ablation
            else args.phase1_transitions
        )
        phase2_energy_interactions = args.phase2_energy_transitions
        total_interactions = source_navigation_transitions + phase2_energy_interactions
        summary = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "training_transition_accounting": {
                "phase1": phase1["actual_training_transitions"],
                "phase2_frozen_energy_learning": energy_collection[
                    "actual_training_transitions"
                ],
                "total": total_interactions,
                "current_run_phase2_total": phase2_energy_interactions,
                "exact_phase2_500k": phase2_energy_interactions == 500_000,
                "exact_formal_1m": formal
                and total_interactions == FORMAL_TOTAL_TRANSITIONS,
                "evaluation_transitions_excluded": True,
                "phase2_navigation_gradient_updates": 0,
            },
            "phase1": compact_summary(phase1),
            "battery_calibration": compact_summary(calibration),
            "battery_validation": compact_summary(validation),
            "frozen_energy_collection": energy_collection,
            "frozen_energy_models": final_models_summary,
            "conformal": conformal,
            "final_navigation": compact_summary(final_navigation),
            "persistent_delivery": persistent,
            "frozen_policy_hash_before": frozen_policy_hash_before,
            "frozen_policy_hash_after": frozen_policy_hash_after_collection,
            "navigation_policy_unchanged": (
                frozen_policy_hash_before == frozen_policy_hash_after_collection
            ),
            "claim_status": (
                "EXPLORATORY_FROZEN_NAVIGATION_ENERGY_LEARNING"
                if args.intermediate_checkpoint_energy_ablation
                else "FORMAL_FROZEN_NAVIGATION_ENERGY_LEARNING_1M"
            ),
            "claim_boundary_risks": [
                "HOCBF remains the final hard safety layer",
                "E1/E2 safety context is an ablation; deployed future-mission switching uses E0 because future J is not observed",
                "coverage is finite-group and exchangeability-conditional",
                "static obstacles only",
                "energy learning does not improve or repair frozen navigation failures",
            ],
        }
        write_json(output / "summary.json", summary)
        write_json(output / "COMPLETED.json", summary)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return summary
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    run(parse_args())
