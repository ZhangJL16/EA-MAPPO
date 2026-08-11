#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import gymnasium
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb3_energy_harness import EnergyMetricsCallback, evaluate_energy_model, make_energy_environment
from scripts.sb3_navigation_harness import model_device_metadata, utc_now, write_json
from scripts.sb3_recovery_teacher import (
    CertifiedRecoveryOracle,
    CertifiedRecoveryRolloutTeacher,
    collect_successful_teacher_trajectories,
    prefill_sb3_replay_buffer,
    supervised_actor_warm_start,
)


def build_model(environment, seed: int, device: str) -> SAC:
    return SAC(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        seed=seed,
        device=device,
        verbose=1,
    )


def run(args: argparse.Namespace) -> None:
    sb3_version = importlib.metadata.version("stable-baselines3")
    if sb3_version != "2.8.0":
        raise RuntimeError(f"2x energy baseline requires stable-baselines3==2.8.0, found {sb3_version}")
    if args.flight_energy_multiplier != 2.0:
        raise ValueError("formal recovery-teacher protocol requires flight_energy_multiplier=2.0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists() or any(output_dir.glob("checkpoint_step_*.zip")):
        raise FileExistsError(f"refusing to overwrite existing run in {output_dir}")
    certificate_artifact = Path(args.certificate_artifact)
    torch.set_num_threads(args.torch_threads)
    environment = make_energy_environment(args)
    monitored_environment = Monitor(environment)
    model = build_model(monitored_environment, args.seed, args.device)
    model.set_logger(configure(str(output_dir / "sb3_logger"), ["stdout", "csv", "json"]))
    callback = EnergyMetricsCallback(args, output_dir)
    oracle = CertifiedRecoveryOracle(args.scenario, args.flight_energy_multiplier, certificate_artifact)
    certificate = oracle.certificate
    teacher_metrics: dict[str, object] = {
        "teacher_cycles_attempted": 0,
        "teacher_cycles_successful": 0,
        "teacher_full_trajectory_success_rate": 0.0,
        "teacher_transitions_generated": 0,
        "teacher_transitions_inserted": 0,
        "teacher_recovery_transitions": 0,
        "teacher_charging_steps": 0,
        "teacher_resume_successes": 0,
        "teacher_original_goal_completions": 0,
        "actor_warmstart_losses": [],
    }
    if args.guidance_mode == "recovery_guided":
        navigation_path = Path(args.navigation_teacher_model)
        if not navigation_path.is_file():
            raise FileNotFoundError(f"navigation teacher checkpoint does not exist: {navigation_path}")
        navigation_teacher = SAC.load(navigation_path, device=args.device)
        teacher = CertifiedRecoveryRolloutTeacher(navigation_teacher, oracle)
        transitions, kappa_observations, kappa_actions, teacher_metrics = collect_successful_teacher_trajectories(
            teacher,
            lambda: make_energy_environment(args, max_episode_steps=args.teacher_max_rollout_steps),
            args.teacher_prefill_transitions,
            args.teacher_max_rollout_steps,
            args.teacher_seed,
            args.teacher_max_cycles,
        )
        inserted = prefill_sb3_replay_buffer(model, transitions)
        if inserted != len(transitions):
            raise RuntimeError("SB3_REPLAY_PREFILL_COUNT_MISMATCH")
        losses = supervised_actor_warm_start(
            model,
            kappa_observations,
            kappa_actions,
            epochs=args.actor_warmstart_epochs,
            batch_size=args.actor_warmstart_batch_size,
            learning_rate=args.actor_warmstart_learning_rate,
            seed=args.teacher_seed,
        )
        teacher_metrics["teacher_transitions_inserted"] = inserted
        teacher_metrics["actor_warmstart_losses"] = losses
        write_json(output_dir / "teacher_initialization.json", teacher_metrics)
    elif args.navigation_teacher_model is not None:
        raise ValueError("unguided control must not receive a navigation teacher model")

    config = vars(args) | {
        "initialization": "FROM_SCRATCH",
        "guidance_difference": (
            "complete_teacher_replay_prefill_plus_actor_only_kappa_warmstart"
            if args.guidance_mode == "recovery_guided"
            else "none"
        ),
        "online_teacher_override_during_model_learn": False,
        "algorithm_class": "stable_baselines3.SAC",
        "stable_baselines3_version": sb3_version,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "numpy_version": np.__version__,
        "environment_class": "PersistentEnergyNavigationEnv",
        "flight_energy_multiplier": args.flight_energy_multiplier,
        "energy_reward_uses_actual_flight_energy": True,
        "certificate": certificate,
        "observation_fields": list(environment.observation_fields),
        "observation_dimension": int(environment.observation_space.shape[0]),
        "action_space_low": environment.action_space.low.tolist(),
        "action_space_high": environment.action_space.high.tolist(),
        "dt": environment.config.dt,
        "physical_acceleration_limit": environment.config.a_max.tolist(),
        "velocity_limit": environment.config.v_max.tolist(),
        "battery_capacity": environment.energy_navigation_config.battery_capacity,
        "charging_rate": environment.energy_navigation_config.charging_rate,
        "sac_hyperparameters": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 5000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": [1, "step"],
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
        },
        "teacher_metrics": teacher_metrics,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": utc_now(),
    }
    config |= model_device_metadata(model, args.device)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "RUNNING.json", {"status": "RUNNING", "started_at": config["started_at"]})
    print("DEVICE_CHECK " + json.dumps(model_device_metadata(model, args.device), sort_keys=True), flush=True)
    try:
        for checkpoint_step in args.checkpoint_steps:
            remaining = checkpoint_step - model.num_timesteps
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    callback=callback,
                    log_interval=1,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            callback.assert_finite_training_state()
            actual_step = int(model.num_timesteps)
            model.save(output_dir / f"checkpoint_step_{actual_step:07d}")
            write_json(
                output_dir / f"checkpoint_summary_step_{actual_step:07d}.json",
                callback.snapshot() | {"requested_checkpoint_step": checkpoint_step},
            )
            for execution_mode in ("POLICY_ONLY", "SYSTEM_WITH_KAPPA"):
                for evaluation_mode in ("deterministic", "stochastic"):
                    for soc_group in ("full", "low_soc"):
                        evaluation = evaluate_energy_model(
                            model,
                            args,
                            checkpoint_step,
                            actual_step,
                            evaluation_mode=evaluation_mode,
                            soc_group=soc_group,
                            execution_mode=execution_mode,
                            certificate_artifact=certificate_artifact,
                        )
                        mode_name = execution_mode.lower()
                        write_json(
                            output_dir / f"heldout_{mode_name}_{evaluation_mode}_{soc_group}_step_{actual_step:07d}.json",
                            evaluation,
                        )
        summary = callback.snapshot() | {
            "status": "COMPLETED",
            "requested_timesteps": args.steps,
            "actual_timesteps": int(model.num_timesteps),
            "completed_at": utc_now(),
            "teacher_metrics": teacher_metrics,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "COMPLETED.json", summary)
        write_json(output_dir / "RUNNING.json", {"status": "COMPLETED", "completed_at": summary["completed_at"]})
    except Exception as error:
        failure = {
            "status": "IMPLEMENTATION_FAILURE",
            "failed_at": utc_now(),
            "actual_timesteps": int(model.num_timesteps),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "FAILED.json", failure)
        write_json(output_dir / "RUNNING.json", failure)
        raise
    finally:
        monitored_environment.close()
        oracle.environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched 2x SB3 SAC energy baseline")
    parser.add_argument("--guidance-mode", choices=["unguided", "recovery_guided"], required=True)
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--goal-radius", type=float, default=0.20)
    parser.add_argument("--minimum-goal-separation", type=float, default=0.60)
    parser.add_argument("--sampling-margin", type=float, default=0.20)
    parser.add_argument("--distance-potential-scale", type=float, default=0.25)
    parser.add_argument("--velocity-reward-weight", type=float, default=0.1)
    parser.add_argument("--time-cost", type=float, default=0.01)
    parser.add_argument("--completion-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=1.2)
    parser.add_argument("--energy-cost-weight", type=float, default=0.01)
    parser.add_argument("--backup-intervention-cost", type=float, default=0.1)
    parser.add_argument("--battery-capacity", type=float, default=30.0)
    parser.add_argument("--charging-rate", type=float, default=2.0)
    parser.add_argument("--charging-radius", type=float, default=0.18)
    parser.add_argument("--charging-velocity-limit", type=float, nargs=3, default=[0.05, 0.05, 0.04])
    parser.add_argument("--initial-energy-fraction-min", type=float, default=0.30)
    parser.add_argument("--initial-energy-fraction-max", type=float, default=1.00)
    parser.add_argument("--flight-energy-multiplier", type=float, default=2.0)
    parser.add_argument("--certificate-artifact", required=True)
    parser.add_argument("--navigation-teacher-model")
    parser.add_argument("--teacher-prefill-transitions", type=int, default=20_000)
    parser.add_argument("--teacher-max-rollout-steps", type=int, default=5000)
    parser.add_argument("--teacher-max-cycles", type=int, default=200)
    parser.add_argument("--teacher-seed", type=int, default=70_000)
    parser.add_argument("--actor-warmstart-epochs", type=int, default=20)
    parser.add_argument("--actor-warmstart-batch-size", type=int, default=256)
    parser.add_argument("--actor-warmstart-learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=5000)
    parser.add_argument("--evaluation-seed-base", type=int, default=193_000_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.gamma = 0.99
    args.dt = 0.2
    if args.guidance_mode == "recovery_guided" and not args.navigation_teacher_model:
        parser.error("recovery_guided mode requires --navigation-teacher-model")
    if not args.smoke and (args.steps != 1_000_000 or args.checkpoint_steps[-1] != 1_000_000):
        parser.error("formal 2x protocol requires a 1M final checkpoint")
    if args.smoke and args.checkpoint_steps[-1] != args.steps:
        parser.error("smoke checkpoint schedule must end at --steps")
    return args


if __name__ == "__main__":
    run(parse_args())
