from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from envs.UAVEnergyDeliverySAC import OnlineScalarTDEnergyEstimator, UAVEnergyDeliverySACEnv


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_clean() -> bool:
    return not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PhaseCheckpointCallback(BaseCallback):
    def __init__(self, output: Path, *, checkpoint_freq: int) -> None:
        super().__init__(verbose=0)
        self.output = output
        self.checkpoint_freq = int(checkpoint_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.checkpoint_freq == 0:
            self.model.save(self.output / f"checkpoint_step_{self.num_timesteps:07d}")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-phase single-UAV SAC protocol")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase1-steps", type=int, default=500_000)
    parser.add_argument("--phase2-steps", type=int, default=500_000)
    parser.add_argument("--episode-limit", type=int, default=10_000)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--operational-energy-capacity", type=float, required=True)
    parser.add_argument("--energy-cost-per-step", type=float, required=True)
    parser.add_argument("--energy-reserve", type=float, required=True)
    parser.add_argument("--energy-critic-initial-output", type=float, required=True)
    parser.add_argument("--energy-learning-rate", type=float, default=3e-4)
    parser.add_argument("--energy-learning-starts", type=int, default=512)
    parser.add_argument("--energy-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.phase1_steps != 500_000 or args.phase2_steps != 500_000:
        parser.error("the formal protocol fixes both phases at 500,000 steps")
    if min(args.operational_energy_capacity, args.energy_cost_per_step, args.energy_critic_initial_output) <= 0.0:
        parser.error("energy capacity, cost, and critic initialization must be positive")
    if not 0.0 <= args.energy_reserve < args.operational_energy_capacity:
        parser.error("energy reserve must lie in [0, operational-energy-capacity)")
    return args


def train(args: argparse.Namespace) -> None:
    if not args.allow_dirty and not git_clean():
        raise RuntimeError("formal two-phase training requires a clean working tree")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    phase1_output = output / "phase1_navigation"
    phase2_output = output / "phase2_energy_managed"
    phase1_output.mkdir()
    phase2_output.mkdir()
    environment = UAVEnergyDeliverySACEnv(
        episode_limit=args.episode_limit,
        operational_energy_capacity=args.operational_energy_capacity,
        energy_cost_per_step=args.energy_cost_per_step,
    )
    check_env(environment, warn=True, skip_render_check=True)
    monitored = Monitor(environment, filename=str(output / "monitor.csv"))
    model = SAC(
        "MlpPolicy",
        monitored,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        verbose=1,
        tensorboard_log=str(output / "tensorboard"),
    )
    command = " ".join([platform.python_implementation(), *os.sys.argv])
    config = {
        "protocol": "UAV_ENERGY_DELIVERY_SAC_TWO_PHASE_V1",
        "status": "RUNNING",
        "started_at": utc_now(),
        "pid": os.getpid(),
        "git_sha": git_sha(),
        "git_clean_at_start": git_clean(),
        "exact_argv": os.sys.argv,
        "command_description": command,
        "seed": args.seed,
        "device": str(model.device),
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps,
        "phase2_replay_policy": "CLEAR_SAC_REPLAY_BEFORE_PHASE2",
        "environment": {
            "map": [4000.0, 4000.0, 4.0],
            "num_agents": 1,
            "num_obstacles": 0,
            "dt": 0.2,
            "horizontal_v_max": 12.0,
            "vertical_v_max": 3.0,
            "horizontal_a_max": 3.0,
            "vertical_a_max": 2.0,
            "goal_radius": 5.0,
            "near_goal_distance": 80.0,
            "observation_dimension": 7,
        },
        "energy": {
            "operational_energy_capacity": args.operational_energy_capacity,
            "energy_cost_per_step": args.energy_cost_per_step,
            "reserve": args.energy_reserve,
            "critic_initial_output": args.energy_critic_initial_output,
            "critic_state": "7D_CHARGER_RELATIVE_NO_SOC",
            "gamma": 1.0,
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": config["started_at"]})
    try:
        model.learn(
            total_timesteps=args.phase1_steps,
            callback=PhaseCheckpointCallback(phase1_output, checkpoint_freq=args.checkpoint_freq),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        phase1_checkpoint = phase1_output / "checkpoint_step_0500000_final.zip"
        model.save(phase1_checkpoint.with_suffix(""))
        if model.replay_buffer is None:
            raise RuntimeError("SAC replay buffer is unavailable at the phase boundary")
        phase1_replay_size = int(model.replay_buffer.size())
        model.replay_buffer.reset()
        if model.replay_buffer.size() != 0:
            raise RuntimeError("SAC replay buffer was not cleared before Phase 2")
        energy_estimator = OnlineScalarTDEnergyEstimator(
            learning_rate=args.energy_learning_rate,
            batch_size=args.energy_batch_size,
            learning_starts=args.energy_learning_starts,
            initial_output=args.energy_critic_initial_output,
            seed=args.seed,
            device=args.device,
        )

        def charger_action_provider(observation: np.ndarray) -> np.ndarray:
            action, _ = model.predict(observation, deterministic=True)
            return np.asarray(action, dtype=np.float32)

        environment.enable_phase_two(
            energy_estimator=energy_estimator,
            charger_action_provider=charger_action_provider,
            reserve=args.energy_reserve,
        )
        model.learn(
            total_timesteps=args.phase2_steps,
            callback=PhaseCheckpointCallback(phase2_output, checkpoint_freq=args.checkpoint_freq),
            reset_num_timesteps=False,
            progress_bar=False,
        )
        final_checkpoint = phase2_output / "checkpoint_step_1000000_final.zip"
        model.save(final_checkpoint.with_suffix(""))
        energy_checkpoint = phase2_output / "scalar_td_energy_estimator.pt"
        torch.save(
            {
                "input_dim": energy_estimator.input_dim,
                "state_dim": energy_estimator.state_dim,
                "action_dim": energy_estimator.action_dim,
                "state_dict": energy_estimator.model.state_dict(),
                "target_state_dict": energy_estimator.target_model.state_dict(),
                "update_count": energy_estimator.update_count,
                "replay_size": len(energy_estimator.replay),
                "gamma": 1.0,
                "state_semantics": "charger_relative_7d_no_soc",
            },
            energy_checkpoint,
        )
        completed = {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "phase1_checkpoint": str(phase1_checkpoint),
            "phase1_checkpoint_sha256": file_sha256(phase1_checkpoint),
            "phase1_replay_size_before_clear": phase1_replay_size,
            "phase2_initial_replay_size": 0,
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "energy_checkpoint": str(energy_checkpoint),
            "energy_checkpoint_sha256": file_sha256(energy_checkpoint),
            "environment_summary": environment.summary(),
        }
        write_json(output / "COMPLETED.json", completed)
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
        raise
    finally:
        monitored.close()


if __name__ == "__main__":
    train(parse_args())
