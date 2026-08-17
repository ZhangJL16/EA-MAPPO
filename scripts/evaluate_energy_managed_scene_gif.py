from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Protocol

import matplotlib
import numpy as np
from PIL import Image
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import (
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
)
from scripts.train_uav_energy_delivery_sac import single_action_provider


matplotlib.use("Agg")
from matplotlib import pyplot as plt


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True): ...


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_managed_lifecycle(trace: list[dict[str, object]]) -> dict[str, object]:
    if not trace:
        raise ValueError("trace cannot be empty")
    recharge_steps = [
        int(row["step"])
        for row in trace
        if bool(row["battery_cycle_end"])
        and row.get("battery_cycle_record") is not None
        and bool(row["battery_cycle_record"]["return_success"])
    ]
    first_recharge_step = recharge_steps[0] if recharge_steps else None
    tasks_before_first_recharge = (
        max(
            int(row["tasks_completed"])
            for row in trace
            if int(row["step"]) <= first_recharge_step
        )
        if first_recharge_step is not None
        else 0
    )
    resumed_task_steps = [
        int(row["step"])
        for row in trace
        if first_recharge_step is not None
        and int(row["step"]) > first_recharge_step
        and str(row["mode"]) == "TASK"
        and not bool(row["switched_now"])
        and not bool(row["battery_cycle_end"])
    ]
    exhaustion_steps = [
        int(row["step"])
        for row in trace
        if bool(row["terminated"]) and str(row["end_reason"]) == "energy_exhausted"
    ]
    exhausted_after_recharge = bool(
        first_recharge_step is not None
        and exhaustion_steps
        and exhaustion_steps[-1] > first_recharge_step
    )
    lifecycle_observed = bool(
        tasks_before_first_recharge > 0
        and recharge_steps
        and resumed_task_steps
        and exhausted_after_recharge
    )
    return {
        "tasks_before_first_recharge": int(tasks_before_first_recharge),
        "successful_recharge_count": len(recharge_steps),
        "successful_recharge_steps": recharge_steps,
        "resumed_task_after_recharge": bool(resumed_task_steps),
        "first_resumed_task_step": resumed_task_steps[0] if resumed_task_steps else None,
        "energy_exhaustion_steps": exhaustion_steps,
        "energy_exhausted_after_recharge": exhausted_after_recharge,
        "requested_task_return_recharge_resume_exhaustion_lifecycle_observed": lifecycle_observed,
    }


def collect_energy_managed_scene(
    policy: PredictPolicy,
    estimator,
    *,
    capacity: float,
    seed: int,
    max_recording_steps: int,
    start_position: np.ndarray,
    task_point: np.ndarray,
    initial_energy_fraction: float = 1.0,
    reserve_fraction: float = 0.10,
    post_first_recharge_steps: int | None = None,
    environment: UAVEnergyDeliverySACEnv | None = None,
) -> tuple[list[dict[str, object]], dict[str, object], UAVEnergyDeliverySACEnv]:
    if not 0.0 < initial_energy_fraction <= 1.0:
        raise ValueError("initial_energy_fraction must lie in (0, 1]")
    if max_recording_steps <= 0:
        raise ValueError("max_recording_steps must be positive")
    if post_first_recharge_steps is not None and post_first_recharge_steps < 0:
        raise ValueError("post_first_recharge_steps cannot be negative")
    selected_environment = environment or UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.ENERGY_MANAGED,
        render_mode="rgb_array",
        operational_energy_capacity=capacity,
        energy_reserve_fraction=reserve_fraction,
        phase2_episode_limit=max(20_000, max_recording_steps + 1),
    )
    selected_environment.bind_energy_learning(
        energy_estimator=estimator,
        goal_action_provider=single_action_provider(policy),
        training_enabled=False,
    )
    observation, _ = selected_environment.reset(
        seed=seed,
        options={
            "start_position": np.asarray(start_position, dtype=np.float32),
            "start_velocity": np.zeros(3, dtype=np.float32),
            "task_point": np.asarray(task_point, dtype=np.float32),
        },
    )
    selected_environment.agent.energy = capacity * initial_energy_fraction
    selected_environment.cycle_start_energy = selected_environment.agent.energy
    initial_estimate = selected_environment.mission_energy_estimate()
    trace: list[dict[str, object]] = []
    first_recharge_step: int | None = None
    stop_reason = "external_recording_budget_reached_without_environment_terminal"
    for step in range(1, max_recording_steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = selected_environment.step(action)
        cycle_record = info["battery_cycle_record"]
        if cycle_record is not None and cycle_record["return_success"] and first_recharge_step is None:
            first_recharge_step = step
        trace.append(
            {
                "step": step,
                "position": selected_environment.agent.pos.copy(),
                "velocity": selected_environment.agent.vel.copy(),
                "speed": float(np.linalg.norm(selected_environment.agent.vel)),
                "mode": info["mode"],
                "active_goal": np.asarray(info["active_goal"], dtype=np.float32),
                "task_point": np.asarray(info["current_task_point"], dtype=np.float32),
                "tasks_completed": info["tasks_completed"],
                "battery_cycle_id": info["battery_cycle_id"],
                "battery_cycle_end": info["battery_cycle_end"],
                "battery_cycle_record": info["battery_cycle_record"],
                "switched_now": info["switched_now"],
                "remaining_energy": info["remaining_energy"],
                "remaining_energy_fraction": info["remaining_energy_fraction"],
                "step_realized_energy": info["realized_energy_cost"],
                "distance_to_active_goal": info["distance_to_active_goal"],
                "Q50": info["Q50"],
                "Q90": info["Q90"],
                "Q95": info["Q95"],
                "Q99": info["Q99"],
                "E_task_95": info["E_task_95"],
                "E_return_after_task_95": info["E_return_after_task_95"],
                "E_mission_95": info["E_mission_95"],
                "E_return_now_95": info["E_return_now_95"],
                "transition_dt": info["transition_dt"],
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "end_reason": info["end_reason"],
            }
        )
        if terminated or truncated:
            stop_reason = str(info["end_reason"])
            break
        if (
            first_recharge_step is not None
            and post_first_recharge_steps is not None
            and step >= first_recharge_step + post_first_recharge_steps
        ):
            stop_reason = "post_first_recharge_recording_complete"
            break
    if not trace:
        raise RuntimeError("energy-managed scene produced no transitions")
    final = trace[-1]
    cycle_records = selected_environment.battery_cycle_records
    zero_task_cycles = sum(
        int(record["tasks_completed_in_cycle"]) == 0 for record in cycle_records
    )
    one_step_zero_task_cycles = sum(
        int(record["tasks_completed_in_cycle"]) == 0
        and int(record["cycle_policy_steps"]) <= 1
        for record in cycle_records
    )
    summary = {
        "scenario": "continuous_energy_managed_delivery_with_trained_td",
        "start_position": np.asarray(start_position, dtype=np.float32),
        "initial_task_point": np.asarray(task_point, dtype=np.float32),
        "calibrated_battery_capacity": capacity,
        "initial_energy_fraction": initial_energy_fraction,
        "initial_energy": capacity * initial_energy_fraction,
        "energy_reserve": capacity * reserve_fraction,
        "initial_task_q95": initial_estimate.task_q95,
        "initial_return_after_task_q95": initial_estimate.return_after_task_q95,
        "initial_mission_q95": initial_estimate.mission_q95_composition,
        "initial_return_now_q95": initial_estimate.return_now_q95,
        "initial_mission_margin": (
            capacity * initial_energy_fraction
            - initial_estimate.mission_q95_composition
            - capacity * reserve_fraction
        ),
        "recorded_policy_steps": len(trace),
        "tasks_completed": int(final["tasks_completed"]),
        "battery_cycles_completed": int(selected_environment.battery_cycles_completed),
        "first_recharge_step": first_recharge_step,
        "switch_count": int(sum(bool(row["switched_now"]) for row in trace)),
        "zero_task_battery_cycles": int(zero_task_cycles),
        "one_step_zero_task_battery_cycles": int(one_step_zero_task_cycles),
        "charger_loop_detected": bool(one_step_zero_task_cycles >= 2),
        "energy_exhausted": bool(final["end_reason"] == "energy_exhausted"),
        "environment_terminated": bool(final["terminated"]),
        "environment_truncated": bool(final["truncated"]),
        "recording_stop_reason": stop_reason,
        "final_remaining_energy": final["remaining_energy"],
        "final_remaining_energy_fraction": final["remaining_energy_fraction"],
        "uses_trained_td": True,
        "forced_switching": False,
        "charger_reach_is_terminal": False,
        "requested_energy_exhaustion_observed": bool(final["end_reason"] == "energy_exhausted"),
    }
    summary.update(audit_managed_lifecycle(trace))
    return trace, summary, selected_environment


def render_energy_managed_gif(
    trace: list[dict[str, object]],
    summary: dict[str, object],
    environment: UAVEnergyDeliverySACEnv,
    output_path: Path,
    *,
    frame_stride: int = 1,
    fps: int = 15,
) -> int:
    if not trace:
        raise ValueError("trace cannot be empty")
    if frame_stride <= 0 or fps <= 0:
        raise ValueError("frame_stride and fps must be positive")
    indices = list(range(0, len(trace), frame_stride))
    if indices[-1] != len(trace) - 1:
        indices.append(len(trace) - 1)
    positions = np.asarray([row["position"] for row in trace], dtype=np.float64)
    steps = np.asarray([row["step"] for row in trace], dtype=np.int64)
    battery = np.asarray([row["remaining_energy"] for row in trace], dtype=np.float64)
    q95 = np.asarray(
        [np.nan if row["Q95"] is None else row["Q95"] for row in trace],
        dtype=np.float64,
    )
    return95 = np.asarray(
        [np.nan if row["E_return_now_95"] is None else row["E_return_now_95"] for row in trace],
        dtype=np.float64,
    )
    capacity = float(summary["calibrated_battery_capacity"])
    reserve = float(summary["energy_reserve"])
    start = np.asarray(summary["start_position"], dtype=np.float64)
    charger = environment.charger_position.astype(np.float64)
    all_positive_td = np.concatenate((q95, return95, [capacity]))
    all_positive_td = all_positive_td[np.isfinite(all_positive_td) & (all_positive_td > 0.0)]
    use_log_td_axis = bool(
        all_positive_td.size and all_positive_td.max() / all_positive_td.min() > 100.0
    )
    td_lower = max(1e-6, float(all_positive_td.min()) * 0.8)
    td_upper = max(td_lower * 1.1, float(all_positive_td.max()) * 1.2)
    frames: list[Image.Image] = []
    for index in indices:
        row = trace[index]
        task = np.asarray(row["task_point"], dtype=np.float64)
        active_goal = np.asarray(row["active_goal"], dtype=np.float64)
        figure = plt.figure(figsize=(10.0, 4.8), dpi=75)
        axis_path = figure.add_subplot(1, 2, 1)
        axis_energy = figure.add_subplot(2, 2, 2)
        axis_td = figure.add_subplot(2, 2, 4)
        axis_path.plot(
            positions[: index + 1, 0],
            positions[: index + 1, 1],
            color="#1f77b4",
            linewidth=1.8,
            label="flight path",
        )
        axis_path.scatter(*start[:2], marker="o", s=70, c="#7f7f7f", label="start")
        axis_path.scatter(*task[:2], marker="X", s=110, c="#2ca02c", label="current task")
        axis_path.scatter(*charger[:2], marker="P", s=125, c="#1f77b4", edgecolors="black", label="charger")
        axis_path.scatter(*active_goal[:2], marker="*", s=150, c="#d62728", label="active goal")
        axis_path.scatter(*positions[index, :2], s=85, c="#ffbf00", edgecolors="black", label="UAV")
        axis_path.set_xlim(0.0, environment.length)
        axis_path.set_ylim(0.0, environment.width)
        axis_path.set_aspect("equal", adjustable="box")
        axis_path.set_xlabel("x (m)")
        axis_path.set_ylabel("y (m)")
        axis_path.grid(alpha=0.2)
        axis_path.legend(loc="upper left", fontsize=7)

        axis_energy.plot(steps[: index + 1], battery[: index + 1], color="#2ca02c", linewidth=1.8)
        axis_energy.axhline(capacity, color="#2ca02c", linestyle=":", linewidth=1.0, label="capacity")
        axis_energy.axhline(reserve, color="#d62728", linestyle="--", linewidth=1.0, label="reserve")
        axis_energy.set_xlim(0.5, max(1.5, float(len(trace))))
        axis_energy.set_ylim(0.0, capacity * 1.05)
        axis_energy.set_ylabel("remaining energy")
        axis_energy.grid(alpha=0.25)
        axis_energy.legend(fontsize=7)

        axis_td.plot(steps[: index + 1], q95[: index + 1], color="#9467bd", linewidth=1.4, label="active-goal Q95")
        axis_td.plot(steps[: index + 1], return95[: index + 1], color="#ff7f0e", linewidth=1.4, label="charger Q95")
        axis_td.axhline(capacity, color="#2ca02c", linestyle=":", linewidth=1.0, label="battery capacity")
        if use_log_td_axis:
            axis_td.set_yscale("log")
        axis_td.set_ylim(td_lower, td_upper)
        axis_td.set_xlim(0.5, max(1.5, float(len(trace))))
        axis_td.set_xlabel("policy step")
        axis_td.set_ylabel("TD energy estimate")
        axis_td.grid(alpha=0.25)
        axis_td.legend(fontsize=7)

        figure.suptitle(
            "TD-managed continuous UAV mission\n"
            f"step={row['step']}/{len(trace)} mode={row['mode']} tasks={row['tasks_completed']} "
            f"cycles={row['battery_cycle_id']} speed={row['speed']:.2f}m/s\n"
            f"battery={row['remaining_energy']:.3f}/{capacity:.3f} "
            f"step_energy={row['step_realized_energy']:.4f} "
            f"Q95={row['Q95']} return_Q95={row['E_return_now_95']}",
            fontsize=10,
        )
        figure.tight_layout()
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        frames.append(
            Image.fromarray(rgba[:, :, :3].copy(), mode="RGB").convert(
                "P",
                palette=Image.Palette.ADAPTIVE,
                colors=128,
            )
        )
        plt.close(figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 / fps)),
        loop=0,
        optimize=False,
    )
    return len(frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a complete continuous energy-managed mission using a trained TD estimator"
    )
    parser.add_argument("--sac-checkpoint", required=True)
    parser.add_argument("--td-checkpoint", required=True)
    parser.add_argument("--battery-calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=150001)
    parser.add_argument("--max-recording-steps", type=int, default=5000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--initial-energy-fraction", type=float, default=1.0)
    parser.add_argument("--post-first-recharge-steps", type=int)
    parser.add_argument("--start-position", nargs=3, type=float, default=[500.0, 500.0, 200.0])
    parser.add_argument("--task-point", nargs=3, type=float, default=[3500.0, 3500.0, 200.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sac_checkpoint = Path(args.sac_checkpoint).resolve()
    td_checkpoint = Path(args.td_checkpoint).resolve()
    calibration_path = Path(args.battery_calibration).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    capacity = float(calibration["calibrated_battery_capacity"])
    policy = SAC.load(sac_checkpoint, device=args.device)
    for parameter in policy.policy.parameters():
        parameter.requires_grad_(False)
    policy.policy.set_training_mode(False)
    estimator = GoalConditionedQuantileTDEnergyEstimator.load(
        td_checkpoint,
        device=args.device,
    )
    trace, summary, environment = collect_energy_managed_scene(
        policy,
        estimator,
        capacity=capacity,
        seed=args.seed,
        max_recording_steps=args.max_recording_steps,
        start_position=np.asarray(args.start_position, dtype=np.float32),
        task_point=np.asarray(args.task_point, dtype=np.float32),
        initial_energy_fraction=args.initial_energy_fraction,
        post_first_recharge_steps=args.post_first_recharge_steps,
    )
    gif_path = output / "energy_managed_scene_full.gif"
    frame_count = render_energy_managed_gif(
        trace,
        summary,
        environment,
        gif_path,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    summary.update(
        {
            "gif_frames": frame_count,
            "gif_path": str(gif_path),
            "trace_path": str(output / "energy_managed_trace.jsonl"),
            "sac_checkpoint": str(sac_checkpoint),
            "sac_checkpoint_sha256": file_sha256(sac_checkpoint),
            "td_checkpoint": str(td_checkpoint),
            "td_checkpoint_sha256": file_sha256(td_checkpoint),
            "battery_calibration": str(calibration_path),
        }
    )
    with (output / "energy_managed_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in trace:
            handle.write(json.dumps(json_value(row), sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(json_value(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment.close()
    print(json.dumps(json_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
