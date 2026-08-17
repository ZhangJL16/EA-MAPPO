from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import (
    ENERGY_QUANTILES,
    GoalConditionedQuantileTDEnergyEstimator,
    SACTrainingPhase,
    UAVEnergyDeliverySACEnv,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_task(path: Path, task_index: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["tasks"] if isinstance(payload, dict) else payload
    if not 0 <= task_index < len(tasks):
        raise IndexError(f"task-index {task_index} is outside [0, {len(tasks)})")
    return dict(tasks[task_index])


def collect_td_flight(
    policy,
    estimator: GoalConditionedQuantileTDEnergyEstimator,
    task: dict[str, object],
    *,
    seed: int,
    max_steps: int = 4000,
) -> tuple[list[dict[str, object]], dict[str, object], UAVEnergyDeliverySACEnv]:
    environment = UAVEnergyDeliverySACEnv(phase=SACTrainingPhase.TD_PRETRAINING)
    start = np.asarray(task["start_position"], dtype=np.float32)
    goal = np.asarray(task["goal_position"], dtype=np.float32)
    initial_velocity = np.asarray(task.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
    observation, _ = environment.reset(
        seed=seed,
        options={
            "start_position": start,
            "start_velocity": initial_velocity,
            "task_point": goal,
        },
    )
    positions = [environment.agent.pos.copy()]
    rows: list[dict[str, object]] = []
    end_info: dict[str, object] | None = None
    for step in range(1, max_steps + 1):
        action, _ = policy.predict(observation, deterministic=True)
        energy_state = environment.energy_state_for_goal(goal)
        quantiles = estimator.predict_quantiles(energy_state, np.asarray(action, dtype=np.float32))
        observation, reward, terminated, truncated, info = environment.step(action)
        positions.append(environment.agent.pos.copy())
        rows.append(
            {
                "step": step,
                "position": environment.agent.pos.copy(),
                "velocity": environment.agent.vel.copy(),
                "speed": float(np.linalg.norm(environment.agent.vel)),
                "distance_to_goal": float(np.linalg.norm(goal - environment.agent.pos)),
                "action": np.asarray(action, dtype=np.float32).copy(),
                "Q50": float(quantiles[0]),
                "Q90": float(quantiles[1]),
                "Q95": float(quantiles[2]),
                "Q99": float(quantiles[3]),
                "realized_energy": float(info["realized_energy_cost"]),
                "transition_dt": float(info["transition_dt"]),
                "boundary_contact": bool(info["boundary_contact"]),
                "reward": float(reward),
            }
        )
        if terminated or truncated:
            end_info = dict(info)
            break
    if end_info is None:
        raise RuntimeError(f"TD flight did not finish within {max_steps} policy steps")
    costs = np.asarray([float(row["realized_energy"]) for row in rows], dtype=np.float64)
    returns = np.cumsum(costs[::-1])[::-1]
    for row, return_to_go in zip(rows, returns):
        row["true_energy_to_go"] = float(return_to_go)
    summary = {
        "success": bool(end_info["is_success"]),
        "end_reason": end_info["end_reason"],
        "policy_steps": len(rows),
        "straight_line_distance": float(np.linalg.norm(goal - start)),
        "path_length": float(
            np.sum(np.linalg.norm(np.diff(np.stack(positions), axis=0), axis=1))
        ),
        "true_total_energy": float(returns[0]),
        "td_q50_mae": float(
            np.mean(np.abs(np.asarray([row["Q50"] for row in rows]) - returns))
        ),
        "td_q95_coverage": float(
            np.mean(returns <= np.asarray([row["Q95"] for row in rows]))
        ),
        "boundary_contact_steps": int(sum(bool(row["boundary_contact"]) for row in rows)),
        "quantile_ordering_valid": bool(
            all(row["Q50"] <= row["Q90"] <= row["Q95"] <= row["Q99"] for row in rows)
        ),
        "start_position": start.tolist(),
        "goal_position": goal.tolist(),
    }
    return rows, summary, environment


def render_td_flight_gif(
    rows: list[dict[str, object]],
    summary: dict[str, object],
    environment: UAVEnergyDeliverySACEnv,
    output_path: Path,
    *,
    frame_stride: int = 1,
    fps: int = 10,
) -> int:
    if not rows:
        raise ValueError("TD flight rows cannot be empty")
    if frame_stride <= 0 or fps <= 0:
        raise ValueError("frame_stride and fps must be positive")
    positions = np.asarray([row["position"] for row in rows], dtype=np.float64)
    steps = np.asarray([row["step"] for row in rows], dtype=int)
    true_energy = np.asarray([row["true_energy_to_go"] for row in rows], dtype=np.float64)
    quantile_values = {
        label: np.asarray([row[label] for row in rows], dtype=np.float64)
        for label in ("Q50", "Q90", "Q95", "Q99")
    }
    goal = np.asarray(summary["goal_position"], dtype=np.float64)
    start = np.asarray(summary["start_position"], dtype=np.float64)
    frame_indices = list(range(0, len(rows), frame_stride))
    if frame_indices[-1] != len(rows) - 1:
        frame_indices.append(len(rows) - 1)
    frames: list[Image.Image] = []
    colors = {"Q50": "#ff7f0e", "Q90": "#9467bd", "Q95": "#d62728", "Q99": "#8c564b"}
    energy_max = max(
        float(np.max(true_energy)),
        *(float(np.max(values)) for values in quantile_values.values()),
    )
    for index in frame_indices:
        figure, (axis_path, axis_td) = plt.subplots(1, 2, figsize=(12, 5.5))
        axis_path.plot(positions[:, 0], positions[:, 1], color="#b7c9e2", linewidth=1.2, label="full path")
        axis_path.plot(
            positions[: index + 1, 0],
            positions[: index + 1, 1],
            color="#1f77b4",
            linewidth=2.2,
            label="flown path",
        )
        axis_path.scatter(*start[:2], marker="o", s=75, c="#2ca02c", edgecolors="black", label="start")
        axis_path.scatter(*goal[:2], marker="X", s=120, c="#d62728", edgecolors="black", label="goal")
        axis_path.scatter(
            *environment.charger_position[:2],
            marker="P",
            s=110,
            c="#1f77b4",
            edgecolors="black",
            label="charger",
        )
        axis_path.scatter(*positions[index, :2], s=90, c="#ffbf00", edgecolors="black", label="UAV")
        axis_path.set_xlim(0.0, environment.length)
        axis_path.set_ylim(0.0, environment.width)
        axis_path.set_aspect("equal", adjustable="box")
        axis_path.set_xlabel("x (m)")
        axis_path.set_ylabel("y (m)")
        axis_path.grid(alpha=0.2)
        axis_path.legend(loc="upper right", fontsize=7)
        axis_td.plot(steps[: index + 1], true_energy[: index + 1], color="black", linewidth=2.2, label="true RTG")
        for label, values in quantile_values.items():
            axis_td.plot(steps[: index + 1], values[: index + 1], color=colors[label], linewidth=1.4, label=label)
        axis_td.set_xlim(1, len(rows))
        axis_td.set_ylim(0.0, energy_max * 1.05 if energy_max > 0.0 else 1.0)
        axis_td.set_xlabel("policy step")
        axis_td.set_ylabel("energy-to-go (synthetic units)")
        axis_td.grid(alpha=0.25)
        axis_td.legend(loc="upper left", fontsize=8)
        row = rows[index]
        figure.suptitle(
            "Goal-conditioned Quantile TD Flight Test\n"
            f"step={row['step']}/{len(rows)}  distance={row['distance_to_goal']:.1f}m  "
            f"speed={row['speed']:.2f}m/s  step_energy={row['realized_energy']:.4f}\n"
            f"true_RTG={row['true_energy_to_go']:.3f}  Q50={row['Q50']:.3e}  "
            f"Q95={row['Q95']:.3e}  boundary={row['boundary_contact']}",
            fontsize=10,
        )
        figure.tight_layout()
        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba[:, :, :3].copy(), mode="RGB"))
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
    parser = argparse.ArgumentParser(description="Render one complete frozen-SAC TD energy flight")
    parser.add_argument("--sac-checkpoint", required=True)
    parser.add_argument("--td-checkpoint", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--task-index", type=int, default=250)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=130001)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sac_checkpoint = Path(args.sac_checkpoint).resolve()
    td_checkpoint = Path(args.td_checkpoint).resolve()
    tasks_path = Path(args.tasks).resolve()
    policy = SAC.load(sac_checkpoint, device=args.device)
    for parameter in policy.policy.parameters():
        parameter.requires_grad_(False)
    policy.policy.set_training_mode(False)
    estimator = GoalConditionedQuantileTDEnergyEstimator.load(
        td_checkpoint,
        device=args.device,
    )
    task = load_task(tasks_path, args.task_index)
    rows, summary, environment = collect_td_flight(
        policy,
        estimator,
        task,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    gif_path = output / "td_flight_full.gif"
    frame_count = render_td_flight_gif(
        rows,
        summary,
        environment,
        gif_path,
        frame_stride=args.frame_stride,
        fps=args.fps,
    )
    environment.close()
    summary.update(
        {
            "task_index": args.task_index,
            "task_distance_bucket": task.get("distance_bucket"),
            "frame_count": frame_count,
            "frame_stride": args.frame_stride,
            "fps": args.fps,
            "full_rollout_recorded": True,
            "gif_path": str(gif_path),
            "trace_path": str(output / "td_flight_trace.jsonl"),
            "sac_checkpoint": str(sac_checkpoint),
            "sac_checkpoint_sha256": file_sha256(sac_checkpoint),
            "td_checkpoint": str(td_checkpoint),
            "td_checkpoint_sha256": file_sha256(td_checkpoint),
            "energy_quantiles": list(ENERGY_QUANTILES),
        }
    )
    with (output / "td_flight_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            serializable = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in row.items()
            }
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
