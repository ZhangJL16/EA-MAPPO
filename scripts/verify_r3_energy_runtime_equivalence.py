from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_bundle.safety.collision import hocbf as hocbf_module
from review_bundle.safety.collision.hocbf import SphericalObstacle
from scripts.diagnose_return_decision_stage_b import diagnostic_args
from scripts.run_return_decision_stage_b import (
    configure_method,
    environment_from_args,
    load_policy,
    load_prerequisite_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate frozen-R3 Python/native QP trajectory equivalence and "
            "same-solver object/array constraint-builder equivalence."
        )
    )
    parser.add_argument("--source-launch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seed", type=int, default=110004)
    parser.add_argument("--soc-threshold", type=float, default=0.20)
    parser.add_argument("--maximum-policy-steps", type=int, default=50_000)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def trace_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def run_cycle(
    stage_args: argparse.Namespace,
    policy,
    *,
    evaluation_seed: int,
    soc_threshold: float,
    maximum_policy_steps: int,
    native_qp_enabled: bool,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    environment = environment_from_args(stage_args, reserve_fraction=0.0)
    configure_method(
        environment,
        policy,
        method="soc",
        parameter=soc_threshold,
        args=stage_args,
    )
    if environment.safety_filter is None:
        raise RuntimeError("equivalence requires the frozen HOCBF filter")
    environment.bind_keyed_task_schedule(evaluation_seed)
    observation, _ = environment.reset(seed=evaluation_seed)
    continuous_rows: list[np.ndarray] = []
    flag_rows: list[np.ndarray] = []
    started = time.perf_counter()
    terminal_reason: str | None = None
    try:
        for policy_step in range(1, maximum_policy_steps + 1):
            nominal_action = policy.action(observation)
            observation, reward, terminated, truncated, info = environment.step(
                nominal_action
            )
            continuous_rows.append(
                np.concatenate(
                    (
                        np.asarray(environment.agent.pos, dtype=np.float64),
                        np.asarray(environment.agent.vel, dtype=np.float64),
                        np.asarray(info["current_task_point"], dtype=np.float64),
                        np.asarray(info["active_goal"], dtype=np.float64),
                        np.asarray(info["nominal_action"], dtype=np.float64),
                        np.asarray(info["executed_action"], dtype=np.float64),
                        np.asarray(info["realized_acceleration"], dtype=np.float64),
                        np.asarray(
                            [
                                float(info["realized_energy_cost"]),
                                float(info["cumulative_virtual_energy"]),
                                float(info["remaining_energy"]),
                                float(reward),
                            ],
                            dtype=np.float64,
                        ),
                    )
                )
            )
            flag_rows.append(
                np.asarray(
                    [
                        int(info["mode"] == "CHARGER_COMMITTED"),
                        int(bool(info["switched_now"])),
                        int(bool(info["task_completed_now"])),
                        int(bool(info["charger_reached_now"])),
                        int(bool(info["boundary_contact"])),
                        int(bool(info["obstacle_collision"])),
                        int(bool(info["hocbf_intervened"])),
                        int(bool(info["hocbf_fallback_used"])),
                        int(bool(info["hocbf_emergency_brake"])),
                        int(bool(terminated)),
                        int(bool(truncated)),
                    ],
                    dtype=np.int8,
                )
            )
            if bool(info["charger_reached_now"]):
                terminal_reason = "charger_reached"
                break
            if terminated:
                terminal_reason = str(info.get("end_reason", "terminated"))
                break
            if truncated:
                terminal_reason = str(info.get("end_reason", "truncated"))
                break
        else:
            terminal_reason = "equivalence_policy_step_limit"
    finally:
        elapsed = time.perf_counter() - started
        summary = {
            "native_qp_enabled": native_qp_enabled,
            "policy_steps": len(continuous_rows),
            "terminal_reason": terminal_reason,
            "wall_clock_seconds": elapsed,
            "tasks_completed": int(environment.tasks_completed),
            "hocbf_filter_calls": int(environment.safety_filter_calls),
            "hocbf_interventions": int(environment.safety_interventions),
            "hocbf_fallbacks": int(environment.safety_fallbacks),
            "hocbf_emergency_brakes": int(environment.safety_emergency_brakes),
            "obstacle_collision_count": int(environment.obstacle_collision_count),
            "boundary_collision_count": int(environment.boundary_collision_count),
            "remaining_energy": float(environment.agent.energy),
            "cumulative_realized_energy": float(environment.cumulative_virtual_energy),
        }
        environment.close()
    if not continuous_rows:
        raise RuntimeError("equivalence trajectory produced no policy steps")
    return {
        "continuous": np.stack(continuous_rows),
        "flags": np.stack(flag_rows),
    }, summary


def compare_constraint_builders(
    stage_args: argparse.Namespace,
    *,
    evaluation_seed: int,
    snapshot_count: int,
    absolute_tolerance: float,
) -> dict[str, object]:
    """Hold the native solver fixed while comparing object/array constraints."""

    environment = environment_from_args(stage_args, reserve_fraction=0.0)
    safety_filter = environment.safety_filter
    environment.close()
    if safety_filter is None:
        raise RuntimeError("constraint comparison requires the frozen HOCBF filter")
    if hocbf_module._NATIVE_QP_FUNCTION is None:
        raise RuntimeError("same-solver snapshot comparison requires the native QP")
    rng = np.random.default_rng(evaluation_seed + 91_337)
    ignored_diagnostics = {
        "constraint_build_seconds",
        "solver_seconds",
        "total_seconds",
        "deadline_missed",
    }
    maximum_action_delta = 0.0
    mismatched_diagnostic_snapshots: list[int] = []
    for snapshot_index in range(snapshot_count):
        position = rng.uniform(-200.0, 200.0, size=3)
        velocity = rng.uniform(-15.0, 15.0, size=3)
        nominal = np.asarray(
            [
                rng.uniform(
                    -safety_filter.config.horizontal_acceleration_limit,
                    safety_filter.config.horizontal_acceleration_limit,
                ),
                rng.uniform(
                    -safety_filter.config.horizontal_acceleration_limit,
                    safety_filter.config.horizontal_acceleration_limit,
                ),
                rng.uniform(
                    -safety_filter.config.vertical_acceleration_limit,
                    safety_filter.config.vertical_acceleration_limit,
                ),
            ],
            dtype=np.float64,
        )
        point_count = (1, 4, 16, 64, 211, 512)[snapshot_index % 6]
        directions = rng.normal(size=(point_count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        distances = rng.uniform(0.25, 150.0, size=(point_count, 1))
        points = position[None, :] + directions * distances
        objects = [
            SphericalObstacle(point, 1e-9, identifier=f"point_{index}")
            for index, point in enumerate(points)
        ]
        object_output = safety_filter.filter(position, velocity, nominal, objects)
        array_output = safety_filter.filter_lidar_points(
            position,
            velocity,
            nominal,
            points,
        )
        maximum_action_delta = max(
            maximum_action_delta,
            float(np.max(np.abs(object_output.acceleration - array_output.acceleration))),
        )
        object_diagnostics = {
            key: value
            for key, value in object_output.diagnostics.__dict__.items()
            if key not in ignored_diagnostics
        }
        array_diagnostics = {
            key: value
            for key, value in array_output.diagnostics.__dict__.items()
            if key not in ignored_diagnostics
        }
        if object_diagnostics != array_diagnostics:
            mismatched_diagnostic_snapshots.append(snapshot_index)
    return {
        "snapshot_count": snapshot_count,
        "same_native_solver": True,
        "maximum_action_absolute_delta": maximum_action_delta,
        "mismatched_diagnostic_snapshot_count": len(
            mismatched_diagnostic_snapshots
        ),
        "mismatched_diagnostic_snapshots": mismatched_diagnostic_snapshots,
        "passed": bool(
            maximum_action_delta <= absolute_tolerance
            and not mismatched_diagnostic_snapshots
        ),
    }


def main() -> None:
    cli = parse_args()
    if cli.output_dir.exists():
        raise FileExistsError(cli.output_dir)
    if not 0.0 < cli.soc_threshold < 1.0:
        raise ValueError("SOC threshold must lie in (0, 1)")
    if cli.maximum_policy_steps <= 0 or cli.absolute_tolerance <= 0.0:
        raise ValueError("step limit and tolerance must be positive")
    cli.output_dir.mkdir(parents=True)
    running = {
        "status": "RUNNING",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_seed": cli.evaluation_seed,
    }
    atomic_json(cli.output_dir / "RUNNING.json", running)
    try:
        stage_args = diagnostic_args(
            cli.source_launch,
            cli.output_dir,
            [cli.evaluation_seed],
            evaluation_num_envs=1,
            formal_seed_retries=0,
        )
        prerequisite = load_prerequisite_audit(stage_args)
        if not prerequisite.passed:
            raise RuntimeError("frozen R3 energy prerequisites no longer pass")
        probe = environment_from_args(stage_args, reserve_fraction=0.0)
        policy = load_policy(stage_args, probe)
        probe.close()

        saved_native = hocbf_module._NATIVE_QP_FUNCTION
        if saved_native is None:
            raise RuntimeError("native QP library is unavailable")
        try:
            hocbf_module._NATIVE_QP_FUNCTION = None
            python_trace, python_summary = run_cycle(
                stage_args,
                policy,
                evaluation_seed=cli.evaluation_seed,
                soc_threshold=cli.soc_threshold,
                maximum_policy_steps=cli.maximum_policy_steps,
                native_qp_enabled=False,
            )
        finally:
            hocbf_module._NATIVE_QP_FUNCTION = saved_native
        native_trace, native_summary = run_cycle(
            stage_args,
            policy,
            evaluation_seed=cli.evaluation_seed,
            soc_threshold=cli.soc_threshold,
            maximum_policy_steps=cli.maximum_policy_steps,
            native_qp_enabled=True,
        )
        builder_comparison = compare_constraint_builders(
            stage_args,
            evaluation_seed=cli.evaluation_seed,
            snapshot_count=96,
            absolute_tolerance=cli.absolute_tolerance,
        )
        np.savez_compressed(cli.output_dir / "python_qp_trace.npz", **python_trace)
        np.savez_compressed(
            cli.output_dir / "native_qp_trace.npz", **native_trace
        )
        same_shape = all(
            python_trace[name].shape == native_trace[name].shape
            for name in python_trace
        )
        continuous_max_abs_delta = None
        flags_equal = False
        if same_shape:
            continuous_max_abs_delta = float(
                np.max(
                    np.abs(
                        python_trace["continuous"]
                        - native_trace["continuous"]
                    )
                )
            )
            flags_equal = bool(
                np.array_equal(python_trace["flags"], native_trace["flags"])
            )
        terminal_equal = (
            python_summary["terminal_reason"] == native_summary["terminal_reason"]
        )
        trajectory_passed = bool(
            same_shape
            and flags_equal
            and terminal_equal
            and continuous_max_abs_delta is not None
            and continuous_max_abs_delta <= cli.absolute_tolerance
        )
        passed = bool(trajectory_passed and builder_comparison["passed"])
        result = {
            "protocol": "r3_energy_runtime_equivalence_v2",
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "evaluation_seed": cli.evaluation_seed,
            "soc_threshold": cli.soc_threshold,
            "absolute_tolerance": cli.absolute_tolerance,
            "qp_backend_trajectory_comparison": {
                "array_constraint_path_fixed": True,
                "same_trace_shape": same_shape,
                "flags_equal": flags_equal,
                "terminal_semantics_equal": terminal_equal,
                "continuous_max_absolute_delta": continuous_max_abs_delta,
                "passed": trajectory_passed,
            },
            "constraint_builder_snapshot_comparison": builder_comparison,
            "python_qp": {
                **python_summary,
                "trace_sha256": trace_digest(python_trace),
            },
            "native_qp": {
                **native_summary,
                "trace_sha256": trace_digest(native_trace),
            },
            "source_launch": str(cli.source_launch.resolve()),
            "source_launch_sha256": file_sha256(cli.source_launch),
            "navigation_checkpoint": str(stage_args.navigation_checkpoint.resolve()),
            "navigation_checkpoint_sha256": file_sha256(
                stage_args.navigation_checkpoint
            ),
            "native_qp_available": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(cli.output_dir / "RESULT.json", result)
        (cli.output_dir / "RUNNING.json").unlink(missing_ok=True)
        terminal_name = "COMPLETED.json" if passed else "FAILED.json"
        atomic_json(cli.output_dir / terminal_name, result)
        if not passed:
            raise SystemExit(2)
    except Exception as error:
        if not (cli.output_dir / "FAILED.json").exists():
            atomic_json(
                cli.output_dir / "FAILED.json",
                {
                    **running,
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        (cli.output_dir / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
