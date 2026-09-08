from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from torch import nn

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from review_bundle.safety.energy.mc_regression import (
    ModelBasedEnergyRolloutEstimator,
    ModelBasedRolloutError,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    environment_from_args,
    generate_stratified_navigation_tasks,
)


DEFAULT_HORIZONS = (1000, 2000, 3000, 4000)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory, seed-held-out learnability Gate for an extended-real "
            "resource-to-recharge CDF head under the frozen R3 policy"
        )
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-tasks", type=int, default=30)
    parser.add_argument("--task-seed", type=int, default=140_001)
    parser.add_argument("--world-seed", type=int, default=150_001)
    parser.add_argument("--max-policy-steps", type=int, default=4000)
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS))
    parser.add_argument("--snapshot-interval", type=int, default=200)
    parser.add_argument("--energy-bins", type=int, default=32)
    parser.add_argument("--support-capacity-multiple", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wait-for-upstream", action="store_true")
    parser.add_argument("--upstream-pid", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_tasks = 5
        args.max_policy_steps = 20
        args.horizons = [5, 10, 20]
        args.snapshot_interval = 10
        args.epochs = 2
        args.batch_size = 16
    if args.num_tasks <= 0 or args.num_tasks % 5:
        parser.error("--num-tasks must be a positive multiple of five")
    if args.max_policy_steps <= 0 or args.snapshot_interval <= 0:
        parser.error("policy-step limits must be positive")
    if not args.horizons or any(value <= 0 for value in args.horizons):
        parser.error("--horizons must contain positive values")
    if any(value > args.max_policy_steps for value in args.horizons):
        parser.error("no horizon may exceed --max-policy-steps")
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("--horizons must be unique")
    if args.energy_bins < 4 or args.support_capacity_multiple <= 1.0:
        parser.error("energy support requires >=4 bins and capacity multiple >1")
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        parser.error("training parameters must be positive")
    if args.upstream_pid is not None and args.upstream_pid <= 0:
        parser.error("--upstream-pid must be positive")
    return args


class ResourceCDFHead(nn.Module):
    """Finite energy atoms plus one explicit collision/deadline failure atom."""

    def __init__(self, input_dim: int, hidden_dim: int, finite_bins: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, finite_bins + 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def energy_class(
    energy: float | None,
    *,
    capacity: float,
    finite_bins: int,
    support_capacity_multiple: float,
) -> int:
    if energy is None or not np.isfinite(energy):
        return finite_bins
    ratio = max(0.0, float(energy) / float(capacity))
    width = float(support_capacity_multiple) / finite_bins
    return min(finite_bins - 1, int(ratio / width))


def cdf_probability(
    probabilities: np.ndarray,
    *,
    budget_fraction: float,
    support_capacity_multiple: float,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    finite_bins = values.shape[1] - 1
    upper_edges = (
        np.arange(1, finite_bins + 1, dtype=np.float64)
        * float(support_capacity_multiple)
        / finite_bins
    )
    included = upper_edges <= float(budget_fraction) + 1e-12
    return values[:, :finite_bins][:, included].sum(axis=1)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("binary metric inputs must be aligned vectors")
    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    positives = int(y.sum())
    negatives = int(y.size - positives)
    auroc = None
    auprc = None
    if positives and negatives:
        positive_scores = p[y == 1]
        negative_scores = p[y == 0]
        auroc = float(
            np.mean(
                (positive_scores[:, None] > negative_scores[None, :])
                + 0.5 * (positive_scores[:, None] == negative_scores[None, :])
            )
        )
        order = np.argsort(-p, kind="stable")
        ordered = y[order]
        precision = np.cumsum(ordered) / np.arange(1, y.size + 1)
        auprc = float(np.sum(precision * ordered) / positives)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (p >= lower) & (p < lower + 0.1 + (1e-12 if lower == 0.9 else 0.0))
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(float(np.mean(p[selected]) - np.mean(y[selected])))
    return {
        "count": int(y.size),
        "positive_count": positives,
        "prevalence": float(np.mean(y)),
        "brier": brier,
        "log_loss": log_loss,
        "auroc": auroc,
        "auprc": auprc,
        "ece_10_bin": float(ece),
        "accuracy_at_0_5": float(np.mean((p >= 0.5) == y)),
    }


def fit_logistic_baseline(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(x.shape[0]), (x - mean) / scale))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(2000):
        logits = np.clip(design @ weights, -30.0, 30.0)
        prediction = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (prediction - y) / y.size
        gradient[1:] += 1e-3 * weights[1:]
        weights -= 0.08 * gradient
    return weights, np.vstack((mean, scale))


def predict_logistic(features: np.ndarray, weights: np.ndarray, normalization: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(features.shape[0]), (features - normalization[0]) / normalization[1]))
    logits = np.clip(design @ weights, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def sanitize_snapshot_velocity(
    velocity: np.ndarray,
    *,
    horizontal_limit: float,
    vertical_limit: float,
) -> np.ndarray:
    """Project float32 integration noise without hiding physical violations."""

    value = np.asarray(velocity, dtype=np.float32).copy()
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("snapshot velocity must be a finite three-vector")
    tolerance = 64.0 * np.finfo(np.float32).eps * max(
        1.0, float(horizontal_limit), float(vertical_limit)
    )
    horizontal_speed = float(np.linalg.norm(value[:2].astype(np.float64)))
    if horizontal_speed > float(horizontal_limit) + tolerance:
        raise ValueError("snapshot horizontal velocity has a non-numerical limit violation")
    if abs(float(value[2])) > float(vertical_limit) + tolerance:
        raise ValueError("snapshot vertical velocity has a non-numerical limit violation")
    if horizontal_speed > float(horizontal_limit):
        value[:2] *= float(horizontal_limit) / horizontal_speed
    value[2] = np.clip(value[2], -float(vertical_limit), float(vertical_limit))
    return value


def snapshot_feature(environment, oracle, position, velocity, task_goal, horizon_fraction):
    snapshot_velocity = sanitize_snapshot_velocity(
        velocity,
        horizontal_limit=environment.horizontal_v_max,
        vertical_limit=environment.vertical_v_max,
    )
    clone = oracle._make_rollout_environment(
        environment,
        start_position=np.asarray(position, dtype=np.float32),
        start_velocity=snapshot_velocity,
        goal=np.asarray(task_goal, dtype=np.float32),
    )
    try:
        task_observation = clone._active_goal_sac_observation()
        charger_state = clone.compact_energy_state_for_goal(clone.charger_position)
        absolute_position = np.asarray(
            [
                clone.agent.pos[0] / clone.length,
                clone.agent.pos[1] / clone.width,
                clone.agent.pos[2] / clone.height,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            (
                task_observation,
                charger_state,
                absolute_position,
                np.asarray([horizon_fraction], dtype=np.float32),
            )
        ).astype(np.float32)
    finally:
        clone.close()


def collect_task(args, task, task_index: int, policy, capacity: float) -> dict[str, object]:
    environment_args, _ = reconstruct_environment_args(
        args.artifact,
        device=args.device,
        seed=args.world_seed + task_index,
    )
    environment = environment_from_args(environment_args, phase=SACTrainingPhase.TD_PRETRAINING)
    environment.reset(
        seed=args.world_seed + task_index,
        options={
            "start_position": task.start_position,
            "start_velocity": task.initial_velocity,
            "task_point": task.goal_position,
        },
    )
    oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=args.max_policy_steps)
    initial_position = environment.agent.pos.copy()
    initial_velocity = environment.agent.vel.copy()
    rows: list[tuple[np.ndarray, int, float, bool, int, float, float]] = []
    status = "finite"
    failure_detail = None
    try:
        task_trace = oracle._rollout_trace(
            environment,
            environment.current_task_point,
            rollout_label="learnability_task",
        )
        task_ok = bool(task_trace.prediction.deadline_feasible)
        return_prediction = None
        if task_ok:
            try:
                return_prediction = oracle.estimate_context(
                    environment,
                    environment.charger_position,
                    position=task_trace.positions[-1],
                    velocity=np.zeros(3, dtype=np.float32),
                    goal_type="learnability_return_after_task",
                )
            except ModelBasedRolloutError as error:
                status = "return_unsuccessful_terminal"
                failure_detail = error.rollout_diagnostics
        else:
            status = "task_deadline_infeasible"
        return_ok = bool(return_prediction is not None and return_prediction.deadline_feasible)
        if return_prediction is not None and not return_ok:
            status = "return_deadline_infeasible"
        indices = [0]
        if task_ok:
            indices = list(range(0, max(1, task_trace.positions.shape[0] - 1), args.snapshot_interval))
            if not indices:
                indices = [0]
        for trace_index in indices:
            remaining_task_steps = max(0, int(task_trace.prediction.rollout_steps) - trace_index)
            base_energy = None
            if task_ok and return_ok:
                base_energy = float(
                    task_trace.suffix_energies[trace_index]
                    + return_prediction.prediction
                )
            for horizon in sorted(args.horizons):
                finite = bool(
                    base_energy is not None
                    and remaining_task_steps <= horizon
                    and int(return_prediction.rollout_steps) <= horizon
                )
                resource = base_energy if finite else None
                feature = snapshot_feature(
                    environment,
                    oracle,
                    task_trace.positions[trace_index],
                    task_trace.velocities[trace_index],
                    environment.current_task_point,
                    horizon / args.max_policy_steps,
                )
                geometry = np.asarray(
                    [
                        np.linalg.norm(environment.current_task_point - task_trace.positions[trace_index]) / environment.d_max,
                        np.linalg.norm(environment.charger_position - environment.current_task_point) / environment.d_max,
                        horizon / args.max_policy_steps,
                    ],
                    dtype=np.float32,
                )
                rows.append(
                    (
                        feature,
                        energy_class(
                            resource,
                            capacity=capacity,
                            finite_bins=args.energy_bins,
                            support_capacity_multiple=args.support_capacity_multiple,
                        ),
                        np.nan if resource is None else resource / capacity,
                        trace_index == 0,
                        int(horizon),
                        float(geometry[0]),
                        float(geometry[1]),
                    )
                )
    except ModelBasedRolloutError as error:
        status = "task_unsuccessful_terminal"
        failure_detail = error.rollout_diagnostics
        for horizon in sorted(args.horizons):
            feature = snapshot_feature(
                environment,
                oracle,
                initial_position,
                initial_velocity,
                environment.current_task_point,
                horizon / args.max_policy_steps,
            )
            rows.append(
                (
                    feature,
                    args.energy_bins,
                    np.nan,
                    True,
                    int(horizon),
                    float(np.linalg.norm(environment.current_task_point - initial_position) / environment.d_max),
                    float(np.linalg.norm(environment.charger_position - environment.current_task_point) / environment.d_max),
                )
            )
    finally:
        environment.close()
    arrays = {
        "features": np.stack([row[0] for row in rows]),
        "classes": np.asarray([row[1] for row in rows], dtype=np.int64),
        "energy_fraction": np.asarray([row[2] for row in rows], dtype=np.float32),
        "is_initial": np.asarray([row[3] for row in rows], dtype=np.bool_),
        "horizon": np.asarray([row[4] for row in rows], dtype=np.int64),
        "geometry": np.asarray([[row[5], row[6], row[4] / args.max_policy_steps] for row in rows], dtype=np.float32),
        "task_index": np.full(len(rows), task_index, dtype=np.int64),
    }
    target = args.output_dir / "tasks" / f"task_{task_index:03d}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target)
    metadata = {
        "task_index": task_index,
        "world_seed": args.world_seed + task_index,
        "distance_bucket": task.distance_bucket,
        "start": np.asarray(task.start_position).tolist(),
        "goal": np.asarray(task.goal_position).tolist(),
        "status": status,
        "num_examples": len(rows),
        "num_initial_examples": int(arrays["is_initial"].sum()),
        "num_finite_examples": int(np.sum(arrays["classes"] < args.energy_bins)),
        "failure_detail": failure_detail,
    }
    atomic_json(args.output_dir / "tasks" / f"task_{task_index:03d}.json", metadata)
    return metadata


def load_dataset(output: Path, num_tasks: int) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {}
    for index in range(num_tasks):
        with np.load(output / "tasks" / f"task_{index:03d}.npz") as data:
            for key in data.files:
                parts.setdefault(key, []).append(data[key])
    return {key: np.concatenate(values, axis=0) for key, values in parts.items()}


def train_fold(args, data: dict[str, np.ndarray], fold: int) -> dict[str, object]:
    task_index = data["task_index"]
    train = task_index % 3 != fold
    test = (task_index % 3 == fold) & data["is_initial"]
    x_train = data["features"][train]
    y_train = data["classes"][train]
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    torch.manual_seed(args.task_seed + fold)
    model = ResourceCDFHead(x_train.shape[1], args.hidden_dim, args.energy_bins).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = np.random.default_rng(args.task_seed + fold)
    for _ in range(args.epochs):
        order = generator.permutation(x_train.shape[0])
        for start in range(0, order.size, args.batch_size):
            selected = order[start : start + args.batch_size]
            features = torch.as_tensor((x_train[selected] - mean) / scale, dtype=torch.float32, device=args.device)
            targets = torch.as_tensor(y_train[selected], dtype=torch.long, device=args.device)
            loss = nn.functional.cross_entropy(model(features), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    x_test = torch.as_tensor((data["features"][test] - mean) / scale, dtype=torch.float32, device=args.device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x_test), dim=1).cpu().numpy()
    checkpoint = args.output_dir / "models" / f"fold_{fold}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_scale": scale,
            "finite_bins": args.energy_bins,
            "support_capacity_multiple": args.support_capacity_multiple,
        },
        checkpoint,
    )
    train_initial = train & data["is_initial"]
    train_budget_label = (
        (data["classes"][train_initial] < args.energy_bins)
        & (data["energy_fraction"][train_initial] <= 1.0)
    ).astype(np.int64)
    test_budget_label = (
        (data["classes"][test] < args.energy_bins)
        & (data["energy_fraction"][test] <= 1.0)
    ).astype(np.int64)
    constant = float(np.mean(train_budget_label))
    logistic_weights, logistic_normalization = fit_logistic_baseline(
        data["geometry"][train_initial], train_budget_label
    )
    return {
        "fold": fold,
        "test_task_indices": sorted(set(int(value) for value in task_index[test])),
        "labels": test_budget_label.tolist(),
        "cdf_probabilities": cdf_probability(
            probabilities,
            budget_fraction=1.0,
            support_capacity_multiple=args.support_capacity_multiple,
        ).tolist(),
        "success_labels": (data["classes"][test] < args.energy_bins).astype(np.int64).tolist(),
        "success_probabilities": (1.0 - probabilities[:, -1]).tolist(),
        "constant_probabilities": np.full(int(test.sum()), constant).tolist(),
        "geometry_probabilities": predict_logistic(
            data["geometry"][test], logistic_weights, logistic_normalization
        ).tolist(),
        "checkpoint": str(checkpoint),
    }


def assess(folds: list[dict[str, object]]) -> dict[str, object]:
    def joined(name: str, dtype=float):
        return np.asarray([value for fold in folds for value in fold[name]], dtype=dtype)

    labels = joined("labels", int)
    cdf = joined("cdf_probabilities")
    constant = joined("constant_probabilities")
    geometry = joined("geometry_probabilities")
    success_labels = joined("success_labels", int)
    success = joined("success_probabilities")
    metrics = {
        "cdf_at_one_capacity": binary_metrics(labels, cdf),
        "explicit_finite_mass": binary_metrics(success_labels, success),
        "constant_budget_baseline": binary_metrics(labels, constant),
        "geometry_logistic_budget_baseline": binary_metrics(labels, geometry),
    }
    main = metrics["cdf_at_one_capacity"]
    baseline = min(
        metrics["constant_budget_baseline"]["brier"],
        metrics["geometry_logistic_budget_baseline"]["brier"],
    )
    checks = {
        "both_budget_classes_have_at_least_15_examples": bool(
            main["positive_count"] >= 15 and main["count"] - main["positive_count"] >= 15
        ),
        "cdf_auroc_at_least_0_70": bool(main["auroc"] is not None and main["auroc"] >= 0.70),
        "cdf_ece_at_most_0_20": bool(main["ece_10_bin"] <= 0.20),
        "cdf_brier_beats_best_simple_baseline_by_2_percent": bool(
            main["brier"] <= 0.98 * baseline
        ),
    }
    promotable = all(checks.values())
    return {
        "status": "PROMOTE_TO_ON_POLICY_CDF_PILOT" if promotable else "DO_NOT_PROMOTE_FROM_LEARNABILITY_PILOT",
        "promotable": promotable,
        "checks": checks,
        "metrics": metrics,
        "interpretation_limit": (
            "Exploratory held-out-task learnability only; this does not establish "
            "closed-loop policy improvement, calibrated lower confidence bounds, "
            "or lifecycle safety."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.artifact = args.artifact.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.upstream_dir = args.upstream_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    upstream = args.upstream_dir / "COMPLETED_DIAGNOSTIC.json"
    if args.wait_for_upstream and not upstream.is_file():
        if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
            raise FileExistsError(f"output is not fresh: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output_dir / "WAITING_UPSTREAM.json",
            {
                "status": "WAITING_UPSTREAM",
                "upstream": str(args.upstream_dir),
                "upstream_pid": args.upstream_pid,
                "started_unix": time.time(),
            },
        )
        while not upstream.is_file():
            terminal_failures = sorted(
                path.name
                for pattern in ("FAILED*.json", "STOPPED*.json")
                for path in args.upstream_dir.glob(pattern)
            )
            process_gone = bool(
                args.upstream_pid is not None
                and not Path(f"/proc/{args.upstream_pid}").exists()
            )
            if terminal_failures or process_gone:
                atomic_json(
                    args.output_dir / "BLOCKED_UPSTREAM.json",
                    {
                        "status": "BLOCKED_UPSTREAM",
                        "terminal_failure_files": terminal_failures,
                        "upstream_process_gone": process_gone,
                    },
                )
                (args.output_dir / "WAITING_UPSTREAM.json").unlink(missing_ok=True)
                return 4
            time.sleep(30.0)
        (args.output_dir / "WAITING_UPSTREAM.json").unlink(missing_ok=True)
    required = [args.artifact / "config.json", args.checkpoint, upstream]
    missing = [str(path) for path in required if not path.is_file()]
    if args.dry_run:
        print(json.dumps({"missing": missing, "would_run": not missing}, indent=2))
        return int(bool(missing))
    if missing:
        raise FileNotFoundError("missing prerequisites: " + ", ".join(missing))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    previous_failure = args.output_dir / "FAILED.json"
    if args.resume and previous_failure.is_file():
        failure_archive = (
            args.output_dir
            / "failure_history"
            / f"FAILED_before_resume_{int(time.time())}.json"
        )
        failure_archive.parent.mkdir(parents=True, exist_ok=True)
        previous_failure.replace(failure_archive)
    torch.set_num_threads(args.torch_threads)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        selected_device = torch.device(args.device)
        torch.cuda.set_device(0 if selected_device.index is None else selected_device.index)
    environment_args, _ = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.task_seed
    )
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.TD_PRETRAINING)
    capacity_payload = json.loads(
        (args.upstream_dir / "prerequisite_audit.json").read_text(encoding="utf-8")
    )
    capacity = float(capacity_payload["calibrated_battery_capacity"])
    policy = SAC.load(args.checkpoint, env=probe, device=args.device, print_system_info=False)
    probe.close()
    manifest = {
        "status": "RUNNING",
        "protocol": "extended_real_resource_cdf_learnability_pilot_v1",
        "formal_evidence": False,
        "claim": "held-out task-seed learnability of joint deadline/resource feasibility",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "upstream": str(upstream),
        "upstream_sha256": file_sha256(upstream),
        "num_tasks": args.num_tasks,
        "task_seed": args.task_seed,
        "world_seed": args.world_seed,
        "horizons": sorted(args.horizons),
        "max_policy_steps": args.max_policy_steps,
        "snapshot_interval": args.snapshot_interval,
        "finite_energy_bins": args.energy_bins,
        "failure_atom": True,
        "capacity": capacity,
        "device": args.device,
        "exact_command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    try:
        tasks = generate_stratified_navigation_tasks(
            num_tasks=args.num_tasks, seed=args.task_seed
        )
        completed = []
        for index, task in enumerate(tasks):
            task_file = args.output_dir / "tasks" / f"task_{index:03d}.npz"
            metadata_file = args.output_dir / "tasks" / f"task_{index:03d}.json"
            if args.resume and task_file.is_file() and metadata_file.is_file():
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            else:
                metadata = collect_task(args, task, index, policy, capacity)
            completed.append(metadata)
            atomic_json(
                args.output_dir / "PROGRESS.json",
                {
                    "status": "COLLECTING",
                    "completed_tasks": len(completed),
                    "total_tasks": args.num_tasks,
                    "last_task": metadata,
                },
            )
        data = load_dataset(args.output_dir, args.num_tasks)
        initial = data["is_initial"]
        finite = data["classes"] < args.energy_bins
        budget_labels = finite & (data["energy_fraction"] <= 1.0)
        label_audit = {
            "num_examples": int(data["classes"].size),
            "num_initial_task_horizon_examples": int(initial.sum()),
            "initial_finite_count": int(np.sum(initial & finite)),
            "initial_failure_atom_count": int(np.sum(initial & ~finite)),
            "initial_within_capacity_count": int(np.sum(initial & budget_labels)),
            "initial_over_capacity_or_failure_count": int(np.sum(initial & ~budget_labels)),
            "feature_dim": int(data["features"].shape[1]),
        }
        atomic_json(args.output_dir / "label_audit.json", label_audit)
        if min(
            label_audit["initial_within_capacity_count"],
            label_audit["initial_over_capacity_or_failure_count"],
        ) < 5:
            result = {
                **manifest,
                "status": "STOPPED_UNINFORMATIVE_LABELS",
                "label_audit": label_audit,
                "reason": "fewer than five examples in one capacity-feasibility class",
            }
            atomic_json(args.output_dir / "RESULT.json", result)
            atomic_json(args.output_dir / "COMPLETED.json", {"status": result["status"]})
            (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
            return 0
        folds = [train_fold(args, data, fold) for fold in range(3)]
        result = {
            **manifest,
            **assess(folds),
            "label_audit": label_audit,
            "folds": folds,
            "finished_unix": time.time(),
        }
        atomic_json(args.output_dir / "RESULT.json", result)
        atomic_json(args.output_dir / "COMPLETED.json", {"status": result["status"]})
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        atomic_json(
            args.output_dir / "FAILED.json",
            {"status": "FAILED", "error_type": type(error).__name__, "error": str(error)},
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
