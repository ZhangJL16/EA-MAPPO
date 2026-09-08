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
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.explore_resource_cdf_learnability import (
    ResourceCDFHead,
    atomic_json,
    binary_metrics,
    cdf_probability,
    collect_task,
    file_sha256,
    fit_logistic_baseline,
    load_dataset,
    predict_logistic,
)
from scripts.train_uav_energy_delivery_sac import (
    environment_from_args,
    generate_stratified_navigation_tasks,
)


DEFAULT_BUDGETS = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36)
DEFAULT_EPOCHS = (25, 100, 400)
DEFAULT_REPRESENTATIONS = ("raw", "frozen_encoder", "compact")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Boundary-balanced scaling study for the R3 resource CDF head"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-tasks", type=int, default=300)
    parser.add_argument("--task-seed", type=int, default=240_001)
    parser.add_argument("--world-seed", type=int, default=250_001)
    parser.add_argument("--max-policy-steps", type=int, default=4000)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1000, 2000, 3000, 4000])
    parser.add_argument("--snapshot-interval", type=int, default=200)
    parser.add_argument("--energy-bins", type=int, default=64)
    parser.add_argument("--support-capacity-multiple", type=float, default=1.5)
    parser.add_argument("--budget-fractions", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--epoch-checkpoints", type=int, nargs="+", default=list(DEFAULT_EPOCHS))
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=list(DEFAULT_REPRESENTATIONS),
        default=list(DEFAULT_REPRESENTATIONS),
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_tasks = 5
        args.max_policy_steps = 20
        args.horizons = [5, 10, 20]
        args.snapshot_interval = 10
        args.budget_fractions = [0.08, 0.20]
        args.epoch_checkpoints = [1, 2]
        args.representations = ["compact"]
        args.batch_size = 16
    if args.num_tasks <= 0 or args.num_tasks % 5:
        parser.error("--num-tasks must be a positive multiple of five")
    if any(value <= 0 or value > args.max_policy_steps for value in args.horizons):
        parser.error("horizons must lie in (0, max-policy-steps]")
    if len(set(args.horizons)) != len(args.horizons):
        parser.error("horizons must be unique")
    if any(not 0.0 < value <= args.support_capacity_multiple for value in args.budget_fractions):
        parser.error("budget fractions must lie inside the finite CDF support")
    if len(set(args.budget_fractions)) != len(args.budget_fractions):
        parser.error("budget fractions must be unique")
    if any(value <= 0 for value in args.epoch_checkpoints):
        parser.error("epoch checkpoints must be positive")
    if args.epoch_checkpoints != sorted(set(args.epoch_checkpoints)):
        parser.error("epoch checkpoints must be unique and increasing")
    if len(set(args.representations)) != len(args.representations):
        parser.error("representations must be unique")
    return args


def representation_arrays(
    raw_features: np.ndarray,
    *,
    observation_dim: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(raw_features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != observation_dim + 11:
        raise ValueError("raw feature layout does not match observation_dim + 11")
    task_observation = values[:, :observation_dim]
    charger_compact = values[:, observation_dim : observation_dim + 7]
    position_and_horizon = values[:, -4:]
    compact = np.concatenate(
        (task_observation[:, :7], charger_compact, position_and_horizon), axis=1
    )
    return {
        "raw": values,
        "task_observation": task_observation,
        "charger_compact": charger_compact,
        "position_and_horizon": position_and_horizon,
        "compact": compact.astype(np.float32),
    }


def frozen_encoder_representation(
    policy: SAC,
    arrays: dict[str, np.ndarray],
    *,
    d_max: float,
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    task_observation = arrays["task_observation"]
    charger_compact = arrays["charger_compact"].copy()
    charger_compact[:, 6] = np.log1p(charger_compact[:, 6] * d_max) / np.log1p(d_max)
    charger_observation = np.concatenate(
        (charger_compact, task_observation[:, 7:]), axis=1
    ).astype(np.float32)
    actor = policy.policy.actor
    actor.eval()

    def encode(observations: np.ndarray) -> np.ndarray:
        encoded = []
        with torch.no_grad():
            for start in range(0, observations.shape[0], batch_size):
                tensor = torch.as_tensor(
                    observations[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                features = actor.extract_features(tensor, actor.features_extractor)
                encoded.append(features.cpu().numpy())
        return np.concatenate(encoded, axis=0).astype(np.float32)

    return np.concatenate(
        (
            encode(task_observation),
            encode(charger_observation),
            arrays["position_and_horizon"],
        ),
        axis=1,
    ).astype(np.float32)


def expand_budget_queries(
    finite: np.ndarray,
    energy_fraction: np.ndarray,
    budgets: np.ndarray,
) -> np.ndarray:
    finite_values = np.asarray(finite, dtype=np.bool_)
    energy = np.asarray(energy_fraction, dtype=np.float64)
    budget_values = np.asarray(budgets, dtype=np.float64)
    return (
        finite_values[:, None]
        & np.isfinite(energy[:, None])
        & (energy[:, None] <= budget_values[None, :])
    ).astype(np.int64)


def query_probabilities(
    atom_probabilities: np.ndarray,
    budgets: np.ndarray,
    *,
    support_capacity_multiple: float,
) -> np.ndarray:
    columns = [
        cdf_probability(
            atom_probabilities,
            budget_fraction=float(budget),
            support_capacity_multiple=support_capacity_multiple,
        )
        for budget in budgets
    ]
    return np.column_stack(columns)


def archive_failure(output: Path) -> None:
    failure = output / "FAILED.json"
    if failure.is_file():
        destination = output / "failure_history" / f"FAILED_before_resume_{int(time.time())}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        failure.replace(destination)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def task_weights(task_indices: np.ndarray) -> np.ndarray:
    task = np.asarray(task_indices, dtype=np.int64)
    unique, counts = np.unique(task, return_counts=True)
    lookup = {int(key): int(value) for key, value in zip(unique, counts)}
    weights = np.asarray([1.0 / lookup[int(index)] for index in task], dtype=np.float32)
    return weights / weights.mean()


def fold_query_data(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    budgets: np.ndarray,
) -> dict[str, np.ndarray]:
    finite = data["classes"][mask] < int(data["finite_bins"][0])
    labels = expand_budget_queries(finite, data["energy_fraction"][mask], budgets)
    geometry = np.repeat(data["geometry"][mask], budgets.size, axis=0)
    geometry = np.column_stack((geometry, np.tile(budgets, int(mask.sum()))))
    return {
        "labels": labels.reshape(-1),
        "geometry": geometry.astype(np.float64),
        "task_index": np.repeat(data["task_index"][mask], budgets.size),
        "horizon": np.repeat(data["horizon"][mask], budgets.size),
        "budget": np.tile(budgets, int(mask.sum())),
    }


def train_fold(
    args: argparse.Namespace,
    representation_name: str,
    features: np.ndarray,
    data: dict[str, np.ndarray],
    fold: int,
    budgets: np.ndarray,
) -> dict[str, object]:
    train_mask = data["task_index"] % 3 != fold
    test_mask = (data["task_index"] % 3 == fold) & data["is_initial"]
    x_train = features[train_mask]
    y_train = data["classes"][train_mask]
    weights = task_weights(data["task_index"][train_mask])
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-6] = 1.0
    torch.manual_seed(args.task_seed + 100 * fold + len(representation_name))
    model = ResourceCDFHead(features.shape[1], args.hidden_dim, args.energy_bins).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = np.random.default_rng(args.task_seed + 100 * fold + len(representation_name))
    checkpoints = set(args.epoch_checkpoints)
    epoch_results: dict[str, object] = {}
    test_tensor = torch.as_tensor(
        (features[test_mask] - mean) / scale,
        dtype=torch.float32,
        device=args.device,
    )
    for epoch in range(1, max(args.epoch_checkpoints) + 1):
        order = generator.permutation(x_train.shape[0])
        model.train()
        for start in range(0, order.size, args.batch_size):
            selected = order[start : start + args.batch_size]
            batch_x = torch.as_tensor(
                (x_train[selected] - mean) / scale,
                dtype=torch.float32,
                device=args.device,
            )
            batch_y = torch.as_tensor(y_train[selected], dtype=torch.long, device=args.device)
            batch_w = torch.as_tensor(weights[selected], dtype=torch.float32, device=args.device)
            losses = nn.functional.cross_entropy(model(batch_x), batch_y, reduction="none")
            loss = torch.sum(losses * batch_w) / torch.sum(batch_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch not in checkpoints:
            continue
        model.eval()
        with torch.no_grad():
            atom_probabilities = torch.softmax(model(test_tensor), dim=1).cpu().numpy()
        query = fold_query_data(data, test_mask, budgets)
        query["probability"] = query_probabilities(
            atom_probabilities,
            budgets,
            support_capacity_multiple=args.support_capacity_multiple,
        ).reshape(-1)
        model_path = args.output_dir / "models" / representation_name / f"fold_{fold}_epoch_{epoch}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_mean": mean,
                "feature_scale": scale,
                "representation": representation_name,
                "epoch": epoch,
                "finite_bins": args.energy_bins,
                "support_capacity_multiple": args.support_capacity_multiple,
            },
            model_path,
        )
        epoch_results[str(epoch)] = {
            "checkpoint": str(model_path),
            **{name: value.tolist() for name, value in query.items()},
        }
    train_initial = train_mask & data["is_initial"]
    train_query = fold_query_data(data, train_initial, budgets)
    test_query = fold_query_data(data, test_mask, budgets)
    logistic_weights, logistic_normalization = fit_logistic_baseline(
        train_query["geometry"], train_query["labels"]
    )
    prevalence = float(np.mean(train_query["labels"]))
    baseline = {
        "labels": test_query["labels"].tolist(),
        "task_index": test_query["task_index"].tolist(),
        "horizon": test_query["horizon"].tolist(),
        "budget": test_query["budget"].tolist(),
        "constant_probability": np.full(test_query["labels"].size, prevalence).tolist(),
        "geometry_probability": predict_logistic(
            test_query["geometry"], logistic_weights, logistic_normalization
        ).tolist(),
    }
    return {
        "representation": representation_name,
        "fold": fold,
        "test_tasks": sorted(set(int(value) for value in data["task_index"][test_mask])),
        "epochs": epoch_results,
        "baseline": baseline,
    }


def grouped_metrics(labels: np.ndarray, probabilities: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    result = {}
    for value in sorted(set(groups.tolist()), key=str):
        mask = groups == value
        result[str(value)] = binary_metrics(labels[mask], probabilities[mask])
    return result


def aggregate_results(
    args: argparse.Namespace,
    fold_results: list[dict[str, object]],
    task_buckets: dict[int, str],
) -> tuple[dict[str, object], dict[str, object]]:
    matrix: dict[str, object] = {}
    baseline_folds = [row for row in fold_results if row["representation"] == args.representations[0]]
    labels = np.asarray([value for row in baseline_folds for value in row["baseline"]["labels"]], dtype=np.int64)
    tasks = np.asarray([value for row in baseline_folds for value in row["baseline"]["task_index"]], dtype=np.int64)
    horizons = np.asarray([value for row in baseline_folds for value in row["baseline"]["horizon"]], dtype=np.int64)
    budgets = np.asarray([value for row in baseline_folds for value in row["baseline"]["budget"]], dtype=np.float64)
    constant = np.asarray([value for row in baseline_folds for value in row["baseline"]["constant_probability"]], dtype=np.float64)
    geometry = np.asarray([value for row in baseline_folds for value in row["baseline"]["geometry_probability"]], dtype=np.float64)
    bucket = np.asarray([task_buckets[int(index)] for index in tasks])
    baseline = {
        "constant": binary_metrics(labels, constant),
        "geometry": binary_metrics(labels, geometry),
        "geometry_by_horizon": grouped_metrics(labels, geometry, horizons),
        "geometry_by_budget": grouped_metrics(labels, geometry, budgets),
        "geometry_by_distance_bucket": grouped_metrics(labels, geometry, bucket),
    }
    for representation in args.representations:
        selected = [row for row in fold_results if row["representation"] == representation]
        for epoch in args.epoch_checkpoints:
            probabilities = np.asarray(
                [value for row in selected for value in row["epochs"][str(epoch)]["probability"]],
                dtype=np.float64,
            )
            key = f"{representation}@{epoch}"
            matrix[key] = {
                "overall": binary_metrics(labels, probabilities),
                "by_horizon": grouped_metrics(labels, probabilities, horizons),
                "by_budget": grouped_metrics(labels, probabilities, budgets),
                "by_distance_bucket": grouped_metrics(labels, probabilities, bucket),
            }
            save_npz_atomic(
                args.output_dir / "predictions" / f"{representation}_epoch_{epoch}.npz",
                label=labels,
                probability=probabilities,
                task_index=tasks,
                horizon=horizons,
                budget=budgets,
                distance_bucket=bucket,
            )
    return matrix, baseline


def decision(
    matrix: dict[str, object],
    baseline: dict[str, object],
    *,
    primary_key: str = "frozen_encoder@400",
) -> dict[str, object]:
    if primary_key not in matrix:
        raise KeyError(f"primary configuration is absent: {primary_key}")
    primary = matrix[primary_key]["overall"]
    geometry = baseline["geometry"]
    checks = {
        "both_classes_have_at_least_200_queries": bool(
            primary["positive_count"] >= 200
            and primary["count"] - primary["positive_count"] >= 200
        ),
        "primary_auroc_at_least_0_75": bool(
            primary["auroc"] is not None and primary["auroc"] >= 0.75
        ),
        "primary_ece_at_most_0_15": bool(primary["ece_10_bin"] <= 0.15),
        "primary_brier_beats_geometry_by_2_percent": bool(
            primary["brier"] <= 0.98 * geometry["brier"]
        ),
    }
    early_key = "frozen_encoder@25"
    brier_25 = (
        matrix[early_key]["overall"]["brier"]
        if early_key in matrix
        else primary["brier"]
    )
    brier_400 = primary["brier"]
    return {
        "status": "PROMOTE_TO_ON_POLICY_CDF_PILOT" if all(checks.values()) else "DO_NOT_PROMOTE_FROM_SCALING_EXPERIMENT",
        "promotable": all(checks.values()),
        "primary_configuration": primary_key,
        "checks": checks,
        "optimization_brier_relative_improvement_25_to_400": float((brier_25 - brier_400) / brier_25),
        "interpretation": (
            "Exploratory boundary-balanced held-out-task evidence only; repeated "
            "budget/horizon queries are not independent experimental seeds."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("artifact", "checkpoint", "upstream_result", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = [args.artifact / "config.json", args.checkpoint, args.upstream_result]
    missing = [str(path) for path in required if not path.is_file()]
    if args.dry_run:
        print(json.dumps({"missing": missing, "would_run": not missing}, indent=2))
        return int(bool(missing))
    if missing:
        raise FileNotFoundError("missing prerequisites: " + ", ".join(missing))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        archive_failure(args.output_dir)
    selected_device = torch.device(args.device)
    if torch.cuda.is_available() and selected_device.type == "cuda":
        torch.cuda.set_device(0 if selected_device.index is None else selected_device.index)
    torch.set_num_threads(args.torch_threads)
    environment_args, _ = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.task_seed
    )
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.TD_PRETRAINING)
    observation_dim = int(probe.observation_space.shape[0])
    d_max = float(probe.d_max)
    policy = SAC.load(args.checkpoint, env=probe, device=args.device, print_system_info=False)
    probe.close()
    upstream = json.loads(args.upstream_result.read_text(encoding="utf-8"))
    capacity = float(upstream["capacity"])
    manifest = {
        "status": "RUNNING",
        "protocol": "resource_cdf_boundary_balanced_scaling_v1",
        "formal_evidence": False,
        "artifact": str(args.artifact),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "upstream_result": str(args.upstream_result),
        "upstream_result_sha256": file_sha256(args.upstream_result),
        "num_tasks": args.num_tasks,
        "task_seed": args.task_seed,
        "world_seed": args.world_seed,
        "horizons": sorted(args.horizons),
        "budget_fractions": sorted(args.budget_fractions),
        "representations": args.representations,
        "epoch_checkpoints": args.epoch_checkpoints,
        "energy_bins": args.energy_bins,
        "support_capacity_multiple": args.support_capacity_multiple,
        "capacity": capacity,
        "observation_dim": observation_dim,
        "device": args.device,
        "exact_command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    try:
        tasks = generate_stratified_navigation_tasks(num_tasks=args.num_tasks, seed=args.task_seed)
        task_buckets = {index: task.distance_bucket for index, task in enumerate(tasks)}
        for index, task in enumerate(tasks):
            task_npz = args.output_dir / "tasks" / f"task_{index:03d}.npz"
            task_json = args.output_dir / "tasks" / f"task_{index:03d}.json"
            if args.resume and task_npz.is_file() and task_json.is_file():
                metadata = json.loads(task_json.read_text(encoding="utf-8"))
            else:
                metadata = collect_task(args, task, index, policy, capacity)
            atomic_json(
                args.output_dir / "PROGRESS.json",
                {
                    "stage": "collection",
                    "completed_tasks": index + 1,
                    "total_tasks": args.num_tasks,
                    "last_task": metadata,
                },
            )
        data = load_dataset(args.output_dir, args.num_tasks)
        data["finite_bins"] = np.full(data["classes"].size, args.energy_bins, dtype=np.int64)
        arrays = representation_arrays(data["features"], observation_dim=observation_dim)
        representation_data = {
            "raw": arrays["raw"],
            "compact": arrays["compact"],
        }
        if "frozen_encoder" in args.representations:
            cache = args.output_dir / "representations" / "frozen_encoder.npz"
            if args.resume and cache.is_file():
                with np.load(cache) as stored:
                    frozen = stored["features"]
            else:
                frozen = frozen_encoder_representation(
                    policy, arrays, d_max=d_max, device=args.device
                )
                save_npz_atomic(cache, features=frozen)
            representation_data["frozen_encoder"] = frozen
        initial = data["is_initial"]
        budgets = np.asarray(sorted(args.budget_fractions), dtype=np.float64)
        finite = data["classes"][initial] < args.energy_bins
        initial_labels = expand_budget_queries(
            finite, data["energy_fraction"][initial], budgets
        ).reshape(-1)
        finite_energy_values = data["energy_fraction"][
            np.isfinite(data["energy_fraction"])
        ]
        label_audit = {
            "num_tasks": args.num_tasks,
            "num_training_examples": int(data["classes"].size),
            "num_initial_task_horizon_states": int(initial.sum()),
            "num_budget_queries": int(initial_labels.size),
            "positive_budget_queries": int(initial_labels.sum()),
            "negative_budget_queries": int(initial_labels.size - initial_labels.sum()),
            "finite_initial_states": int(finite.sum()),
            "failure_atom_initial_states": int((~finite).sum()),
            "maximum_finite_energy_fraction": (
                None
                if finite_energy_values.size == 0
                else float(np.max(finite_energy_values))
            ),
            "representation_dimensions": {
                name: int(representation_data[name].shape[1])
                for name in args.representations
            },
        }
        atomic_json(args.output_dir / "label_audit.json", label_audit)
        fold_results = []
        for representation in args.representations:
            for fold in range(3):
                result_path = args.output_dir / "fold_results" / f"{representation}_fold_{fold}.json"
                if args.resume and result_path.is_file():
                    fold_result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    fold_result = train_fold(
                        args,
                        representation,
                        representation_data[representation],
                        data,
                        fold,
                        budgets,
                    )
                    atomic_json(result_path, fold_result)
                fold_results.append(fold_result)
                atomic_json(
                    args.output_dir / "PROGRESS.json",
                    {
                        "stage": "training",
                        "completed_representation_folds": len(fold_results),
                        "total_representation_folds": len(args.representations) * 3,
                        "last_representation": representation,
                        "last_fold": fold,
                    },
                )
        matrix, baselines = aggregate_results(args, fold_results, task_buckets)
        primary_key = (
            "frozen_encoder@400"
            if not args.smoke
            else f"{args.representations[0]}@{max(args.epoch_checkpoints)}"
        )
        result = {
            **manifest,
            **decision(matrix, baselines, primary_key=primary_key),
            "label_audit": label_audit,
            "baselines": baselines,
            "experiment_matrix": matrix,
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
