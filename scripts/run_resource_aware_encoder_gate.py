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
import copy
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
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.explore_resource_cdf_learnability import (
    ResourceCDFHead,
    atomic_json,
    binary_metrics,
    file_sha256,
    fit_logistic_baseline,
    load_dataset,
    predict_logistic,
)
from scripts.run_resource_cdf_scaling_experiment import (
    archive_failure,
    expand_budget_queries,
    grouped_metrics,
    representation_arrays,
    save_npz_atomic,
    task_weights,
)
from scripts.train_uav_energy_delivery_sac import environment_from_args


DEFAULT_BUDGETS = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36)
DEFAULT_VARIANTS = ("frozen_ce", "trainable_ce", "trainable_calibrated")
PRIMARY_VARIANT = "trainable_calibrated"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task-disjoint resource-aware R3 encoder learnability Gate"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=260_001)
    parser.add_argument("--model-seed", type=int, default=270_001)
    parser.add_argument("--uniform-energy-bins", type=int, default=64)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--minimum-epochs", type=int, default=25)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--patience-checks", type=int, default=30)
    parser.add_argument("--validation-tasks-per-bucket", type=int, default=6)
    parser.add_argument("--calibration-tasks-per-bucket", type=int, default=6)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--calibration-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(DEFAULT_VARIANTS),
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.max_epochs = 2
        args.minimum_epochs = 1
        args.validation_every = 1
        args.patience_checks = 2
        args.validation_tasks_per_bucket = 2
        args.calibration_tasks_per_bucket = 2
        args.batch_size = 64
        args.variants = [PRIMARY_VARIANT]
    if args.uniform_energy_bins < 8:
        parser.error("--uniform-energy-bins must be at least 8")
    if not args.budgets or any(value <= 0.0 for value in args.budgets):
        parser.error("--budgets must contain positive fractions")
    if args.budgets != sorted(set(args.budgets)):
        parser.error("--budgets must be unique and increasing")
    if args.max_epochs <= 0 or not 1 <= args.minimum_epochs <= args.max_epochs:
        parser.error("epoch limits are invalid")
    if args.validation_every <= 0 or args.patience_checks <= 0:
        parser.error("validation cadence and patience must be positive")
    if args.validation_tasks_per_bucket <= 0 or args.calibration_tasks_per_bucket <= 0:
        parser.error("nested split sizes must be positive")
    if args.head_learning_rate <= 0.0 or args.encoder_learning_rate <= 0.0:
        parser.error("learning rates must be positive")
    if args.calibration_weight < 0.0 or args.weight_decay < 0.0:
        parser.error("loss and regularization weights must be non-negative")
    if args.batch_size <= 0 or args.hidden_dim <= 0 or args.torch_threads <= 0:
        parser.error("batch, hidden, and thread sizes must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    if not args.smoke and PRIMARY_VARIANT not in args.variants:
        parser.error(f"formal exploratory run requires {PRIMARY_VARIANT}")
    return args


def stable_task_rank(task_index: int, *, fold: int, bucket: str, seed: int) -> bytes:
    payload = f"{seed}:{fold}:{bucket}:{task_index}".encode("utf-8")
    return hashlib.sha256(payload).digest()


def nested_task_split(
    task_indices: np.ndarray,
    task_buckets: dict[int, str],
    *,
    fold: int,
    split_seed: int,
    validation_per_bucket: int,
    calibration_per_bucket: int,
) -> dict[str, list[int]]:
    unique_tasks = sorted(set(int(value) for value in np.asarray(task_indices).tolist()))
    test = {index for index in unique_tasks if index % 3 == fold}
    development = set(unique_tasks) - test
    validation: set[int] = set()
    calibration: set[int] = set()
    for bucket in sorted(set(task_buckets[index] for index in unique_tasks)):
        candidates = [index for index in development if task_buckets[index] == bucket]
        candidates.sort(
            key=lambda index: stable_task_rank(
                index, fold=fold, bucket=bucket, seed=split_seed
            )
        )
        required = validation_per_bucket + calibration_per_bucket + 1
        if len(candidates) < required:
            raise ValueError(
                f"bucket {bucket!r} has {len(candidates)} development tasks; "
                f"at least {required} are required"
            )
        validation.update(candidates[:validation_per_bucket])
        calibration.update(
            candidates[
                validation_per_bucket : validation_per_bucket + calibration_per_bucket
            ]
        )
    selection_train = development - validation - calibration
    parts = {
        "selection_train": sorted(selection_train),
        "validation": sorted(validation),
        "calibration": sorted(calibration),
        "test": sorted(test),
    }
    joined = [index for values in parts.values() for index in values]
    if len(joined) != len(set(joined)) or set(joined) != set(unique_tasks):
        raise AssertionError("nested task split is not a disjoint partition")
    return parts


def threshold_aligned_edges(
    *,
    support: float,
    uniform_bins: int,
    budgets: np.ndarray,
) -> np.ndarray:
    if support <= 0.0 or uniform_bins <= 0:
        raise ValueError("finite support and uniform bin count must be positive")
    query = np.asarray(budgets, dtype=np.float64)
    if query.ndim != 1 or query.size == 0 or np.any(query <= 0.0):
        raise ValueError("budgets must be a nonempty positive vector")
    if float(np.max(query)) > support:
        raise ValueError("budget lies beyond finite resource support")
    uniform = np.linspace(support / uniform_bins, support, uniform_bins)
    edges = np.unique(np.round(np.concatenate((uniform, query)), decimals=12))
    return edges.astype(np.float64)


def energy_targets(energy_fraction: np.ndarray, edges: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_fraction, dtype=np.float64)
    finite = np.isfinite(energy)
    if np.any(energy[finite] < 0.0):
        raise ValueError("finite energy targets must be non-negative")
    if np.any(energy[finite] > float(edges[-1]) + 1e-7):
        raise ValueError("finite energy target exceeds the registered support")
    target = np.full(energy.shape, edges.size, dtype=np.int64)
    target[finite] = np.minimum(
        np.searchsorted(edges, energy[finite], side="left"), edges.size - 1
    )
    return target


def cdf_query_mask(edges: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    return (
        np.asarray(edges, dtype=np.float64)[:, None]
        <= np.asarray(budgets, dtype=np.float64)[None, :] + 1e-12
    ).astype(np.float32)


def cdf_from_logits(
    logits: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if logits.ndim != 2 or query_mask.ndim != 2:
        raise ValueError("logits and query mask must be matrices")
    if logits.shape[1] != query_mask.shape[0] + 1:
        raise ValueError("the final logit must be the explicit failure atom")
    finite_probability = torch.softmax(logits / float(temperature), dim=1)[:, :-1]
    return finite_probability @ query_mask


def calibration_aware_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    budget_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    query_mask: torch.Tensor,
    *,
    calibration_weight: float,
    normalize_nll: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = sample_weights / torch.sum(sample_weights)
    nll_rows = nn.functional.cross_entropy(logits, targets, reduction="none")
    nll = torch.sum(weights * nll_rows)
    probabilities = cdf_from_logits(logits, query_mask)
    brier_rows = torch.mean((probabilities - budget_labels) ** 2, dim=1)
    brier = torch.sum(weights * brier_rows)
    nll_term = nll / math.log(logits.shape[1]) if normalize_nll else nll
    loss = nll_term + float(calibration_weight) * brier
    return loss, {
        "loss": float(loss.detach().cpu()),
        "nll": float(nll.detach().cpu()),
        "brier": float(brier.detach().cpu()),
    }


class ResourceAwareCDFModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        context_dim: int,
        hidden_dim: int,
        finite_atoms: int,
        encoder_trainable: bool,
    ) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.encoder_trainable = bool(encoder_trainable)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(self.encoder_trainable)
        feature_dim = int(getattr(self.encoder, "features_dim"))
        self.head = ResourceCDFHead(
            2 * feature_dim + int(context_dim), hidden_dim, finite_atoms
        )

    def forward(
        self,
        task_observation: torch.Tensor,
        charger_observation: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        task_features = self.encoder(task_observation)
        charger_features = self.encoder(charger_observation)
        return self.head(torch.cat((task_features, charger_features, context), dim=1))

    def train(self, mode: bool = True) -> ResourceAwareCDFModel:
        super().train(mode)
        if not self.encoder_trainable:
            self.encoder.eval()
        return self


def observation_views(
    raw_features: np.ndarray,
    *,
    observation_dim: int,
    d_max: float,
) -> dict[str, np.ndarray]:
    arrays = representation_arrays(raw_features, observation_dim=observation_dim)
    charger_compact = arrays["charger_compact"].copy()
    charger_compact[:, 6] = np.log1p(charger_compact[:, 6] * d_max) / np.log1p(d_max)
    charger_observation = np.concatenate(
        (charger_compact, arrays["task_observation"][:, 7:]), axis=1
    )
    return {
        "task_observation": arrays["task_observation"].astype(np.float32),
        "charger_observation": charger_observation.astype(np.float32),
        "context": arrays["position_and_horizon"].astype(np.float32),
    }


def mask_for_tasks(task_index: np.ndarray, tasks: list[int]) -> np.ndarray:
    return np.isin(np.asarray(task_index, dtype=np.int64), np.asarray(tasks, dtype=np.int64))


def task_averaged_brier(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task_index: np.ndarray,
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    task = np.asarray(task_index, dtype=np.int64)
    if y.shape != p.shape or y.ndim != 2 or task.shape != (y.shape[0],):
        raise ValueError("task Brier inputs are not aligned")
    per_row = np.mean((p - y) ** 2, axis=1)
    values = [float(np.mean(per_row[task == value])) for value in np.unique(task)]
    return float(np.mean(values))


def select_best_epoch(
    history: list[dict[str, float | int]],
) -> int:
    if not history:
        raise ValueError("validation history is empty")
    return int(min(history, key=lambda row: (float(row["validation_brier"]), int(row["epoch"])))["epoch"])


def variant_spec(name: str, calibration_weight: float) -> dict[str, object]:
    if name == "frozen_ce":
        return {"encoder_trainable": False, "calibration_weight": 0.0, "temperature": False}
    if name == "trainable_ce":
        return {"encoder_trainable": True, "calibration_weight": 0.0, "temperature": False}
    if name == PRIMARY_VARIANT:
        return {
            "encoder_trainable": True,
            "calibration_weight": float(calibration_weight),
            "temperature": True,
        }
    raise KeyError(name)


def build_model_and_optimizer(
    base_encoder: nn.Module,
    args: argparse.Namespace,
    spec: dict[str, object],
    *,
    finite_atoms: int,
    seed: int,
) -> tuple[ResourceAwareCDFModel, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = ResourceAwareCDFModel(
        base_encoder,
        context_dim=4,
        hidden_dim=args.hidden_dim,
        finite_atoms=finite_atoms,
        encoder_trainable=bool(spec["encoder_trainable"]),
    ).to(args.device)
    groups: list[dict[str, object]] = [
        {"params": list(model.head.parameters()), "lr": args.head_learning_rate}
    ]
    if bool(spec["encoder_trainable"]):
        groups.append(
            {"params": list(model.encoder.parameters()), "lr": args.encoder_learning_rate}
        )
    optimizer = torch.optim.AdamW(
        groups, weight_decay=args.weight_decay
    )
    return model, optimizer


def run_epoch(
    model: ResourceAwareCDFModel,
    optimizer: torch.optim.Optimizer,
    indices: np.ndarray,
    views: dict[str, np.ndarray],
    targets: np.ndarray,
    budget_labels: np.ndarray,
    weights: np.ndarray,
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    query_mask: torch.Tensor,
    args: argparse.Namespace,
    spec: dict[str, object],
    generator: np.random.Generator,
) -> dict[str, float]:
    order = generator.permutation(indices)
    totals = {"loss": 0.0, "nll": 0.0, "brier": 0.0, "count": 0.0}
    model.train()
    for start in range(0, order.size, args.batch_size):
        selected = order[start : start + args.batch_size]
        task = torch.as_tensor(
            views["task_observation"][selected], dtype=torch.float32, device=args.device
        )
        charger = torch.as_tensor(
            views["charger_observation"][selected], dtype=torch.float32, device=args.device
        )
        context = torch.as_tensor(
            (views["context"][selected] - context_mean) / context_scale,
            dtype=torch.float32,
            device=args.device,
        )
        target = torch.as_tensor(targets[selected], dtype=torch.long, device=args.device)
        labels = torch.as_tensor(
            budget_labels[selected], dtype=torch.float32, device=args.device
        )
        sample_weight = torch.as_tensor(
            weights[selected], dtype=torch.float32, device=args.device
        )
        logits = model(task, charger, context)
        loss, pieces = calibration_aware_loss(
            logits,
            target,
            labels,
            sample_weight,
            query_mask,
            calibration_weight=float(spec["calibration_weight"]),
            normalize_nll=float(spec["calibration_weight"]) > 0.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        count = float(selected.size)
        for key in ("loss", "nll", "brier"):
            totals[key] += pieces[key] * count
        totals["count"] += count
    return {key: totals[key] / totals["count"] for key in ("loss", "nll", "brier")}


def predict_logits(
    model: ResourceAwareCDFModel,
    indices: np.ndarray,
    views: dict[str, np.ndarray],
    context_mean: np.ndarray,
    context_scale: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            selected = indices[start : start + batch_size]
            task = torch.as_tensor(
                views["task_observation"][selected], dtype=torch.float32, device=device
            )
            charger = torch.as_tensor(
                views["charger_observation"][selected], dtype=torch.float32, device=device
            )
            context = torch.as_tensor(
                (views["context"][selected] - context_mean) / context_scale,
                dtype=torch.float32,
                device=device,
            )
            output.append(model(task, charger, context).cpu().numpy())
    return np.concatenate(output, axis=0)


def numpy_cdf_from_logits(
    logits: np.ndarray,
    query_mask: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    scaled -= np.max(scaled, axis=1, keepdims=True)
    atom = np.exp(scaled)
    atom /= np.sum(atom, axis=1, keepdims=True)
    return atom[:, :-1] @ np.asarray(query_mask, dtype=np.float64)


def fit_temperature(
    logits: np.ndarray,
    query_mask: np.ndarray,
    labels: np.ndarray,
    task_index: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    candidates = np.unique(
        np.concatenate((np.asarray([1.0]), np.geomspace(0.35, 5.0, 121)))
    )
    curve = []
    for temperature in candidates:
        probabilities = numpy_cdf_from_logits(
            logits, query_mask, temperature=float(temperature)
        )
        score = task_averaged_brier(labels, probabilities, task_index)
        curve.append({"temperature": float(temperature), "brier": score})
    best = min(curve, key=lambda row: (row["brier"], abs(math.log(row["temperature"]))))
    return float(best["temperature"]), curve


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def initial_indices(data: dict[str, np.ndarray], tasks: list[int]) -> np.ndarray:
    mask = mask_for_tasks(data["task_index"], tasks) & data["is_initial"]
    return np.flatnonzero(mask)


def all_indices(data: dict[str, np.ndarray], tasks: list[int]) -> np.ndarray:
    return np.flatnonzero(mask_for_tasks(data["task_index"], tasks))


def query_rows(
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    budget_labels: np.ndarray,
    probabilities: np.ndarray,
    budgets: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "labels": budget_labels[indices].reshape(-1),
        "probability": probabilities.reshape(-1),
        "task_index": np.repeat(data["task_index"][indices], budgets.size),
        "horizon": np.repeat(data["horizon"][indices], budgets.size),
        "budget": np.tile(budgets, indices.size),
    }


def train_variant_fold(
    args: argparse.Namespace,
    variant: str,
    fold: int,
    split: dict[str, list[int]],
    data: dict[str, np.ndarray],
    views: dict[str, np.ndarray],
    targets: np.ndarray,
    budget_labels: np.ndarray,
    edges: np.ndarray,
    budgets: np.ndarray,
    base_encoder: nn.Module,
) -> dict[str, object]:
    spec = variant_spec(variant, args.calibration_weight)
    query_mask_numpy = cdf_query_mask(edges, budgets)
    query_mask = torch.as_tensor(
        query_mask_numpy, dtype=torch.float32, device=args.device
    )
    selection_indices = all_indices(data, split["selection_train"])
    validation_indices = initial_indices(data, split["validation"])
    selection_weights = np.ones(data["task_index"].shape, dtype=np.float32)
    selection_weights[selection_indices] = task_weights(
        data["task_index"][selection_indices]
    )
    context_mean = views["context"][selection_indices].mean(axis=0)
    context_scale = views["context"][selection_indices].std(axis=0)
    context_scale[context_scale < 1e-6] = 1.0
    seed = args.model_seed + 1000 * fold + 37 * list(DEFAULT_VARIANTS).index(variant)
    model, optimizer = build_model_and_optimizer(
        base_encoder, args, spec, finite_atoms=edges.size, seed=seed
    )
    generator = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best_brier = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_checks = 0
    for epoch in range(1, args.max_epochs + 1):
        training = run_epoch(
            model,
            optimizer,
            selection_indices,
            views,
            targets,
            budget_labels,
            selection_weights,
            context_mean,
            context_scale,
            query_mask,
            args,
            spec,
            generator,
        )
        if epoch < args.minimum_epochs or epoch % args.validation_every:
            continue
        validation_logits = predict_logits(
            model,
            validation_indices,
            views,
            context_mean,
            context_scale,
            device=args.device,
            batch_size=args.batch_size,
        )
        validation_probability = numpy_cdf_from_logits(
            validation_logits, query_mask_numpy, temperature=1.0
        )
        validation_brier = task_averaged_brier(
            budget_labels[validation_indices],
            validation_probability,
            data["task_index"][validation_indices],
        )
        history.append(
            {
                "epoch": epoch,
                "validation_brier": validation_brier,
                "training_loss": training["loss"],
                "training_nll": training["nll"],
                "training_brier": training["brier"],
            }
        )
        if validation_brier < best_brier - 1e-5:
            best_brier = validation_brier
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_checks = 0
        else:
            stale_checks += 1
        if stale_checks >= args.patience_checks:
            break
    if best_state is None:
        raise RuntimeError("early stopping did not produce a checkpoint")
    best_epoch = select_best_epoch(history)

    # Refit from the registered initialization for exactly the selected epoch.
    refit_tasks = sorted(split["selection_train"] + split["validation"])
    refit_indices = all_indices(data, refit_tasks)
    refit_weights = np.ones(data["task_index"].shape, dtype=np.float32)
    refit_weights[refit_indices] = task_weights(data["task_index"][refit_indices])
    refit_context_mean = views["context"][refit_indices].mean(axis=0)
    refit_context_scale = views["context"][refit_indices].std(axis=0)
    refit_context_scale[refit_context_scale < 1e-6] = 1.0
    refit_model, refit_optimizer = build_model_and_optimizer(
        base_encoder, args, spec, finite_atoms=edges.size, seed=seed
    )
    refit_generator = np.random.default_rng(seed)
    for _ in range(best_epoch):
        run_epoch(
            refit_model,
            refit_optimizer,
            refit_indices,
            views,
            targets,
            budget_labels,
            refit_weights,
            refit_context_mean,
            refit_context_scale,
            query_mask,
            args,
            spec,
            refit_generator,
        )

    calibration_indices = initial_indices(data, split["calibration"])
    calibration_logits = predict_logits(
        refit_model,
        calibration_indices,
        views,
        refit_context_mean,
        refit_context_scale,
        device=args.device,
        batch_size=args.batch_size,
    )
    if bool(spec["temperature"]):
        temperature, temperature_curve = fit_temperature(
            calibration_logits,
            query_mask_numpy,
            budget_labels[calibration_indices],
            data["task_index"][calibration_indices],
        )
    else:
        temperature = 1.0
        temperature_curve = []
    test_indices = initial_indices(data, split["test"])
    test_logits = predict_logits(
        refit_model,
        test_indices,
        views,
        refit_context_mean,
        refit_context_scale,
        device=args.device,
        batch_size=args.batch_size,
    )
    probability = numpy_cdf_from_logits(
        test_logits, query_mask_numpy, temperature=temperature
    )
    uncalibrated_probability = numpy_cdf_from_logits(
        test_logits, query_mask_numpy, temperature=1.0
    )
    checkpoint = args.output_dir / "models" / variant / f"fold_{fold}.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model_state_dict": {
                name: value.detach().cpu() for name, value in refit_model.state_dict().items()
            },
            "variant": variant,
            "fold": fold,
            "best_epoch": best_epoch,
            "temperature": temperature,
            "context_mean": refit_context_mean,
            "context_scale": refit_context_scale,
            "finite_edges": edges,
            "budgets": budgets,
            "split": split,
        },
    )
    query = query_rows(
        data, test_indices, budget_labels, probability, budgets
    )
    query["uncalibrated_probability"] = uncalibrated_probability.reshape(-1)
    return {
        "variant": variant,
        "fold": fold,
        "split": split,
        "best_epoch": best_epoch,
        "stopped_epoch": int(history[-1]["epoch"]),
        "temperature": temperature,
        "temperature_curve": temperature_curve,
        "selection_history": history,
        "checkpoint": str(checkpoint),
        "test": {name: value.tolist() for name, value in query.items()},
    }


def baseline_for_fold(
    data: dict[str, np.ndarray],
    split: dict[str, list[int]],
    budget_labels: np.ndarray,
    budgets: np.ndarray,
) -> dict[str, object]:
    development_tasks = sorted(
        split["selection_train"] + split["validation"] + split["calibration"]
    )
    train_indices = initial_indices(data, development_tasks)
    test_indices = initial_indices(data, split["test"])
    train_geometry = np.repeat(data["geometry"][train_indices], budgets.size, axis=0)
    train_geometry = np.column_stack(
        (train_geometry, np.tile(budgets, train_indices.size))
    )
    test_geometry = np.repeat(data["geometry"][test_indices], budgets.size, axis=0)
    test_geometry = np.column_stack(
        (test_geometry, np.tile(budgets, test_indices.size))
    )
    train_labels = budget_labels[train_indices].reshape(-1)
    test_labels = budget_labels[test_indices].reshape(-1)
    weights, normalization = fit_logistic_baseline(train_geometry, train_labels)
    prevalence = float(np.mean(train_labels))
    return {
        "labels": test_labels.tolist(),
        "task_index": np.repeat(data["task_index"][test_indices], budgets.size).tolist(),
        "horizon": np.repeat(data["horizon"][test_indices], budgets.size).tolist(),
        "budget": np.tile(budgets, test_indices.size).tolist(),
        "constant_probability": np.full(test_labels.size, prevalence).tolist(),
        "geometry_probability": predict_logistic(
            test_geometry, weights, normalization
        ).tolist(),
    }


def aggregate_results(
    args: argparse.Namespace,
    fold_results: list[dict[str, object]],
    baselines_by_fold: list[dict[str, object]],
    task_buckets: dict[int, str],
) -> tuple[dict[str, object], dict[str, object]]:
    labels = np.asarray(
        [value for fold in baselines_by_fold for value in fold["labels"]], dtype=np.int64
    )
    tasks = np.asarray(
        [value for fold in baselines_by_fold for value in fold["task_index"]], dtype=np.int64
    )
    horizons = np.asarray(
        [value for fold in baselines_by_fold for value in fold["horizon"]], dtype=np.int64
    )
    budgets = np.asarray(
        [value for fold in baselines_by_fold for value in fold["budget"]], dtype=np.float64
    )
    constant = np.asarray(
        [value for fold in baselines_by_fold for value in fold["constant_probability"]],
        dtype=np.float64,
    )
    geometry = np.asarray(
        [value for fold in baselines_by_fold for value in fold["geometry_probability"]],
        dtype=np.float64,
    )
    distance = np.asarray([task_buckets[int(index)] for index in tasks])
    baselines = {
        "constant": binary_metrics(labels, constant),
        "geometry": binary_metrics(labels, geometry),
        "geometry_by_horizon": grouped_metrics(labels, geometry, horizons),
        "geometry_by_budget": grouped_metrics(labels, geometry, budgets),
        "geometry_by_distance_bucket": grouped_metrics(labels, geometry, distance),
    }
    matrix: dict[str, object] = {}
    for variant in args.variants:
        selected = sorted(
            (row for row in fold_results if row["variant"] == variant),
            key=lambda row: int(row["fold"]),
        )
        variant_labels = np.asarray(
            [value for row in selected for value in row["test"]["labels"]], dtype=np.int64
        )
        if not np.array_equal(variant_labels, labels):
            raise AssertionError("variant and baseline labels are not aligned")
        probability = np.asarray(
            [value for row in selected for value in row["test"]["probability"]],
            dtype=np.float64,
        )
        uncalibrated = np.asarray(
            [
                value
                for row in selected
                for value in row["test"]["uncalibrated_probability"]
            ],
            dtype=np.float64,
        )
        matrix[variant] = {
            "overall": binary_metrics(labels, probability),
            "uncalibrated_overall": binary_metrics(labels, uncalibrated),
            "by_horizon": grouped_metrics(labels, probability, horizons),
            "by_budget": grouped_metrics(labels, probability, budgets),
            "by_distance_bucket": grouped_metrics(labels, probability, distance),
            "best_epochs": [int(row["best_epoch"]) for row in selected],
            "temperatures": [float(row["temperature"]) for row in selected],
        }
        save_npz_atomic(
            args.output_dir / "predictions" / f"{variant}.npz",
            label=labels,
            probability=probability,
            uncalibrated_probability=uncalibrated,
            task_index=tasks,
            horizon=horizons,
            budget=budgets,
            distance_bucket=distance,
            geometry_probability=geometry,
        )
    return matrix, baselines


def gate_decision(
    matrix: dict[str, object],
    baselines: dict[str, object],
    fold_results: list[dict[str, object]],
    baselines_by_fold: list[dict[str, object]],
    *,
    primary_variant: str,
) -> dict[str, object]:
    primary = matrix[primary_variant]["overall"]
    geometry = baselines["geometry"]
    primary_folds = sorted(
        (row for row in fold_results if row["variant"] == primary_variant),
        key=lambda row: int(row["fold"]),
    )
    fold_comparisons = []
    for row, baseline in zip(primary_folds, baselines_by_fold):
        model_metric = binary_metrics(
            np.asarray(row["test"]["labels"], dtype=np.int64),
            np.asarray(row["test"]["probability"], dtype=np.float64),
        )
        geometry_metric = binary_metrics(
            np.asarray(baseline["labels"], dtype=np.int64),
            np.asarray(baseline["geometry_probability"], dtype=np.float64),
        )
        fold_comparisons.append(
            {
                "fold": int(row["fold"]),
                "primary_brier": model_metric["brier"],
                "geometry_brier": geometry_metric["brier"],
                "primary_better": bool(model_metric["brier"] < geometry_metric["brier"]),
            }
        )
    checks = {
        "both_classes_have_at_least_200_queries": bool(
            primary["positive_count"] >= 200
            and primary["count"] - primary["positive_count"] >= 200
        ),
        "primary_brier_beats_geometry_by_2_percent": bool(
            primary["brier"] <= 0.98 * geometry["brier"]
        ),
        "primary_ece_at_most_0_10": bool(primary["ece_10_bin"] <= 0.10),
        "primary_auroc_within_0_01_of_geometry": bool(
            primary["auroc"] is not None
            and geometry["auroc"] is not None
            and primary["auroc"] >= geometry["auroc"] - 0.01
        ),
        "primary_brier_better_on_at_least_two_folds": bool(
            sum(row["primary_better"] for row in fold_comparisons) >= 2
        ),
    }
    promotable = all(checks.values())
    return {
        "status": (
            "PROMOTE_TO_CLOSED_LOOP_RESOURCE_POLICY_PILOT"
            if promotable
            else "DO_NOT_PROMOTE_RESOURCE_AWARE_ENCODER"
        ),
        "promotable": promotable,
        "primary_variant": primary_variant,
        "checks": checks,
        "fold_comparisons": fold_comparisons,
        "interpretation": (
            "Exploratory held-out-task representation evidence only; repeated "
            "budget/horizon queries are not independent experimental seeds."
        ),
    }


def selected_task_indices(
    task_buckets: dict[int, str], *, smoke: bool
) -> np.ndarray:
    if not smoke:
        return np.asarray(sorted(task_buckets), dtype=np.int64)
    selected = []
    for bucket in sorted(set(task_buckets.values())):
        members = [index for index in sorted(task_buckets) if task_buckets[index] == bucket]
        selected.extend(members[:15])
    return np.asarray(sorted(selected), dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("source_dir", "artifact", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    protocol = ROOT / "docs" / "RESOURCE_AWARE_ENCODER_GATE_PROTOCOL.md"
    implementation = Path(__file__).resolve()
    required = [
        args.source_dir / "RESULT.json",
        args.artifact / "config.json",
        args.checkpoint,
        protocol,
    ]
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
    source_result = json.loads((args.source_dir / "RESULT.json").read_text(encoding="utf-8"))
    num_source_tasks = int(source_result["num_tasks"])
    task_buckets = {
        index: json.loads(
            (args.source_dir / "tasks" / f"task_{index:03d}.json").read_text(
                encoding="utf-8"
            )
        )["distance_bucket"]
        for index in range(num_source_tasks)
    }
    chosen_tasks = selected_task_indices(task_buckets, smoke=args.smoke)
    data = load_dataset(args.source_dir, num_source_tasks)
    chosen_rows = np.isin(data["task_index"], chosen_tasks)
    data = {name: value[chosen_rows] for name, value in data.items()}
    budgets = np.asarray(args.budgets, dtype=np.float64)
    support = float(source_result["support_capacity_multiple"])
    edges = threshold_aligned_edges(
        support=support, uniform_bins=args.uniform_energy_bins, budgets=budgets
    )
    targets = energy_targets(data["energy_fraction"], edges)
    budget_labels = expand_budget_queries(
        np.isfinite(data["energy_fraction"]), data["energy_fraction"], budgets
    ).astype(np.float32)
    environment_args, _ = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.model_seed
    )
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.TD_PRETRAINING)
    observation_dim = int(probe.observation_space.shape[0])
    d_max = float(probe.d_max)
    policy = SAC.load(
        args.checkpoint, env=probe, device=args.device, print_system_info=False
    )
    base_encoder = copy.deepcopy(policy.policy.actor.features_extractor).cpu()
    probe.close()
    del policy
    views = observation_views(
        data["features"], observation_dim=observation_dim, d_max=d_max
    )
    manifest = {
        "status": "RUNNING",
        "protocol": "resource_aware_encoder_gate_v1",
        "formal_evidence": False,
        "source_dir": str(args.source_dir),
        "source_result_sha256": file_sha256(args.source_dir / "RESULT.json"),
        "artifact": str(args.artifact),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "protocol_sha256": file_sha256(protocol),
        "implementation_sha256": file_sha256(implementation),
        "selected_num_tasks": int(chosen_tasks.size),
        "source_num_tasks": num_source_tasks,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "budgets": budgets.tolist(),
        "support_capacity_multiple": support,
        "uniform_energy_bins": args.uniform_energy_bins,
        "finite_atom_count": int(edges.size),
        "finite_edges": edges.tolist(),
        "variants": args.variants,
        "max_epochs": args.max_epochs,
        "minimum_epochs": args.minimum_epochs,
        "validation_every": args.validation_every,
        "patience_checks": args.patience_checks,
        "validation_tasks_per_bucket": args.validation_tasks_per_bucket,
        "calibration_tasks_per_bucket": args.calibration_tasks_per_bucket,
        "head_learning_rate": args.head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "calibration_weight": args.calibration_weight,
        "device": args.device,
        "exact_command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    try:
        fold_results: list[dict[str, object]] = []
        baselines_by_fold: list[dict[str, object]] = []
        for fold in range(3):
            split = nested_task_split(
                chosen_tasks,
                task_buckets,
                fold=fold,
                split_seed=args.split_seed,
                validation_per_bucket=args.validation_tasks_per_bucket,
                calibration_per_bucket=args.calibration_tasks_per_bucket,
            )
            baseline_path = args.output_dir / "fold_results" / f"baseline_fold_{fold}.json"
            if args.resume and baseline_path.is_file():
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            else:
                baseline = baseline_for_fold(data, split, budget_labels, budgets)
                atomic_json(baseline_path, baseline)
            baselines_by_fold.append(baseline)
            for variant in args.variants:
                result_path = (
                    args.output_dir / "fold_results" / f"{variant}_fold_{fold}.json"
                )
                if args.resume and result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    result = train_variant_fold(
                        args,
                        variant,
                        fold,
                        split,
                        data,
                        views,
                        targets,
                        budget_labels,
                        edges,
                        budgets,
                        base_encoder,
                    )
                    atomic_json(result_path, result)
                fold_results.append(result)
                atomic_json(
                    args.output_dir / "PROGRESS.json",
                    {
                        "stage": "training",
                        "completed_variant_folds": len(fold_results),
                        "total_variant_folds": 3 * len(args.variants),
                        "last_fold": fold,
                        "last_variant": variant,
                        "best_epoch": result["best_epoch"],
                        "stopped_epoch": result["stopped_epoch"],
                    },
                )
        matrix, baselines = aggregate_results(
            args, fold_results, baselines_by_fold, task_buckets
        )
        primary_variant = PRIMARY_VARIANT
        result = {
            **manifest,
            **gate_decision(
                matrix,
                baselines,
                fold_results,
                baselines_by_fold,
                primary_variant=primary_variant,
            ),
            "label_audit": {
                "num_tasks": int(chosen_tasks.size),
                "num_examples": int(targets.size),
                "num_initial_states": int(data["is_initial"].sum()),
                "num_budget_queries": int(data["is_initial"].sum() * budgets.size),
                "positive_budget_queries": int(budget_labels[data["is_initial"]].sum()),
                "negative_budget_queries": int(
                    budget_labels[data["is_initial"]].size
                    - budget_labels[data["is_initial"]].sum()
                ),
                "maximum_finite_energy_fraction": float(
                    np.max(data["energy_fraction"][np.isfinite(data["energy_fraction"])])
                ),
            },
            "baselines": baselines,
            "experiment_matrix": matrix,
            "finished_unix": time.time(),
        }
        if args.smoke:
            result["status"] = "SMOKE_COMPLETED_NOT_SCIENTIFIC_EVIDENCE"
            result["promotable"] = False
        atomic_json(args.output_dir / "RESULT.json", result)
        atomic_json(args.output_dir / "COMPLETED.json", {"status": result["status"]})
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        atomic_json(
            args.output_dir / "FAILED.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
