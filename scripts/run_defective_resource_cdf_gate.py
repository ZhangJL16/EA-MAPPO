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
    atomic_json,
    binary_metrics,
    file_sha256,
    load_dataset,
)
from scripts.run_resource_aware_encoder_gate import (
    all_indices,
    atomic_torch_save,
    baseline_for_fold,
    gate_decision,
    initial_indices,
    nested_task_split,
    observation_views,
    selected_task_indices,
    task_averaged_brier,
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
DEFAULT_VARIANTS = (
    "compact_hurdle",
    "encoder_hurdle",
    "fusion_no_cdf",
    "fusion_hurdle",
)
PRIMARY_VARIANT = "fusion_hurdle"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identifiable defective continuous resource-CDF Gate"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=280_001)
    parser.add_argument("--model-seed", type=int, default=290_001)
    parser.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--minimum-epochs", type=int, default=5)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--patience-checks", type=int, default=30)
    parser.add_argument("--validation-tasks-per-bucket", type=int, default=6)
    parser.add_argument("--calibration-tasks-per-bucket", type=int, default=6)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--cdf-loss-weight", type=float, default=2.0)
    parser.add_argument("--huber-beta", type=float, default=0.25)
    parser.add_argument("--minimum-scale", type=float, default=0.03)
    parser.add_argument("--initial-scale", type=float, default=0.25)
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
    if not args.budgets or args.budgets != sorted(set(args.budgets)):
        parser.error("--budgets must be nonempty, unique, and increasing")
    if any(value <= 0.0 for value in args.budgets):
        parser.error("--budgets must be positive for the log-resource CDF")
    if args.max_epochs <= 0 or not 1 <= args.minimum_epochs <= args.max_epochs:
        parser.error("epoch limits are invalid")
    if args.validation_every <= 0 or args.patience_checks <= 0:
        parser.error("validation cadence and patience must be positive")
    if args.validation_tasks_per_bucket <= 0 or args.calibration_tasks_per_bucket <= 0:
        parser.error("nested split sizes must be positive")
    if args.head_learning_rate <= 0.0 or args.encoder_learning_rate <= 0.0:
        parser.error("learning rates must be positive")
    if args.cdf_loss_weight < 0.0 or args.huber_beta <= 0.0:
        parser.error("loss weights are invalid")
    if not 0.0 < args.minimum_scale < args.initial_scale:
        parser.error("scales must satisfy 0 < minimum < initial")
    if args.weight_decay < 0.0 or args.batch_size <= 0 or args.hidden_dim <= 0:
        parser.error("optimization parameters are invalid")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if len(set(args.variants)) != len(args.variants):
        parser.error("variants must be unique")
    if not args.smoke and PRIMARY_VARIANT not in args.variants:
        parser.error(f"formal exploratory run requires {PRIMARY_VARIANT}")
    return args


def inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("inverse-softplus input must be positive")
    return float(math.log(math.expm1(value)))


def variant_spec(name: str, cdf_loss_weight: float) -> dict[str, object]:
    if name == "compact_hurdle":
        return {"use_encoder": False, "use_compact": True, "cdf_weight": cdf_loss_weight}
    if name == "encoder_hurdle":
        return {"use_encoder": True, "use_compact": False, "cdf_weight": cdf_loss_weight}
    if name == "fusion_no_cdf":
        return {"use_encoder": True, "use_compact": True, "cdf_weight": 0.0}
    if name == PRIMARY_VARIANT:
        return {"use_encoder": True, "use_compact": True, "cdf_weight": cdf_loss_weight}
    raise KeyError(name)


class DefectiveResourceCDF(nn.Module):
    """Failure mass plus a conditional log-energy location family."""

    def __init__(
        self,
        base_encoder: nn.Module,
        *,
        variant: str,
        hidden_dim: int,
        compact_dim: int,
        context_dim: int,
        minimum_scale: float,
        initial_scale: float,
        initial_log_energy: float,
    ) -> None:
        super().__init__()
        spec = variant_spec(variant, cdf_loss_weight=1.0)
        self.use_encoder = bool(spec["use_encoder"])
        self.use_compact = bool(spec["use_compact"])
        self.minimum_scale = float(minimum_scale)
        input_dim = 0
        if self.use_encoder:
            self.encoder = copy.deepcopy(base_encoder)
            input_dim += 2 * int(getattr(self.encoder, "features_dim")) + (
                0 if self.use_compact else context_dim
            )
        else:
            self.encoder = None
        if self.use_compact:
            input_dim += compact_dim
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.failure_head = nn.Linear(hidden_dim, 1)
        self.energy_head = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.energy_head.bias.fill_(float(initial_log_energy))
        raw_scale = inverse_softplus(initial_scale - minimum_scale)
        self.raw_scale = nn.Parameter(torch.tensor(raw_scale, dtype=torch.float32))

    @property
    def scale(self) -> torch.Tensor:
        return self.minimum_scale + nn.functional.softplus(self.raw_scale)

    def forward(
        self,
        task_observation: torch.Tensor,
        charger_observation: torch.Tensor,
        context: torch.Tensor,
        compact: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pieces = []
        if self.encoder is not None:
            pieces.extend(
                (self.encoder(task_observation), self.encoder(charger_observation))
            )
            if not self.use_compact:
                pieces.append(context)
        if self.use_compact:
            pieces.append(compact)
        latent = self.backbone(torch.cat(pieces, dim=1))
        return (
            self.failure_head(latent).squeeze(1),
            self.energy_head(latent).squeeze(1),
            self.scale,
        )


def defective_cdf_torch(
    failure_logit: torch.Tensor,
    log_energy_location: torch.Tensor,
    scale: torch.Tensor,
    budgets: torch.Tensor,
    *,
    failure_temperature: float = 1.0,
    location_shift: float = 0.0,
    scale_multiplier: float = 1.0,
) -> torch.Tensor:
    if failure_temperature <= 0.0 or scale_multiplier <= 0.0:
        raise ValueError("calibration temperatures must be positive")
    if torch.any(scale <= 0.0) or torch.any(budgets <= 0.0):
        raise ValueError("scale and budgets must be positive")
    failure = torch.sigmoid(failure_logit / float(failure_temperature))
    standardized = (
        torch.log(budgets)[None, :]
        - (log_energy_location[:, None] + float(location_shift))
    ) / (scale * float(scale_multiplier))
    conditional = 0.5 * (1.0 + torch.erf(standardized / math.sqrt(2.0)))
    return (1.0 - failure[:, None]) * conditional


def defective_cdf_numpy(
    failure_logit: np.ndarray,
    log_energy_location: np.ndarray,
    scale: float,
    budgets: np.ndarray,
    *,
    failure_temperature: float = 1.0,
    location_shift: float = 0.0,
    scale_multiplier: float = 1.0,
) -> np.ndarray:
    failure_tensor = torch.as_tensor(failure_logit, dtype=torch.float64)
    location_tensor = torch.as_tensor(log_energy_location, dtype=torch.float64)
    scale_tensor = torch.as_tensor(float(scale), dtype=torch.float64)
    budget_tensor = torch.as_tensor(budgets, dtype=torch.float64)
    with torch.no_grad():
        result = defective_cdf_torch(
            failure_tensor,
            location_tensor,
            scale_tensor,
            budget_tensor,
            failure_temperature=failure_temperature,
            location_shift=location_shift,
            scale_multiplier=scale_multiplier,
        )
    return result.numpy()


def defective_resource_loss(
    failure_logit: torch.Tensor,
    log_energy_location: torch.Tensor,
    scale: torch.Tensor,
    energy_fraction: torch.Tensor,
    budget_labels: torch.Tensor,
    sample_weights: torch.Tensor,
    budgets: torch.Tensor,
    *,
    cdf_weight: float,
    huber_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    finite = torch.isfinite(energy_fraction)
    failure_target = (~finite).to(torch.float32)
    normalized = sample_weights / torch.sum(sample_weights)
    failure_rows = nn.functional.binary_cross_entropy_with_logits(
        failure_logit, failure_target, reduction="none"
    )
    failure_loss = torch.sum(normalized * failure_rows)
    if torch.any(finite):
        finite_weights = sample_weights[finite]
        finite_weights = finite_weights / torch.sum(finite_weights)
        energy_rows = nn.functional.smooth_l1_loss(
            log_energy_location[finite],
            torch.log(energy_fraction[finite]),
            beta=float(huber_beta),
            reduction="none",
        )
        energy_loss = torch.sum(finite_weights * energy_rows)
    else:
        energy_loss = log_energy_location.sum() * 0.0
    probability = defective_cdf_torch(
        failure_logit, log_energy_location, scale, budgets
    )
    brier_rows = torch.mean((probability - budget_labels) ** 2, dim=1)
    brier_loss = torch.sum(normalized * brier_rows)
    total = failure_loss + energy_loss + float(cdf_weight) * brier_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "failure_bce": float(failure_loss.detach().cpu()),
        "energy_huber": float(energy_loss.detach().cpu()),
        "cdf_brier": float(brier_loss.detach().cpu()),
        "scale": float(scale.detach().cpu()),
    }


def add_compact_view(
    views: dict[str, np.ndarray],
    raw_features: np.ndarray,
    *,
    observation_dim: int,
) -> dict[str, np.ndarray]:
    result = dict(views)
    result["compact"] = representation_arrays(
        raw_features, observation_dim=observation_dim
    )["compact"]
    return result


def fit_normalization(
    views: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, np.ndarray]:
    result = {}
    for name in ("context", "compact"):
        mean = views[name][indices].mean(axis=0)
        scale = views[name][indices].std(axis=0)
        scale[scale < 1e-6] = 1.0
        result[f"{name}_mean"] = mean.astype(np.float32)
        result[f"{name}_scale"] = scale.astype(np.float32)
    return result


def build_model_and_optimizer(
    base_encoder: nn.Module,
    args: argparse.Namespace,
    variant: str,
    *,
    initial_log_energy: float,
    seed: int,
) -> tuple[DefectiveResourceCDF, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DefectiveResourceCDF(
        base_encoder,
        variant=variant,
        hidden_dim=args.hidden_dim,
        compact_dim=18,
        context_dim=4,
        minimum_scale=args.minimum_scale,
        initial_scale=args.initial_scale,
        initial_log_energy=initial_log_energy,
    ).to(args.device)
    head_parameters = (
        list(model.backbone.parameters())
        + list(model.failure_head.parameters())
        + list(model.energy_head.parameters())
        + [model.raw_scale]
    )
    groups: list[dict[str, object]] = [
        {"params": head_parameters, "lr": args.head_learning_rate}
    ]
    if model.encoder is not None:
        groups.append(
            {"params": list(model.encoder.parameters()), "lr": args.encoder_learning_rate}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return model, optimizer


def normalized_batch(
    views: dict[str, np.ndarray],
    indices: np.ndarray,
    normalization: dict[str, np.ndarray],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    task = torch.as_tensor(
        views["task_observation"][indices], dtype=torch.float32, device=device
    )
    charger = torch.as_tensor(
        views["charger_observation"][indices], dtype=torch.float32, device=device
    )
    context = torch.as_tensor(
        (views["context"][indices] - normalization["context_mean"])
        / normalization["context_scale"],
        dtype=torch.float32,
        device=device,
    )
    compact = torch.as_tensor(
        (views["compact"][indices] - normalization["compact_mean"])
        / normalization["compact_scale"],
        dtype=torch.float32,
        device=device,
    )
    return task, charger, context, compact


def run_epoch(
    model: DefectiveResourceCDF,
    optimizer: torch.optim.Optimizer,
    indices: np.ndarray,
    views: dict[str, np.ndarray],
    data: dict[str, np.ndarray],
    budget_labels: np.ndarray,
    weights: np.ndarray,
    normalization: dict[str, np.ndarray],
    budgets: torch.Tensor,
    args: argparse.Namespace,
    *,
    cdf_weight: float,
    generator: np.random.Generator,
) -> dict[str, float]:
    order = generator.permutation(indices)
    totals = {
        "loss": 0.0,
        "failure_bce": 0.0,
        "energy_huber": 0.0,
        "cdf_brier": 0.0,
        "scale": 0.0,
        "count": 0.0,
    }
    model.train()
    for start in range(0, order.size, args.batch_size):
        selected = order[start : start + args.batch_size]
        batch = normalized_batch(
            views, selected, normalization, device=args.device
        )
        failure_logit, location, scale = model(*batch)
        energy = torch.as_tensor(
            data["energy_fraction"][selected], dtype=torch.float32, device=args.device
        )
        labels = torch.as_tensor(
            budget_labels[selected], dtype=torch.float32, device=args.device
        )
        sample_weight = torch.as_tensor(
            weights[selected], dtype=torch.float32, device=args.device
        )
        loss, pieces = defective_resource_loss(
            failure_logit,
            location,
            scale,
            energy,
            labels,
            sample_weight,
            budgets,
            cdf_weight=cdf_weight,
            huber_beta=args.huber_beta,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        count = float(selected.size)
        for key in ("loss", "failure_bce", "energy_huber", "cdf_brier", "scale"):
            totals[key] += pieces[key] * count
        totals["count"] += count
    return {
        key: totals[key] / totals["count"]
        for key in ("loss", "failure_bce", "energy_huber", "cdf_brier", "scale")
    }


def predict_parameters(
    model: DefectiveResourceCDF,
    indices: np.ndarray,
    views: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    failure_logits = []
    locations = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            selected = indices[start : start + batch_size]
            batch = normalized_batch(
                views, selected, normalization, device=device
            )
            failure_logit, location, _ = model(*batch)
            failure_logits.append(failure_logit.cpu().numpy())
            locations.append(location.cpu().numpy())
        scale = float(model.scale.cpu())
    return np.concatenate(failure_logits), np.concatenate(locations), scale


def calibration_score(
    failure_logit: np.ndarray,
    location: np.ndarray,
    scale: float,
    budgets: np.ndarray,
    labels: np.ndarray,
    task_index: np.ndarray,
    parameters: dict[str, float],
) -> float:
    probability = defective_cdf_numpy(
        failure_logit,
        location,
        scale,
        budgets,
        failure_temperature=parameters["failure_temperature"],
        location_shift=parameters["location_shift"],
        scale_multiplier=parameters["scale_multiplier"],
    )
    return task_averaged_brier(labels, probability, task_index)


def fit_calibration(
    failure_logit: np.ndarray,
    location: np.ndarray,
    scale: float,
    budgets: np.ndarray,
    labels: np.ndarray,
    task_index: np.ndarray,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    parameters = {
        "failure_temperature": 1.0,
        "location_shift": 0.0,
        "scale_multiplier": 1.0,
    }
    candidates = {
        "failure_temperature": np.geomspace(0.5, 3.0, 31),
        "location_shift": np.linspace(-0.35, 0.35, 57),
        "scale_multiplier": np.geomspace(0.35, 3.5, 41),
    }
    history: list[dict[str, float | str]] = []
    for iteration in range(3):
        for name in ("failure_temperature", "location_shift", "scale_multiplier"):
            rows = []
            for value in np.unique(
                np.concatenate((np.asarray([parameters[name]]), candidates[name]))
            ):
                proposal = dict(parameters)
                proposal[name] = float(value)
                score = calibration_score(
                    failure_logit,
                    location,
                    scale,
                    budgets,
                    labels,
                    task_index,
                    proposal,
                )
                rows.append((score, abs(float(value) - parameters[name]), float(value)))
            score, _, value = min(rows)
            parameters[name] = value
            history.append(
                {
                    "iteration": iteration,
                    "parameter": name,
                    "value": value,
                    "task_brier": score,
                }
            )
    return parameters, history


def select_best_epoch(history: list[dict[str, float | int]]) -> int:
    if not history:
        raise ValueError("validation history is empty")
    best = min(
        history,
        key=lambda row: (float(row["validation_brier"]), int(row["epoch"])),
    )
    return int(best["epoch"])


def train_variant_fold(
    args: argparse.Namespace,
    variant: str,
    fold: int,
    split: dict[str, list[int]],
    data: dict[str, np.ndarray],
    views: dict[str, np.ndarray],
    budget_labels: np.ndarray,
    budgets_numpy: np.ndarray,
    base_encoder: nn.Module,
) -> dict[str, object]:
    spec = variant_spec(variant, args.cdf_loss_weight)
    budgets = torch.as_tensor(
        budgets_numpy, dtype=torch.float32, device=args.device
    )
    selection_indices = all_indices(data, split["selection_train"])
    validation_indices = initial_indices(data, split["validation"])
    normalization = fit_normalization(views, selection_indices)
    weights = np.ones(data["task_index"].shape, dtype=np.float32)
    weights[selection_indices] = task_weights(data["task_index"][selection_indices])
    finite_selection = np.isfinite(data["energy_fraction"][selection_indices])
    initial_location = float(
        np.median(np.log(data["energy_fraction"][selection_indices][finite_selection]))
    )
    seed = args.model_seed + 1000 * fold + 41 * list(DEFAULT_VARIANTS).index(variant)
    model, optimizer = build_model_and_optimizer(
        base_encoder,
        args,
        variant,
        initial_log_energy=initial_location,
        seed=seed,
    )
    generator = np.random.default_rng(seed)
    history: list[dict[str, float | int]] = []
    best_brier = float("inf")
    stale_checks = 0
    for epoch in range(1, args.max_epochs + 1):
        training = run_epoch(
            model,
            optimizer,
            selection_indices,
            views,
            data,
            budget_labels,
            weights,
            normalization,
            budgets,
            args,
            cdf_weight=float(spec["cdf_weight"]),
            generator=generator,
        )
        if epoch < args.minimum_epochs or epoch % args.validation_every:
            continue
        failure_logit, location, scale = predict_parameters(
            model,
            validation_indices,
            views,
            normalization,
            device=args.device,
            batch_size=args.batch_size,
        )
        probability = defective_cdf_numpy(
            failure_logit, location, scale, budgets_numpy
        )
        validation_brier = task_averaged_brier(
            budget_labels[validation_indices],
            probability,
            data["task_index"][validation_indices],
        )
        history.append(
            {
                "epoch": epoch,
                "validation_brier": validation_brier,
                **{f"training_{name}": value for name, value in training.items()},
            }
        )
        if validation_brier < best_brier - 1e-5:
            best_brier = validation_brier
            stale_checks = 0
        else:
            stale_checks += 1
        if stale_checks >= args.patience_checks:
            break
    best_epoch = select_best_epoch(history)

    refit_tasks = sorted(split["selection_train"] + split["validation"])
    refit_indices = all_indices(data, refit_tasks)
    refit_normalization = fit_normalization(views, refit_indices)
    refit_weights = np.ones(data["task_index"].shape, dtype=np.float32)
    refit_weights[refit_indices] = task_weights(data["task_index"][refit_indices])
    finite_refit = np.isfinite(data["energy_fraction"][refit_indices])
    refit_location = float(
        np.median(np.log(data["energy_fraction"][refit_indices][finite_refit]))
    )
    refit_model, refit_optimizer = build_model_and_optimizer(
        base_encoder,
        args,
        variant,
        initial_log_energy=refit_location,
        seed=seed,
    )
    refit_generator = np.random.default_rng(seed)
    for _ in range(best_epoch):
        run_epoch(
            refit_model,
            refit_optimizer,
            refit_indices,
            views,
            data,
            budget_labels,
            refit_weights,
            refit_normalization,
            budgets,
            args,
            cdf_weight=float(spec["cdf_weight"]),
            generator=refit_generator,
        )

    calibration_indices = initial_indices(data, split["calibration"])
    calibration_output = predict_parameters(
        refit_model,
        calibration_indices,
        views,
        refit_normalization,
        device=args.device,
        batch_size=args.batch_size,
    )
    calibration, calibration_history = fit_calibration(
        *calibration_output,
        budgets_numpy,
        budget_labels[calibration_indices],
        data["task_index"][calibration_indices],
    )
    test_indices = initial_indices(data, split["test"])
    failure_logit, location, scale = predict_parameters(
        refit_model,
        test_indices,
        views,
        refit_normalization,
        device=args.device,
        batch_size=args.batch_size,
    )
    probability = defective_cdf_numpy(
        failure_logit,
        location,
        scale,
        budgets_numpy,
        failure_temperature=calibration["failure_temperature"],
        location_shift=calibration["location_shift"],
        scale_multiplier=calibration["scale_multiplier"],
    )
    uncalibrated = defective_cdf_numpy(
        failure_logit, location, scale, budgets_numpy
    )
    calibrated_failure = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                failure_logit / calibration["failure_temperature"], -30.0, 30.0
            )
        )
    )
    predicted_energy = np.exp(location + calibration["location_shift"])
    finite = np.isfinite(data["energy_fraction"][test_indices])
    checkpoint = args.output_dir / "models" / variant / f"fold_{fold}.pt"
    atomic_torch_save(
        checkpoint,
        {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in refit_model.state_dict().items()
            },
            "variant": variant,
            "fold": fold,
            "best_epoch": best_epoch,
            "base_scale": scale,
            "calibration": calibration,
            "normalization": refit_normalization,
            "budgets": budgets_numpy,
            "split": split,
        },
    )
    return {
        "variant": variant,
        "fold": fold,
        "split": split,
        "best_epoch": best_epoch,
        "stopped_epoch": int(history[-1]["epoch"]),
        "selection_history": history,
        "base_scale": scale,
        "calibration": calibration,
        "calibration_history": calibration_history,
        "checkpoint": str(checkpoint),
        "test": {
            "labels": budget_labels[test_indices].reshape(-1).astype(np.int64).tolist(),
            "probability": probability.reshape(-1).tolist(),
            "uncalibrated_probability": uncalibrated.reshape(-1).tolist(),
            "task_index": np.repeat(
                data["task_index"][test_indices], budgets_numpy.size
            ).tolist(),
            "horizon": np.repeat(data["horizon"][test_indices], budgets_numpy.size).tolist(),
            "budget": np.tile(budgets_numpy, test_indices.size).tolist(),
            "failure_label": (~finite).astype(np.int64).tolist(),
            "failure_probability": calibrated_failure.tolist(),
            "finite_energy_target": data["energy_fraction"][test_indices][finite].tolist(),
            "finite_energy_prediction": predicted_energy[finite].tolist(),
        },
    }


def finite_energy_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("finite-energy arrays must be aligned and nonempty")
    error = p - y
    log_error = np.log(p) - np.log(y)
    return {
        "count": int(y.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
    }


def aggregate_results(
    args: argparse.Namespace,
    fold_results: list[dict[str, object]],
    baselines_by_fold: list[dict[str, object]],
    task_buckets: dict[int, str],
) -> tuple[dict[str, object], dict[str, object]]:
    labels = np.asarray(
        [value for fold in baselines_by_fold for value in fold["labels"]],
        dtype=np.int64,
    )
    tasks = np.asarray(
        [value for fold in baselines_by_fold for value in fold["task_index"]],
        dtype=np.int64,
    )
    horizons = np.asarray(
        [value for fold in baselines_by_fold for value in fold["horizon"]],
        dtype=np.int64,
    )
    budgets = np.asarray(
        [value for fold in baselines_by_fold for value in fold["budget"]],
        dtype=np.float64,
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
            [value for row in selected for value in row["test"]["labels"]],
            dtype=np.int64,
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
        failure_label = np.asarray(
            [value for row in selected for value in row["test"]["failure_label"]],
            dtype=np.int64,
        )
        failure_probability = np.asarray(
            [
                value
                for row in selected
                for value in row["test"]["failure_probability"]
            ],
            dtype=np.float64,
        )
        finite_target = np.asarray(
            [
                value
                for row in selected
                for value in row["test"]["finite_energy_target"]
            ],
            dtype=np.float64,
        )
        finite_prediction = np.asarray(
            [
                value
                for row in selected
                for value in row["test"]["finite_energy_prediction"]
            ],
            dtype=np.float64,
        )
        matrix[variant] = {
            "overall": binary_metrics(labels, probability),
            "uncalibrated_overall": binary_metrics(labels, uncalibrated),
            "failure_metrics": binary_metrics(failure_label, failure_probability),
            "finite_energy_metrics": finite_energy_metrics(
                finite_target, finite_prediction
            ),
            "by_horizon": grouped_metrics(labels, probability, horizons),
            "by_budget": grouped_metrics(labels, probability, budgets),
            "by_distance_bucket": grouped_metrics(
                labels, probability, distance
            ),
            "best_epochs": [int(row["best_epoch"]) for row in selected],
            "stopped_epochs": [int(row["stopped_epoch"]) for row in selected],
            "base_scales": [float(row["base_scale"]) for row in selected],
            "calibrations": [row["calibration"] for row in selected],
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("source_dir", "artifact", "checkpoint", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    protocol = ROOT / "docs" / "DEFECTIVE_RESOURCE_CDF_GATE_PROTOCOL.md"
    derivation = ROOT / "docs" / "DEFECTIVE_RESOURCE_CDF_DERIVATION.md"
    implementation = Path(__file__).resolve()
    required = [
        args.source_dir / "RESULT.json",
        args.artifact / "config.json",
        args.checkpoint,
        protocol,
        derivation,
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
    source_result = json.loads(
        (args.source_dir / "RESULT.json").read_text(encoding="utf-8")
    )
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
    finite_energy = data["energy_fraction"][np.isfinite(data["energy_fraction"])]
    if finite_energy.size == 0 or np.any(finite_energy <= 0.0):
        raise ValueError("defective log-resource model requires positive finite energy")
    budgets = np.asarray(args.budgets, dtype=np.float64)
    budget_labels = expand_budget_queries(
        np.isfinite(data["energy_fraction"]),
        data["energy_fraction"],
        budgets,
    ).astype(np.float32)
    environment_args, _ = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.model_seed
    )
    probe = environment_from_args(
        environment_args, phase=SACTrainingPhase.TD_PRETRAINING
    )
    observation_dim = int(probe.observation_space.shape[0])
    d_max = float(probe.d_max)
    policy = SAC.load(
        args.checkpoint, env=probe, device=args.device, print_system_info=False
    )
    base_encoder = copy.deepcopy(policy.policy.actor.features_extractor).cpu()
    probe.close()
    del policy
    views = add_compact_view(
        observation_views(
            data["features"], observation_dim=observation_dim, d_max=d_max
        ),
        data["features"],
        observation_dim=observation_dim,
    )
    manifest = {
        "status": "RUNNING",
        "protocol": "defective_continuous_resource_cdf_gate_v1",
        "formal_evidence": False,
        "source_dir": str(args.source_dir),
        "source_result_sha256": file_sha256(args.source_dir / "RESULT.json"),
        "artifact": str(args.artifact),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "protocol_sha256": file_sha256(protocol),
        "derivation_sha256": file_sha256(derivation),
        "implementation_sha256": file_sha256(implementation),
        "selected_num_tasks": int(chosen_tasks.size),
        "source_num_tasks": num_source_tasks,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "budgets": budgets.tolist(),
        "variants": args.variants,
        "primary_variant": PRIMARY_VARIANT,
        "max_epochs": args.max_epochs,
        "minimum_epochs": args.minimum_epochs,
        "validation_every": args.validation_every,
        "patience_checks": args.patience_checks,
        "validation_tasks_per_bucket": args.validation_tasks_per_bucket,
        "calibration_tasks_per_bucket": args.calibration_tasks_per_bucket,
        "head_learning_rate": args.head_learning_rate,
        "encoder_learning_rate": args.encoder_learning_rate,
        "cdf_loss_weight": args.cdf_loss_weight,
        "huber_beta": args.huber_beta,
        "minimum_scale": args.minimum_scale,
        "initial_scale": args.initial_scale,
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
            baseline_path = (
                args.output_dir / "fold_results" / f"baseline_fold_{fold}.json"
            )
            if args.resume and baseline_path.is_file():
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            else:
                baseline = baseline_for_fold(
                    data, split, budget_labels, budgets
                )
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
                        budget_labels,
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
        result = {
            **manifest,
            **gate_decision(
                matrix,
                baselines,
                fold_results,
                baselines_by_fold,
                primary_variant=PRIMARY_VARIANT,
            ),
            "label_audit": {
                "num_tasks": int(chosen_tasks.size),
                "num_examples": int(data["energy_fraction"].size),
                "num_initial_states": int(data["is_initial"].sum()),
                "num_budget_queries": int(data["is_initial"].sum() * budgets.size),
                "positive_budget_queries": int(
                    budget_labels[data["is_initial"]].sum()
                ),
                "negative_budget_queries": int(
                    budget_labels[data["is_initial"]].size
                    - budget_labels[data["is_initial"]].sum()
                ),
                "minimum_finite_energy_fraction": float(np.min(finite_energy)),
                "maximum_finite_energy_fraction": float(np.max(finite_energy)),
            },
            "baselines": baselines,
            "experiment_matrix": matrix,
            "finished_unix": time.time(),
        }
        if args.smoke:
            result["status"] = "SMOKE_COMPLETED_NOT_SCIENTIFIC_EVIDENCE"
            result["promotable"] = False
        atomic_json(args.output_dir / "RESULT.json", result)
        atomic_json(
            args.output_dir / "COMPLETED.json", {"status": result["status"]}
        )
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
