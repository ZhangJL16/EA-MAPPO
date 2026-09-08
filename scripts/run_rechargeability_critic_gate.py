from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.rechargeability_safety.core import (
    CURRENT_ACTION_CONDITIONING,
    FEATURE_NAMES,
    GEOMETRY_FEATURES,
    MonotoneBudgetCritic,
    bellman_target,
    build_pre_action_context,
    decode_active_goal_state,
    grouped_nested_split,
    probability_metrics,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    environment_from_args,
    generate_stratified_navigation_tasks,
)


PROTOCOL = "ACTION_CONDITIONED_RECHARGEABILITY_GATE_V1"
VARIANTS = ("geometry", "no_action", "no_bellman", "full")


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remaining_horizon(legs: np.ndarray, maximum_per_leg: int) -> np.ndarray:
    result = np.empty(legs.size, dtype=np.float32)
    task_elapsed = 0
    return_elapsed = 0
    for index, leg in enumerate(legs):
        if int(leg) == 0:
            result[index] = (2 * maximum_per_leg - task_elapsed) / (2 * maximum_per_leg)
            task_elapsed += 1
        else:
            result[index] = (maximum_per_leg - return_elapsed) / (2 * maximum_per_leg)
            return_elapsed += 1
    return np.clip(result, 0.0, 1.0)


def build_transition_cache(
    source: Path,
    cache_path: Path,
    *,
    transitions_per_rollout: int,
) -> dict[str, Any]:
    result_path = source / "RESULT.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    num_scenes = int(result["num_scenes"])
    task_seed = int(result["task_seed"])
    maximum_steps = int(result["max_policy_steps_per_leg"])
    capacity = float(result["capacity"])
    artifact = Path(result["artifact"])
    environment_args, _ = reconstruct_environment_args(
        artifact, device="cpu", seed=int(result["world_seed"])
    )
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    try:
        d_max = float(probe.d_max)
        charger = probe.charger_position.copy().astype(np.float32)
        extent = np.asarray([probe.length, probe.width, probe.height], dtype=np.float32)
        horizontal_v_max = float(probe.horizontal_v_max)
        vertical_v_max = float(probe.vertical_v_max)
    finally:
        probe.close()
    tasks = generate_stratified_navigation_tasks(num_tasks=num_scenes, seed=task_seed)
    contexts: list[np.ndarray] = []
    next_contexts: list[np.ndarray] = []
    suffix_energy: list[np.ndarray] = []
    realized_energy: list[np.ndarray] = []
    terminal_success: list[np.ndarray] = []
    terminal_failure: list[np.ndarray] = []
    scenes: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    transition_indices: list[np.ndarray] = []
    reconstruction_errors: list[float] = []
    rollout_count = 0
    for metadata_path in sorted(source.glob("rollouts/scene_*/*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene = int(metadata["scene_index"])
        intervention = int(metadata["intervention_index"])
        task = tasks[scene]
        arrays = np.load(metadata_path.with_suffix(".npz"))
        compact = np.asarray(arrays["compact_observations"], dtype=np.float32)
        legs = np.asarray(arrays["leg"], dtype=np.int8)
        frozen = np.asarray(arrays["frozen_actions"], dtype=np.float32)
        proposed = np.asarray(arrays["behavior_actions"], dtype=np.float32)
        energy = np.asarray(arrays["realized_energy"], dtype=np.float32) / capacity
        filter_norm = np.asarray(arrays["hocbf_intervention_norm"], dtype=np.float32)
        length = compact.shape[0]
        if not all(value.shape[0] == length for value in (legs, frozen, proposed, energy, filter_norm)):
            raise RuntimeError(f"unaligned rollout arrays: {metadata_path}")
        descriptor = np.asarray(
            [
                metadata["intervention"]["action_gain"],
                metadata["intervention"]["residual_sigma"],
                metadata["intervention"]["residual_rho"],
            ],
            dtype=np.float32,
        )
        horizon = _remaining_horizon(legs, maximum_steps)
        rollout_context = np.empty((length, len(FEATURE_NAMES)), dtype=np.float32)
        previous_action = np.zeros(3, dtype=np.float32)
        previous_energy = 0.0
        previous_filter = 0.0
        for step in range(length):
            active_goal = task.goal_position if int(legs[step]) == 0 else charger
            position, velocity = decode_active_goal_state(
                compact[step],
                active_goal,
                d_max=d_max,
                horizontal_v_max=horizontal_v_max,
                vertical_v_max=vertical_v_max,
            )
            rollout_context[step] = build_pre_action_context(
                position=position,
                velocity=velocity,
                task_goal=task.goal_position,
                charger_goal=charger,
                world_extent=extent,
                d_max=d_max,
                horizontal_v_max=horizontal_v_max,
                vertical_v_max=vertical_v_max,
                leg=int(legs[step]),
                remaining_horizon_fraction=float(horizon[step]),
                frozen_action=frozen[step],
                proposed_action=proposed[step],
                previous_action=previous_action,
                previous_energy_fraction=previous_energy,
                previous_filter_intervention=previous_filter,
                intervention_descriptor=descriptor,
            )
            if step == 0:
                reconstruction_errors.append(float(np.max(np.abs(position - task.start_position))))
            previous_action = proposed[step]
            previous_energy = float(energy[step])
            previous_filter = float(filter_norm[step])
        next_rollout_context = np.zeros_like(rollout_context)
        next_rollout_context[:-1] = rollout_context[1:]
        suffix = np.cumsum(energy[::-1], dtype=np.float64)[::-1].astype(np.float32)
        selected = np.unique(
            np.linspace(0, length - 1, min(length, transitions_per_rollout), dtype=np.int64)
        )
        is_last = selected == length - 1
        success = bool(metadata["mission_success"])
        contexts.append(rollout_context[selected])
        next_contexts.append(next_rollout_context[selected])
        suffix_energy.append(suffix[selected])
        realized_energy.append(energy[selected])
        terminal_success.append(is_last & success)
        terminal_failure.append(is_last & (not success))
        scenes.append(np.full(selected.size, scene, dtype=np.int64))
        interventions.append(np.full(selected.size, intervention, dtype=np.int8))
        transition_indices.append(selected)
        rollout_count += 1
    if rollout_count != num_scenes * 5:
        raise RuntimeError(f"expected {num_scenes * 5} rollouts, found {rollout_count}")
    payload = {
        "context": np.concatenate(contexts),
        "next_context": np.concatenate(next_contexts),
        "suffix_energy": np.concatenate(suffix_energy),
        "realized_energy": np.concatenate(realized_energy),
        "terminal_success": np.concatenate(terminal_success),
        "terminal_failure": np.concatenate(terminal_failure),
        "scene_index": np.concatenate(scenes),
        "intervention_index": np.concatenate(interventions),
        "transition_index": np.concatenate(transition_indices),
    }
    save_npz_atomic(cache_path, **payload)
    return {
        "num_rollouts": rollout_count,
        "num_transitions": int(payload["scene_index"].size),
        "maximum_initial_position_reconstruction_error": float(max(reconstruction_errors)),
        "capacity": capacity,
        "d_max": d_max,
        "world_extent": extent,
        "horizontal_v_max": horizontal_v_max,
        "vertical_v_max": vertical_v_max,
        "source_result_sha256": sha256(result_path),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as arrays:
        return {key: arrays[key] for key in arrays.files}


def variant_view(
    context: np.ndarray,
    *,
    variant: str,
    mean: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant == "geometry":
        selected = GEOMETRY_FEATURES
    elif variant in {"no_action", "no_bellman", "full"}:
        selected = np.arange(context.shape[1], dtype=np.int64)
    else:
        raise ValueError(f"unknown variant: {variant}")
    values = np.asarray(context[:, selected], dtype=np.float32)
    if mean is None or scale is None:
        mean = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
        scale = np.where(scale < 1e-5, 1.0, scale).astype(np.float32)
    normalized = (values - mean) / scale
    if variant == "no_action":
        positions = np.flatnonzero(np.isin(selected, CURRENT_ACTION_CONDITIONING))
        normalized[:, positions] = 0.0
    return normalized.astype(np.float32), mean, scale


def query_labels(
    suffix_energy: np.ndarray,
    terminal_mission_success: np.ndarray,
    budgets: np.ndarray,
) -> np.ndarray:
    return (
        terminal_mission_success[:, None]
        & (suffix_energy[:, None] <= budgets[None, :] + 1e-9)
    ).astype(np.float32)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with exact average ranks for tied scores."""

    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


def probabilities(
    model: MonotoneBudgetCritic,
    context: np.ndarray,
    budgets: np.ndarray,
    *,
    device: torch.device,
    temperature: float = 1.0,
    chunk: int = 65_536,
) -> np.ndarray:
    rows = context.shape[0]
    tiled_context = np.repeat(context, budgets.size, axis=0)
    tiled_budget = np.tile(budgets, rows).astype(np.float32)
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tiled_budget.size, chunk):
            end = min(start + chunk, tiled_budget.size)
            batch_context = torch.as_tensor(tiled_context[start:end], device=device)
            batch_budget = torch.as_tensor(tiled_budget[start:end], device=device)
            value = model(batch_context, batch_budget)
            if temperature != 1.0:
                value = torch.sigmoid(torch.logit(value.clamp(1e-6, 1 - 1e-6)) / temperature)
            predictions.append(value.cpu().numpy())
    return np.concatenate(predictions).reshape(rows, budgets.size)


def calibrate_temperature(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.reshape(-1).astype(np.float64)
    logits = np.log(np.clip(predictions.reshape(-1), 1e-6, 1 - 1e-6)) - np.log1p(
        -np.clip(predictions.reshape(-1), 1e-6, 1 - 1e-6)
    )
    candidates = np.geomspace(0.35, 3.0, 41)
    scores = []
    for temperature in candidates:
        calibrated = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
        scores.append(float(np.mean((calibrated - labels) ** 2)))
    return float(candidates[int(np.argmin(scores))])


def train_one(
    *,
    variant: str,
    context: np.ndarray,
    next_context: np.ndarray,
    suffix_energy: np.ndarray,
    realized_energy: np.ndarray,
    mission_success: np.ndarray,
    terminal_success: np.ndarray,
    terminal_failure: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    budgets: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[MonotoneBudgetCritic, np.ndarray, np.ndarray, float, dict[str, Any]]:
    train_context, mean, scale = variant_view(context[train_mask], variant=variant)
    validation_context, _, _ = variant_view(
        context[validation_mask], variant=variant, mean=mean, scale=scale
    )
    train_next, _, _ = variant_view(
        next_context[train_mask], variant=variant, mean=mean, scale=scale
    )
    model = MonotoneBudgetCritic(train_context.shape[1]).to(device)
    target_model = copy.deepcopy(model).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    train_indices = np.arange(train_context.shape[0])
    train_success = mission_success[train_mask]
    validation_labels = query_labels(
        suffix_energy[validation_mask], mission_success[validation_mask], budgets
    )
    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    bellman_weight = 0.0 if variant == "no_bellman" else 0.5
    for epoch in range(1, epochs + 1):
        model.train()
        target_model.load_state_dict(model.state_dict())
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, order.size, batch_size):
            rows = order[start : start + batch_size]
            selected_budget = rng.choice(budgets, size=rows.size).astype(np.float32)
            labels = (
                train_success[rows]
                & (suffix_energy[train_mask][rows] <= selected_budget + 1e-9)
            ).astype(np.float32)
            x = torch.as_tensor(train_context[rows], device=device)
            nx = torch.as_tensor(train_next[rows], device=device)
            b = torch.as_tensor(selected_budget, device=device)
            e = torch.as_tensor(realized_energy[train_mask][rows], device=device)
            success_terminal = torch.as_tensor(terminal_success[train_mask][rows], device=device)
            failure_terminal = torch.as_tensor(terminal_failure[train_mask][rows], device=device)
            label = torch.as_tensor(labels, device=device)
            prediction = model(x, b)
            monte_carlo_loss = functional.binary_cross_entropy(prediction, label)
            with torch.no_grad():
                next_probability = target_model(nx, b - e)
                target = bellman_target(
                    next_probability=next_probability,
                    budget=b,
                    realized_energy=e,
                    terminal_success=success_terminal,
                    terminal_failure=failure_terminal,
                )
            consistency = functional.mse_loss(prediction, target)
            loss = monte_carlo_loss + bellman_weight * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_prediction = probabilities(
            model, validation_context, budgets, device=device
        )
        validation_brier = float(np.mean((validation_prediction - validation_labels) ** 2))
        history.append(
            {"epoch": epoch, "loss": float(np.mean(losses)), "validation_brier": validation_brier}
        )
        if validation_brier < best_score - 1e-5:
            best_score = validation_brier
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = probabilities(model, validation_context, budgets, device=device)
    temperature = calibrate_temperature(validation_labels, validation_prediction)
    return model, mean, scale, temperature, {
        "best_epoch": best_epoch,
        "stopped_epoch": history[-1]["epoch"],
        "best_validation_brier": best_score,
        "temperature": temperature,
        "history": history,
    }


def scene_averaged_brier(
    labels: np.ndarray, predictions: np.ndarray, scenes: np.ndarray
) -> float:
    per_scene = []
    for scene in np.unique(scenes):
        mask = scenes == scene
        per_scene.append(float(np.mean((predictions[mask] - labels[mask]) ** 2)))
    return float(np.mean(per_scene))


def evaluate_one(
    *,
    model: MonotoneBudgetCritic,
    variant: str,
    mean: np.ndarray,
    scale: np.ndarray,
    temperature: float,
    arrays: dict[str, np.ndarray],
    mask: np.ndarray,
    budgets: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    context, _, _ = variant_view(arrays["context"][mask], variant=variant, mean=mean, scale=scale)
    next_context, _, _ = variant_view(
        arrays["next_context"][mask], variant=variant, mean=mean, scale=scale
    )
    labels = query_labels(
        arrays["suffix_energy"][mask], arrays["mission_success"][mask], budgets
    )
    prediction = probabilities(
        model, context, budgets, device=device, temperature=temperature
    )
    flat_metrics = probability_metrics(labels.reshape(-1), prediction.reshape(-1))
    auroc = binary_auroc(labels.reshape(-1), prediction.reshape(-1))
    next_budget = budgets[None, :] - arrays["realized_energy"][mask, None]
    tiled_next = np.repeat(next_context, budgets.size, axis=0)
    flat_next_budget = next_budget.reshape(-1).astype(np.float32)
    next_prediction = []
    model.eval()
    with torch.no_grad():
        for start in range(0, flat_next_budget.size, 65_536):
            end = min(start + 65_536, flat_next_budget.size)
            p = model(
                torch.as_tensor(tiled_next[start:end], device=device),
                torch.as_tensor(flat_next_budget[start:end], device=device),
            )
            if temperature != 1.0:
                p = torch.sigmoid(torch.logit(p.clamp(1e-6, 1 - 1e-6)) / temperature)
            next_prediction.append(p.cpu().numpy())
    next_prediction_array = np.concatenate(next_prediction).reshape(labels.shape)
    terminal_success = np.repeat(arrays["terminal_success"][mask, None], budgets.size, axis=1)
    terminal_failure = np.repeat(arrays["terminal_failure"][mask, None], budgets.size, axis=1)
    target = np.where(
        terminal_success,
        next_budget >= 0.0,
        np.where(terminal_failure, 0.0, np.where(next_budget >= 0.0, next_prediction_array, 0.0)),
    ).astype(np.float32)
    bellman_residual = float(np.mean((prediction - target) ** 2))
    monotone_violations = int(np.sum(np.diff(prediction, axis=1) < -1e-7))
    return {
        "brier": flat_metrics.brier,
        "scene_averaged_brier": scene_averaged_brier(
            labels, prediction, arrays["scene_index"][mask]
        ),
        "ece": flat_metrics.ece,
        "dangerous_false_safe_rate": flat_metrics.dangerous_false_safe_rate,
        "prevalence": flat_metrics.prevalence,
        "auroc": auroc,
        "bellman_residual": bellman_residual,
        "budget_monotonicity_violations": monotone_violations,
        "num_transitions": int(np.sum(mask)),
        "num_queries": int(labels.size),
    }


def action_sensitivity(
    *,
    model: MonotoneBudgetCritic,
    mean: np.ndarray,
    scale: np.ndarray,
    context: np.ndarray,
    scene: np.ndarray,
    intervention: np.ndarray,
    transition: np.ndarray,
    device: torch.device,
    budget: float = 0.20,
) -> float:
    values = []
    for scene_index in np.unique(scene):
        rows = np.flatnonzero((scene == scene_index) & (transition == 0))
        if rows.size < 2:
            continue
        nominal_rows = rows[intervention[rows] == 0]
        if nominal_rows.size != 1:
            continue
        counterfactual = np.repeat(context[nominal_rows[0]][None, :], rows.size, axis=0)
        counterfactual[:, 19:22] = context[rows, 19:22]
        normalized, _, _ = variant_view(counterfactual, variant="full", mean=mean, scale=scale)
        prediction = probabilities(
            model,
            normalized,
            np.asarray([budget], dtype=np.float32),
            device=device,
        ).reshape(-1)
        values.append(float(np.std(prediction)))
    return float(np.mean(values)) if values else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transitions-per-rollout", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=350_001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.transitions_per_rollout = min(args.transitions_per_rollout, 4)
        args.epochs = min(args.epochs, 2)
        args.patience = 2
    if min(args.transitions_per_rollout, args.epochs, args.patience, args.batch_size) <= 0:
        parser.error("sampling and optimization arguments must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.source = args.source.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.source / "RESULT.json", args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        (args.output_dir / "FAILED.json").unlink(missing_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    cache_path = args.output_dir / "transition_cache.npz"
    cache_summary_path = args.output_dir / "transition_cache.json"
    if not cache_path.is_file():
        cache_summary = build_transition_cache(
            args.source,
            cache_path,
            transitions_per_rollout=args.transitions_per_rollout,
        )
        atomic_json(cache_summary_path, cache_summary)
    else:
        cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8"))
    arrays = load_cache(cache_path)
    rollout_success = {}
    for path in args.source.glob("rollouts/scene_*/*.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        rollout_success[(int(metadata["scene_index"]), int(metadata["intervention_index"]))] = bool(
            metadata["mission_success"]
        )
    arrays["mission_success"] = np.asarray(
        [rollout_success[(int(scene), int(intervention))] for scene, intervention in zip(
            arrays["scene_index"], arrays["intervention_index"], strict=True
        )],
        dtype=np.bool_,
    )
    budgets = np.linspace(0.01, 0.70, 15, dtype=np.float32)
    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": not args.smoke,
        "source": args.source,
        "source_result_sha256": sha256(args.source / "RESULT.json"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256(args.protocol),
        "feature_names": FEATURE_NAMES,
        "budgets": budgets,
        "variants": VARIANTS,
        "transitions_per_rollout": args.transitions_per_rollout,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": args.device,
        "cache": cache_summary,
        "started_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    fold_results = []
    started = time.time()
    try:
        for fold in range(3):
            train_mask, validation_mask, test_mask = grouped_nested_split(
                arrays["scene_index"], outer_fold=fold
            )
            split_scenes = {
                "train": sorted(np.unique(arrays["scene_index"][train_mask]).tolist()),
                "validation": sorted(np.unique(arrays["scene_index"][validation_mask]).tolist()),
                "test": sorted(np.unique(arrays["scene_index"][test_mask]).tolist()),
            }
            for variant_index, variant in enumerate(VARIANTS):
                unit_path = args.output_dir / "folds" / f"fold_{fold}_{variant}.json"
                checkpoint_path = args.output_dir / "checkpoints" / f"fold_{fold}_{variant}.pt"
                if args.resume and unit_path.is_file() and checkpoint_path.is_file():
                    fold_results.append(json.loads(unit_path.read_text(encoding="utf-8")))
                    continue
                torch.manual_seed(args.seed + fold * 100 + variant_index)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + fold * 100 + variant_index)
                model, mean, scale, temperature, training = train_one(
                    variant=variant,
                    context=arrays["context"],
                    next_context=arrays["next_context"],
                    suffix_energy=arrays["suffix_energy"],
                    realized_energy=arrays["realized_energy"],
                    mission_success=arrays["mission_success"],
                    terminal_success=arrays["terminal_success"],
                    terminal_failure=arrays["terminal_failure"],
                    train_mask=train_mask,
                    validation_mask=validation_mask,
                    budgets=budgets,
                    device=device,
                    seed=args.seed + fold * 1000 + variant_index,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                metrics = evaluate_one(
                    model=model,
                    variant=variant,
                    mean=mean,
                    scale=scale,
                    temperature=temperature,
                    arrays=arrays,
                    mask=test_mask,
                    budgets=budgets,
                    device=device,
                )
                sensitivity = (
                    action_sensitivity(
                        model=model,
                        mean=mean,
                        scale=scale,
                        context=arrays["context"],
                        scene=arrays["scene_index"],
                        intervention=arrays["intervention_index"],
                        transition=arrays["transition_index"],
                        device=device,
                    )
                    if variant in {"full", "no_bellman"}
                    else 0.0
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_checkpoint = checkpoint_path.with_suffix(".tmp.pt")
                torch.save(
                    {
                        "protocol": PROTOCOL,
                        "variant": variant,
                        "fold": fold,
                        "model_state_dict": model.state_dict(),
                        "context_dim": model.context_dim,
                        "mean": mean,
                        "scale": scale,
                        "temperature": temperature,
                        "feature_indices": (
                            GEOMETRY_FEATURES
                            if variant == "geometry"
                            else np.arange(len(FEATURE_NAMES), dtype=np.int64)
                        ),
                    },
                    temporary_checkpoint,
                )
                temporary_checkpoint.replace(checkpoint_path)
                unit = {
                    "fold": fold,
                    "variant": variant,
                    "split_scenes": split_scenes,
                    "training": training,
                    "metrics": metrics,
                    "matched_initial_action_sensitivity": sensitivity,
                    "checkpoint": checkpoint_path,
                }
                atomic_json(unit_path, unit)
                fold_results.append(unit)
                atomic_json(
                    args.output_dir / "PROGRESS.json",
                    {
                        "completed_units": len(fold_results),
                        "total_units": 12,
                        "last_fold": fold,
                        "last_variant": variant,
                        "wall_clock_seconds": time.time() - started,
                    },
                )
        aggregate = {}
        for variant in VARIANTS:
            rows = [row for row in fold_results if row["variant"] == variant]
            aggregate[variant] = {
                key: float(np.mean([row["metrics"][key] for row in rows]))
                for key in (
                    "brier",
                    "scene_averaged_brier",
                    "ece",
                    "dangerous_false_safe_rate",
                    "prevalence",
                    "auroc",
                    "bellman_residual",
                    "budget_monotonicity_violations",
                )
            }
            aggregate[variant]["matched_initial_action_sensitivity"] = float(
                np.mean([row["matched_initial_action_sensitivity"] for row in rows])
            )
        full_brier = aggregate["full"]["scene_averaged_brier"]
        geometry_brier = aggregate["geometry"]["scene_averaged_brier"]
        fold_wins = sum(
            next(row for row in fold_results if row["fold"] == fold and row["variant"] == "full")[
                "metrics"
            ]["scene_averaged_brier"]
            < next(
                row
                for row in fold_results
                if row["fold"] == fold and row["variant"] == "geometry"
            )["metrics"]["scene_averaged_brier"]
            for fold in range(3)
        )
        checks = {
            "position_reconstruction_at_1e_2": cache_summary[
                "maximum_initial_position_reconstruction_error"
            ]
            <= 1e-2,
            "pooled_full_brier_beats_geometry_by_2_percent": full_brier
            <= 0.98 * geometry_brier,
            "full_beats_geometry_on_two_of_three_folds": fold_wins >= 2,
            "full_bellman_residual_beats_no_bellman": aggregate["full"][
                "bellman_residual"
            ]
            < aggregate["no_bellman"]["bellman_residual"],
            "matched_action_sensitivity_nonzero": aggregate["full"][
                "matched_initial_action_sensitivity"
            ]
            > 1e-5,
            "budget_monotonicity_exact": aggregate["full"][
                "budget_monotonicity_violations"
            ]
            == 0.0,
        }
        promotable = bool(all(checks.values()) and not args.smoke)
        final = {
            **manifest,
            "status": (
                "PROMOTE_TO_LIDAR_FORKED_ACTION_COLLECTION"
                if promotable
                else "DO_NOT_START_ACTOR_TRAINING"
            ),
            "promotable": promotable,
            "checks": checks,
            "aggregate": aggregate,
            "fold_results": fold_results,
            "finished_unix": time.time(),
            "wall_clock_seconds": time.time() - started,
        }
        atomic_json(args.output_dir / "RESULT.json", final)
        atomic_json(
            args.output_dir / "COMPLETED.json",
            {"status": final["status"], "result": args.output_dir / "RESULT.json"},
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        atomic_json(
            args.output_dir / "FAILED.json",
            {"type": type(error).__name__, "message": str(error), "failed_unix": time.time()},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
