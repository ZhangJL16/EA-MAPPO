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
from stable_baselines3 import SAC
from torch.nn import functional

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.forked_action_safety.core import nearest_obstacle_direction
from experiments.rechargeability_safety.core import (
    MonotoneBudgetCritic,
    ResidualMonotoneBudgetCritic,
    grouped_nested_split,
    probability_metrics,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import environment_from_args


PROTOCOL = "LARGE_SCALE_LIDAR_RESIDUAL_CRITIC_GATE_V1"
VARIANTS = ("geometry_action", "lidar_no_action", "lidar_action_direct", "geometry_lidar_action_residual")
ACTION_FEATURE_SLICE = slice(17, 26)
GEOMETRY_NAMES = (
    "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z",
    "task_direction_x", "task_direction_y", "task_direction_z", "task_distance",
    "charger_direction_x", "charger_direction_y", "charger_direction_z", "charger_distance",
    "return_leg", "remaining_horizon", "nearest_obstacle_clearance",
    "action_x", "action_y", "action_z",
    "action_delta_x", "action_delta_y", "action_delta_z",
    "active_goal_alignment", "obstacle_alignment", "action_norm",
)


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
    temporary.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_anchor_lookup(
    anchor_document: dict[str, object],
    anchor_observations: np.ndarray,
) -> dict[str, tuple[int, dict[str, object]]]:
    """Bind anchor IDs to their canonical NPZ row, ignoring legacy bad indices."""

    items = list(anchor_document["anchors"])
    if len(items) != int(anchor_observations.shape[0]):
        raise RuntimeError("anchor metadata and observation row counts differ")
    lookup: dict[str, tuple[int, dict[str, object]]] = {}
    for canonical_index, item in enumerate(items):
        anchor = dict(item)
        anchor_id = str(anchor["anchor_id"])
        if anchor_id in lookup:
            raise RuntimeError(f"duplicate anchor ID: {anchor_id}")
        digest = hashlib.sha256(
            np.asarray(anchor_observations[canonical_index], dtype=np.float32).tobytes()
        ).hexdigest()
        if digest != str(anchor["anchor_observation_sha256"]):
            raise RuntimeError(f"anchor observation row mismatch: {anchor_id}")
        lookup[anchor_id] = (canonical_index, anchor)
    return lookup


def unit(delta: np.ndarray) -> tuple[np.ndarray, float]:
    delta = np.asarray(delta, dtype=np.float32)
    distance = float(np.linalg.norm(delta))
    return (np.zeros(3, dtype=np.float32) if distance <= 1e-8 else delta / distance), distance


def build_dataset(data_dir: Path, *, device: torch.device, output: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    gate_result = json.loads((data_dir / "RESULT.json").read_text(encoding="utf-8"))
    if not gate_result.get("promotable", False):
        raise RuntimeError("forked-action data Gate did not authorize critic fitting")
    source_result = json.loads((Path(gate_result["source"]) / "RESULT.json").read_text(encoding="utf-8"))
    capacity = float(source_result["capacity"])
    maximum_steps = int(source_result["max_policy_steps_per_leg"])
    environment_args, _ = reconstruct_environment_args(
        Path(source_result["artifact"]), device=str(device), seed=int(source_result["world_seed"])
    )
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    policy = SAC.load(Path(source_result["checkpoint"]), env=probe, device=str(device), print_system_info=False)
    with np.load(data_dir / "anchors.npz") as payload:
        anchor_observations = np.asarray(payload["observation"], dtype=np.float32)
        anchor_nominal = np.asarray(payload["nominal_action"], dtype=np.float32)
    if anchor_nominal.shape[0] != anchor_observations.shape[0]:
        raise RuntimeError("anchor observation and nominal-action row counts differ")
    encoder = copy.deepcopy(policy.policy.actor.features_extractor).to(device).eval()
    embedding_chunks = []
    with torch.no_grad():
        for start in range(0, anchor_observations.shape[0], 256):
            tensor = torch.as_tensor(anchor_observations[start : start + 256], device=device)
            embedding_chunks.append(encoder(tensor).cpu().numpy())
    anchor_embeddings = np.concatenate(embedding_chunks).astype(np.float32)
    del encoder, policy
    anchor_document = json.loads((data_dir / "anchors.json").read_text(encoding="utf-8"))
    anchors = canonical_anchor_lookup(anchor_document, anchor_observations)
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(data_dir.glob("branches/*/*.json"))]
    geometry = []
    embeddings = []
    safe = []
    energy = []
    scene = []
    anchor_indices = []
    buckets = []
    actions = []
    for row in rows:
        anchor_id = str(row["anchor_id"])
        if anchor_id not in anchors:
            raise RuntimeError(f"branch refers to unknown anchor: {anchor_id}")
        anchor_index, anchor = anchors[anchor_id]
        if int(row["scene_index"]) != int(anchor["scene_index"]):
            raise RuntimeError(f"branch scene does not match anchor: {anchor_id}")
        if str(row["anchor_observation_sha256"]) != str(anchor["anchor_observation_sha256"]):
            raise RuntimeError(f"branch observation digest does not match anchor: {anchor_id}")
        position = np.asarray(anchor["position"], dtype=np.float32)
        velocity = np.asarray(anchor["velocity"], dtype=np.float32)
        task_goal = np.asarray(anchor["task_goal"], dtype=np.float32)
        charger_goal = np.asarray(anchor["charger_goal"], dtype=np.float32)
        active_goal = np.asarray(anchor["active_goal"], dtype=np.float32)
        action = np.asarray(row["candidate_action"], dtype=np.float32)
        task_direction, task_distance = unit(task_goal - position)
        charger_direction, charger_distance = unit(charger_goal - position)
        active_direction, _ = unit(active_goal - position)
        obstacle_direction = nearest_obstacle_direction(position, list(anchor["obstacle_layout"]))
        elapsed = int(anchor["elapsed_policy_steps"])
        leg = int(anchor["leg"])
        horizon = (
            (2 * maximum_steps - elapsed) / (2 * maximum_steps)
            if leg == 0
            else (maximum_steps - elapsed) / (2 * maximum_steps)
        )
        geometry.append(
            np.concatenate(
                [
                    position / np.asarray([probe.length, probe.width, probe.height], dtype=np.float32),
                    velocity / np.asarray([probe.horizontal_v_max, probe.horizontal_v_max, probe.vertical_v_max], dtype=np.float32),
                    task_direction,
                    [task_distance / probe.d_max],
                    charger_direction,
                    [charger_distance / probe.d_max],
                    [float(leg), horizon, float(anchor["nearest_obstacle_clearance"]) / probe.lidar_max_range],
                    action,
                    action - anchor_nominal[anchor_index],
                    [float(np.dot(action, active_direction)), float(np.dot(action, obstacle_direction)), float(np.linalg.norm(action))],
                ]
            ).astype(np.float32)
        )
        embeddings.append(anchor_embeddings[anchor_index])
        safe.append(bool(row["safe_recharge"]))
        energy.append(float(row["total_realized_energy"]) / capacity)
        scene.append(int(row["scene_index"]))
        anchor_indices.append(anchor_index)
        buckets.append(str(row["distance_bucket"]))
        actions.append(action)
    probe.close()
    if len(rows) != int(gate_result["num_branches"]):
        raise RuntimeError("branch count does not match terminal data result")
    arrays = {
        "geometry": np.asarray(geometry, dtype=np.float32),
        "embedding": np.asarray(embeddings, dtype=np.float32),
        "safe": np.asarray(safe, dtype=np.bool_),
        "energy": np.asarray(energy, dtype=np.float32),
        "scene": np.asarray(scene, dtype=np.int64),
        "anchor": np.asarray(anchor_indices, dtype=np.int64),
        "bucket": np.asarray(buckets),
        "action": np.asarray(actions, dtype=np.float32),
    }
    save_npz_atomic(output / "dataset.npz", **arrays)
    summary = {
        "branches": len(rows),
        "anchors": int(np.unique(arrays["anchor"]).size),
        "scenes": int(np.unique(arrays["scene"]).size),
        "geometry_dim": arrays["geometry"].shape[1],
        "embedding_dim": arrays["embedding"].shape[1],
        "capacity": capacity,
        "unsafe_prevalence": float(np.mean(~arrays["safe"])),
        "data_result_sha256": sha256_file(data_dir / "RESULT.json"),
        "checkpoint": source_result["checkpoint"],
        "checkpoint_sha256": source_result["checkpoint_sha256"],
    }
    atomic_json(output / "dataset.json", summary)
    return arrays, summary


def load_dataset(output: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(output / "dataset.npz") as payload:
        arrays = {key: payload[key] for key in payload.files}
    return arrays, json.loads((output / "dataset.json").read_text(encoding="utf-8"))


def normalize(train: np.ndarray, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale < 1e-5, 1.0, scale).astype(np.float32)
    target = train if values is None else values
    return ((target - mean) / scale).astype(np.float32), mean, scale


def apply_normalization(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((values - mean) / scale).astype(np.float32)


def labels_for_budgets(safe: np.ndarray, energy: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    return (safe[:, None] & (energy[:, None] <= budgets[None, :] + 1e-9)).astype(np.float32)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def standard_probabilities(model, context: np.ndarray, budgets: np.ndarray, device: torch.device, temperature: float = 1.0) -> np.ndarray:
    rows = np.repeat(np.arange(context.shape[0]), budgets.size)
    query_budget = np.tile(budgets, context.shape[0])
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, 65536):
            end = min(start + 65536, rows.size)
            probability = model(
                torch.as_tensor(context[rows[start:end]], device=device),
                torch.as_tensor(query_budget[start:end], device=device),
            )
            if temperature != 1.0:
                probability = torch.sigmoid(torch.logit(probability.clamp(1e-6, 1 - 1e-6)) / temperature)
            output.append(probability.cpu().numpy())
    return np.concatenate(output).reshape(context.shape[0], budgets.size)


def residual_probabilities(model, geometry: np.ndarray, residual: np.ndarray, budgets: np.ndarray, device: torch.device, temperature: float = 1.0) -> np.ndarray:
    rows = np.repeat(np.arange(geometry.shape[0]), budgets.size)
    query_budget = np.tile(budgets, geometry.shape[0])
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, 65536):
            end = min(start + 65536, rows.size)
            probability = model(
                torch.as_tensor(geometry[rows[start:end]], device=device),
                torch.as_tensor(residual[rows[start:end]], device=device),
                torch.as_tensor(query_budget[start:end], device=device),
            )
            if temperature != 1.0:
                probability = torch.sigmoid(torch.logit(probability.clamp(1e-6, 1 - 1e-6)) / temperature)
            output.append(probability.cpu().numpy())
    return np.concatenate(output).reshape(geometry.shape[0], budgets.size)


def select_temperature(labels: np.ndarray, prediction: np.ndarray) -> float:
    clipped = np.clip(prediction.reshape(-1), 1e-6, 1 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    target = labels.reshape(-1)
    candidates = np.geomspace(0.35, 3.0, 41)
    scores = [
        np.mean((1.0 / (1.0 + np.exp(-np.clip(logits / value, -40, 40))) - target) ** 2)
        for value in candidates
    ]
    return float(candidates[int(np.argmin(scores))])


def fit_standard(train_context: np.ndarray, validation_context: np.ndarray, train_labels: np.ndarray, validation_labels: np.ndarray, budgets: np.ndarray, *, device: torch.device, seed: int, epochs: int, patience: int) -> tuple[MonotoneBudgetCritic, float, dict[str, object]]:
    torch.manual_seed(seed)
    model = MonotoneBudgetCritic(train_context.shape[1], hidden_dim=96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    train_rows = np.repeat(np.arange(train_context.shape[0]), budgets.size)
    train_budgets = np.tile(budgets, train_context.shape[0]).astype(np.float32)
    flat_labels = train_labels.reshape(-1)
    rng = np.random.default_rng(seed)
    best = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows.size)
        for start in range(0, order.size, 4096):
            query = order[start : start + 4096]
            probability = model(
                torch.as_tensor(train_context[train_rows[query]], device=device),
                torch.as_tensor(train_budgets[query], device=device),
            )
            loss = functional.binary_cross_entropy(probability, torch.as_tensor(flat_labels[query], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        prediction = standard_probabilities(model, validation_context, budgets, device)
        score = float(np.mean((prediction - validation_labels) ** 2))
        if score < best - 1e-6:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("standard model did not produce a checkpoint")
    model.load_state_dict(best_state)
    raw = standard_probabilities(model, validation_context, budgets, device)
    temperature = select_temperature(validation_labels, raw)
    return model, temperature, {"best_epoch": best_epoch, "stopped_epoch": epoch, "validation_brier": best, "temperature": temperature}


def fit_residual(geometry_model: MonotoneBudgetCritic, train_geometry: np.ndarray, validation_geometry: np.ndarray, train_residual: np.ndarray, validation_residual: np.ndarray, train_labels: np.ndarray, validation_labels: np.ndarray, budgets: np.ndarray, *, device: torch.device, seed: int, epochs: int, patience: int) -> tuple[ResidualMonotoneBudgetCritic, float, dict[str, object]]:
    torch.manual_seed(seed)
    model = ResidualMonotoneBudgetCritic(copy.deepcopy(geometry_model), train_residual.shape[1], hidden_dim=96).to(device)
    optimizer = torch.optim.AdamW(model.residual_network.parameters(), lr=1e-3, weight_decay=1e-5)
    train_rows = np.repeat(np.arange(train_geometry.shape[0]), budgets.size)
    train_budgets = np.tile(budgets, train_geometry.shape[0]).astype(np.float32)
    flat_labels = train_labels.reshape(-1)
    rng = np.random.default_rng(seed)
    initial = residual_probabilities(model, validation_geometry, validation_residual, budgets, device)
    best = float(np.mean((initial - validation_labels) ** 2))
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(train_rows.size)
        for start in range(0, order.size, 4096):
            query = order[start : start + 4096]
            probability = model(
                torch.as_tensor(train_geometry[train_rows[query]], device=device),
                torch.as_tensor(train_residual[train_rows[query]], device=device),
                torch.as_tensor(train_budgets[query], device=device),
            )
            loss = functional.binary_cross_entropy(probability, torch.as_tensor(flat_labels[query], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        prediction = residual_probabilities(model, validation_geometry, validation_residual, budgets, device)
        score = float(np.mean((prediction - validation_labels) ** 2))
        if score < best - 1e-6:
            best, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    raw = residual_probabilities(model, validation_geometry, validation_residual, budgets, device)
    temperature = select_temperature(validation_labels, raw)
    return model, temperature, {"best_epoch": best_epoch, "stopped_epoch": epoch, "validation_brier": best, "temperature": temperature}


def metrics(labels: np.ndarray, prediction: np.ndarray, scenes: np.ndarray, anchors: np.ndarray, buckets: np.ndarray) -> dict[str, object]:
    flat = probability_metrics(labels.reshape(-1), prediction.reshape(-1))
    per_scene = [float(np.mean((prediction[scenes == scene] - labels[scenes == scene]) ** 2)) for scene in np.unique(scenes)]
    high_budget_prediction = prediction[:, -1]
    pair_scores = []
    for anchor in np.unique(anchors):
        mask = anchors == anchor
        safe_at_high = labels[mask, -1] > 0.5
        if np.any(safe_at_high) and np.any(~safe_at_high):
            pair_scores.extend(
                float(left > right)
                for left in high_budget_prediction[mask][safe_at_high]
                for right in high_budget_prediction[mask][~safe_at_high]
            )
    bucket_brier = {
        str(bucket): float(np.mean((prediction[buckets == bucket] - labels[buckets == bucket]) ** 2))
        for bucket in np.unique(buckets)
    }
    return {
        "brier": flat.brier,
        "scene_averaged_brier": float(np.mean(per_scene)),
        "auroc": binary_auroc(labels, prediction),
        "ece": flat.ece,
        "dangerous_false_safe_rate": flat.dangerous_false_safe_rate,
        "prevalence": flat.prevalence,
        "budget_monotonicity_violations": int(np.sum(np.diff(prediction, axis=1) < -1e-7)),
        "mixed_anchor_pair_ranking_accuracy": float(np.mean(pair_scores)) if pair_scores else None,
        "mixed_anchor_pair_count": len(pair_scores),
        "bucket_brier": bucket_brier,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=370001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.data_dir = args.data_dir.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in (args.data_dir / "RESULT.json", args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    if args.resume and (args.output_dir / "dataset.npz").is_file():
        arrays, dataset_summary = load_dataset(args.output_dir)
    else:
        arrays, dataset_summary = build_dataset(args.data_dir, device=device, output=args.output_dir)
    budgets = np.linspace(0.01, 0.70, 15, dtype=np.float32)
    labels = labels_for_budgets(arrays["safe"], arrays["energy"], budgets)
    manifest = {
        "status": "RUNNING", "protocol": PROTOCOL, "formal_evidence": False,
        "data_dir": args.data_dir, "data_result_sha256": sha256_file(args.data_dir / "RESULT.json"),
        "protocol_document": args.protocol, "protocol_sha256": sha256_file(args.protocol),
        "variants": VARIANTS, "budgets": budgets, "epochs": args.epochs,
        "patience": args.patience, "seed": args.seed, "device": args.device,
        "dataset": dataset_summary, "started_unix": time.time(), "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    fold_results = []
    started = time.time()
    try:
        for fold in range(3):
            fold_path = args.output_dir / "folds" / f"fold_{fold}.json"
            checkpoint_path = args.output_dir / "checkpoints" / f"fold_{fold}.pt"
            if args.resume and fold_path.is_file() and checkpoint_path.is_file():
                fold_results.append(json.loads(fold_path.read_text(encoding="utf-8")))
                continue
            train_mask, validation_mask, test_mask = grouped_nested_split(arrays["scene"], outer_fold=fold)
            geometry_train, geometry_mean, geometry_scale = normalize(arrays["geometry"][train_mask])
            geometry_validation = apply_normalization(arrays["geometry"][validation_mask], geometry_mean, geometry_scale)
            geometry_test = apply_normalization(arrays["geometry"][test_mask], geometry_mean, geometry_scale)
            embedding_train, embedding_mean, embedding_scale = normalize(arrays["embedding"][train_mask])
            embedding_validation = apply_normalization(arrays["embedding"][validation_mask], embedding_mean, embedding_scale)
            embedding_test = apply_normalization(arrays["embedding"][test_mask], embedding_mean, embedding_scale)
            no_action_train = geometry_train.copy(); no_action_train[:, ACTION_FEATURE_SLICE] = 0.0
            no_action_validation = geometry_validation.copy(); no_action_validation[:, ACTION_FEATURE_SLICE] = 0.0
            no_action_test = geometry_test.copy(); no_action_test[:, ACTION_FEATURE_SLICE] = 0.0
            full_train = np.concatenate([geometry_train, embedding_train], axis=1)
            full_validation = np.concatenate([geometry_validation, embedding_validation], axis=1)
            full_test = np.concatenate([geometry_test, embedding_test], axis=1)
            no_action_full_train = np.concatenate([no_action_train, embedding_train], axis=1)
            no_action_full_validation = np.concatenate([no_action_validation, embedding_validation], axis=1)
            no_action_full_test = np.concatenate([no_action_test, embedding_test], axis=1)
            train_labels, validation_labels, test_labels = labels[train_mask], labels[validation_mask], labels[test_mask]
            trained = {}
            geometry_model, geometry_temperature, geometry_training = fit_standard(
                geometry_train, geometry_validation, train_labels, validation_labels, budgets,
                device=device, seed=args.seed + fold * 100, epochs=args.epochs, patience=args.patience,
            )
            trained["geometry_action"] = (geometry_model, geometry_temperature, geometry_training, geometry_test, None)
            no_action_model, no_action_temperature, no_action_training = fit_standard(
                no_action_full_train, no_action_full_validation, train_labels, validation_labels, budgets,
                device=device, seed=args.seed + fold * 100 + 1, epochs=args.epochs, patience=args.patience,
            )
            trained["lidar_no_action"] = (no_action_model, no_action_temperature, no_action_training, no_action_full_test, None)
            direct_model, direct_temperature, direct_training = fit_standard(
                full_train, full_validation, train_labels, validation_labels, budgets,
                device=device, seed=args.seed + fold * 100 + 2, epochs=args.epochs, patience=args.patience,
            )
            trained["lidar_action_direct"] = (direct_model, direct_temperature, direct_training, full_test, None)
            residual_model, residual_temperature, residual_training = fit_residual(
                geometry_model, geometry_train, geometry_validation, full_train, full_validation,
                train_labels, validation_labels, budgets, device=device, seed=args.seed + fold * 100 + 3,
                epochs=args.epochs, patience=args.patience,
            )
            trained["geometry_lidar_action_residual"] = (
                residual_model, residual_temperature, residual_training, geometry_test, full_test
            )
            variants = {}
            checkpoint_payload = {"fold": fold, "geometry_mean": geometry_mean, "geometry_scale": geometry_scale, "embedding_mean": embedding_mean, "embedding_scale": embedding_scale, "models": {}}
            for name, (model, temperature, training, first_context, second_context) in trained.items():
                prediction = (
                    standard_probabilities(model, first_context, budgets, device, temperature)
                    if second_context is None
                    else residual_probabilities(model, first_context, second_context, budgets, device, temperature)
                )
                variants[name] = {
                    "training": training,
                    "metrics": metrics(
                        test_labels, prediction, arrays["scene"][test_mask], arrays["anchor"][test_mask], arrays["bucket"][test_mask]
                    ),
                    "matched_action_sensitivity": float(np.mean([
                        np.std(prediction[arrays["anchor"][test_mask] == anchor, 4])
                        for anchor in np.unique(arrays["anchor"][test_mask])
                    ])),
                }
                checkpoint_payload["models"][name] = {"state_dict": model.state_dict(), "temperature": temperature}
            fold_result = {
                "fold": fold,
                "train_scenes": sorted(np.unique(arrays["scene"][train_mask]).tolist()),
                "validation_scenes": sorted(np.unique(arrays["scene"][validation_mask]).tolist()),
                "test_scenes": sorted(np.unique(arrays["scene"][test_mask]).tolist()),
                "variants": variants,
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp.pt")
            torch.save(checkpoint_payload, temporary); temporary.replace(checkpoint_path)
            atomic_json(fold_path, fold_result)
            fold_results.append(fold_result)
            atomic_json(args.output_dir / "PROGRESS.json", {"completed_folds": len(fold_results), "total_folds": 3, "wall_clock_seconds": time.time() - started})
        aggregate = {}
        for variant in VARIANTS:
            aggregate[variant] = {
                key: float(np.mean([fold["variants"][variant]["metrics"][key] for fold in fold_results]))
                for key in ("brier", "scene_averaged_brier", "auroc", "ece", "dangerous_false_safe_rate", "budget_monotonicity_violations")
            }
            aggregate[variant]["matched_action_sensitivity"] = float(np.mean([fold["variants"][variant]["matched_action_sensitivity"] for fold in fold_results]))
        primary = "geometry_lidar_action_residual"
        residual_brier = aggregate[primary]["scene_averaged_brier"]
        geometry_brier = aggregate["geometry_action"]["scene_averaged_brier"]
        fold_wins = sum(
            fold["variants"][primary]["metrics"]["scene_averaged_brier"] < fold["variants"]["geometry_action"]["metrics"]["scene_averaged_brier"]
            for fold in fold_results
        )
        checks = {
            "residual_brier_beats_geometry_by_2_percent": residual_brier <= 0.98 * geometry_brier,
            "residual_beats_geometry_on_two_folds": fold_wins >= 2,
            "budget_monotonicity_exact": aggregate[primary]["budget_monotonicity_violations"] == 0.0,
            "matched_action_sensitivity_nonzero": aggregate[primary]["matched_action_sensitivity"] > 1e-5,
            "dangerous_false_safe_not_worse": aggregate[primary]["dangerous_false_safe_rate"] <= aggregate["geometry_action"]["dangerous_false_safe_rate"] + 1e-12,
        }
        promotable = bool(all(checks.values()) and not args.smoke)
        final = {
            **manifest, "status": "PROMOTE_TO_CLOSED_LOOP_ACTOR_PILOT" if promotable else "DO_NOT_MODIFY_R3_ACTOR",
            "promotable": promotable, "checks": checks, "aggregate": aggregate,
            "fold_results": fold_results, "finished_unix": time.time(), "wall_clock_seconds": time.time() - started,
        }
        atomic_json(args.output_dir / "RESULT.json", final)
        atomic_json(args.output_dir / "COMPLETED.json", {"status": final["status"], "result": args.output_dir / "RESULT.json"})
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        atomic_json(args.output_dir / "FAILED.json", {"type": type(error).__name__, "message": str(error), "failed_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
