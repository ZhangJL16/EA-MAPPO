from __future__ import annotations

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
from torch.nn import functional

from experiments.rechargeability_safety.core import (
    ConservativeHazardResidualCritic,
    MonotoneBudgetCritic,
)
from scripts.run_forked_lidar_residual_critic_gate import (
    apply_normalization,
    atomic_json,
    fit_standard,
    labels_for_budgets,
    metrics,
    normalize,
    standard_probabilities,
)


PROTOCOL = "CONSERVATIVE_HAZARD_FUSION_GATE_V1"
VARIANTS = (
    "geometry_action",
    "lidar_action_direct",
    "geometry_direct_intersection",
    "geometry_only_hazard",
    "lidar_action_hazard",
)


def four_way_scene_split(scene_indices: np.ndarray, *, outer_fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scenes = np.asarray(scene_indices, dtype=np.int64)
    if scenes.ndim != 1 or not 0 <= outer_fold < 5:
        raise ValueError("invalid scene split request")
    rank = np.searchsorted(np.unique(scenes), scenes)
    role = rank % 5
    test = role == outer_fold
    calibration = role == (outer_fold + 1) % 5
    validation = role == (outer_fold + 2) % 5
    train = ~(test | calibration | validation)
    return train, validation, calibration, test


def hazard_probabilities(
    model: ConservativeHazardResidualCritic,
    geometry: np.ndarray,
    hazard: np.ndarray,
    budgets: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    rows = np.repeat(np.arange(geometry.shape[0]), budgets.size)
    query_budgets = np.tile(budgets, geometry.shape[0])
    output = []
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, 65536):
            end = min(start + 65536, rows.size)
            output.append(
                model(
                    torch.as_tensor(geometry[rows[start:end]], device=device),
                    torch.as_tensor(hazard[rows[start:end]], device=device),
                    torch.as_tensor(query_budgets[start:end], device=device),
                ).cpu().numpy()
            )
    return np.concatenate(output).reshape(geometry.shape[0], budgets.size)


def fit_hazard(
    geometry_model: MonotoneBudgetCritic,
    geometry_temperature: float,
    train_geometry: np.ndarray,
    validation_geometry: np.ndarray,
    train_hazard: np.ndarray,
    validation_hazard: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    budgets: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
) -> tuple[ConservativeHazardResidualCritic, dict[str, object]]:
    torch.manual_seed(seed)
    model = ConservativeHazardResidualCritic(
        copy.deepcopy(geometry_model),
        train_hazard.shape[1],
        hidden_dim=96,
        geometry_temperature=geometry_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.hazard_network.parameters(), lr=2e-3, weight_decay=1e-5)
    rows = np.repeat(np.arange(train_geometry.shape[0]), budgets.size)
    query_budgets = np.tile(budgets, train_geometry.shape[0]).astype(np.float32)
    flat_labels = train_labels.reshape(-1)
    rng = np.random.default_rng(seed)
    initial = hazard_probabilities(model, validation_geometry, validation_hazard, budgets, device)
    best = float(np.mean((initial - validation_labels) ** 2))
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale = 0
    epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(rows.size)
        for start in range(0, order.size, 4096):
            query = order[start : start + 4096]
            logits = model.logits(
                torch.as_tensor(train_geometry[rows[query]], device=device),
                torch.as_tensor(train_hazard[rows[query]], device=device),
                torch.as_tensor(query_budgets[query], device=device),
            )
            loss = functional.binary_cross_entropy_with_logits(
                logits, torch.as_tensor(flat_labels[query], device=device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        prediction = hazard_probabilities(model, validation_geometry, validation_hazard, budgets, device)
        score = float(np.mean((prediction - validation_labels) ** 2))
        if score < best - 1e-6:
            best = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "stopped_epoch": epoch, "validation_brier": best}


def per_scene_false_safe(
    labels: np.ndarray,
    prediction: np.ndarray,
    scenes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    values = []
    for scene in np.unique(scenes):
        mask = scenes == scene
        infeasible = labels[mask] < 0.5
        values.append(
            float(np.mean((prediction[mask] >= threshold)[infeasible]))
            if np.any(infeasible)
            else 0.0
        )
    return np.asarray(values, dtype=np.float64)


def conformal_risk_threshold(
    labels: np.ndarray,
    prediction: np.ndarray,
    scenes: np.ndarray,
    *,
    alpha: float,
    grid_size: int = 2001,
) -> dict[str, float | int]:
    unique_scenes = np.unique(scenes)
    if unique_scenes.size < 2 or not 0.0 < alpha < 1.0:
        raise ValueError("CRC needs multiple scenes and a proper risk target")
    thresholds = np.concatenate(
        [np.linspace(0.0, 1.0, grid_size, dtype=np.float64), [np.inf]]
    )
    selected = thresholds[-1]
    empirical = 0.0
    corrected = 1.0 / (unique_scenes.size + 1)
    for threshold in thresholds:
        losses = per_scene_false_safe(labels, prediction, scenes, float(threshold))
        risk = float(np.mean(losses))
        upper = unique_scenes.size / (unique_scenes.size + 1) * risk + 1.0 / (unique_scenes.size + 1)
        if upper <= alpha:
            selected, empirical, corrected = float(threshold), risk, upper
            break
    return {
        "threshold": selected,
        "empirical_scene_false_safe": empirical,
        "corrected_risk": corrected,
        "calibration_scenes": int(unique_scenes.size),
    }


def load_combined(paths: list[Path]) -> dict[str, np.ndarray]:
    chunks = []
    scene_sets = []
    anchor_offset = 0
    for path in paths:
        with np.load(path) as payload:
            item = {key: payload[key] for key in payload.files}
        scenes = set(map(int, np.unique(item["scene"])))
        if any(scenes & prior for prior in scene_sets):
            raise RuntimeError("combined datasets contain overlapping scenes")
        scene_sets.append(scenes)
        item["anchor"] = item["anchor"].astype(np.int64) + anchor_offset
        anchor_offset = int(item["anchor"].max()) + 1
        chunks.append(item)
    common = set.intersection(*(set(item) for item in chunks))
    arrays = {key: np.concatenate([item[key] for item in chunks]) for key in common}
    if len(np.unique(arrays["scene"])) != 150 or arrays["geometry"].shape[0] != 6000:
        raise RuntimeError("hazard development Gate requires 150 scenes and 6,000 branches")
    return arrays


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=410001)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if len(args.dataset) != 2:
        parser.error("exactly two disjoint datasets are required")
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 2)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.dataset = [path.expanduser().resolve() for path in args.dataset]
    args.protocol = args.protocol.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    for path in [*args.dataset, args.protocol]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    arrays = load_combined(args.dataset)
    budgets = np.linspace(0.01, 0.70, 15, dtype=np.float32)
    labels = labels_for_budgets(arrays["safe"], arrays["energy"], budgets)
    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": False,
        "datasets": args.dataset,
        "protocol_document": args.protocol,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
        "started_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    started = time.time()
    try:
        oof = {name: np.full(labels.shape, np.nan, dtype=np.float32) for name in VARIANTS}
        accepted = {name: np.zeros(labels.shape, dtype=np.bool_) for name in VARIANTS}
        folds = []
        for fold in range(5):
            train, validation, calibration, test = four_way_scene_split(arrays["scene"], outer_fold=fold)
            geometry_train, geometry_mean, geometry_scale = normalize(arrays["geometry"][train])
            geometry_validation = apply_normalization(arrays["geometry"][validation], geometry_mean, geometry_scale)
            geometry_calibration = apply_normalization(arrays["geometry"][calibration], geometry_mean, geometry_scale)
            geometry_test = apply_normalization(arrays["geometry"][test], geometry_mean, geometry_scale)
            embedding_train, embedding_mean, embedding_scale = normalize(arrays["embedding"][train])
            embedding_validation = apply_normalization(arrays["embedding"][validation], embedding_mean, embedding_scale)
            embedding_calibration = apply_normalization(arrays["embedding"][calibration], embedding_mean, embedding_scale)
            embedding_test = apply_normalization(arrays["embedding"][test], embedding_mean, embedding_scale)
            full_train = np.concatenate([geometry_train, embedding_train], axis=1)
            full_validation = np.concatenate([geometry_validation, embedding_validation], axis=1)
            full_calibration = np.concatenate([geometry_calibration, embedding_calibration], axis=1)
            full_test = np.concatenate([geometry_test, embedding_test], axis=1)
            y_train, y_validation, y_calibration, y_test = labels[train], labels[validation], labels[calibration], labels[test]

            geometry_model, geometry_temperature, geometry_training = fit_standard(
                geometry_train, geometry_validation, y_train, y_validation, budgets,
                device=device, seed=args.seed + 100 * fold, epochs=args.epochs, patience=args.patience,
            )
            direct_model, direct_temperature, direct_training = fit_standard(
                full_train, full_validation, y_train, y_validation, budgets,
                device=device, seed=args.seed + 100 * fold + 1, epochs=args.epochs, patience=args.patience,
            )
            geometry_hazard, geometry_hazard_training = fit_hazard(
                geometry_model, geometry_temperature, geometry_train, geometry_validation,
                geometry_train, geometry_validation, y_train, y_validation, budgets,
                device=device, seed=args.seed + 100 * fold + 2, epochs=args.epochs, patience=args.patience,
            )
            lidar_hazard, lidar_hazard_training = fit_hazard(
                geometry_model, geometry_temperature, geometry_train, geometry_validation,
                full_train, full_validation, y_train, y_validation, budgets,
                device=device, seed=args.seed + 100 * fold + 3, epochs=args.epochs, patience=args.patience,
            )
            geometry_predictions = {
                "calibration": standard_probabilities(geometry_model, geometry_calibration, budgets, device, geometry_temperature),
                "test": standard_probabilities(geometry_model, geometry_test, budgets, device, geometry_temperature),
            }
            direct_predictions = {
                "calibration": standard_probabilities(direct_model, full_calibration, budgets, device, direct_temperature),
                "test": standard_probabilities(direct_model, full_test, budgets, device, direct_temperature),
            }
            predictions = {
                "geometry_action": geometry_predictions,
                "lidar_action_direct": direct_predictions,
                "geometry_direct_intersection": {
                    split: np.minimum(geometry_predictions[split], direct_predictions[split])
                    for split in ("calibration", "test")
                },
                "geometry_only_hazard": {
                    "calibration": hazard_probabilities(geometry_hazard, geometry_calibration, geometry_calibration, budgets, device),
                    "test": hazard_probabilities(geometry_hazard, geometry_test, geometry_test, budgets, device),
                },
                "lidar_action_hazard": {
                    "calibration": hazard_probabilities(lidar_hazard, geometry_calibration, full_calibration, budgets, device),
                    "test": hazard_probabilities(lidar_hazard, geometry_test, full_test, budgets, device),
                },
            }
            training = {
                "geometry_action": geometry_training,
                "lidar_action_direct": direct_training,
                "geometry_direct_intersection": {"derived": True},
                "geometry_only_hazard": geometry_hazard_training,
                "lidar_action_hazard": lidar_hazard_training,
            }
            fold_variants = {}
            for name in VARIANTS:
                calibration_result = conformal_risk_threshold(
                    y_calibration, predictions[name]["calibration"], arrays["scene"][calibration], alpha=args.alpha
                )
                threshold = float(calibration_result["threshold"])
                test_prediction = predictions[name]["test"]
                test_losses = per_scene_false_safe(y_test, test_prediction, arrays["scene"][test], threshold)
                oof[name][test] = test_prediction
                accepted[name][test] = test_prediction >= threshold
                fold_variants[name] = {
                    "training": training[name],
                    "calibration": calibration_result,
                    "test_scene_false_safe": float(np.mean(test_losses)),
                    "test_scene_false_safe_std": float(np.std(test_losses, ddof=1)),
                    "test_coverage": float(np.mean(test_prediction >= threshold)),
                    "metrics_at_0_90": metrics(
                        y_test, test_prediction, arrays["scene"][test], arrays["anchor"][test], arrays["bucket"][test]
                    ),
                }
            folds.append({
                "fold": fold,
                "train_scenes": int(np.unique(arrays["scene"][train]).size),
                "validation_scenes": int(np.unique(arrays["scene"][validation]).size),
                "calibration_scenes": int(np.unique(arrays["scene"][calibration]).size),
                "test_scenes": int(np.unique(arrays["scene"][test]).size),
                "variants": fold_variants,
            })
            atomic_json(args.output_dir / "PROGRESS.json", {"completed_folds": fold + 1, "total_folds": 5, "wall_clock_seconds": time.time() - started})

        if any(not np.all(np.isfinite(value)) for value in oof.values()):
            raise RuntimeError("out-of-fold predictions are incomplete or non-finite")
        aggregate = {}
        for name in VARIANTS:
            report = metrics(labels, oof[name], arrays["scene"], arrays["anchor"], arrays["bucket"])
            report["crc_scene_false_safe"] = float(np.mean([
                np.mean(((accepted[name][arrays["scene"] == scene]) & (labels[arrays["scene"] == scene] < 0.5))[labels[arrays["scene"] == scene] < 0.5])
                if np.any(labels[arrays["scene"] == scene] < 0.5) else 0.0
                for scene in np.unique(arrays["scene"])
            ]))
            report["crc_coverage"] = float(np.mean(accepted[name]))
            aggregate[name] = report
        primary = aggregate["lidar_action_hazard"]
        geometry = aggregate["geometry_action"]
        fold_wins = sum(
            fold["variants"]["lidar_action_hazard"]["metrics_at_0_90"]["scene_averaged_brier"]
            < fold["variants"]["geometry_action"]["metrics_at_0_90"]["scene_averaged_brier"]
            for fold in folds
        )
        dominance_violations = int(np.sum(oof["lidar_action_hazard"] > oof["geometry_action"] + 1e-7))
        checks = {
            "hazard_brier_beats_geometry_by_2_percent": primary["scene_averaged_brier"] <= 0.98 * geometry["scene_averaged_brier"],
            "hazard_wins_three_folds": fold_wins >= 3,
            "probability_dominance_exact": dominance_violations == 0,
            "budget_monotonicity_exact": primary["budget_monotonicity_violations"] == 0,
            "fixed_threshold_danger_not_worse": primary["dangerous_false_safe_rate"] <= geometry["dangerous_false_safe_rate"] + 1e-12,
            "crc_test_risk_at_most_alpha": primary["crc_scene_false_safe"] <= args.alpha,
            "crc_coverage_beats_geometry_by_2_points": primary["crc_coverage"] >= geometry["crc_coverage"] + 0.02,
        }
        promotable = bool(all(checks.values()) and not args.smoke)
        final = {
            **manifest,
            "status": "PROMOTE_TO_FRESH_SCENE_CONFIRMATION" if promotable else "DO_NOT_COLLECT_FRESH_CONFIRMATION",
            "promotable": promotable,
            "checks": checks,
            "fold_wins": fold_wins,
            "dominance_violations": dominance_violations,
            "aggregate": aggregate,
            "folds": folds,
            "finished_unix": time.time(),
            "wall_clock_seconds": time.time() - started,
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
