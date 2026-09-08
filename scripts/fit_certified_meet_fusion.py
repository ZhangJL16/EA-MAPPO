from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.run_conservative_hazard_fusion_gate import (
    conformal_risk_threshold,
    load_combined,
    per_scene_false_safe,
)
from scripts.run_forked_lidar_residual_critic_gate import (
    apply_normalization,
    atomic_json,
    fit_standard,
    labels_for_budgets,
    metrics,
    normalize,
    sha256_file,
    standard_probabilities,
)


PROTOCOL = "CERTIFIED_MEET_FUSION_CONFIRMATION_V2"


def development_split(scene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(scene, dtype=np.int64)
    unique = np.unique(values)
    rank = np.searchsorted(unique, values)
    role = rank % 5
    return role <= 2, role == 3, role == 4


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=440001)
    args = parser.parse_args(argv)
    if len(args.dataset) != 2:
        parser.error("exactly two disjoint development datasets are required")
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
    train, validation, calibration = development_split(arrays["scene"])
    role_counts = [int(np.unique(arrays["scene"][mask]).size) for mask in (train, validation, calibration)]
    if role_counts != [90, 30, 30]:
        raise RuntimeError(f"unexpected development split: {role_counts}")

    geometry_train, geometry_mean, geometry_scale = normalize(arrays["geometry"][train])
    geometry_validation = apply_normalization(arrays["geometry"][validation], geometry_mean, geometry_scale)
    geometry_calibration = apply_normalization(arrays["geometry"][calibration], geometry_mean, geometry_scale)
    embedding_train, embedding_mean, embedding_scale = normalize(arrays["embedding"][train])
    embedding_validation = apply_normalization(arrays["embedding"][validation], embedding_mean, embedding_scale)
    embedding_calibration = apply_normalization(arrays["embedding"][calibration], embedding_mean, embedding_scale)
    full_train = np.concatenate([geometry_train, embedding_train], axis=1)
    full_validation = np.concatenate([geometry_validation, embedding_validation], axis=1)
    full_calibration = np.concatenate([geometry_calibration, embedding_calibration], axis=1)

    started = time.time()
    geometry_model, geometry_temperature, geometry_training = fit_standard(
        geometry_train, geometry_validation, labels[train], labels[validation], budgets,
        device=device, seed=args.seed, epochs=args.epochs, patience=args.patience,
    )
    direct_model, direct_temperature, direct_training = fit_standard(
        full_train, full_validation, labels[train], labels[validation], budgets,
        device=device, seed=args.seed + 1, epochs=args.epochs, patience=args.patience,
    )
    geometry_prediction = standard_probabilities(
        geometry_model, geometry_calibration, budgets, device, geometry_temperature
    )
    direct_prediction = standard_probabilities(
        direct_model, full_calibration, budgets, device, direct_temperature
    )
    meet_prediction = np.minimum(geometry_prediction, direct_prediction)
    predictions = {"geometry_action": geometry_prediction, "certified_meet": meet_prediction}
    thresholds = {
        name: conformal_risk_threshold(
            labels[calibration], prediction, arrays["scene"][calibration], alpha=args.alpha
        )
        for name, prediction in predictions.items()
    }
    calibration_report = {}
    for name, prediction in predictions.items():
        threshold = float(thresholds[name]["threshold"])
        report = metrics(
            labels[calibration], prediction, arrays["scene"][calibration],
            arrays["anchor"][calibration], arrays["bucket"][calibration],
        )
        report["crc_scene_false_safe"] = float(np.mean(per_scene_false_safe(
            labels[calibration], prediction, arrays["scene"][calibration], threshold
        )))
        report["crc_coverage"] = float(np.mean(prediction >= threshold))
        calibration_report[name] = report

    dominance = int(np.sum(meet_prediction > geometry_prediction + 1e-7))
    checkpoint = {
        "protocol": PROTOCOL,
        "geometry_state_dict": geometry_model.state_dict(),
        "direct_state_dict": direct_model.state_dict(),
        "geometry_dim": int(geometry_train.shape[1]),
        "embedding_dim": int(embedding_train.shape[1]),
        "geometry_mean": geometry_mean,
        "geometry_scale": geometry_scale,
        "embedding_mean": embedding_mean,
        "embedding_scale": embedding_scale,
        "geometry_temperature": float(geometry_temperature),
        "direct_temperature": float(direct_temperature),
        "budgets": budgets,
        "thresholds": thresholds,
        "train_scenes": np.unique(arrays["scene"][train]),
        "validation_scenes": np.unique(arrays["scene"][validation]),
        "calibration_scenes": np.unique(arrays["scene"][calibration]),
    }
    checkpoint_path = args.output_dir / "frozen_meet.pt"
    torch.save(checkpoint, checkpoint_path)
    result = {
        "status": "FROZEN_FOR_FRESH_CONFIRMATION",
        "protocol": PROTOCOL,
        "formal_evidence": False,
        "datasets": args.dataset,
        "dataset_sha256": {str(path): sha256_file(path) for path in args.dataset},
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "role_scene_counts": role_counts,
        "alpha": args.alpha,
        "budgets": budgets,
        "thresholds": thresholds,
        "training": {"geometry_action": geometry_training, "lidar_action_direct": direct_training},
        "calibration": calibration_report,
        "structural_checks": {
            "meet_probability_dominance_exact": dominance == 0,
            "geometry_budget_monotone": calibration_report["geometry_action"]["budget_monotonicity_violations"] == 0,
            "meet_budget_monotone": calibration_report["certified_meet"]["budget_monotonicity_violations"] == 0,
        },
        "wall_clock_seconds": time.time() - started,
        "finished_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    if not all(result["structural_checks"].values()):
        result["status"] = "DO_NOT_LAUNCH_FRESH_CONFIRMATION"
    atomic_json(args.output_dir / "RESULT.json", result)
    atomic_json(args.output_dir / "COMPLETED.json", {"status": result["status"], "result": args.output_dir / "RESULT.json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
