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

from experiments.rechargeability_safety.core import MonotoneBudgetCritic
from experiments.rechargeability_safety.risk_control import select_rcps_threshold
from scripts.run_forked_lidar_residual_critic_gate import (
    apply_normalization,
    atomic_json,
    build_dataset,
    labels_for_budgets,
    save_npz_atomic,
    sha256_file,
    standard_probabilities,
)


PROTOCOL = "SCENE_GROUPED_RCPS_MEET_CALIBRATION_V1"


def scene_loss_table(
    labels: np.ndarray,
    prediction: np.ndarray,
    scenes: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    table = np.empty((np.unique(scenes).size, thresholds.size), dtype=np.float64)
    for row, scene in enumerate(np.unique(scenes)):
        mask = scenes == scene
        unsafe_scores = np.sort(np.asarray(prediction[mask][labels[mask] < 0.5], dtype=np.float64))
        if unsafe_scores.size == 0:
            table[row] = 0.0
        else:
            table[row] = (
                unsafe_scores.size
                - np.searchsorted(unsafe_scores, thresholds, side="left")
            ) / unsafe_scores.size
    return table


def coverage_curve(prediction: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    scores = np.sort(np.asarray(prediction, dtype=np.float64).reshape(-1))
    return (scores.size - np.searchsorted(scores, thresholds, side="left")) / scores.size


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scenes", type=int, default=450)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--grid-size", type=int, default=2001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("data_dir", "freeze_dir", "protocol", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_result_path = args.data_dir / "RESULT.json"
    freeze_result_path = args.freeze_dir / "RESULT.json"
    for path in (data_result_path, freeze_result_path, args.freeze_dir / "frozen_meet.pt", args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_result = json.loads(data_result_path.read_text(encoding="utf-8"))
    source_result_path = Path(data_result["source"]) / "RESULT.json"
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    freeze_result = json.loads(freeze_result_path.read_text(encoding="utf-8"))
    if not data_result.get("promotable", False):
        raise RuntimeError("paired-action causal data Gate did not pass")
    if int(data_result.get("num_scenes", 0)) != args.num_scenes:
        raise RuntimeError("calibration scene count does not match the frozen protocol")
    if int(source_result.get("task_seed", -1)) != args.task_seed or int(source_result.get("world_seed", -1)) != args.world_seed:
        raise RuntimeError("calibration task/world seed contract was not honored")
    if source_result.get("state_snapshot_format") != "simulator_pre_action_float32_v1":
        raise RuntimeError("calibration source lacks exact simulator state snapshots")
    if freeze_result.get("status") != "FROZEN_FOR_FRESH_CONFIRMATION":
        raise RuntimeError("meet model was not frozen before calibration")
    if freeze_result.get("checkpoint_sha256") != sha256_file(args.freeze_dir / "frozen_meet.pt"):
        raise RuntimeError("frozen meet checkpoint hash mismatch")

    started = time.time()
    device = torch.device(args.device)
    arrays, dataset_summary = build_dataset(args.data_dir, device=device, output=args.output_dir)
    frozen = torch.load(args.freeze_dir / "frozen_meet.pt", map_location=device, weights_only=False)
    budgets = np.asarray(frozen["budgets"], dtype=np.float32)
    labels = labels_for_budgets(arrays["safe"], arrays["energy"], budgets)
    geometry = apply_normalization(
        arrays["geometry"], np.asarray(frozen["geometry_mean"]), np.asarray(frozen["geometry_scale"])
    )
    embedding = apply_normalization(
        arrays["embedding"], np.asarray(frozen["embedding_mean"]), np.asarray(frozen["embedding_scale"])
    )
    full = np.concatenate([geometry, embedding], axis=1)
    geometry_model = MonotoneBudgetCritic(int(frozen["geometry_dim"]), hidden_dim=96).to(device)
    direct_model = MonotoneBudgetCritic(
        int(frozen["geometry_dim"]) + int(frozen["embedding_dim"]), hidden_dim=96
    ).to(device)
    geometry_model.load_state_dict(frozen["geometry_state_dict"])
    direct_model.load_state_dict(frozen["direct_state_dict"])
    geometry_prediction = standard_probabilities(
        geometry_model, geometry, budgets, device, float(frozen["geometry_temperature"])
    )
    direct_prediction = standard_probabilities(
        direct_model, full, budgets, device, float(frozen["direct_temperature"])
    )
    meet_prediction = np.minimum(geometry_prediction, direct_prediction)
    thresholds = np.concatenate(
        [np.linspace(0.0, 1.0, args.grid_size, dtype=np.float64), [np.inf]]
    )
    reports: dict[str, object] = {}
    saved: dict[str, np.ndarray] = {"thresholds": thresholds}
    for name, prediction in {
        "geometry_action": geometry_prediction,
        "certified_meet": meet_prediction,
    }.items():
        loss_table = scene_loss_table(labels, prediction, arrays["scene"], thresholds)
        coverages = coverage_curve(prediction, thresholds)
        selected, upper_bounds = select_rcps_threshold(
            loss_table, coverages, thresholds, alpha=args.alpha, delta=args.delta
        )
        reports[name] = {
            "threshold": selected.threshold,
            "empirical_scene_false_safe": selected.empirical_risk,
            "hbb_risk_upper_bound": selected.risk_upper_bound,
            "coverage": selected.coverage,
            "grid_index": selected.grid_index,
            "nonvacuous": selected.nonvacuous,
        }
        saved[f"{name}_scene_losses"] = loss_table
        saved[f"{name}_risk_upper_bounds"] = upper_bounds
        saved[f"{name}_coverages"] = coverages
    dominance_violations = {
        "geometry": int(np.sum(meet_prediction > geometry_prediction + 1e-7)),
        "direct": int(np.sum(meet_prediction > direct_prediction + 1e-7)),
    }
    primary = reports["certified_meet"]
    baseline = reports["geometry_action"]
    checks = {
        "exact_simulator_snapshots": True,
        "meet_probability_dominance_exact": all(value == 0 for value in dominance_violations.values()),
        "meet_certificate_nonvacuous": bool(primary["nonvacuous"]),
        "meet_hbb_upper_bound_at_most_alpha": float(primary["hbb_risk_upper_bound"]) <= args.alpha,
        "geometry_certificate_nonvacuous": bool(baseline["nonvacuous"]),
        "meet_coverage_beats_geometry_by_2_points": float(primary["coverage"]) >= float(baseline["coverage"]) + 0.02,
    }
    promotable = bool(all(checks.values()))
    save_npz_atomic(args.output_dir / "risk_curves.npz", **saved)
    result = {
        "status": "FROZEN_FOR_RCPS_CONFIRMATION" if promotable else "RCPS_CALIBRATION_NOT_PROMOTABLE",
        "promotable": promotable,
        "protocol": PROTOCOL,
        "alpha": args.alpha,
        "delta": args.delta,
        "grid_size": args.grid_size,
        "num_scenes": args.num_scenes,
        "reports": reports,
        "checks": checks,
        "dominance_violations": dominance_violations,
        "dataset": dataset_summary,
        "data_dir": args.data_dir,
        "data_result_sha256": sha256_file(data_result_path),
        "source_result_sha256": sha256_file(source_result_path),
        "freeze_dir": args.freeze_dir,
        "frozen_checkpoint_sha256": sha256_file(args.freeze_dir / "frozen_meet.pt"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "risk_curves_sha256": sha256_file(args.output_dir / "risk_curves.npz"),
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
