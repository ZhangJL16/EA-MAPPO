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
from scripts.run_conservative_hazard_fusion_gate import per_scene_false_safe
from scripts.run_forked_lidar_residual_critic_gate import (
    apply_normalization,
    atomic_json,
    build_dataset,
    labels_for_budgets,
    metrics,
    save_npz_atomic,
    sha256_file,
    standard_probabilities,
)


PROTOCOL = "CERTIFIED_MEET_FUSION_CONFIRMATION_V2"


def paired_bootstrap(values: np.ndarray, *, seed: int, draws: int = 10_000) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("paired bootstrap requires a finite scene vector")
    rng = np.random.default_rng(seed)
    sampled = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        indices = rng.integers(0, data.size, size=(count, data.size))
        sampled[start : start + count] = np.mean(data[indices], axis=1)
    return {
        "scene_count": int(data.size),
        "draws": int(draws),
        "mean_difference": float(np.mean(data)),
        "lower_95": float(np.quantile(sampled, 0.025)),
        "upper_95": float(np.quantile(sampled, 0.975)),
    }


def selective_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    scenes: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | None], np.ndarray, np.ndarray]:
    scene_risk = per_scene_false_safe(labels, prediction, scenes, threshold)
    scene_coverage = np.asarray(
        [float(np.mean(prediction[scenes == scene] >= threshold)) for scene in np.unique(scenes)],
        dtype=np.float64,
    )
    accepted = prediction >= threshold
    infeasible = labels < 0.5
    result = {
        "threshold": float(threshold),
        "scene_false_safe": float(np.mean(scene_risk)),
        "coverage": float(np.mean(accepted)),
        "conditional_false_safe_given_infeasible": float(np.mean(accepted[infeasible])) if np.any(infeasible) else None,
        "unsafe_among_accepted": float(np.mean(infeasible[accepted])) if np.any(accepted) else None,
    }
    return result, scene_risk, scene_coverage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=450001)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("data_dir", "freeze_dir", "protocol", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = [
        args.data_dir / "RESULT.json",
        args.freeze_dir / "RESULT.json",
        args.freeze_dir / "frozen_meet.pt",
        args.protocol,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    data_result = json.loads((args.data_dir / "RESULT.json").read_text())
    freeze_result = json.loads((args.freeze_dir / "RESULT.json").read_text())
    if freeze_result.get("status") != "FROZEN_FOR_FRESH_CONFIRMATION":
        raise RuntimeError("development model was not frozen for confirmation")
    if freeze_result.get("protocol") != PROTOCOL:
        raise RuntimeError("frozen model protocol does not match evaluator protocol")
    if freeze_result.get("protocol_sha256") != sha256_file(args.protocol):
        raise RuntimeError("frozen protocol hash does not match the supplied protocol")
    if freeze_result.get("checkpoint_sha256") != sha256_file(args.freeze_dir / "frozen_meet.pt"):
        raise RuntimeError("frozen checkpoint hash mismatch")
    if not data_result.get("promotable", False):
        raise RuntimeError("fresh paired-action data Gate did not pass")
    if int(data_result.get("num_scenes", 0)) != 75 or int(data_result.get("num_branches", 0)) != 3000:
        raise RuntimeError("fresh confirmation requires exactly 75 scenes and 3,000 branches")
    source_result_path = Path(data_result["source"]) / "RESULT.json"
    source_result = json.loads(source_result_path.read_text())
    if int(source_result.get("task_seed", -1)) != 420001 or int(source_result.get("world_seed", -1)) != 430001:
        raise RuntimeError("fresh task/world seed contract was not honored")

    started = time.time()
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
    predictions = {"geometry_action": geometry_prediction, "certified_meet": meet_prediction}
    reports = {}
    risks = {}
    coverages = {}
    for name, prediction in predictions.items():
        report = metrics(labels, prediction, arrays["scene"], arrays["anchor"], arrays["bucket"])
        selective, scene_risk, scene_coverage = selective_metrics(
            labels, prediction, arrays["scene"], float(frozen["thresholds"][name]["threshold"])
        )
        report["selective"] = selective
        reports[name] = report
        risks[name] = scene_risk
        coverages[name] = scene_coverage
    intervals = {
        "coverage_meet_minus_geometry": paired_bootstrap(
            coverages["certified_meet"] - coverages["geometry_action"],
            seed=args.seed, draws=args.bootstrap_draws,
        ),
        "risk_meet_minus_geometry": paired_bootstrap(
            risks["certified_meet"] - risks["geometry_action"],
            seed=args.seed + 1, draws=args.bootstrap_draws,
        ),
    }
    primary = reports["certified_meet"]
    baseline = reports["geometry_action"]
    dominance_geometry = int(np.sum(meet_prediction > geometry_prediction + 1e-7))
    dominance_direct = int(np.sum(meet_prediction > direct_prediction + 1e-7))
    checks = {
        "fresh_data_gate_passed": True,
        "meet_probability_dominance_exact": dominance_geometry == 0 and dominance_direct == 0,
        "meet_budget_monotonicity_exact": primary["budget_monotonicity_violations"] == 0,
        "meet_test_risk_at_most_0_05": primary["selective"]["scene_false_safe"] <= 0.05,
        "meet_coverage_beats_geometry_by_2_points": primary["selective"]["coverage"] >= baseline["selective"]["coverage"] + 0.02,
        "paired_coverage_interval_lower_above_zero": intervals["coverage_meet_minus_geometry"]["lower_95"] > 0.0,
        "fixed_0_90_danger_not_worse": primary["dangerous_false_safe_rate"] <= baseline["dangerous_false_safe_rate"] + 1e-12,
    }
    promotable = bool(all(checks.values()) and not args.smoke)
    save_npz_atomic(
        args.output_dir / "predictions.npz",
        labels=labels,
        geometry_action=geometry_prediction,
        lidar_action_direct=direct_prediction,
        certified_meet=meet_prediction,
        scene=arrays["scene"],
        anchor=arrays["anchor"],
        bucket=arrays["bucket"],
    )
    result = {
        "status": (
            "SMOKE_COMPLETE_NOT_EVIDENCE"
            if args.smoke
            else "PROMOTE_TO_BOUNDED_CLOSED_LOOP_PILOT" if promotable
            else "DO_NOT_MODIFY_R3_ACTOR"
        ),
        "promotable": promotable,
        "formal_evidence": not args.smoke,
        "protocol": PROTOCOL,
        "checks": checks,
        "reports": reports,
        "paired_intervals": intervals,
        "dominance_violations": {"geometry": dominance_geometry, "direct": dominance_direct},
        "dataset": dataset_summary,
        "data_dir": args.data_dir,
        "data_result_sha256": sha256_file(args.data_dir / "RESULT.json"),
        "freeze_dir": args.freeze_dir,
        "freeze_result_sha256": sha256_file(args.freeze_dir / "RESULT.json"),
        "frozen_checkpoint_sha256": sha256_file(args.freeze_dir / "frozen_meet.pt"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "bootstrap_draws": args.bootstrap_draws,
        "wall_clock_seconds": time.time() - started,
        "finished_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    atomic_json(args.output_dir / "COMPLETED.json", {"status": result["status"], "result": args.output_dir / "RESULT.json"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
