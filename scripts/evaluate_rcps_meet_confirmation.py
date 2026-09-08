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
from scripts.evaluate_certified_meet_fusion import paired_bootstrap, selective_metrics
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


PROTOCOL = "SCENE_GROUPED_RCPS_MEET_CONFIRMATION_V1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scenes", type=int, default=150)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--world-seed", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=650001)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("data_dir", "freeze_dir", "calibration_dir", "protocol", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_result_path = args.data_dir / "RESULT.json"
    freeze_result_path = args.freeze_dir / "RESULT.json"
    calibration_result_path = args.calibration_dir / "RESULT.json"
    for path in (data_result_path, freeze_result_path, calibration_result_path, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_result = json.loads(data_result_path.read_text(encoding="utf-8"))
    freeze_result = json.loads(freeze_result_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_result_path.read_text(encoding="utf-8"))
    source_result_path = Path(data_result["source"]) / "RESULT.json"
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if not data_result.get("promotable", False):
        raise RuntimeError("confirmation paired-action data Gate did not pass")
    if int(data_result.get("num_scenes", 0)) != args.num_scenes:
        raise RuntimeError("confirmation scene count does not match protocol")
    if int(source_result.get("task_seed", -1)) != args.task_seed or int(source_result.get("world_seed", -1)) != args.world_seed:
        raise RuntimeError("confirmation task/world seed contract was not honored")
    if source_result.get("state_snapshot_format") != "simulator_pre_action_float32_v1":
        raise RuntimeError("confirmation source lacks exact simulator snapshots")
    if calibration.get("status") != "FROZEN_FOR_RCPS_CONFIRMATION" or not calibration.get("promotable", False):
        raise RuntimeError("RCPS calibration did not pass")
    if calibration.get("protocol_sha256") != sha256_file(args.protocol):
        raise RuntimeError("calibration protocol hash mismatch")
    if calibration.get("frozen_checkpoint_sha256") != sha256_file(args.freeze_dir / "frozen_meet.pt"):
        raise RuntimeError("calibration and evaluator model hashes differ")
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
    predictions = {"geometry_action": geometry_prediction, "certified_meet": meet_prediction}
    reports: dict[str, object] = {}
    risks: dict[str, np.ndarray] = {}
    coverages: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        report = metrics(labels, prediction, arrays["scene"], arrays["anchor"], arrays["bucket"])
        threshold = float(calibration["reports"][name]["threshold"])
        selective, scene_risk, scene_coverage = selective_metrics(
            labels, prediction, arrays["scene"], threshold
        )
        report["selective"] = selective
        reports[name] = report
        risks[name] = scene_risk
        coverages[name] = scene_coverage
    intervals = {
        "coverage_meet_minus_geometry": paired_bootstrap(
            coverages["certified_meet"] - coverages["geometry_action"],
            seed=args.seed,
            draws=args.bootstrap_draws,
        ),
        "risk_meet_minus_geometry": paired_bootstrap(
            risks["certified_meet"] - risks["geometry_action"],
            seed=args.seed + 1,
            draws=args.bootstrap_draws,
        ),
    }
    primary = reports["certified_meet"]
    baseline = reports["geometry_action"]
    dominance = {
        "geometry": int(np.sum(meet_prediction > geometry_prediction + 1e-7)),
        "direct": int(np.sum(meet_prediction > direct_prediction + 1e-7)),
    }
    checks = {
        "exact_simulator_snapshots": True,
        "meet_probability_dominance_exact": all(value == 0 for value in dominance.values()),
        "meet_budget_monotonicity_exact": primary["budget_monotonicity_violations"] == 0,
        "meet_confirmation_risk_at_most_0_05": primary["selective"]["scene_false_safe"] <= 0.05,
        "meet_coverage_beats_geometry_by_2_points": primary["selective"]["coverage"] >= baseline["selective"]["coverage"] + 0.02,
        "paired_coverage_interval_lower_above_zero": intervals["coverage_meet_minus_geometry"]["lower_95"] > 0.0,
        "fixed_0_90_danger_not_worse": primary["dangerous_false_safe_rate"] <= baseline["dangerous_false_safe_rate"] + 1e-12,
    }
    promotable = bool(all(checks.values()))
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
        "status": "PROMOTE_TO_BOUNDED_CLOSED_LOOP_PILOT" if promotable else "DO_NOT_MODIFY_R3_ACTOR",
        "promotable": promotable,
        "protocol": PROTOCOL,
        "checks": checks,
        "reports": reports,
        "paired_intervals": intervals,
        "dominance_violations": dominance,
        "dataset": dataset_summary,
        "data_dir": args.data_dir,
        "data_result_sha256": sha256_file(data_result_path),
        "calibration_dir": args.calibration_dir,
        "calibration_result_sha256": sha256_file(calibration_result_path),
        "freeze_dir": args.freeze_dir,
        "frozen_checkpoint_sha256": sha256_file(args.freeze_dir / "frozen_meet.pt"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "wall_clock_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
