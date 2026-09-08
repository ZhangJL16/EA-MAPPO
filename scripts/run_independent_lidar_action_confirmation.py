from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
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

from experiments.rechargeability_safety.core import MonotoneBudgetCritic, ResidualMonotoneBudgetCritic
from scripts.run_forked_lidar_residual_critic_gate import (
    ACTION_FEATURE_SLICE,
    VARIANTS,
    apply_normalization,
    atomic_json,
    build_dataset,
    labels_for_budgets,
    load_dataset,
    metrics,
    residual_probabilities,
    save_npz_atomic,
    sha256_file,
    standard_probabilities,
)


PROTOCOL = "INDEPENDENT_LIDAR_ACTION_CONFIRMATION_V1"
PRIMARY = "lidar_action_direct"
BASELINE = "geometry_action"


def acceptance_metrics(labels: np.ndarray, prediction: np.ndarray, threshold: float = 0.90) -> dict[str, float | None]:
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.asarray(prediction, dtype=np.float64).reshape(-1)
    accepted = probability >= threshold
    infeasible = truth < 0.5
    dangerous = accepted & infeasible
    return {
        "joint_dangerous_false_safe": float(np.mean(dangerous)),
        "conditional_false_safe_given_infeasible": (
            float(np.mean(accepted[infeasible])) if np.any(infeasible) else None
        ),
        "unsafe_among_accepted": (
            float(np.mean(infeasible[accepted])) if np.any(accepted) else None
        ),
        "declaration_coverage": float(np.mean(accepted)),
    }


def scene_paired_differences(
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    scenes: np.ndarray,
    *,
    threshold: float = 0.90,
) -> tuple[np.ndarray, np.ndarray]:
    brier = []
    dangerous = []
    for scene in np.unique(scenes):
        mask = scenes == scene
        truth = labels[mask]
        left = candidate[mask]
        right = baseline[mask]
        brier.append(float(np.mean((left - truth) ** 2) - np.mean((right - truth) ** 2)))
        left_danger = (left >= threshold) & (truth < 0.5)
        right_danger = (right >= threshold) & (truth < 0.5)
        dangerous.append(float(np.mean(left_danger) - np.mean(right_danger)))
    return np.asarray(brier, dtype=np.float64), np.asarray(dangerous, dtype=np.float64)


def paired_bootstrap_interval(values: np.ndarray, *, seed: int, draws: int = 10_000) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or data.size < 2 or not np.all(np.isfinite(data)):
        raise ValueError("paired bootstrap requires finite scene-level differences")
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 1000):
        count = min(1000, draws - start)
        indices = rng.integers(0, data.size, size=(count, data.size))
        estimates[start : start + count] = np.mean(data[indices], axis=1)
    return {
        "scene_count": int(data.size),
        "draws": int(draws),
        "mean_difference": float(np.mean(data)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def load_member_predictions(
    checkpoint_path: Path,
    arrays: dict[str, np.ndarray],
    budgets: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    geometry = apply_normalization(
        arrays["geometry"],
        np.asarray(payload["geometry_mean"], dtype=np.float32),
        np.asarray(payload["geometry_scale"], dtype=np.float32),
    )
    embedding = apply_normalization(
        arrays["embedding"],
        np.asarray(payload["embedding_mean"], dtype=np.float32),
        np.asarray(payload["embedding_scale"], dtype=np.float32),
    )
    no_action = geometry.copy()
    no_action[:, ACTION_FEATURE_SLICE] = 0.0
    full = np.concatenate([geometry, embedding], axis=1)
    no_action_full = np.concatenate([no_action, embedding], axis=1)

    geometry_model = MonotoneBudgetCritic(geometry.shape[1], hidden_dim=96).to(device)
    no_action_model = MonotoneBudgetCritic(no_action_full.shape[1], hidden_dim=96).to(device)
    direct_model = MonotoneBudgetCritic(full.shape[1], hidden_dim=96).to(device)
    residual_model = ResidualMonotoneBudgetCritic(
        MonotoneBudgetCritic(geometry.shape[1], hidden_dim=96), full.shape[1], hidden_dim=96
    ).to(device)
    models = {
        "geometry_action": (geometry_model, geometry, None),
        "lidar_no_action": (no_action_model, no_action_full, None),
        "lidar_action_direct": (direct_model, full, None),
        "geometry_lidar_action_residual": (residual_model, geometry, full),
    }
    predictions: dict[str, np.ndarray] = {}
    for name, (model, first, second) in models.items():
        record = payload["models"][name]
        model.load_state_dict(record["state_dict"])
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise RuntimeError(f"non-finite frozen checkpoint: {checkpoint_path} {name}")
        temperature = float(record["temperature"])
        predictions[name] = (
            standard_probabilities(model, first, budgets, device, temperature)
            if second is None
            else residual_probabilities(model, first, second, budgets, device, temperature)
        )
    return predictions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-data-dir", type=Path, required=True)
    parser.add_argument("--development-model-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=390001)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.confirmation_data_dir = args.confirmation_data_dir.expanduser().resolve()
    args.development_model_dir = args.development_model_dir.expanduser().resolve()
    args.protocol = args.protocol.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    required = [
        args.confirmation_data_dir / "RESULT.json",
        args.development_model_dir / "RESULT.json",
        args.protocol,
        *[args.development_model_dir / "checkpoints" / f"fold_{fold}.pt" for fold in range(3)],
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    if args.resume and (args.output_dir / "RESULT.json").is_file():
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    development = json.loads((args.development_model_dir / "RESULT.json").read_text(encoding="utf-8"))
    confirmation = json.loads((args.confirmation_data_dir / "RESULT.json").read_text(encoding="utf-8"))
    development_data = json.loads((Path(development["data_dir"]) / "RESULT.json").read_text(encoding="utf-8"))
    development_scenes = set(map(int, development_data["scene_indices"]))
    confirmation_scenes = set(map(int, confirmation["scene_indices"]))
    if development_scenes & confirmation_scenes:
        raise RuntimeError("development and confirmation scenes overlap")
    if len(confirmation_scenes) != 75 or int(confirmation["num_branches"]) != 3000:
        raise RuntimeError("confirmation data does not match the locked scale")
    if not bool(confirmation.get("promotable", False)):
        raise RuntimeError("confirmation data Gate did not pass")

    manifest = {
        "status": "RUNNING",
        "protocol": PROTOCOL,
        "formal_evidence": True,
        "primary": PRIMARY,
        "baseline": BASELINE,
        "confirmation_data_dir": args.confirmation_data_dir,
        "confirmation_result_sha256": sha256_file(args.confirmation_data_dir / "RESULT.json"),
        "development_model_dir": args.development_model_dir,
        "development_result_sha256": sha256_file(args.development_model_dir / "RESULT.json"),
        "protocol_document": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint_sha256": {
            str(fold): sha256_file(args.development_model_dir / "checkpoints" / f"fold_{fold}.pt")
            for fold in range(3)
        },
        "development_scene_count": len(development_scenes),
        "confirmation_scene_count": len(confirmation_scenes),
        "scene_overlap": 0,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "device": args.device,
        "started_unix": time.time(),
        "exact_command": [sys.executable, *sys.argv],
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    started = time.time()
    try:
        if args.resume and (args.output_dir / "dataset.npz").is_file():
            arrays, dataset_summary = load_dataset(args.output_dir)
        else:
            arrays, dataset_summary = build_dataset(
                args.confirmation_data_dir, device=device, output=args.output_dir
            )
        budgets = np.asarray(development["budgets"], dtype=np.float32)
        labels = labels_for_budgets(arrays["safe"], arrays["energy"], budgets)
        members: list[dict[str, np.ndarray]] = []
        member_metrics = []
        for fold in range(3):
            prediction = load_member_predictions(
                args.development_model_dir / "checkpoints" / f"fold_{fold}.pt",
                arrays,
                budgets,
                device,
            )
            members.append(prediction)
            member_metrics.append(
                {
                    "fold": fold,
                    "variants": {
                        name: {
                            **metrics(labels, prediction[name], arrays["scene"], arrays["anchor"], arrays["bucket"]),
                            **acceptance_metrics(labels, prediction[name]),
                        }
                        for name in VARIANTS
                    },
                }
            )
        ensemble = {
            name: np.mean(np.stack([member[name] for member in members]), axis=0)
            for name in VARIANTS
        }
        ensemble_metrics = {}
        for name in VARIANTS:
            result = metrics(labels, ensemble[name], arrays["scene"], arrays["anchor"], arrays["bucket"])
            result.update(acceptance_metrics(labels, ensemble[name]))
            result["matched_action_sensitivity"] = float(
                np.mean(
                    [
                        np.std(ensemble[name][arrays["anchor"] == anchor, 4])
                        for anchor in np.unique(arrays["anchor"])
                    ]
                )
            )
            ensemble_metrics[name] = result
        brier_difference, danger_difference = scene_paired_differences(
            labels, ensemble[PRIMARY], ensemble[BASELINE], arrays["scene"]
        )
        intervals = {
            "brier_direct_minus_geometry": paired_bootstrap_interval(
                brier_difference, seed=args.seed, draws=args.bootstrap_draws
            ),
            "dangerous_false_safe_direct_minus_geometry": paired_bootstrap_interval(
                danger_difference, seed=args.seed + 1, draws=args.bootstrap_draws
            ),
        }
        member_wins = sum(
            item["variants"][PRIMARY]["scene_averaged_brier"]
            < item["variants"][BASELINE]["scene_averaged_brier"]
            for item in member_metrics
        )
        direct = ensemble_metrics[PRIMARY]
        geometry = ensemble_metrics[BASELINE]
        checks = {
            "direct_brier_beats_geometry_by_2_percent": (
                direct["scene_averaged_brier"] <= 0.98 * geometry["scene_averaged_brier"]
            ),
            "brier_bootstrap_upper_below_zero": (
                intervals["brier_direct_minus_geometry"]["upper_95"] < 0.0
            ),
            "dangerous_false_safe_point_not_worse": (
                direct["joint_dangerous_false_safe"] <= geometry["joint_dangerous_false_safe"]
            ),
            "dangerous_false_safe_noninferiority_upper": (
                intervals["dangerous_false_safe_direct_minus_geometry"]["upper_95"] <= 0.005
            ),
            "direct_wins_two_frozen_members": member_wins >= 2,
            "budget_monotonicity_exact": direct["budget_monotonicity_violations"] == 0,
            "matched_action_sensitivity_nonzero": direct["matched_action_sensitivity"] > 1e-5,
            "confirmation_data_gate_passed": bool(confirmation["promotable"]),
        }
        promotable = bool(all(checks.values()))
        save_npz_atomic(
            args.output_dir / "predictions.npz",
            labels=labels,
            budgets=budgets,
            scene=arrays["scene"],
            anchor=arrays["anchor"],
            **{f"prediction_{name}": value for name, value in ensemble.items()},
        )
        final = {
            **manifest,
            "status": "AUTHORIZE_BOUNDED_ACTOR_PILOT" if promotable else "DO_NOT_MODIFY_R3_ACTOR",
            "promotable": promotable,
            "dataset": dataset_summary,
            "checks": checks,
            "ensemble": ensemble_metrics,
            "member_metrics": member_metrics,
            "paired_scene_bootstrap": intervals,
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
