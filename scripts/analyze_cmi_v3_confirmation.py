#!/usr/bin/env python3
"""Frozen fixed-N CMI-v3 confirmation analysis; never authorizes METHOD-TRAIN."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cmi_v3_dev import atomic_json
from scripts.cmi_v3_models import (
    ACTION_ORDER, CHANNELS, FAMILIES, legal_features, load_states,
    predict_frozen_ensemble,
)
from scripts.validate_identification_contract import file_hash


BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2_026_091_225


def load_confirmation(contract: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    pre = contract["pre_h"]
    assigned = pre["master_splits"]["CMI_V3_CONFIRM"]
    status = json.loads((run_dir / "status.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    if status.get("status") != "COMPLETE_AWAITING_CMI_V3_CONFIRM_ANALYSIS":
        raise ValueError("CMI-v3 confirmation is not complete")
    if results.get("completed") != len(assigned):
        raise ValueError("CMI-v3 confirmation count is incomplete")
    if [row["world_identity"] for row in results["records"]] != assigned:
        raise ValueError("CMI-v3 confirmation order differs from assignment")
    expected = pre["resamples_per_anchor_action"]["CMI_V3_CONFIRM"]
    feature_rows = {channel: [] for channel in CHANNELS}
    targets, paired, worlds, steps = [], [], [], []
    for record in results["records"]:
        json_path = run_dir / "records" / record["json"]
        npz_path = run_dir / "records" / record["npz"]
        if file_hash(json_path).removeprefix("sha256:") != record["json_sha256"]:
            raise ValueError("CMI-v3 confirmation JSON hash mismatch")
        if file_hash(npz_path).removeprefix("sha256:") != record["npz_sha256"]:
            raise ValueError("CMI-v3 confirmation NPZ hash mismatch")
        meta = json.loads(json_path.read_text())
        included = [anchor for anchor in meta["anchors"] if anchor.get("included")]
        with np.load(npz_path, allow_pickle=False) as archive:
            features = legal_features({name: archive[name] for name in archive.files})
        for index, anchor in enumerate(included):
            values = {action: [] for action in ACTION_ORDER}
            for row in anchor["resamples"]:
                values[row["action"]].append(float(row["outcome"]["utility"]))
            if any(len(values[action]) != expected for action in ACTION_ORDER):
                raise ValueError("CMI-v3 confirmation resample count mismatch")
            for channel in CHANNELS:
                feature_rows[channel].append(features[channel][index])
            targets.append([np.mean(values[action]) for action in ACTION_ORDER])
            paired.append([values[action] for action in ACTION_ORDER])
            worlds.append(meta["world_identity"])
            steps.append(int(anchor["anchor_step"]))
    return {
        "features": {channel: np.asarray(rows, np.float64) for channel, rows in feature_rows.items()},
        "targets": np.asarray(targets, np.float64),
        "paired": np.asarray(paired, np.float64),
        "worlds": np.asarray(worlds),
        "steps": np.asarray(steps, np.int64),
    }


def world_means(values: np.ndarray, worlds: np.ndarray) -> np.ndarray:
    identities = list(dict.fromkeys(worlds.tolist()))
    return np.asarray([values[worlds == world].mean() for world in identities])


def interval(values: np.ndarray) -> dict[str, float]:
    estimate = float(values.mean())
    centered = values - estimate
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap = np.empty(BOOTSTRAP_REPLICATES, np.float64)
    for draw in range(BOOTSTRAP_REPLICATES):
        bootstrap[draw] = estimate + float(np.mean(centered * rng.standard_normal(len(values))))
    scale = float(bootstrap.std(ddof=1))
    critical = float(np.quantile(np.abs(bootstrap - bootstrap.mean()) / max(scale, 1e-15), 0.95))
    return {"estimate": estimate, "lcb": estimate - critical * scale, "ucb": estimate + critical * scale}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--confirmation-child", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    contract = json.loads(args.contract.resolve().read_text())
    child = json.loads(args.confirmation_child.resolve().read_text())
    adequacy = json.loads((args.dev_analysis / "adequacy.json").read_text())
    analysis = json.loads((args.dev_analysis / "analysis.json").read_text())
    if not adequacy.get("passed") or not child.get("CMI_V3_CONFIRM_READY"):
        raise ValueError("CMI-v3 confirmation was not authorized")
    for name, digest in child["dev_artifact_hashes"].items():
        if file_hash(args.dev_analysis / name) != digest:
            raise ValueError("CMI-v3 DEV artifact binding changed")
    if child.get("dev_analysis_hash") != file_hash(args.dev_analysis / "analysis.json"):
        raise ValueError("CMI-v3 DEV analysis hash changed")
    if not analysis.get("adequacy_passed"):
        raise ValueError("CMI-v3 DEV analysis does not pass adequacy")
    data = load_confirmation(contract, args.run_dir.resolve())
    model_meta = json.loads((args.dev_analysis / "frozen_models.json").read_text())
    states = load_states(str(args.dev_analysis / "frozen_models.npz"), model_meta["states"])
    ensemble, family = predict_frozen_ensemble(states, data["features"])
    target, index = data["targets"], np.arange(len(data["targets"]))
    choices = {channel: np.argmax(ensemble[channel], axis=1) for channel in CHANNELS}
    selected = {channel: target[index, choices[channel]] for channel in CHANNELS}
    delta_anchor = selected["full"] - selected["latest"]
    delta_world = world_means(delta_anchor, data["worlds"])
    delta_interval = interval(delta_world)

    calibration = json.loads((args.dev_analysis / "calibration.json").read_text())
    support_meta = json.loads((args.dev_analysis / "support_calibration.json").read_text())
    support_evidence: dict[str, Any] = {}
    support_passed = True
    with np.load(args.dev_analysis / "support_calibration.npz", allow_pickle=False) as support:
        for channel in CHANNELS:
            mean, std = support[f"{channel}_mean"], support[f"{channel}_std"]
            components = support[f"{channel}_components"]
            reference = support[f"{channel}_representation"]
            representation = ((data["features"][channel] - mean) / std) @ components.T
            distances = np.linalg.norm(
                representation[:, None] - reference[None, :], axis=2
            ).min(axis=1)
            radius = float(support_meta["channels"][channel]["confirmation_radius"])
            overall = float(np.mean(distances <= radius))
            strata = {
                str(step): float(np.mean(distances[data["steps"] == step] <= radius))
                for step in (256, 768)
            }
            passed = bool(overall >= 0.95 and all(value >= 0.90 for value in strata.values()))
            support_evidence[channel] = {
                "passed": passed, "overall_fraction": overall,
                "anchor_step_fraction": strata, "maximum": float(distances.max()),
                "radius": radius,
            }
            support_passed &= passed
    q_evidence = {}
    calibration_passed = True
    for channel in CHANNELS:
        width = float(calibration["channels"][channel]["confirmation_half_width"])
        coverage = float(np.mean(np.abs(ensemble[channel] - target) <= width))
        rmse = float(np.sqrt(np.mean(np.square(ensemble[channel] - target))))
        q_evidence[channel] = {"rmse": rmse, "coverage_90": coverage, "half_width": width}
        calibration_passed &= coverage >= 0.85
    family_delta = {
        name: float(np.mean(
            target[index, np.argmax(family["full"][name], axis=1)]
            - target[index, np.argmax(family["latest"][name], axis=1)]
        ))
        for name in FAMILIES
    }
    finite = bool(
        all(np.isfinite(value).all() for value in ensemble.values())
        and np.isfinite(target).all() and math.isfinite(delta_interval["estimate"])
    )
    valid = bool(finite and support_passed and calibration_passed)
    by_step = {
        str(step): float(np.mean(delta_anchor[data["steps"] == step]))
        for step in (256, 768)
    }
    if not valid:
        verdict = "INCONCLUSIVE"
        reason = "confirmation validity, support, or calibration failed"
    elif (
        delta_interval["lcb"] > 0 and delta_interval["estimate"] >= 0.05
        and all(value > 0 for value in by_step.values())
        and all(value > 0 for value in family_delta.values())
    ):
        verdict = "HISTORY_INFORMATION_PRESENT"
        reason = "stable positive fixed-policy full-history increment"
    elif delta_interval["lcb"] >= -0.025 and delta_interval["ucb"] <= 0.025:
        verdict = "HISTORY_EQUIVALENT_WITHIN_MARGIN"
        reason = "full and latest fixed policies are equivalent within the frozen margin"
    else:
        verdict = "INCONCLUSIVE"
        reason = "frozen information-value boundaries are not separated"
    output.mkdir(parents=True)
    atomic_json(output / "analysis.json", {
        "schema_version": "cmi-v3-confirm-analysis-v1",
        "worlds": len(delta_world), "anchors": len(target),
        "Delta_H": delta_interval,
        "anchor_step_Delta_H": by_step,
        "family_Delta_H": family_delta,
        "conditional_q": q_evidence,
        "support": support_evidence,
        "valid": valid,
        "verdict": verdict,
        "reason": reason,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "method_train_authorized": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
