"""Final validation of independent single-environment CPU replication seeds."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from analysis2d.raw_frozen_replay import replay as raw_replay
from dual_constraint_2d.evaluate_matrix import evaluate as shielded_evaluate
from dual_constraint_2d.train_ppo import ROOT, VALIDATION_MAP_IDS
from .mode_policy import ModeMaskedPolicy  # noqa: F401; frozen policy loader
from .train_shield_feedback_cpu import check_arm_cpu_runtime
from .train_shield_feedback_replicates_cpu import PROTOCOL as TRAIN_PROTOCOL, SEEDS


PROTOCOL = "dual_constraint_2d_shield_feedback_cpu_replication_validation_v1"
MODES = ("shielded", "raw")
SOURCE_FILES = (
    "analysis2d/raw_frozen_replay.py",
    "dual_constraint_2d/evaluate_matrix.py",
    "learning2d/evaluate_shield_feedback_replicates_cpu.py",
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def evaluate(output: Path, *, mode: str, map_id: int, training_output: Path) -> dict:
    if mode not in MODES or map_id not in VALIDATION_MAP_IDS:
        raise ValueError("mode must be shielded/raw and map must be validation 32–39")
    check_arm_cpu_runtime()
    training_manifest_path = training_output / "manifest.json"
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    status_path = training_output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (training_manifest["protocol"] != TRAIN_PROTOCOL
            or training_manifest["seed"] not in SEEDS
            or training_manifest["device"] != "cpu"
            or training_manifest["architecture"] not in ("aarch64", "arm64")
            or status["timesteps"] != training_manifest["total_timesteps"]
            or not status["complete"]):
        raise RuntimeError("completed ARM CPU replication is required")
    for relative, digest in training_manifest["source_sha256"].items():
        if _hash(ROOT / relative) != digest:
            raise RuntimeError(f"training source changed: {relative}")
    model_path = training_output / status["latest_model"]
    provenance = {
        "protocol": PROTOCOL, "condition": training_manifest["condition"],
        "seed": training_manifest["seed"], "mode": mode, "map_id": map_id,
        "device": "cpu", "training_manifest_sha256": _hash(training_manifest_path),
        "training_status_sha256": _hash(status_path), "model_sha256": _hash(model_path),
        "source_sha256": {relative: _hash(ROOT / relative) for relative in SOURCE_FILES},
    }
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("evaluation provenance changed")
    else:
        _atomic_json(provenance_path, provenance)
    if mode == "shielded":
        return shielded_evaluate(output, algorithm="ppo", map_id=map_id,
                                 model_path=model_path)
    result = raw_replay(model_path, (map_id,))
    summary_path = output / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != result:
            raise RuntimeError("raw evaluation result changed")
    else:
        _atomic_json(summary_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output, mode=args.mode, map_id=args.map_id,
                              training_output=args.training_output), sort_keys=True))


if __name__ == "__main__":
    main()
