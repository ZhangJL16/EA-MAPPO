"""Evaluate a completed mode-masked PPO checkpoint on frozen 2D maps.

The execution path and metrics come from the existing paired evaluator.  This
entry point adds method provenance and rejects unfinished training checkpoints
so that startup-health snapshots cannot accidentally enter a result matrix.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from dual_constraint_2d.evaluate_matrix import evaluate as evaluate_existing
from dual_constraint_2d.train_ppo import ROOT, VALIDATION_MAP_IDS

from .mode_policy import ModeMaskedPolicy  # noqa: F401; needed when loading models


EVAL_SOURCE_FILES = (
    "dual_constraint_2d/evaluate_matrix.py",
    "dual_constraint_2d/gym_adapter.py",
    "dual_constraint_2d/action_adapter.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/shield.py",
    "learning2d/mode_policy.py",
    "learning2d/evaluate_mode_ppo.py",
)


def evaluate(output: Path, *, map_id: int, training_output: Path,
             max_new_decisions: int | None = None) -> dict:
    if map_id not in VALIDATION_MAP_IDS and map_id not in range(56, 72):
        raise ValueError("evaluation map must be validation 32–39 or fresh holdout 56–71")
    training_manifest_path = training_output / "manifest.json"
    training_status_path = training_output / "status.json"
    training_manifest = json.loads(training_manifest_path.read_text())
    training_status = json.loads(training_status_path.read_text())
    if training_manifest["protocol"] != "dual_constraint_2d_mode_masked_ppo_v2_horizon_close":
        raise ValueError("not a mode-masked PPO training run")
    if not training_status["complete"]:
        raise ValueError("training must complete before result evaluation")
    model_path = training_output / training_status["latest_model"]
    provenance = {
        "protocol": "dual_constraint_2d_mode_masked_eval_v1",
        "map_id": map_id,
        "training_manifest_sha256": sha256(training_manifest_path.read_bytes()).hexdigest(),
        "training_status_sha256": sha256(training_status_path.read_bytes()).hexdigest(),
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "model_path": str(model_path.resolve()),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in EVAL_SOURCE_FILES
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "method_provenance.json"
    if path.exists():
        if json.loads(path.read_text()) != provenance:
            raise RuntimeError("evaluation provenance changed; refusing mixed result")
    else:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)
    return evaluate_existing(
        output, algorithm="ppo", map_id=map_id, model_path=model_path,
        max_new_decisions=max_new_decisions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--max-new-decisions", type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.output, map_id=args.map_id, training_output=args.training_output,
        max_new_decisions=args.max_new_decisions,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
