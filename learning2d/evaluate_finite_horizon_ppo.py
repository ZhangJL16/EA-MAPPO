"""Evaluate completed finite-horizon PPO using the frozen common safety layer."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

from dual_constraint_2d.evaluate_matrix import evaluate as evaluate_existing
from dual_constraint_2d.train_ppo import ROOT, VALIDATION_MAP_IDS

from .mode_policy import ModeMaskedPolicy  # noqa: F401; needed when loading models


SOURCE_FILES = (
    "dual_constraint_2d/evaluate_matrix.py",
    "dual_constraint_2d/gym_adapter.py",
    "dual_constraint_2d/action_adapter.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/shield.py",
    "learning2d/mode_policy.py",
    "learning2d/finite_horizon_gym.py",
    "learning2d/evaluate_finite_horizon_ppo.py",
)


def evaluate(output: Path, *, map_id: int, training_output: Path,
             max_new_decisions: int | None = None) -> dict:
    if map_id not in VALIDATION_MAP_IDS and map_id not in range(56, 72):
        raise ValueError("evaluation map must be validation 32–39 or fresh holdout 56–71")
    manifest_path = training_output / "manifest.json"
    status_path = training_output / "status.json"
    manifest = json.loads(manifest_path.read_text())
    status = json.loads(status_path.read_text())
    if manifest["protocol"] != "dual_constraint_2d_mode_masked_finite_horizon_v3":
        raise ValueError("not a finite-horizon PPO training run")
    if not status["complete"]:
        raise ValueError("training must complete before result evaluation")
    model_path = training_output / status["latest_model"]
    provenance = {
        "protocol": "dual_constraint_2d_finite_horizon_eval_v3",
        "map_id": map_id,
        "training_gamma": manifest["ppo"]["gamma"],
        "training_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "training_status_sha256": sha256(status_path.read_bytes()).hexdigest(),
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "model_path": str(model_path.resolve()),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES
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
