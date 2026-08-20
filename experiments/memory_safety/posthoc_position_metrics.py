from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.memory_safety.estimation_benchmark import (
    _current_position_estimate,
    _future_position_at,
)


HORIZONS = (0.25, 0.5, 1.0, 2.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(source_artifact: Path, output_dir: Path, dt: float = 0.1) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    test = dict(np.load(source_artifact / "test.npz"))
    predictions = np.load(source_artifact / "test_predictions.npz")
    current_position = _current_position_estimate(test)
    methods: dict[str, object] = {}
    for name in predictions.files:
        prediction = predictions[name]
        horizons = {}
        for horizon in HORIZONS:
            center = (
                current_position
                + prediction[:, :3] * horizon
                + 0.5 * prediction[:, 3:] * horizon**2
            )
            error = _future_position_at(test, horizon, dt) - center
            horizons[str(horizon)] = {
                "position_mae": float(np.mean(np.abs(error))),
                "position_rmse": float(np.sqrt(np.mean(error**2))),
                "p95_position_error_l2": float(
                    np.quantile(np.linalg.norm(error, axis=1), 0.95)
                ),
            }
        methods[name] = horizons
    result = {
        "scope": (
            "post-hoc future-position metrics on unchanged held-out test "
            "predictions; no new trajectories, training, or model selection"
        ),
        "formal_500k": False,
        "source_artifact": str(source_artifact),
        "source_sha256": _sha256(Path(__file__)),
        "input_sha256": {
            name: _sha256(source_artifact / name)
            for name in ("test.npz", "test_predictions.npz", "summary.json")
        },
        "methods": methods,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
