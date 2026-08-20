from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.memory_safety.estimation_benchmark import (
    EstimationBenchmarkConfig,
    PhysicsResidualEstimator,
    _features,
    _metrics,
    _mhe_estimate,
    _predict_model,
    _targets,
    _train_model,
)


@dataclass(frozen=True)
class HiddenSizeAblationConfig:
    seed: int = 20260819
    source_artifact: str = "artifacts/memory_estimation_5k_20260819_v2"
    hidden_sizes: tuple[int, ...] = (16, 32, 64, 128)
    epochs: int = 25
    output_dir: str = "artifacts/memory_hidden_size_ablation"

    def __post_init__(self) -> None:
        if tuple(self.hidden_sizes) != (16, 32, 64, 128):
            raise ValueError("the pre-registered hidden sizes are 16, 32, 64, 128")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: HiddenSizeAblationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    source = Path(config.source_artifact)
    train = dict(np.load(source / "train.npz"))
    validation = dict(np.load(source / "validation.npz"))
    test = dict(np.load(source / "test.npz"))
    if len(train["regime_index"]) + len(validation["regime_index"]) + len(
        test["regime_index"]
    ) > 5000:
        raise ValueError("hidden-size ablation exceeds the 5000-trajectory ceiling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_config = EstimationBenchmarkConfig(
        seed=config.seed,
        train_trajectories=len(train["regime_index"]),
        validation_trajectories=len(validation["regime_index"]),
        test_trajectories=len(test["regime_index"]),
        epochs=config.epochs,
        output_dir=str(output),
    )
    train_features = _features(train, ego_compensated=True)
    validation_features = _features(validation, ego_compensated=True)
    test_features = _features(test, ego_compensated=True)
    train_target = _targets(train)
    validation_target = _targets(validation)
    test_target = _targets(test)
    train_base = _mhe_estimate(train, benchmark_config.dt)
    validation_base = _mhe_estimate(validation, benchmark_config.dt)
    test_base = _mhe_estimate(test, benchmark_config.dt)
    rows: dict[str, object] = {}
    for contractive in (False, True):
        family = "Contractive_Physics_Memory" if contractive else "Physics_GRU"
        for hidden_size in config.hidden_sizes:
            torch.manual_seed(config.seed + hidden_size + 1000 * int(contractive))
            model = PhysicsResidualEstimator(hidden_size, contractive=contractive)
            model, training = _train_model(
                model,
                train_features,
                train_base,
                train_target,
                validation_features,
                validation_base,
                validation_target,
                benchmark_config,
                device,
            )
            prediction, latency = _predict_model(
                model,
                test_features,
                test_base,
                training,
                device,
            )
            rows[f"{family}_H{hidden_size}"] = {
                "hidden_size": hidden_size,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "metrics": _metrics(prediction, test_target, test["regime_index"]),
                "latency": latency,
                "training_wall_seconds": training["wall_seconds"],
            }
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "git_sha": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "device": str(device),
        "matched_trajectory_count": (
            len(train["regime_index"])
            + len(validation["regime_index"])
            + len(test["regime_index"])
        ),
        "source_data_sha256": {
            name: _sha256(source / name)
            for name in ("train.npz", "validation.npz", "test.npz")
        },
        "methods": rows,
        "formal_500k": False,
    }
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
