from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.memory_safety.estimation_benchmark import (
    DiagonalSSMEstimator,
    FlattenMLP,
    PhysicsResidualEstimator,
    REGIMES,
    RecurrentEstimator,
    TemporalConvEstimator,
    _features,
    _imm_estimate,
    _kalman_estimate,
    _mhe_estimate,
)


CHANGE_STEPS = {
    "sudden_velocity_change": 12,
    "sudden_direction_change": 12,
    "stop_go": 13,
    "crossing_like_turn": 12,
}


@dataclass(frozen=True)
class AdaptationConfig:
    artifact_dir: str = "artifacts/memory_estimation_5k_20260819_v2"
    output_path: str = "artifacts/memory_estimation_5k_20260819_v2/adaptation_summary.json"
    velocity_l2_threshold: float = 1.0


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def prefix_split(data: dict[str, np.ndarray], end_step: int, length: int) -> dict[str, np.ndarray]:
    if not 0 <= end_step < data["relative_measurement"].shape[1]:
        raise ValueError("end_step is outside the trajectory")
    count = data["relative_measurement"].shape[0]
    selected_length = end_step + 1
    padding = length - selected_length
    if padding < 0:
        raise ValueError("requested prefix exceeds model history length")
    first_relative = np.nan_to_num(data["relative_measurement"][:, :1], nan=0.0)
    first_ego = data["ego_position"][:, :1]
    relative = np.concatenate(
        (
            np.repeat(first_relative, padding, axis=1),
            data["relative_measurement"][:, :selected_length],
        ),
        axis=1,
    )
    ego = np.concatenate(
        (
            np.repeat(first_ego, padding, axis=1),
            data["ego_position"][:, :selected_length],
        ),
        axis=1,
    )
    valid = np.concatenate(
        (
            np.zeros((count, padding), dtype=bool),
            data["valid"][:, :selected_length],
        ),
        axis=1,
    )
    return {
        "relative_measurement": relative.astype(np.float32),
        "ego_position": ego.astype(np.float32),
        "valid": valid,
    }


def _load_neural_models(artifact: Path, history_length: int, hidden_size: int, device: torch.device):
    specs = {
        "B_raw_L16_MLP": FlattenMLP(history_length, hidden_size),
        "C_ego_L16_MLP": FlattenMLP(history_length, hidden_size),
        "D_RNN": RecurrentEstimator("rnn", hidden_size),
        "E_GRU": RecurrentEstimator("gru", hidden_size),
        "F_LSTM": RecurrentEstimator("lstm", hidden_size),
        "G_TCN": TemporalConvEstimator(hidden_size),
        "H_SSM": DiagonalSSMEstimator(hidden_size),
        "I_Physics_GRU": PhysicsResidualEstimator(hidden_size, contractive=False),
        "M_Contractive_Physics_Memory": PhysicsResidualEstimator(
            hidden_size, contractive=True
        ),
    }
    result = {}
    for name, model in specs.items():
        checkpoint = torch.load(
            artifact / "models" / f"{name}.pt", map_location=device
        )
        model.load_state_dict(checkpoint["state_dict"])
        result[name] = (model.to(device).eval(), checkpoint["training"])
    return result


def _predict(
    split: dict[str, np.ndarray],
    models: dict[str, tuple[torch.nn.Module, dict]],
    dt: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    count = split["valid"].shape[0]
    zero = np.zeros((count, 6), dtype=np.float32)
    mhe = _mhe_estimate(split, dt)
    predictions = {
        "A_current_only": zero,
        "CV_Kalman": _kalman_estimate(split, dt, constant_acceleration=False),
        "CA_Kalman": _kalman_estimate(split, dt, constant_acceleration=True),
        "EKF_linear_equivalent": _kalman_estimate(
            split, dt, constant_acceleration=True
        ),
        "IMM": _imm_estimate(split, dt),
        "MHE": mhe,
    }
    for name, (model, training) in models.items():
        features = _features(
            split,
            ego_compensated=name != "B_raw_L16_MLP",
        )
        mean = np.asarray(training["target_mean"], dtype=np.float32)
        scale = np.asarray(training["target_scale"], dtype=np.float32)
        base = (
            mhe
            if name in {"I_Physics_GRU", "M_Contractive_Physics_Memory"}
            else zero
        )
        with torch.no_grad():
            output = model(
                torch.from_numpy(features).to(device),
                torch.from_numpy((base - mean) / scale).to(device),
            )
        predictions[name] = output.cpu().numpy() * scale + mean
    return predictions


def _adaptation_statistics(errors: np.ndarray, dt: float, threshold: float) -> dict[str, object]:
    sustained = np.zeros(errors.shape, dtype=bool)
    for lag in range(errors.shape[1]):
        sustained[:, lag] = np.all(errors[:, lag:] <= threshold, axis=1)
    first = np.full(errors.shape[0], np.nan)
    for trajectory in range(errors.shape[0]):
        indices = np.flatnonzero(sustained[trajectory])
        if indices.size:
            first[trajectory] = indices[0] * dt
    adapted = np.isfinite(first)
    return {
        "velocity_l2_threshold": threshold,
        "adapted_within_observed_window_fraction": float(np.mean(adapted)),
        "median_sustained_adaptation_seconds": (
            None if not np.any(adapted) else float(np.median(first[adapted]))
        ),
        "p90_sustained_adaptation_seconds": (
            None if not np.any(adapted) else float(np.quantile(first[adapted], 0.90))
        ),
        "velocity_l2_error_by_lag": [float(np.mean(errors[:, lag])) for lag in range(errors.shape[1])],
        "lags_seconds": [float(lag * dt) for lag in range(errors.shape[1])],
    }


def run(config: AdaptationConfig) -> dict[str, object]:
    artifact = Path(config.artifact_dir)
    benchmark = json.loads((artifact / "summary.json").read_text())
    history_length = int(benchmark["config"]["history_length"])
    hidden_size = int(benchmark["config"]["hidden_size"])
    dt = float(benchmark["config"]["dt"])
    loaded = np.load(artifact / "test.npz")
    data = {key: loaded[key] for key in loaded.files}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _load_neural_models(artifact, history_length, hidden_size, device)
    result: dict[str, dict[str, object]] = {}
    for regime, change_step in CHANGE_STEPS.items():
        selected = data["regime_index"] == REGIMES.index(regime)
        regime_data = {
            key: value[selected]
            for key, value in data.items()
            if key not in {"regime_index", "future_position"}
        }
        errors: dict[str, list[np.ndarray]] = {}
        for end_step in range(change_step, history_length):
            split = prefix_split(regime_data, end_step, history_length)
            target_velocity = regime_data["obstacle_velocity"][:, end_step]
            for name, prediction in _predict(split, models, dt, device).items():
                errors.setdefault(name, []).append(
                    np.linalg.norm(prediction[:, :3] - target_velocity, axis=1)
                )
        result[regime] = {
            name: _adaptation_statistics(
                np.stack(values, axis=1), dt, config.velocity_l2_threshold
            )
            for name, values in errors.items()
        }
    summary = {
        "artifact": str(artifact),
        "source_sha256": _source_hash(),
        "device": str(device),
        "evaluation_split": "held-out test trajectories only",
        "new_training": False,
        "threshold_semantics": (
            "first post-change lag at which velocity L2 error is <= threshold "
            "and remains so for every later observed lag"
        ),
        "regimes": result,
    }
    output = Path(config.output_path)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
