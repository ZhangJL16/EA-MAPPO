from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from review_bundle.safety.memory import ContractiveResidualMemory


REGIMES = (
    "constant_velocity",
    "constant_acceleration",
    "sinusoidal_acceleration",
    "piecewise_acceleration",
    "sudden_velocity_change",
    "sudden_direction_change",
    "stop_go",
    "dropout_burst",
    "crossing_like_turn",
)
ABRUPT_REGIMES = {
    "sudden_velocity_change",
    "sudden_direction_change",
    "stop_go",
    "crossing_like_turn",
}


@dataclass(frozen=True)
class EstimationBenchmarkConfig:
    seed: int = 20260819
    train_trajectories: int = 3000
    validation_trajectories: int = 1000
    test_trajectories: int = 1000
    history_length: int = 16
    future_steps: int = 20
    dt: float = 0.1
    sensor_error_bound: float = 0.10
    dropout_probability: float = 0.05
    hidden_size: int = 32
    epochs: int = 25
    batch_size: int = 128
    learning_rate: float = 1e-3
    output_dir: str = "artifacts/memory_estimation_5k"

    def __post_init__(self) -> None:
        total = self.train_trajectories + self.validation_trajectories + self.test_trajectories
        if total > 5000:
            raise ValueError("controlled protocol permits at most 5000 matched trajectories")
        if min(
            self.train_trajectories,
            self.validation_trajectories,
            self.test_trajectories,
        ) <= 0:
            raise ValueError("all dataset splits must be nonempty")
        if self.history_length < 4 or self.future_steps < 20 or self.dt <= 0.0:
            raise ValueError("history_length, future_steps and dt must be valid")
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0.0:
            raise ValueError("training parameters must be positive")


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sample_regime(index: int) -> str:
    return REGIMES[index % len(REGIMES)]


def _generate_split(
    count: int,
    seed: int,
    config: EstimationBenchmarkConfig,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    length = config.history_length
    obstacle_position = np.zeros((count, length, 3), dtype=np.float32)
    obstacle_velocity = np.zeros_like(obstacle_position)
    obstacle_acceleration = np.zeros_like(obstacle_position)
    ego_position = np.zeros_like(obstacle_position)
    relative_measurement = np.zeros_like(obstacle_position)
    valid = np.ones((count, length), dtype=bool)
    regime_index = np.empty(count, dtype=np.int64)
    future_position = np.zeros((count, config.future_steps + 1, 3), dtype=np.float32)
    for sequence in range(count):
        regime = _sample_regime(sequence)
        regime_index[sequence] = REGIMES.index(regime)
        position = rng.uniform(-20.0, 20.0, size=3)
        velocity = rng.uniform(-3.0, 3.0, size=3)
        acceleration = rng.uniform(-1.0, 1.0, size=3)
        ego = rng.uniform(-10.0, 10.0, size=3)
        ego_velocity = rng.uniform(-5.0, 5.0, size=3)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        amplitude = rng.uniform(0.3, 1.5, size=3)
        switch_step = length - 4
        restart_velocity = rng.uniform(-4.0, 4.0, size=3)
        for step in range(length):
            if regime == "constant_velocity":
                acceleration = np.zeros(3)
            elif regime == "constant_acceleration":
                pass
            elif regime == "sinusoidal_acceleration":
                acceleration = amplitude * np.sin(0.45 * step + phase)
            elif regime == "piecewise_acceleration" and step % 5 == 0:
                acceleration = rng.uniform(-2.0, 2.0, size=3)
            elif regime == "sudden_velocity_change" and step == switch_step:
                velocity += rng.uniform(-8.0, 8.0, size=3)
                acceleration = rng.uniform(-2.0, 2.0, size=3)
            elif regime == "sudden_direction_change" and step == switch_step:
                velocity = -velocity + rng.uniform(-1.0, 1.0, size=3)
                acceleration = rng.uniform(-1.0, 1.0, size=3)
            elif regime == "stop_go":
                if step == switch_step - 2:
                    velocity = np.zeros(3)
                    acceleration = np.zeros(3)
                if step == switch_step + 1:
                    velocity = restart_velocity.copy()
            elif regime == "crossing_like_turn" and step >= switch_step:
                angle = 0.7
                rotation = np.array(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                velocity = rotation @ velocity
                acceleration = rng.uniform(-0.5, 0.5, size=3)
            obstacle_position[sequence, step] = position
            obstacle_velocity[sequence, step] = velocity
            obstacle_acceleration[sequence, step] = acceleration
            ego_position[sequence, step] = ego
            noise = rng.uniform(
                -config.sensor_error_bound,
                config.sensor_error_bound,
                size=3,
            )
            relative_measurement[sequence, step] = position - ego + noise
            position = (
                position
                + velocity * config.dt
                + 0.5 * acceleration * config.dt**2
            )
            velocity = velocity + acceleration * config.dt
            ego_acceleration = 0.3 * np.sin(0.2 * step + phase)
            ego = ego + ego_velocity * config.dt + 0.5 * ego_acceleration * config.dt**2
            ego_velocity = ego_velocity + ego_acceleration * config.dt
        future_position[sequence, 0] = obstacle_position[sequence, -1]
        future_position[sequence, 1] = position
        for future_step in range(1, config.future_steps):
            global_step = length + future_step
            if regime == "sinusoidal_acceleration":
                acceleration = amplitude * np.sin(0.45 * global_step + phase)
            elif regime == "piecewise_acceleration" and global_step % 5 == 0:
                acceleration = rng.uniform(-2.0, 2.0, size=3)
            elif regime == "crossing_like_turn":
                angle = 0.7
                rotation = np.array(
                    [
                        [math.cos(angle), -math.sin(angle), 0.0],
                        [math.sin(angle), math.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                velocity = rotation @ velocity
                acceleration = rng.uniform(-0.5, 0.5, size=3)
            position = (
                position
                + velocity * config.dt
                + 0.5 * acceleration * config.dt**2
            )
            velocity = velocity + acceleration * config.dt
            future_position[sequence, future_step + 1] = position
        valid[sequence] = rng.random(length) >= config.dropout_probability
        valid[sequence, 0] = True
        if regime == "dropout_burst":
            valid[sequence, -5:-1] = False
        relative_measurement[sequence, ~valid[sequence]] = np.nan
    return {
        "obstacle_position": obstacle_position,
        "obstacle_velocity": obstacle_velocity,
        "obstacle_acceleration": obstacle_acceleration,
        "ego_position": ego_position,
        "relative_measurement": relative_measurement,
        "valid": valid,
        "regime_index": regime_index,
        "future_position": future_position,
    }


def _forward_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = np.asarray(values, dtype=np.float32).copy()
    for step in range(1, filled.shape[1]):
        missing = ~valid[:, step]
        filled[missing, step] = filled[missing, step - 1]
    return filled


def _features(split: dict[str, np.ndarray], *, ego_compensated: bool) -> np.ndarray:
    filled = _forward_fill(split["relative_measurement"], split["valid"])
    if ego_compensated:
        filled = filled + split["ego_position"]
    filled = filled - filled[:, :1, :]
    mask = split["valid"].astype(np.float32)[..., None]
    return np.concatenate((filled, mask), axis=2).astype(np.float32)


def _targets(split: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            split["obstacle_velocity"][:, -1],
            split["obstacle_acceleration"][:, -1],
        ),
        axis=1,
    ).astype(np.float32)


def _mhe_estimate(split: dict[str, np.ndarray], dt: float) -> np.ndarray:
    absolute = _forward_fill(split["relative_measurement"], split["valid"])
    absolute += split["ego_position"]
    count, length, _ = absolute.shape
    result = np.zeros((count, 6), dtype=np.float32)
    time = np.arange(length, dtype=np.float64) * dt
    for sequence in range(count):
        selected = split["valid"][sequence]
        selected_time = time[selected]
        for axis in range(3):
            values = absolute[sequence, selected, axis]
            degree = min(2, len(values) - 1)
            coefficients = np.polyfit(selected_time, values, degree)
            evaluation = time[-1]
            if degree == 2:
                result[sequence, axis] = 2.0 * coefficients[0] * evaluation + coefficients[1]
                result[sequence, axis + 3] = 2.0 * coefficients[0]
            elif degree == 1:
                result[sequence, axis] = coefficients[0]
    return result


def _kalman_estimate(
    split: dict[str, np.ndarray],
    dt: float,
    *,
    constant_acceleration: bool,
) -> np.ndarray:
    measurement = _forward_fill(split["relative_measurement"], split["valid"])
    measurement += split["ego_position"]
    count, length, _ = measurement.shape
    dimension = 9 if constant_acceleration else 6
    identity = np.eye(3)
    if constant_acceleration:
        transition = np.block(
            [
                [identity, dt * identity, 0.5 * dt**2 * identity],
                [np.zeros((3, 3)), identity, dt * identity],
                [np.zeros((3, 3)), np.zeros((3, 3)), identity],
            ]
        )
    else:
        transition = np.block(
            [[identity, dt * identity], [np.zeros((3, 3)), identity]]
        )
    observation = np.concatenate((identity, np.zeros((3, dimension - 3))), axis=1)
    process = np.eye(dimension) * (0.03 if constant_acceleration else 0.08)
    measurement_covariance = np.eye(3) * 0.01
    output = np.zeros((count, 6), dtype=np.float32)
    for sequence in range(count):
        state = np.zeros(dimension)
        state[:3] = measurement[sequence, 0]
        covariance = np.eye(dimension) * 10.0
        for step in range(1, length):
            state = transition @ state
            covariance = transition @ covariance @ transition.T + process
            if split["valid"][sequence, step]:
                innovation = measurement[sequence, step] - observation @ state
                innovation_covariance = (
                    observation @ covariance @ observation.T + measurement_covariance
                )
                gain = covariance @ observation.T @ np.linalg.inv(innovation_covariance)
                state = state + gain @ innovation
                covariance = (np.eye(dimension) - gain @ observation) @ covariance
        output[sequence, :3] = state[3:6]
        if constant_acceleration:
            output[sequence, 3:] = state[6:9]
    return output


def _imm_estimate(split: dict[str, np.ndarray], dt: float) -> np.ndarray:
    cv = _kalman_estimate(split, dt, constant_acceleration=False)
    ca = _kalman_estimate(split, dt, constant_acceleration=True)
    mhe = _mhe_estimate(split, dt)
    cv_residual = np.linalg.norm(cv[:, :3] - mhe[:, :3], axis=1)
    ca_residual = np.linalg.norm(ca - mhe, axis=1)
    ca_weight = np.exp(-ca_residual) / (
        np.exp(-ca_residual) + np.exp(-cv_residual) + 1e-12
    )
    return ((1.0 - ca_weight[:, None]) * cv + ca_weight[:, None] * ca).astype(np.float32)


class FlattenMLP(nn.Module):
    def __init__(self, history_length: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(history_length * 4, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 6),
        )

    def forward(self, features: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        return self.net(features)


class RecurrentEstimator(nn.Module):
    def __init__(self, cell: str, hidden_size: int) -> None:
        super().__init__()
        cells = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        self.recurrent = cells[cell](4, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 6)

    def forward(self, features: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        sequence, _ = self.recurrent(features)
        return self.output(sequence[:, -1])


class TemporalConvEstimator(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.Linear(hidden_size, 6)

    def forward(self, features: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.net(features.transpose(1, 2))
        return self.output(hidden[:, :, -1])


class DiagonalSSMEstimator(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input = nn.Linear(4, hidden_size)
        self.logit_decay = nn.Parameter(torch.zeros(hidden_size))
        self.output = nn.Linear(hidden_size, 6)

    def forward(self, features: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        hidden = torch.zeros(
            features.shape[0],
            self.input.out_features,
            device=features.device,
            dtype=features.dtype,
        )
        decay = torch.sigmoid(self.logit_decay)
        for step in range(features.shape[1]):
            hidden = decay * hidden + (1.0 - decay) * torch.tanh(
                self.input(features[:, step])
            )
        return self.output(hidden)


class PhysicsResidualEstimator(nn.Module):
    def __init__(self, hidden_size: int, *, contractive: bool) -> None:
        super().__init__()
        self.contractive = contractive
        if contractive:
            self.memory = ContractiveResidualMemory(
                input_size=4,
                hidden_size=hidden_size,
                output_size=6,
                recurrent_norm_bound=0.75,
                leak=0.5,
                output_bound=4.0,
            )
        else:
            self.memory = nn.GRU(4, hidden_size, batch_first=True)
            self.output = nn.Linear(hidden_size, 6)

    def forward(self, features: torch.Tensor, base: torch.Tensor | None = None) -> torch.Tensor:
        if base is None:
            raise ValueError("physics residual model requires a base estimate")
        if self.contractive:
            hidden = self.memory.initial_hidden(features.shape[0], device=features.device)
            residual = None
            for step in range(features.shape[1]):
                residual, hidden = self.memory(features[:, step], hidden)
            assert residual is not None
        else:
            sequence, _ = self.memory(features)
            residual = self.output(sequence[:, -1])
        return base + residual


def _metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    regime_index: np.ndarray,
) -> dict[str, object]:
    error = prediction - target
    result: dict[str, object] = {
        "velocity_mae": float(np.mean(np.abs(error[:, :3]))),
        "velocity_rmse": float(np.sqrt(np.mean(error[:, :3] ** 2))),
        "acceleration_mae": float(np.mean(np.abs(error[:, 3:]))),
        "acceleration_rmse": float(np.sqrt(np.mean(error[:, 3:] ** 2))),
        "joint_mae": float(np.mean(np.abs(error))),
        "joint_rmse": float(np.sqrt(np.mean(error**2))),
        "regimes": {},
    }
    for index, regime in enumerate(REGIMES):
        selected = regime_index == index
        regime_error = error[selected]
        result["regimes"][regime] = {
            "count": int(np.sum(selected)),
            "velocity_mae": float(np.mean(np.abs(regime_error[:, :3]))),
            "acceleration_mae": float(np.mean(np.abs(regime_error[:, 3:]))),
        }
    abrupt = np.isin(regime_index, [REGIMES.index(item) for item in ABRUPT_REGIMES])
    result["abrupt_velocity_mae"] = float(np.mean(np.abs(error[abrupt, :3])))
    result["steady_velocity_mae"] = float(np.mean(np.abs(error[~abrupt, :3])))
    return result


def _future_position_at(
    split: dict[str, np.ndarray],
    horizon: float,
    dt: float,
) -> np.ndarray:
    index = horizon / dt
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if upper >= split["future_position"].shape[1]:
        raise ValueError("requested horizon exceeds generated future")
    weight = index - lower
    return (
        (1.0 - weight) * split["future_position"][:, lower]
        + weight * split["future_position"][:, upper]
    )


def _current_position_estimate(split: dict[str, np.ndarray]) -> np.ndarray:
    relative = _forward_fill(split["relative_measurement"], split["valid"])
    return relative[:, -1] + split["ego_position"][:, -1]


def _conformal_tube_metrics(
    validation_prediction: np.ndarray,
    test_prediction: np.ndarray,
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    dt: float,
) -> dict[str, object]:
    validation_position = _current_position_estimate(validation)
    test_position = _current_position_estimate(test)
    output: dict[str, object] = {
        "calibration_unit": "complete_3d_future_position_at_fixed_horizon",
        "coverage_type": "split_conformal_marginal_joint_axis_orthotope",
        "horizons": {},
    }
    for horizon in (0.25, 0.5, 1.0, 2.0):
        validation_center = (
            validation_position
            + validation_prediction[:, :3] * horizon
            + 0.5 * validation_prediction[:, 3:] * horizon**2
        )
        test_center = (
            test_position
            + test_prediction[:, :3] * horizon
            + 0.5 * test_prediction[:, 3:] * horizon**2
        )
        validation_error = _future_position_at(validation, horizon, dt) - validation_center
        test_error = _future_position_at(test, horizon, dt) - test_center
        scale = np.maximum(np.median(np.abs(validation_error), axis=0), 1e-5)
        scores = np.max(np.abs(validation_error) / scale[None, :], axis=1)
        levels: dict[str, object] = {}
        for coverage in (0.90, 0.95, 0.99):
            rank = min(
                len(scores) - 1,
                int(np.ceil((len(scores) + 1) * coverage)) - 1,
            )
            quantile = float(np.sort(scores)[rank])
            radius = quantile * scale
            contained = np.all(np.abs(test_error) <= radius[None, :] + 1e-12, axis=1)
            directional_support = np.sum(
                np.abs(test_center)
                / np.maximum(np.linalg.norm(test_center, axis=1, keepdims=True), 1e-12)
                * radius[None, :],
                axis=1,
            )
            levels[f"q{int(coverage * 100)}"] = {
                "target_coverage": coverage,
                "empirical_coverage": float(np.mean(contained)),
                "underbound_count": int(np.sum(~contained)),
                "radius": radius.tolist(),
                "mean_full_width_l2": float(2.0 * np.linalg.norm(radius)),
                "p95_directional_support": float(
                    np.quantile(directional_support, 0.95)
                ),
            }
        output["horizons"][str(horizon)] = levels
    return output


def _train_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_base: np.ndarray,
    train_target: np.ndarray,
    validation_features: np.ndarray,
    validation_base: np.ndarray,
    validation_target: np.ndarray,
    config: EstimationBenchmarkConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    target_mean = train_target.mean(axis=0)
    target_scale = np.maximum(train_target.std(axis=0), 1e-3)
    base_train = (train_base - target_mean) / target_scale
    base_validation = (validation_base - target_mean) / target_scale
    target_train = (train_target - target_mean) / target_scale
    target_validation = (validation_target - target_mean) / target_scale
    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(base_train.astype(np.float32)),
        torch.from_numpy(target_train.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    validation_x = torch.from_numpy(validation_features).to(device)
    validation_b = torch.from_numpy(base_validation.astype(np.float32)).to(device)
    validation_y = torch.from_numpy(target_validation.astype(np.float32)).to(device)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    started = perf_counter()
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for features, base, target in loader:
            features = features.to(device)
            base = base.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, base)
            loss = torch.mean((prediction - target) ** 2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_x, validation_b)
            validation_loss = float(
                torch.mean((validation_prediction - validation_y) ** 2).cpu()
            )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    return model, {
        "best_validation_loss": best_loss,
        "epochs": config.epochs,
        "wall_seconds": perf_counter() - started,
        "history": history,
        "target_mean": target_mean.tolist(),
        "target_scale": target_scale.tolist(),
    }


def _predict_model(
    model: nn.Module,
    features: np.ndarray,
    base: np.ndarray,
    train_info: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    mean = np.asarray(train_info["target_mean"], dtype=np.float32)
    scale = np.asarray(train_info["target_scale"], dtype=np.float32)
    normalized_base = (base - mean) / scale
    tensor = torch.from_numpy(features).to(device)
    base_tensor = torch.from_numpy(normalized_base.astype(np.float32)).to(device)
    timings = []
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), 128):
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = perf_counter()
            output = model(tensor[start : start + 128], base_tensor[start : start + 128])
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((perf_counter() - began) * 1000.0)
            outputs.append(output.cpu().numpy())
    prediction = np.concatenate(outputs, axis=0) * scale + mean
    per_trajectory = np.repeat(
        np.asarray(timings) / np.minimum(128, len(features)),
        [min(128, len(features) - start) for start in range(0, len(features), 128)],
    )
    return prediction, {
        "mean_ms": float(np.mean(per_trajectory)),
        "p95_ms": float(np.quantile(per_trajectory, 0.95)),
        "p99_ms": float(np.quantile(per_trajectory, 0.99)),
    }


def run(config: EstimationBenchmarkConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = _generate_split(config.train_trajectories, config.seed + 11, config)
    validation = _generate_split(config.validation_trajectories, config.seed + 23, config)
    test = _generate_split(config.test_trajectories, config.seed + 37, config)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        np.savez_compressed(output / f"{name}.npz", **split)
    train_target = _targets(train)
    validation_target = _targets(validation)
    test_target = _targets(test)
    train_mhe = _mhe_estimate(train, config.dt)
    validation_mhe = _mhe_estimate(validation, config.dt)
    test_mhe = _mhe_estimate(test, config.dt)
    methods: dict[str, dict[str, object]] = {}
    classical_predictions = {
        "A_current_only": (
            np.zeros_like(validation_target),
            np.zeros_like(test_target),
        ),
        "CV_Kalman": (
            _kalman_estimate(validation, config.dt, constant_acceleration=False),
            _kalman_estimate(test, config.dt, constant_acceleration=False),
        ),
        "CA_Kalman": (
            _kalman_estimate(validation, config.dt, constant_acceleration=True),
            _kalman_estimate(test, config.dt, constant_acceleration=True),
        ),
        "EKF_linear_equivalent": (
            _kalman_estimate(validation, config.dt, constant_acceleration=True),
            _kalman_estimate(test, config.dt, constant_acceleration=True),
        ),
        "IMM": (
            _imm_estimate(validation, config.dt),
            _imm_estimate(test, config.dt),
        ),
        "MHE": (validation_mhe, test_mhe),
    }
    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    for name, (validation_prediction, prediction) in classical_predictions.items():
        validation_predictions[name] = validation_prediction
        test_predictions[name] = prediction
        methods[name] = {
            "metrics": _metrics(prediction, test_target, test["regime_index"]),
            "tube": _conformal_tube_metrics(
                validation_prediction,
                prediction,
                validation,
                test,
                config.dt,
            ),
            "latency": None,
            "trained": False,
        }
    train_raw = _features(train, ego_compensated=False)
    validation_raw = _features(validation, ego_compensated=False)
    test_raw = _features(test, ego_compensated=False)
    train_ego = _features(train, ego_compensated=True)
    validation_ego = _features(validation, ego_compensated=True)
    test_ego = _features(test, ego_compensated=True)
    models: list[tuple[str, nn.Module, np.ndarray, np.ndarray, np.ndarray, bool]] = [
        (
            "B_raw_L16_MLP",
            FlattenMLP(config.history_length, config.hidden_size),
            train_raw,
            validation_raw,
            test_raw,
            False,
        ),
        (
            "C_ego_L16_MLP",
            FlattenMLP(config.history_length, config.hidden_size),
            train_ego,
            validation_ego,
            test_ego,
            False,
        ),
        ("D_RNN", RecurrentEstimator("rnn", config.hidden_size), train_ego, validation_ego, test_ego, False),
        ("E_GRU", RecurrentEstimator("gru", config.hidden_size), train_ego, validation_ego, test_ego, False),
        ("F_LSTM", RecurrentEstimator("lstm", config.hidden_size), train_ego, validation_ego, test_ego, False),
        ("G_TCN", TemporalConvEstimator(config.hidden_size), train_ego, validation_ego, test_ego, False),
        ("H_SSM", DiagonalSSMEstimator(config.hidden_size), train_ego, validation_ego, test_ego, False),
        (
            "I_Physics_GRU",
            PhysicsResidualEstimator(config.hidden_size, contractive=False),
            train_ego,
            validation_ego,
            test_ego,
            True,
        ),
        (
            "M_Contractive_Physics_Memory",
            PhysicsResidualEstimator(config.hidden_size, contractive=True),
            train_ego,
            validation_ego,
            test_ego,
            True,
        ),
    ]
    checkpoint_dir = output / "models"
    checkpoint_dir.mkdir()
    for name, model, train_x, validation_x, test_x, uses_base in models:
        zero_train = train_mhe if uses_base else np.zeros_like(train_target)
        zero_validation = validation_mhe if uses_base else np.zeros_like(validation_target)
        zero_test = test_mhe if uses_base else np.zeros_like(test_target)
        trained, train_info = _train_model(
            model,
            train_x,
            zero_train,
            train_target,
            validation_x,
            zero_validation,
            validation_target,
            config,
            device,
        )
        prediction, latency = _predict_model(
            trained,
            test_x,
            zero_test,
            train_info,
            device,
        )
        validation_prediction, _ = _predict_model(
            trained,
            validation_x,
            zero_validation,
            train_info,
            device,
        )
        validation_predictions[name] = validation_prediction
        test_predictions[name] = prediction
        torch.save(
            {
                "state_dict": trained.state_dict(),
                "training": train_info,
                "config": asdict(config),
            },
            checkpoint_dir / f"{name}.pt",
        )
        methods[name] = {
            "metrics": _metrics(prediction, test_target, test["regime_index"]),
            "tube": _conformal_tube_metrics(
                validation_prediction,
                prediction,
                validation,
                test,
                config.dt,
            ),
            "latency": latency,
            "trained": True,
            "training": train_info,
            "uses_mhe_physical_base": uses_base,
        }
    ranked = sorted(
        methods,
        key=lambda name: methods[name]["metrics"]["joint_mae"],
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "git_sha": _git_sha(),
        "source_sha256": _source_hash(),
        "device": str(device),
        "total_matched_trajectories": (
            config.train_trajectories
            + config.validation_trajectories
            + config.test_trajectories
        ),
        "split_seed_separation": {
            "train": config.seed + 11,
            "validation": config.seed + 23,
            "test": config.seed + 37,
        },
        "methods": methods,
        "ranking_by_joint_mae": ranked,
        "formal_500k": False,
    }
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        output / "validation_predictions.npz",
        **validation_predictions,
    )
    np.savez_compressed(
        output / "test_predictions.npz",
        **test_predictions,
    )
    return summary
