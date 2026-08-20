from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.memory_safety.closed_loop import _apply_velocity_limits
from experiments.memory_safety.grounded_real_dataset import (
    FEATURE_DIM,
    ROUTE_CLASSES,
    GroundedDatasetConfig,
    _boundary_project,
    _energy_state,
    _make_filter,
    _orthogonal,
    _transition_feature,
    make_scenario,
)
from experiments.uav_safety_filter.benchmark import (
    GOAL_RADIUS,
    HORIZONTAL_A_MAX,
    HORIZONTAL_V_MAX,
    SAFETY_DT,
    UAV_RADIUS,
    VERTICAL_A_MAX,
    VERTICAL_V_MAX,
    normalized_action_to_acceleration,
    sac_observation,
)
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel
from review_bundle.safety.collision.hocbf import SphericalObstacle
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.train_uav_energy_mc import load_frozen_sac


HISTORY_LENGTH = 16
MODEL_FEATURE_DIM = FEATURE_DIM + 9
ROUTE_REGRESSION_TARGETS = (
    "future_safe_path_length",
    "detour_ratio",
    "future_goal_progress",
    "future_intervention_count",
    "freeze_risk",
    "trajectory_curvature",
    "time_to_clear_obstacle",
)


@dataclass(frozen=True)
class PredictionDiagnosticConfig:
    dataset_dir: str
    output_dir: str
    training_seeds: tuple[int, ...] = (11, 23, 37)
    history_length: int = HISTORY_LENGTH
    hidden_size: int = 48
    epochs: int = 12
    batch_size: int = 256
    maximum_train_samples: int = 30_000
    maximum_validation_samples: int = 8_000
    maximum_test_samples: int = 16_000
    route_closed_loop_scenarios: int = 16
    route_closed_loop_max_steps: int = 800
    sac_checkpoint: str = (
        "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
        "phase1_navigation/checkpoint_transition_500000.zip"
    )
    device: str = "cpu"


class CurrentJointModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.route = nn.Linear(hidden_size, len(ROUTE_CLASSES))
        self.energy = nn.Linear(hidden_size, 2)
        self.route_regression = nn.Linear(hidden_size, len(ROUTE_REGRESSION_TARGETS))

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.backbone(history[:, -1])
        return self.route(latent), self.energy(latent), self.route_regression(latent)


class GenericJointGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.memory = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.route = nn.Linear(hidden_size, len(ROUTE_CLASSES))
        self.energy = nn.Linear(hidden_size, 2)
        self.route_regression = nn.Linear(hidden_size, len(ROUTE_REGRESSION_TARGETS))

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, hidden = self.memory(history)
        latent = hidden[-1]
        return self.route(latent), self.energy(latent), self.route_regression(latent)


class FOGMJointModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        branch = max(8, hidden_size // 3)
        self.object_memory = nn.GRU(8, branch, batch_first=True)
        self.route_memory = nn.GRU(8, branch, batch_first=True)
        self.energy_memory = nn.GRU(9, branch, batch_first=True)
        total = 3 * branch
        self.route = nn.Linear(total, len(ROUTE_CLASSES))
        self.energy = nn.Linear(total, 2)
        self.route_regression = nn.Linear(total, len(ROUTE_REGRESSION_TARGETS))

    @staticmethod
    def typed_channels(history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = history[..., :FEATURE_DIM]
        energy_state = history[..., FEATURE_DIM : FEATURE_DIM + 7]
        step_energy = history[..., FEATURE_DIM + 7 : FEATURE_DIM + 9]
        object_channel = torch.cat((feature[..., 7:13], feature[..., 21:23]), dim=-1)
        route_channel = torch.cat((feature[..., 13:19], feature[..., 19:21]), dim=-1)
        energy_channel = torch.cat((energy_state, step_energy), dim=-1)
        return object_channel, route_channel, energy_channel

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        object_channel, route_channel, energy_channel = self.typed_channels(history)
        _, object_hidden = self.object_memory(object_channel)
        _, route_hidden = self.route_memory(route_channel)
        _, energy_hidden = self.energy_memory(energy_channel)
        latent = torch.cat((object_hidden[-1], route_hidden[-1], energy_hidden[-1]), dim=-1)
        return self.route(latent), self.energy(latent), self.route_regression(latent)


def _load_dataset(dataset_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, set[int]], dict[int, dict[str, object]]]:
    with np.load(dataset_dir / "transitions.npz", allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    split_payload = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    splits = {
        name: set(int(value) for value in split_payload[name])
        for name in ("train", "validation", "test")
    }
    metadata = {
        int(row["trajectory_id"]): row
        for row in (
            json.loads(line)
            for line in (dataset_dir / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    return arrays, splits, metadata


def _model_features(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate(
        (
            arrays["feature"].astype(np.float32),
            arrays["energy_state"].astype(np.float32),
            arrays["safe_energy"][:, None].astype(np.float32),
            arrays["nominal_energy"][:, None].astype(np.float32),
        ),
        axis=1,
    )


def _split_indices(
    arrays: dict[str, np.ndarray],
    ids: set[int],
    metadata: dict[int, dict[str, object]],
    *,
    successful_only: bool,
) -> np.ndarray:
    accepted = ids
    if successful_only:
        accepted = {trajectory_id for trajectory_id in ids if bool(metadata[trajectory_id]["success"])}
    return np.flatnonzero(np.isin(arrays["trajectory_id"], sorted(accepted)))


def _sample_indices(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if indices.size <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, maximum, replace=False))


def build_history_batch(
    features: np.ndarray,
    trajectory_ids: np.ndarray,
    step_indices: np.ndarray,
    indices: np.ndarray,
    history_length: int,
) -> np.ndarray:
    output = np.zeros((indices.size, history_length, features.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        available = min(history_length, int(step_indices[index]) + 1)
        start = int(index) - available + 1
        candidate = features[start : int(index) + 1]
        if not np.all(trajectory_ids[start : int(index) + 1] == trajectory_ids[index]):
            raise AssertionError("history crossed a trajectory boundary")
        output[row, -available:] = candidate
    return output


def _target_scales(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = np.stack(
        (
            arrays["safe_energy_to_go"],
            arrays["future_energy_overhead_h40"],
        ),
        axis=1,
    ).astype(np.float32)
    route = np.stack(
        [np.clip(arrays[name], -20.0, 20.0) for name in ROUTE_REGRESSION_TARGETS],
        axis=1,
    ).astype(np.float32)
    energy_mean = np.mean(energy[train_indices], axis=0)
    energy_scale = np.maximum(np.std(energy[train_indices], axis=0), 1e-4)
    route_mean = np.mean(route[train_indices], axis=0)
    route_scale = np.maximum(np.std(route[train_indices], axis=0), 1e-4)
    return energy_mean, energy_scale, route_mean, route_scale


def _fit_joint(
    model: nn.Module,
    history: np.ndarray,
    route_class: np.ndarray,
    energy_target: np.ndarray,
    route_regression: np.ndarray,
    config: PredictionDiagnosticConfig,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(config.epochs):
        order = rng.permutation(history.shape[0])
        for start in range(0, order.size, config.batch_size):
            index = order[start : start + config.batch_size]
            x = torch.from_numpy(history[index])
            route_y = torch.from_numpy(route_class[index]).long()
            energy_y = torch.from_numpy(energy_target[index])
            regression_y = torch.from_numpy(route_regression[index])
            logits, energy, regression = model(x)
            loss = (
                torch.nn.functional.cross_entropy(logits, route_y)
                + 0.5 * torch.nn.functional.huber_loss(energy, energy_y)
                + 0.25 * torch.nn.functional.huber_loss(regression, regression_y)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    model.eval()
    return {"final_loss": float(np.mean(losses[-max(1, len(losses) // config.epochs) :]))}


def _macro_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    scores = []
    for label in range(len(ROUTE_CLASSES)):
        true_positive = np.sum((prediction == label) & (target == label))
        false_positive = np.sum((prediction == label) & (target != label))
        false_negative = np.sum((prediction != label) & (target == label))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def _metrics(
    model: nn.Module,
    history: np.ndarray,
    route_class: np.ndarray,
    energy_truth: np.ndarray,
    route_truth: np.ndarray,
    energy_mean: np.ndarray,
    energy_scale: np.ndarray,
    route_mean: np.ndarray,
    route_scale: np.ndarray,
) -> dict[str, object]:
    logits_parts = []
    energy_parts = []
    route_parts = []
    with torch.no_grad():
        for start in range(0, history.shape[0], 1024):
            logits, energy, route = model(torch.from_numpy(history[start : start + 1024]))
            logits_parts.append(logits.numpy())
            energy_parts.append(energy.numpy())
            route_parts.append(route.numpy())
    prediction = np.concatenate(logits_parts).argmax(axis=1)
    energy_prediction = np.concatenate(energy_parts) * energy_scale + energy_mean
    route_prediction = np.concatenate(route_parts) * route_scale + route_mean
    safe_error = energy_prediction[:, 0] - energy_truth[:, 0]
    overhead_error = energy_prediction[:, 1] - energy_truth[:, 1]
    return {
        "route": {
            "accuracy": float(np.mean(prediction == route_class)),
            "macro_f1": _macro_f1(prediction, route_class),
            "future_safe_path_mae": float(np.mean(np.abs(route_prediction[:, 0] - route_truth[:, 0]))),
            "detour_ratio_mae": float(np.mean(np.abs(route_prediction[:, 1] - route_truth[:, 1]))),
            "future_progress_mae": float(np.mean(np.abs(route_prediction[:, 2] - route_truth[:, 2]))),
            "future_intervention_count_mae": float(np.mean(np.abs(route_prediction[:, 3] - route_truth[:, 3]))),
            "freeze_risk_mae": float(np.mean(np.abs(route_prediction[:, 4] - route_truth[:, 4]))),
        },
        "energy": {
            "safe_energy_to_go_mae": float(np.mean(np.abs(safe_error))),
            "safe_energy_to_go_rmse": float(np.sqrt(np.mean(safe_error**2))),
            "safe_energy_to_go_bias": float(np.mean(safe_error)),
            "safe_energy_underestimation_rate": float(np.mean(safe_error < 0.0)),
            "mean_underestimation_magnitude": float(np.mean(np.maximum(-safe_error, 0.0))),
            "future_h40_overhead_mae": float(np.mean(np.abs(overhead_error))),
        },
    }


def _fit_distance_baseline(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, float]:
    speed = np.linalg.norm(arrays["velocity"], axis=1)
    distance = arrays["energy_state"][:, -1] * np.linalg.norm([4000.0, 4000.0, 400.0])
    design = np.stack((np.ones(distance.shape), distance, speed, arrays["intervention"]), axis=1)
    coefficients, *_ = np.linalg.lstsq(design[train_indices], arrays["safe_energy_to_go"][train_indices], rcond=None)
    prediction = np.maximum(0.0, design[test_indices] @ coefficients)
    truth = arrays["safe_energy_to_go"][test_indices]
    error = prediction - truth
    return {
        "safe_energy_to_go_mae": float(np.mean(np.abs(error))),
        "safe_energy_to_go_rmse": float(np.sqrt(np.mean(error**2))),
        "safe_energy_to_go_bias": float(np.mean(error)),
        "safe_energy_underestimation_rate": float(np.mean(error < 0.0)),
        "mean_underestimation_magnitude": float(np.mean(np.maximum(-error, 0.0))),
        "coefficients": coefficients.tolist(),
    }


def _fit_existing_mc(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict[str, float]:
    target_max = float(np.max(arrays["safe_energy_to_go"][train_indices]))
    estimator = EnergyToGoRegressor(
        battery_capacity=max(2.0 * target_max, 1.0),
        hidden_dim=48,
        batch_size=256,
        seed=seed,
        device="cpu",
    )
    estimator.fit(
        arrays["energy_state"][train_indices],
        arrays["safe_energy_to_go"][train_indices],
        arrays["energy_state"][validation_indices],
        arrays["safe_energy_to_go"][validation_indices],
        max_epochs=30,
        patience=5,
    )
    prediction = estimator.predict_batch(arrays["energy_state"][test_indices])
    truth = arrays["safe_energy_to_go"][test_indices]
    error = prediction - truth
    return {
        "safe_energy_to_go_mae": float(np.mean(np.abs(error))),
        "safe_energy_to_go_rmse": float(np.sqrt(np.mean(error**2))),
        "safe_energy_to_go_bias": float(np.mean(error)),
        "safe_energy_underestimation_rate": float(np.mean(error < 0.0)),
        "mean_underestimation_magnitude": float(np.mean(np.maximum(-error, 0.0))),
    }


def _route_correction(route_class: int, position: np.ndarray, velocity: np.ndarray, goal: np.ndarray) -> np.ndarray:
    direction = goal - position
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    lateral = _orthogonal(direction)
    if route_class == 1:
        return 1.5 * lateral
    if route_class == 2:
        return -1.5 * lateral
    if route_class == 3:
        return np.array([0.0, 0.0, 1.0])
    if route_class == 4:
        return np.array([0.0, 0.0, -1.0])
    if route_class == 5:
        return -0.15 * velocity
    return np.zeros(3)


def _live_history(features: list[np.ndarray], history_length: int) -> np.ndarray:
    history = np.zeros((1, history_length, MODEL_FEATURE_DIM), dtype=np.float32)
    selected = features[-history_length:]
    if selected:
        history[0, -len(selected) :] = np.asarray(selected)
    return history


def _predict_route(model: nn.Module, history: list[np.ndarray], history_length: int) -> int:
    if not history:
        return 0
    with torch.no_grad():
        logits, _, _ = model(torch.from_numpy(_live_history(history, history_length)))
    return int(torch.argmax(logits[0]).item())


def _closed_loop_route_rollout(
    policy,
    scenario_index: int,
    config: PredictionDiagnosticConfig,
    method: str,
    model: nn.Module | None,
) -> dict[str, float | int | bool]:
    scenario = make_scenario(scenario_index, 20260820 + 90_000, 20.0)
    telemetry = TelemetryCostModel()
    action_filter = _make_filter(telemetry)
    position = scenario.start.copy()
    velocity = np.zeros(3)
    obstacle_centers = [item.center.copy() for item in scenario.obstacles]
    obstacle_velocities = [item.velocity.copy() for item in scenario.obstacles]
    obstacle_accelerations = [item.acceleration.copy() for item in scenario.obstacles]
    policy_nominal = np.zeros(3)
    previous_executed = np.zeros(3)
    history: list[np.ndarray] = []
    route_sequence = []
    interventions = []
    path = 0.0
    energy = 0.0
    freeze = 0
    collision_steps = 0
    infeasible_steps = 0
    fallback_unsafe_steps = 0
    boundary_steps = 0
    success = False
    hysteresis_side = 0
    hysteresis_age = 0
    for step in range(config.route_closed_loop_max_steps):
        if step % 4 == 0:
            action, _ = policy.predict(sac_observation(position, velocity, scenario.goal), deterministic=True)
            policy_nominal = normalized_action_to_acceleration(action)
        routed_nominal = policy_nominal.copy()
        route_class = 0
        if method == "previous_action_only":
            routed_nominal = routed_nominal + 0.10 * previous_executed
        elif method == "previous_safe_trajectory":
            if len(history) >= 2:
                smoothed = np.mean(np.asarray(history[-8:])[:, 16:19], axis=0)
                routed_nominal = routed_nominal + 0.35 * smoothed * np.array([HORIZONTAL_A_MAX, HORIZONTAL_A_MAX, VERTICAL_A_MAX])
        elif method == "explicit_side_hysteresis":
            if hysteresis_age > 0:
                route_class = 1 if hysteresis_side > 0 else 2
                hysteresis_age -= 1
        elif method in {"generic_GRU", "FOGM_route"}:
            if model is None:
                raise ValueError("learned route method requires a model")
            route_class = _predict_route(model, history, config.history_length)
        routed_nominal = routed_nominal + _route_correction(route_class, position, velocity, scenario.goal)
        obstacles = tuple(
            SphericalObstacle(
                obstacle_centers[index],
                item.radius,
                obstacle_velocities[index],
                obstacle_accelerations[index],
                item.identifier,
            )
            for index, item in enumerate(scenario.obstacles)
        )
        filtered = action_filter.filter(
            position,
            velocity,
            routed_nominal,
            obstacles,
            progress_direction=scenario.goal - position,
        )
        next_velocity, realized_acceleration, _ = _apply_velocity_limits(velocity, filtered.acceleration, SAFETY_DT)
        nominal_velocity, nominal_realized_acceleration, _ = _apply_velocity_limits(
            velocity,
            policy_nominal,
            SAFETY_DT,
        )
        next_position = position + velocity * SAFETY_DT + 0.5 * realized_acceleration * SAFETY_DT**2
        next_position, projected_velocity, boundary = _boundary_project(next_position, next_velocity)
        progress = float(np.linalg.norm(scenario.goal - position) - np.linalg.norm(scenario.goal - next_position))
        safe_energy = telemetry.realized_cost(
            NavigationState(next_position.copy(), next_velocity.copy(), 0.0, (step + 1) * SAFETY_DT),
            realized_acceleration,
            SAFETY_DT,
        )
        nominal_energy = telemetry.realized_cost(
            NavigationState(
                position + velocity * SAFETY_DT + 0.5 * nominal_realized_acceleration * SAFETY_DT**2,
                nominal_velocity,
                0.0,
                (step + 1) * SAFETY_DT,
            ),
            nominal_realized_acceleration,
            SAFETY_DT,
        )
        model_feature = np.concatenate((_transition_feature(
            position,
            velocity,
            scenario.goal,
            obstacle_centers[0],
            obstacle_velocities[0],
            policy_nominal,
            realized_acceleration,
            filtered.diagnostics.intervention_norm,
            progress,
            True,
            boundary,
        ), _energy_state(position, velocity, scenario.goal), [safe_energy, nominal_energy])).astype(np.float32)
        history.append(model_feature)
        interventions.append(filtered.diagnostics.intervention_norm)
        infeasible_steps += int(not filtered.diagnostics.feasible)
        fallback_unsafe_steps += int(
            filtered.diagnostics.fallback_used
            and not filtered.diagnostics.fallback_satisfies_constraints
        )
        if method == "explicit_side_hysteresis" and filtered.diagnostics.intervention_norm > 0.05 and hysteresis_age == 0:
            goal_direction = scenario.goal - position
            goal_direction /= max(float(np.linalg.norm(goal_direction)), 1e-12)
            lateral = _orthogonal(goal_direction)
            obstacle_side = float((obstacle_centers[0] - position) @ lateral)
            hysteresis_side = -1 if obstacle_side > 0.0 else 1
            hysteresis_age = 40
            route_class = 1 if hysteresis_side > 0 else 2
        route_sequence.append(route_class)
        previous_position = position.copy()
        position = next_position
        velocity = projected_velocity
        previous_executed = realized_acceleration
        path += float(np.linalg.norm(position - previous_position))
        energy += safe_energy
        freeze += int(np.linalg.norm(velocity) < 0.2 and np.linalg.norm(position - scenario.goal) > GOAL_RADIUS)
        boundary_steps += int(boundary)
        collision_steps += int(
            min(
                np.linalg.norm(position - obstacle.center) - obstacle.radius - UAV_RADIUS
                for obstacle in obstacles
            )
            <= 0.0
        )
        from experiments.memory_safety.grounded_real_dataset import _advance_obstacle
        for obstacle_index, item in enumerate(scenario.obstacles):
            obstacle_centers[obstacle_index], obstacle_velocities[obstacle_index], obstacle_accelerations[obstacle_index], _ = _advance_obstacle(
                item,
                obstacle_centers[obstacle_index],
                obstacle_velocities[obstacle_index],
                obstacle_accelerations[obstacle_index],
                step,
            )
        if np.linalg.norm(position - scenario.goal) <= GOAL_RADIUS:
            success = True
            break
    steps = step + 1
    route_array = np.asarray(route_sequence)
    return {
        "method": method,
        "scenario": scenario.identifier,
        "success": success,
        "steps": steps,
        "collision_steps": collision_steps,
        "infeasible_steps": infeasible_steps,
        "fallback_unsafe_steps": fallback_unsafe_steps,
        "path_ratio": path / max(float(np.linalg.norm(scenario.goal - scenario.start)), 1e-12),
        "freeze_fraction": freeze / max(steps, 1),
        "intervention_fraction": float(np.mean(np.asarray(interventions) > 1e-6)),
        "mean_intervention": float(np.mean(interventions)),
        "route_switches": int(np.sum(route_array[1:] != route_array[:-1])) if route_array.size > 1 else 0,
        "boundary_contact_fraction": boundary_steps / max(steps, 1),
        "realized_energy": energy,
    }


def _aggregate_closed_loop(rows: list[dict[str, object]]) -> dict[str, object]:
    methods = sorted({str(row["method"]) for row in rows})
    return {
        method: {
            "rollouts": sum(row["method"] == method for row in rows),
            "success_rate": float(np.mean([row["success"] for row in rows if row["method"] == method])),
            "collision_steps": int(sum(row["collision_steps"] for row in rows if row["method"] == method)),
            "infeasible_steps": int(sum(row["infeasible_steps"] for row in rows if row["method"] == method)),
            "fallback_unsafe_steps": int(sum(row["fallback_unsafe_steps"] for row in rows if row["method"] == method)),
            "mean_path_ratio": float(np.mean([row["path_ratio"] for row in rows if row["method"] == method])),
            "mean_freeze_fraction": float(np.mean([row["freeze_fraction"] for row in rows if row["method"] == method])),
            "mean_intervention_fraction": float(np.mean([row["intervention_fraction"] for row in rows if row["method"] == method])),
            "mean_intervention": float(np.mean([row["mean_intervention"] for row in rows if row["method"] == method])),
            "mean_route_switches": float(np.mean([row["route_switches"] for row in rows if row["method"] == method])),
            "mean_energy": float(np.mean([row["realized_energy"] for row in rows if row["method"] == method])),
        }
        for method in methods
    }


def run_prediction_diagnostic(config: PredictionDiagnosticConfig) -> dict[str, object]:
    torch.set_num_threads(1)
    dataset_dir = Path(config.dataset_dir)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    arrays, splits, metadata = _load_dataset(dataset_dir)
    model_features = _model_features(arrays)
    train_indices = _sample_indices(
        _split_indices(arrays, splits["train"], metadata, successful_only=True),
        config.maximum_train_samples,
        101,
    )
    validation_indices = _sample_indices(
        _split_indices(arrays, splits["validation"], metadata, successful_only=True),
        config.maximum_validation_samples,
        102,
    )
    test_indices = _sample_indices(
        _split_indices(arrays, splits["test"], metadata, successful_only=True),
        config.maximum_test_samples,
        103,
    )
    if min(train_indices.size, validation_indices.size, test_indices.size) == 0:
        raise RuntimeError("successful trajectory-level split is empty")
    train_history = build_history_batch(
        model_features,
        arrays["trajectory_id"],
        arrays["step_in_trajectory"],
        train_indices,
        config.history_length,
    )
    test_history = build_history_batch(
        model_features,
        arrays["trajectory_id"],
        arrays["step_in_trajectory"],
        test_indices,
        config.history_length,
    )
    energy_mean, energy_scale, route_mean, route_scale = _target_scales(arrays, train_indices)

    def targets(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        energy = np.stack(
            (arrays["safe_energy_to_go"][indices], arrays["future_energy_overhead_h40"][indices]),
            axis=1,
        ).astype(np.float32)
        route = np.stack(
            [np.clip(arrays[name][indices], -20.0, 20.0) for name in ROUTE_REGRESSION_TARGETS],
            axis=1,
        ).astype(np.float32)
        return (
            arrays["route_class"][indices].astype(np.int64),
            (energy - energy_mean) / energy_scale,
            (route - route_mean) / route_scale,
        )

    train_route, train_energy, train_route_regression = targets(train_indices)
    test_route, _, _ = targets(test_indices)
    test_energy_truth = np.stack(
        (arrays["safe_energy_to_go"][test_indices], arrays["future_energy_overhead_h40"][test_indices]),
        axis=1,
    )
    test_route_truth = np.stack(
        [np.clip(arrays[name][test_indices], -20.0, 20.0) for name in ROUTE_REGRESSION_TARGETS],
        axis=1,
    )
    seed_results = []
    saved_models: list[tuple[int, nn.Module, nn.Module]] = []
    for seed in config.training_seeds:
        torch.manual_seed(seed)
        models: dict[str, nn.Module] = {
            "current_safety_state": CurrentJointModel(MODEL_FEATURE_DIM, config.hidden_size),
            "generic_GRU_history": GenericJointGRU(MODEL_FEATURE_DIM, config.hidden_size),
            "FOGM_route_safety_energy": FOGMJointModel(config.hidden_size),
        }
        method_result = {}
        for name, model in models.items():
            training = _fit_joint(
                model,
                train_history,
                train_route,
                train_energy,
                train_route_regression,
                config,
                seed,
            )
            method_result[name] = {
                "training": training,
                "test": _metrics(
                    model,
                    test_history,
                    test_route,
                    test_energy_truth,
                    test_route_truth,
                    energy_mean,
                    energy_scale,
                    route_mean,
                    route_scale,
                ),
            }
        method_result["distance_velocity"] = {
            "test": {"energy": _fit_distance_baseline(arrays, train_indices, test_indices)}
        }
        method_result["existing_MC_energy_to_go"] = {
            "test": {
                "energy": _fit_existing_mc(
                    arrays,
                    train_indices,
                    validation_indices,
                    test_indices,
                    seed,
                )
            }
        }
        seed_results.append({"seed": seed, "methods": method_result})
        saved_models.append((seed, models["generic_GRU_history"], models["FOGM_route_safety_energy"]))
        torch.save(
            {
                "seed": seed,
                "generic_state": models["generic_GRU_history"].state_dict(),
                "fogm_state": models["FOGM_route_safety_energy"].state_dict(),
                "energy_mean": energy_mean,
                "energy_scale": energy_scale,
                "route_mean": route_mean,
                "route_scale": route_scale,
            },
            output / f"models_seed_{seed}.pt",
        )
    policy = load_frozen_sac(Path(config.sac_checkpoint), config.device)
    closed_loop_rows = []
    for seed, generic, fogm in saved_models:
        for scenario_index in range(config.route_closed_loop_scenarios):
            for method, model in (
                ("no_route_memory", None),
                ("previous_action_only", None),
                ("previous_safe_trajectory", None),
                ("explicit_side_hysteresis", None),
                ("generic_GRU", generic),
                ("FOGM_route", fogm),
            ):
                row = _closed_loop_route_rollout(
                    policy,
                    scenario_index,
                    config,
                    method,
                    model,
                )
                row["training_seed"] = seed
                closed_loop_rows.append(row)
    result = {
        "config": asdict(config),
        "split_unit": "whole_trajectory",
        "samples": {
            "train": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
        },
        "route_labels": ["corridor_identity", *ROUTE_REGRESSION_TARGETS],
        "energy_labels": ["E_safe_terminal_MC", "E_nominal", "E_overhead", "future_H40_E_overhead"],
        "seed_results": seed_results,
        "closed_loop_route": {
            "hard_safety_certificate": "identical exact-state sampled-data HOCBF across route methods",
            "rows": closed_loop_rows,
            "aggregate": _aggregate_closed_loop(closed_loop_rows),
        },
    }
    (output / "prediction_diagnostic.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["closed_loop_route"]["aggregate"], indent=2), flush=True)
    return result
