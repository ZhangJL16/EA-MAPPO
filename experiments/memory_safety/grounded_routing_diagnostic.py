from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.memory_safety.estimation_benchmark import (
    EstimationBenchmarkConfig,
    _features,
    _generate_split,
    _targets,
)
from review_bundle.safety.grounded_memory import (
    CertifiedSafetyMemory,
    GroundedMemoryGraph,
    LearnedModuleMessages,
)


@dataclass(frozen=True)
class GroundedDiagnosticConfig:
    seed: int = 20260820
    train_trajectories: int = 600
    validation_trajectories: int = 200
    test_trajectories: int = 200
    epochs: int = 10
    hidden_size: int = 16
    adversarial_message_draws: int = 2000
    output_dir: str = "artifacts/grounded_memory_short_diagnostic"


class GroundedObjectSafetyProbe(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.object_update = nn.GRU(4, hidden_size, batch_first=True)
        self.motion_head = nn.Linear(hidden_size, 6)
        self.safety_head = nn.Linear(hidden_size, 2)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.object_update(history)
        latent = hidden[-1]
        return self.motion_head(latent), self.safety_head(latent)


class GenericJointProbe(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.memory = nn.GRU(4, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 8)

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, hidden = self.memory(history)
        output = self.head(hidden[-1])
        return output[:, :6], output[:, 6:]


def _safety_targets(split: dict[str, np.ndarray], dt: float) -> np.ndarray:
    relative_position = split["obstacle_position"][:, -1] - split["ego_position"][:, -1]
    obstacle_velocity = split["obstacle_velocity"][:, -1]
    ego_velocity = (
        split["ego_position"][:, -1] - split["ego_position"][:, -2]
    ) / dt
    relative_velocity = obstacle_velocity - ego_velocity
    distance = np.linalg.norm(relative_position, axis=1)
    closing_speed = -np.sum(relative_position * relative_velocity, axis=1) / np.maximum(
        distance, 1e-6
    )
    horizons = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    ego_future = (
        split["ego_position"][:, -1, None, :]
        + horizons[None, :, None] * ego_velocity[:, None, :]
    )
    obstacle_future = split["future_position"][:, :11]
    minimum_distance = np.min(np.linalg.norm(obstacle_future - ego_future, axis=2), axis=1)
    return np.stack((closing_speed, minimum_distance), axis=1).astype(np.float32)


def _normalized_targets(
    train_motion: np.ndarray,
    train_safety: np.ndarray,
    motion: np.ndarray,
    safety: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motion_scale = np.maximum(np.std(train_motion, axis=0), 1e-3)
    safety_scale = np.maximum(np.std(train_safety, axis=0), 1e-3)
    return motion / motion_scale, safety / safety_scale, motion_scale, safety_scale


def _fit_probe(
    model: nn.Module,
    train_x: np.ndarray,
    train_motion: np.ndarray,
    train_safety: np.ndarray,
    epochs: int,
) -> list[dict[str, float]]:
    torch.manual_seed(7)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    history = []
    x = torch.from_numpy(train_x)
    y_motion = torch.from_numpy(train_motion)
    y_safety = torch.from_numpy(train_safety)
    for epoch in range(epochs):
        permutation = torch.randperm(x.shape[0])
        losses = []
        for start in range(0, x.shape[0], 64):
            index = permutation[start : start + 64]
            predicted_motion, predicted_safety = model(x[index])
            motion_loss = torch.mean((predicted_motion - y_motion[index]) ** 2)
            safety_loss = torch.mean((predicted_safety - y_safety[index]) ** 2)
            loss = 0.5 * (motion_loss + safety_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append((float(loss.detach()), float(motion_loss.detach()), float(safety_loss.detach())))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean([item[0] for item in losses])),
                "motion_loss": float(np.mean([item[1] for item in losses])),
                "safety_loss": float(np.mean([item[2] for item in losses])),
            }
        )
    return history


def _evaluate_probe(
    model: nn.Module,
    features: np.ndarray,
    motion: np.ndarray,
    safety: np.ndarray,
    motion_scale: np.ndarray,
    safety_scale: np.ndarray,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        predicted_motion, predicted_safety = model(torch.from_numpy(features))
    motion_prediction = predicted_motion.numpy() * motion_scale
    safety_prediction = predicted_safety.numpy() * safety_scale
    return {
        "motion_mae": float(np.mean(np.abs(motion_prediction - motion))),
        "safety_mae": float(np.mean(np.abs(safety_prediction - safety))),
        "closing_speed_mae": float(np.mean(np.abs(safety_prediction[:, 0] - safety[:, 0]))),
        "future_min_distance_mae": float(np.mean(np.abs(safety_prediction[:, 1] - safety[:, 1]))),
    }


def _routing_diagnostic(draws: int, seed: int) -> dict[str, float | int | bool]:
    rng = np.random.default_rng(seed)
    rows = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    certified = CertifiedSafetyMemory(
        constraint_rows=rows,
        nominal_lower_bounds=np.full(6, -2.0),
        uncertainty_support=np.full(6, 0.25),
        provenance="short_diagnostic_fixed_box",
    )
    graph = GroundedMemoryGraph()
    digest = None
    violations = []
    interventions = []
    nominal_changes = []
    base = np.array([2.5, -2.5, 1.5])
    for _ in range(draws):
        messages = LearnedModuleMessages(
            object_message=rng.normal(0.0, 100.0, 3),
            route_message=rng.normal(0.0, 100.0, 3),
            energy_message=rng.normal(0.0, 100.0, 3),
        )
        result = graph.execute(base, certified, messages)
        if digest is None:
            digest = result.certified_constraint_digest
        elif result.certified_constraint_digest != digest:
            raise AssertionError("learned messages altered the certified constraint channel")
        violations.append(result.projection.max_violation)
        interventions.append(result.projection.intervention_norm)
        nominal_changes.append(float(np.linalg.norm(result.nominal_action - base)))
        if not result.projection.feasible:
            raise AssertionError("routing diagnostic unexpectedly produced an infeasible projection")
    smaller = CertifiedSafetyMemory(
        constraint_rows=rows,
        nominal_lower_bounds=np.full(6, -2.0),
        uncertainty_support=np.full(6, 0.10),
        provenance="nested_smaller_uncertainty",
    )
    zero = LearnedModuleMessages(np.zeros(3), np.zeros(3), np.zeros(3))
    large_result = graph.execute(base, certified, zero)
    small_result = graph.execute(base, smaller, zero)
    return {
        "adversarial_draws": draws,
        "all_actions_feasible": True,
        "max_constraint_violation": float(np.max(violations)),
        "mean_intervention": float(np.mean(interventions)),
        "mean_nominal_message_effect": float(np.mean(nominal_changes)),
        "certified_constraints_message_invariant": True,
        "nested_set_intervention_nonincrease": bool(
            small_result.projection.intervention_norm
            <= large_result.projection.intervention_norm + 1e-8
        ),
        "large_uncertainty_intervention": large_result.projection.intervention_norm,
        "small_uncertainty_intervention": small_result.projection.intervention_norm,
    }


def run_grounded_diagnostic(config: GroundedDiagnosticConfig) -> dict[str, object]:
    torch.set_num_threads(1)
    base = EstimationBenchmarkConfig(
        seed=config.seed,
        train_trajectories=config.train_trajectories,
        validation_trajectories=config.validation_trajectories,
        test_trajectories=config.test_trajectories,
        hidden_size=config.hidden_size,
        epochs=config.epochs,
        output_dir=config.output_dir,
    )
    train = _generate_split(config.train_trajectories, config.seed, base)
    validation = _generate_split(config.validation_trajectories, config.seed + 1, base)
    test = _generate_split(config.test_trajectories, config.seed + 2, base)
    train_x = _features(train, ego_compensated=True)
    validation_x = _features(validation, ego_compensated=True)
    test_x = _features(test, ego_compensated=True)
    train_motion = _targets(train)
    validation_motion = _targets(validation)
    test_motion = _targets(test)
    train_safety = _safety_targets(train, base.dt)
    validation_safety = _safety_targets(validation, base.dt)
    test_safety = _safety_targets(test, base.dt)
    normalized_train_motion, normalized_train_safety, motion_scale, safety_scale = _normalized_targets(
        train_motion, train_safety, train_motion, train_safety
    )
    normalized_validation_motion = validation_motion / motion_scale
    normalized_validation_safety = validation_safety / safety_scale
    candidate = GroundedObjectSafetyProbe(config.hidden_size)
    generic = GenericJointProbe(config.hidden_size)
    candidate_curve = _fit_probe(
        candidate,
        train_x,
        normalized_train_motion,
        normalized_train_safety,
        config.epochs,
    )
    generic_curve = _fit_probe(
        generic,
        train_x,
        normalized_train_motion,
        normalized_train_safety,
        config.epochs,
    )
    result = {
        "config": asdict(config),
        "candidate": {
            "validation": _evaluate_probe(
                candidate,
                validation_x,
                validation_motion,
                validation_safety,
                motion_scale,
                safety_scale,
            ),
            "test": _evaluate_probe(
                candidate,
                test_x,
                test_motion,
                test_safety,
                motion_scale,
                safety_scale,
            ),
            "curve": candidate_curve,
        },
        "generic_gru": {
            "validation": _evaluate_probe(
                generic,
                validation_x,
                validation_motion,
                validation_safety,
                motion_scale,
                safety_scale,
            ),
            "test": _evaluate_probe(
                generic,
                test_x,
                test_motion,
                test_safety,
                motion_scale,
                safety_scale,
            ),
            "curve": generic_curve,
        },
        "routing": _routing_diagnostic(config.adversarial_message_draws, config.seed + 17),
        "grounding_coverage": {
            "motion": "observed",
            "safety_proxy": "observed_from_simulator_state",
            "route": "missing_real_route_labels",
            "energy": "missing_safety_filtered_energy_labels",
        },
        "formal_500k": False,
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
