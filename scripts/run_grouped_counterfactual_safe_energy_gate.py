from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from torch import nn
from torch.nn import functional

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.counterfactual_safe_energy.core import (
    DEFAULT_INTERVENTIONS,
    CorrelatedActionIntervention,
    InterventionSpec,
    grouped_outer_fold,
    intervention_seed,
    relative_range,
)
from experiments.uav_energy_parallel import (
    ParallelUAVEnvPool,
    WorkerReset,
    WorkerRetarget,
    WorkerStep,
)
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    NavigationTask,
    environment_from_args,
    environment_kwargs_from_args,
    generate_stratified_navigation_tasks,
)


PROTOCOL = "GROUPED_COUNTERFACTUAL_SAFE_ENERGY_DATA_GATE_V1"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def rollout_paths(output: Path, scene_index: int, name: str) -> tuple[Path, Path]:
    root = output / "rollouts" / f"scene_{scene_index:03d}"
    return root / f"{name}.npz", root / f"{name}.json"


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


@dataclass
class ActiveRollout:
    scene_index: int
    intervention_index: int
    task: NavigationTask
    spec: InterventionSpec
    behavior: CorrelatedActionIntervention
    observation: np.ndarray
    initial_observation: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    leg: int = 0
    task_success: bool = False
    return_success: bool = False
    task_steps: int = 0
    return_steps: int = 0
    task_path_length: float = 0.0
    return_path_length: float = 0.0
    end_reason: str = ""
    total_reward: float = 0.0
    compact_observations: list[np.ndarray] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    velocities: list[np.ndarray] = field(default_factory=list)
    frozen_actions: list[np.ndarray] = field(default_factory=list)
    behavior_actions: list[np.ndarray] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    realized_energy: list[float] = field(default_factory=list)
    progress: list[float] = field(default_factory=list)
    collision: list[bool] = field(default_factory=list)
    boundary_contact: list[bool] = field(default_factory=list)
    intervened: list[bool] = field(default_factory=list)
    emergency: list[bool] = field(default_factory=list)
    intervention_norm: list[float] = field(default_factory=list)
    nominal_safe: list[bool] = field(default_factory=list)
    nominal_slack: list[float] = field(default_factory=list)
    executed_slack: list[float] = field(default_factory=list)
    legs: list[int] = field(default_factory=list)

    def add_step(
        self,
        frozen_action: np.ndarray,
        behavior_action: np.ndarray,
        result: WorkerStep,
    ) -> None:
        info = result.info
        geometry = info.get("projection_geometry")
        geometry = geometry if isinstance(geometry, dict) else {}
        self.compact_observations.append(
            np.asarray(self.observation[:7], dtype=np.float32)
        )
        # Store the simulator's exact pre-action state.  The compact
        # observation is intentionally float32 and goal-relative, so its
        # inverse is not an exact snapshot near physical boundaries.
        self.positions.append(np.asarray(self.position, dtype=np.float32))
        self.velocities.append(np.asarray(self.velocity, dtype=np.float32))
        self.frozen_actions.append(np.asarray(frozen_action, dtype=np.float32))
        self.behavior_actions.append(np.asarray(behavior_action, dtype=np.float32))
        self.executed_actions.append(
            np.asarray(info.get("executed_action", behavior_action), dtype=np.float32)
        )
        self.realized_energy.append(finite_float(info.get("realized_energy_cost"), 0.0))
        self.progress.append(finite_float(info.get("progress"), 0.0))
        self.collision.append(bool(info.get("obstacle_collision", False)))
        self.boundary_contact.append(bool(info.get("boundary_contact", False)))
        self.intervened.append(bool(info.get("hocbf_intervened", False)))
        self.emergency.append(bool(info.get("hocbf_emergency_brake", False)))
        self.intervention_norm.append(
            finite_float(info.get("hocbf_intervention_norm"), 0.0)
        )
        self.nominal_safe.append(bool(geometry.get("nominal_safe", False)))
        self.nominal_slack.append(finite_float(geometry.get("minimum_nominal_slack")))
        self.executed_slack.append(finite_float(geometry.get("minimum_executed_slack")))
        self.legs.append(self.leg)
        self.total_reward += result.reward
        if self.leg == 0:
            self.task_steps += 1
        else:
            self.return_steps += 1
        self.observation = result.observation
        self.position = result.position
        self.velocity = result.velocity

    @property
    def transition_count(self) -> int:
        return len(self.realized_energy)


def _numeric(values: list[float], *, dtype=np.float32) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def save_rollout(
    output: Path,
    rollout: ActiveRollout,
    *,
    capacity: float,
    d_max: float,
    charger_position: np.ndarray,
) -> dict[str, object]:
    if rollout.transition_count == 0:
        raise RuntimeError("cannot save an empty counterfactual rollout")
    frozen_actions = np.stack(rollout.frozen_actions)
    behavior_actions = np.stack(rollout.behavior_actions)
    executed_actions = np.stack(rollout.executed_actions)
    energy = _numeric(rollout.realized_energy)
    interventions = _numeric(rollout.intervention_norm)
    mission_success = bool(rollout.task_success and rollout.return_success)
    task_distance = float(rollout.task.straight_line_distance)
    return_distance = float(
        np.linalg.norm(rollout.task.goal_position - charger_position)
    )
    total_energy = float(np.sum(energy, dtype=np.float64))
    path_length = rollout.task_path_length + rollout.return_path_length
    action_delta = behavior_actions - frozen_actions
    arrays = {
        "initial_observation": rollout.initial_observation.astype(np.float32),
        "compact_observations": np.stack(rollout.compact_observations).astype(np.float32),
        "positions": np.stack(rollout.positions).astype(np.float32),
        "velocities": np.stack(rollout.velocities).astype(np.float32),
        "frozen_actions": frozen_actions.astype(np.float32),
        "behavior_actions": behavior_actions.astype(np.float32),
        "executed_actions": executed_actions.astype(np.float32),
        "realized_energy": energy,
        "progress": _numeric(rollout.progress),
        "collision": np.asarray(rollout.collision, dtype=np.bool_),
        "boundary_contact": np.asarray(rollout.boundary_contact, dtype=np.bool_),
        "hocbf_intervened": np.asarray(rollout.intervened, dtype=np.bool_),
        "hocbf_emergency": np.asarray(rollout.emergency, dtype=np.bool_),
        "hocbf_intervention_norm": interventions,
        "nominal_safe": np.asarray(rollout.nominal_safe, dtype=np.bool_),
        "minimum_nominal_slack": _numeric(rollout.nominal_slack),
        "minimum_executed_slack": _numeric(rollout.executed_slack),
        "leg": np.asarray(rollout.legs, dtype=np.int8),
    }
    npz_path, json_path = rollout_paths(
        output, rollout.scene_index, rollout.spec.name
    )
    save_npz_atomic(npz_path, **arrays)
    summary = {
        "protocol": PROTOCOL,
        "scene_index": rollout.scene_index,
        "intervention_index": rollout.intervention_index,
        "intervention": rollout.spec.as_dict(),
        "task_success": rollout.task_success,
        "return_success": rollout.return_success,
        "mission_success": mission_success,
        "task_steps": rollout.task_steps,
        "return_steps": rollout.return_steps,
        "transitions": rollout.transition_count,
        "task_path_length": rollout.task_path_length,
        "return_path_length": rollout.return_path_length,
        "path_ratio": path_length / max(task_distance + return_distance, 1e-9),
        "task_distance_fraction": task_distance / d_max,
        "return_distance_fraction": return_distance / d_max,
        "total_realized_energy": total_energy,
        "energy_fraction": total_energy / capacity,
        "total_reward": rollout.total_reward,
        "collision_steps": int(np.sum(arrays["collision"])),
        "boundary_contact_steps": int(np.sum(arrays["boundary_contact"])),
        "hocbf_intervention_fraction": float(np.mean(arrays["hocbf_intervened"])),
        "hocbf_emergency_fraction": float(np.mean(arrays["hocbf_emergency"])),
        "mean_intervention_norm": float(np.mean(interventions)),
        "nominal_unsafe_fraction": float(1.0 - np.mean(arrays["nominal_safe"])),
        "behavior_deviation_rms": float(np.sqrt(np.mean(action_delta**2))),
        "executed_deviation_rms": float(
            np.sqrt(np.mean((executed_actions - frozen_actions) ** 2))
        ),
        "minimum_nominal_slack": (
            None
            if np.all(np.isnan(arrays["minimum_nominal_slack"]))
            else float(np.nanmin(arrays["minimum_nominal_slack"]))
        ),
        "minimum_executed_slack": (
            None
            if np.all(np.isnan(arrays["minimum_executed_slack"]))
            else float(np.nanmin(arrays["minimum_executed_slack"]))
        ),
        "end_reason": rollout.end_reason,
        "npz": str(npz_path),
    }
    atomic_json(json_path, summary)
    return summary


class OutcomeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def fit_predict_regression(
    features: np.ndarray,
    targets: np.ndarray,
    scene_indices: np.ndarray,
    *,
    outer_fold: int,
    seed: int,
    device: str,
    max_epochs: int = 400,
    patience: int = 30,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    outer = grouped_outer_fold(scene_indices)
    test = outer == outer_fold
    candidates = ~test
    validation = candidates & (((scene_indices * 17 + outer_fold) % 5) == 0)
    train = candidates & ~validation
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) == 0:
        raise ValueError("grouped split produced an empty regression partition")
    mean = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = ((features - mean) / scale).astype(np.float32)
    target_mean = float(targets[train].mean())
    target_scale = float(targets[train].std())
    if target_scale < 1e-6:
        target_scale = 1.0
    y = ((targets - target_mean) / target_scale).astype(np.float32)
    torch.manual_seed(seed)
    model = OutcomeMLP(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x_tensor = torch.as_tensor(x, device=device)
    y_tensor = torch.as_tensor(y, device=device)
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train)
    validation_indices = np.flatnonzero(validation)
    best_state = None
    best_epoch = 0
    best_mae = math.inf
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        for start in range(0, order.size, 64):
            index = torch.as_tensor(order[start : start + 64], device=device)
            prediction = model(x_tensor[index])
            loss = functional.huber_loss(prediction, y_tensor[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            index = torch.as_tensor(validation_indices, device=device)
            predicted = model(x_tensor[index]) * target_scale + target_mean
            mae = float(
                torch.mean(
                    torch.abs(
                        predicted
                        - torch.as_tensor(
                            targets[validation_indices], dtype=torch.float32, device=device
                        )
                    )
                ).cpu()
            )
        if mae < best_mae - 1e-7:
            best_mae = mae
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if epoch >= 20 and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("outcome MLP failed to select a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_indices = np.flatnonzero(test)
    with torch.no_grad():
        index = torch.as_tensor(test_indices, device=device)
        prediction = (
            model(x_tensor[index]) * target_scale + target_mean
        ).cpu().numpy()
    return test_indices, prediction.astype(np.float64), {
        "fold": outer_fold,
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
        "best_epoch": best_epoch,
        "stopped_epoch": epoch,
        "best_validation_mae": best_mae,
    }


def crossfit_regression(
    features: np.ndarray,
    targets: np.ndarray,
    scene_indices: np.ndarray,
    *,
    seed: int,
    device: str,
) -> dict[str, object]:
    predictions = np.empty(targets.shape, dtype=np.float64)
    folds = []
    for fold in range(3):
        indices, values, metadata = fit_predict_regression(
            features,
            targets,
            scene_indices,
            outer_fold=fold,
            seed=seed + fold,
            device=device,
        )
        predictions[indices] = values
        metadata["test_mae"] = float(np.mean(np.abs(values - targets[indices])))
        folds.append(metadata)
    return {
        "mae": float(np.mean(np.abs(predictions - targets))),
        "rmse": float(np.sqrt(np.mean((predictions - targets) ** 2))),
        "folds": folds,
        "predictions": predictions,
    }


def frozen_embeddings(policy: SAC, observations: np.ndarray, *, device: str) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float32)
    actor = policy.policy.actor
    actor.eval()
    encoded = []
    with torch.no_grad():
        for start in range(0, values.shape[0], 512):
            batch = torch.as_tensor(values[start : start + 512], device=device)
            features = actor.extract_features(batch, actor.features_extractor)
            encoded.append(features.cpu().numpy())
    return np.concatenate(encoded).astype(np.float32)


def load_completed_rows(
    output: Path,
    *,
    num_scenes: int,
    interventions: tuple[InterventionSpec, ...],
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    observations = []
    for scene_index in range(num_scenes):
        for spec in interventions:
            npz_path, json_path = rollout_paths(output, scene_index, spec.name)
            if not npz_path.is_file() or not json_path.is_file():
                raise FileNotFoundError(f"incomplete rollout: scene={scene_index} {spec.name}")
            rows.append(json.loads(json_path.read_text(encoding="utf-8")))
            with np.load(npz_path) as values:
                observations.append(values["initial_observation"])
    return rows, np.stack(observations).astype(np.float32)


def aggregate_gate(
    rows: list[dict[str, object]],
    initial_observations: np.ndarray,
    policy: SAC,
    *,
    num_scenes: int,
    interventions: tuple[InterventionSpec, ...],
    device: str,
    model_seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    scene = np.asarray([int(row["scene_index"]) for row in rows], dtype=np.int64)
    descriptor = np.stack(
        [
            interventions[int(row["intervention_index"])].descriptor()
            for row in rows
        ]
    )
    geometry = np.asarray(
        [
            [row["task_distance_fraction"], row["return_distance_fraction"]]
            for row in rows
        ],
        dtype=np.float32,
    )
    embeddings = frozen_embeddings(policy, initial_observations, device=device)
    geometry_features = np.concatenate((geometry, descriptor), axis=1)
    learned_features = np.concatenate((embeddings, descriptor), axis=1)
    energy = np.asarray([row["energy_fraction"] for row in rows], dtype=np.float64)
    mission_success = np.asarray([row["mission_success"] for row in rows], dtype=np.bool_)
    unsafe = np.asarray(
        [row["nominal_unsafe_fraction"] for row in rows], dtype=np.float64
    )
    successful = mission_success & np.isfinite(energy)
    if successful.sum() < 30 or np.unique(scene[successful]).size < 15:
        raise RuntimeError("too few successful grouped missions for the energy Gate")
    energy_geometry = crossfit_regression(
        geometry_features[successful],
        energy[successful],
        scene[successful],
        seed=model_seed,
        device=device,
    )
    energy_learned = crossfit_regression(
        learned_features[successful],
        energy[successful],
        scene[successful],
        seed=model_seed,
        device=device,
    )
    safety_geometry = crossfit_regression(
        geometry_features,
        unsafe,
        scene,
        seed=model_seed + 100,
        device=device,
    )
    safety_learned = crossfit_regression(
        learned_features,
        unsafe,
        scene,
        seed=model_seed + 100,
        device=device,
    )
    maximum_replay_error = 0.0
    energy_spreads = []
    safety_changes = []
    for scene_index in range(num_scenes):
        mask = scene == scene_index
        reference = initial_observations[np.flatnonzero(mask)[0]]
        maximum_replay_error = max(
            maximum_replay_error,
            float(np.max(np.abs(initial_observations[mask] - reference))),
        )
        success_mask = mask & successful
        if success_mask.sum() >= 2:
            energy_spreads.append(relative_range(energy[success_mask]))
        scene_rows = [rows[index] for index in np.flatnonzero(mask)]
        unsafe_range = float(np.ptp(unsafe[mask]))
        categorical_change = len(
            {
                (
                    bool(row["mission_success"]),
                    int(row["collision_steps"]) > 0,
                )
                for row in scene_rows
            }
        ) > 1
        safety_changes.append(unsafe_range >= 0.05 or categorical_change)
    energy_spreads_array = np.asarray(energy_spreads, dtype=np.float64)
    intervention_medians = {
        spec.name: float(
            np.median(
                [
                    row["behavior_deviation_rms"]
                    for row in rows
                    if row["intervention"]["name"] == spec.name
                ]
            )
        )
        for spec in interventions
    }
    energy_improvement = 1.0 - energy_learned["mae"] / energy_geometry["mae"]
    safety_improvement = 1.0 - safety_learned["mae"] / safety_geometry["mae"]
    checks = {
        "all_rollouts_complete": len(rows) == num_scenes * len(interventions),
        "matched_scene_replay_at_1e_6": maximum_replay_error <= 1e-6,
        "nonnominal_interventions_realized": all(
            intervention_medians[spec.name] > 0.01
            for spec in interventions
            if spec.name != "nominal"
        ),
        "energy_spread_in_at_least_25_percent_of_eligible_scenes": bool(
            energy_spreads_array.size > 0
            and np.mean(energy_spreads_array >= 0.05) >= 0.25
        ),
        "safety_or_outcome_change_in_at_least_25_percent_of_scenes": bool(
            np.mean(safety_changes) >= 0.25
        ),
        "learned_energy_mae_beats_geometry_by_2_percent": energy_improvement >= 0.02,
        "learned_safety_mae_beats_geometry_by_2_percent": safety_improvement >= 0.02,
    }
    for result in (energy_geometry, energy_learned, safety_geometry, safety_learned):
        result.pop("predictions", None)
    report = {
        "checks": checks,
        "promotable": all(checks.values()),
        "status": (
            "PROMOTE_TO_JOINT_POLICY_PILOT"
            if all(checks.values())
            else "DO_NOT_MODIFY_R3_ACTOR"
        ),
        "num_scenes": num_scenes,
        "num_rollouts": len(rows),
        "num_successful_missions": int(successful.sum()),
        "successful_scene_count": int(np.unique(scene[successful]).size),
        "maximum_matched_initial_observation_error": maximum_replay_error,
        "intervention_behavior_deviation_rms_median": intervention_medians,
        "eligible_energy_spread_scenes": int(energy_spreads_array.size),
        "fraction_energy_spread_at_least_5_percent": float(
            np.mean(energy_spreads_array >= 0.05)
        ),
        "fraction_safety_or_outcome_change": float(np.mean(safety_changes)),
        "energy_prediction": {
            "geometry": energy_geometry,
            "learned": energy_learned,
            "relative_mae_improvement": float(energy_improvement),
        },
        "nominal_unsafe_prediction": {
            "geometry": safety_geometry,
            "learned": safety_learned,
            "relative_mae_improvement": float(safety_improvement),
        },
        "outcome_prevalence": {
            "mission_success": float(np.mean(mission_success)),
            "any_collision": float(
                np.mean([int(row["collision_steps"]) > 0 for row in rows])
            ),
        },
    }
    prediction_arrays = {
        "scene_index": scene,
        "mission_success": mission_success,
        "energy_fraction": energy,
        "nominal_unsafe_fraction": unsafe,
        "intervention_descriptor": descriptor,
        "geometry_features": geometry_features,
        "frozen_r3_embedding": embeddings,
    }
    return report, prediction_arrays


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--battery-calibration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-scenes", type=int, default=150)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--task-seed", type=int, default=310_001)
    parser.add_argument("--world-seed", type=int, default=320_001)
    parser.add_argument("--intervention-seed", type=int, default=330_001)
    parser.add_argument("--model-seed", type=int, default=340_001)
    parser.add_argument("--max-policy-steps", type=int, default=4000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--nominal-source-only",
        action="store_true",
        help="collect only nominal rollouts for a downstream fresh fork experiment",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_scenes = 5
        args.num_workers = min(args.num_workers, 5)
        # Long enough for the nearest-distance task to reach its goal and
        # exercise the matched-obstacle retarget path; still not evidence.
        args.max_policy_steps = 256
    if args.num_scenes <= 0 or args.num_scenes % 5:
        parser.error("--num-scenes must be a positive multiple of five")
    if args.num_workers <= 0 or args.max_policy_steps <= 0:
        parser.error("worker count and policy-step limit must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for name in ("artifact", "checkpoint", "battery_calibration", "protocol", "output_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    required = [
        args.artifact / "config.json",
        args.checkpoint,
        args.battery_calibration,
        args.protocol,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if args.dry_run:
        print(json.dumps({"missing": missing, "would_run": not missing}, indent=2))
        return int(bool(missing))
    if missing:
        raise FileNotFoundError("missing prerequisites: " + ", ".join(missing))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not fresh: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        (args.output_dir / "FAILED.json").unlink(missing_ok=True)
        (args.output_dir / "COMPLETED.json").unlink(missing_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    environment_args, reconstruction = reconstruct_environment_args(
        args.artifact, device=args.device, seed=args.world_seed
    )
    environment_args.phase1_episode_max_policy_steps = args.max_policy_steps
    environment_args.phase1_max_steps = args.max_policy_steps
    probe = environment_from_args(environment_args, phase=SACTrainingPhase.NAVIGATION)
    d_max = float(probe.d_max)
    charger_position = probe.charger_position.copy()
    policy = SAC.load(
        args.checkpoint, env=probe, device=args.device, print_system_info=False
    )
    probe.close()
    calibration = json.loads(args.battery_calibration.read_text(encoding="utf-8"))
    capacity = float(calibration["calibrated_battery_capacity"])
    tasks = generate_stratified_navigation_tasks(
        num_tasks=args.num_scenes, seed=args.task_seed
    )
    interventions = (
        (DEFAULT_INTERVENTIONS[0],)
        if args.nominal_source_only
        else DEFAULT_INTERVENTIONS
    )
    protocol = "NOMINAL_FORK_SOURCE_V1" if args.nominal_source_only else PROTOCOL
    manifest = {
        "status": "RUNNING",
        "protocol": protocol,
        "formal_evidence": not args.smoke and not args.nominal_source_only,
        "artifact": args.artifact,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "battery_calibration": args.battery_calibration,
        "battery_calibration_sha256": file_sha256(args.battery_calibration),
        "protocol_document": args.protocol,
        "protocol_sha256": file_sha256(args.protocol),
        "num_scenes": args.num_scenes,
        "num_workers": args.num_workers,
        "max_policy_steps_per_leg": args.max_policy_steps,
        "task_seed": args.task_seed,
        "world_seed": args.world_seed,
        "intervention_seed": args.intervention_seed,
        "model_seed": args.model_seed,
        "interventions": [spec.as_dict() for spec in interventions],
        "capacity": capacity,
        "device": args.device,
        "environment_reconstruction": reconstruction,
        "state_snapshot_format": "simulator_pre_action_float32_v1",
        "exact_command": [sys.executable, *sys.argv],
        "started_unix": time.time(),
    }
    atomic_json(args.output_dir / "RUNNING.json", manifest)
    completed = 0
    try:
        pending = deque()
        for scene_index, task in enumerate(tasks):
            for intervention_index, spec in enumerate(interventions):
                npz_path, json_path = rollout_paths(
                    args.output_dir, scene_index, spec.name
                )
                if args.resume and npz_path.is_file() and json_path.is_file():
                    completed += 1
                else:
                    pending.append((scene_index, intervention_index, task, spec))
        total = args.num_scenes * len(interventions)
        if pending:
            kwargs = environment_kwargs_from_args(
                environment_args, phase=SACTrainingPhase.NAVIGATION
            )
            kwargs["max_steps_per_task"] = args.max_policy_steps
            kwargs["phase1_episode_max_policy_steps"] = args.max_policy_steps
            workers = min(args.num_workers, len(pending))
            with ParallelUAVEnvPool(kwargs, num_workers=workers) as pool:
                active: dict[int, ActiveRollout] = {}

                def assign(worker_ids: list[int]) -> None:
                    requests = []
                    assigned = []
                    for worker_id in worker_ids:
                        if not pending:
                            break
                        job = pending.popleft()
                        scene_index, intervention_index, task, spec = job
                        requests.append(
                            WorkerReset(
                                worker_id,
                                args.world_seed + scene_index,
                                {
                                    "start_position": task.start_position,
                                    "start_velocity": task.initial_velocity,
                                    "task_point": task.goal_position,
                                },
                            )
                        )
                        assigned.append((worker_id, job))
                    if not requests:
                        return
                    resets = pool.reset_many(requests)
                    for worker_id, job in assigned:
                        scene_index, intervention_index, task, spec = job
                        observation = resets[worker_id].observation
                        active[worker_id] = ActiveRollout(
                            scene_index=scene_index,
                            intervention_index=intervention_index,
                            task=task,
                            spec=spec,
                            behavior=CorrelatedActionIntervention(
                                spec,
                                seed=intervention_seed(
                                    args.intervention_seed,
                                    scene_index,
                                    intervention_index,
                                ),
                            ),
                            observation=observation,
                            initial_observation=observation.copy(),
                            position=resets[worker_id].position.copy(),
                            velocity=resets[worker_id].velocity.copy(),
                        )

                assign(list(range(workers)))
                while active:
                    worker_ids = list(active)
                    observations = np.stack(
                        [active[worker_id].observation for worker_id in worker_ids]
                    )
                    frozen_actions, _ = policy.predict(
                        observations, deterministic=True
                    )
                    behavior_actions = np.stack(
                        [
                            active[worker_id].behavior(frozen_action)
                            for worker_id, frozen_action in zip(
                                worker_ids, frozen_actions, strict=True
                            )
                        ]
                    )
                    results = pool.step_many(worker_ids, behavior_actions)
                    retargets = []
                    available = []
                    for position, worker_id in enumerate(worker_ids):
                        rollout = active[worker_id]
                        result = results[worker_id]
                        rollout.add_step(
                            frozen_actions[position],
                            behavior_actions[position],
                            result,
                        )
                        if not (result.terminated or result.truncated):
                            continue
                        success = bool(result.info.get("is_success", False))
                        if rollout.leg == 0 and success:
                            rollout.task_success = True
                            rollout.task_path_length = result.current_goal_path_length
                            rollout.leg = 1
                            rollout.behavior.reset_leg()
                            retargets.append(
                                WorkerRetarget(
                                    worker_id=worker_id,
                                    seed=args.world_seed + rollout.scene_index,
                                    start_position=result.position,
                                    start_velocity=np.zeros(3, dtype=np.float32),
                                    goal_position=charger_position,
                                )
                            )
                            continue
                        if rollout.leg == 1:
                            rollout.return_success = success
                            rollout.return_path_length = result.current_goal_path_length
                        rollout.end_reason = str(
                            result.info.get("end_reason", "unknown")
                        )
                        save_rollout(
                            args.output_dir,
                            rollout,
                            capacity=capacity,
                            d_max=d_max,
                            charger_position=charger_position,
                        )
                        completed += 1
                        del active[worker_id]
                        available.append(worker_id)
                    if retargets:
                        reset_results = pool.retarget_many(retargets)
                        for request in retargets:
                            reset_result = reset_results[request.worker_id]
                            active_rollout = active[request.worker_id]
                            active_rollout.observation = reset_result.observation
                            active_rollout.position = reset_result.position
                            active_rollout.velocity = reset_result.velocity
                    if available:
                        assign(available)
                    if completed % 5 == 0 or completed == total:
                        atomic_json(
                            args.output_dir / "PROGRESS.json",
                            {
                                "stage": "collection",
                                "completed_rollouts": completed,
                                "total_rollouts": total,
                                "pending_rollouts": len(pending),
                                "active_rollouts": len(active),
                                "wall_clock_seconds": time.time()
                                - manifest["started_unix"],
                            },
                        )
        rows, initial_observations = load_completed_rows(
            args.output_dir,
            num_scenes=args.num_scenes,
            interventions=interventions,
        )
        if args.nominal_source_only:
            result = {
                **manifest,
                "status": "NOMINAL_SOURCE_COMPLETE",
                "promotable": False,
                "source_only": True,
                "num_rollouts": len(rows),
                "num_transitions": int(sum(row["transitions"] for row in rows)),
                "all_rollouts_complete": len(rows) == args.num_scenes,
            }
        elif args.smoke:
            result = {
                **manifest,
                "status": "SMOKE_COMPLETED",
                "promotable": False,
                "smoke_only": True,
                "num_rollouts": len(rows),
                "num_transitions": int(sum(row["transitions"] for row in rows)),
                "matched_replay_max_error": float(
                    max(
                        np.max(
                            np.abs(
                                initial_observations[
                                    np.arange(len(DEFAULT_INTERVENTIONS))
                                + scene_index * len(interventions)
                                ]
                                - initial_observations[
                                    scene_index * len(interventions)
                                ]
                            )
                        )
                        for scene_index in range(args.num_scenes)
                    )
                ),
            }
        else:
            atomic_json(
                args.output_dir / "PROGRESS.json",
                {
                    "stage": "data_gate",
                    "completed_rollouts": len(rows),
                    "total_rollouts": len(rows),
                },
            )
            gate, arrays = aggregate_gate(
                rows,
                initial_observations,
                policy,
                num_scenes=args.num_scenes,
                interventions=interventions,
                device=args.device,
                model_seed=args.model_seed,
            )
            save_npz_atomic(args.output_dir / "crossfit_data.npz", **arrays)
            result = {**manifest, **gate}
        result["finished_unix"] = time.time()
        result["wall_clock_seconds"] = (
            result["finished_unix"] - manifest["started_unix"]
        )
        atomic_json(args.output_dir / "RESULT.json", result)
        atomic_json(
            args.output_dir / "COMPLETED.json", {"status": result["status"]}
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except BaseException as error:
        atomic_json(
            args.output_dir / "FAILED.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (args.output_dir / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
