from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.energy_mc.adaptive_uncertainty import chunked_point_predictions
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    PositiveResidualQuantileModel,
    reconstruct_positions,
    trajectory_balanced_state_indices,
)
from experiments.energy_mc.core import PackedEnergyDataset
from experiments.energy_mc.final_risk import (
    MondrianTrajectoryCalibration,
    future_max_underestimation_target,
    primary_goal_group,
)
from experiments.energy_mc.trajectory_context import (
    ShortFrozenSACRolloutContext,
    ShortRolloutConfig,
)
from review_bundle.safety.energy.mc_regression import (
    EnergyToGoRegressor,
    ModelBasedEnergyRolloutEstimator,
)
from scripts.run_energy_risk_method_ladder import goal_metrics
from scripts.run_energy_uncertainty_v4 import (
    BATTERY_CAPACITY,
    D_MAX,
    POINT_CHECKPOINT,
    SAC_CHECKPOINT,
    SOURCE,
    V2,
)
from scripts.train_uav_energy_mc import load_frozen_sac


ROOT = Path(__file__).resolve().parents[1]
RESERVE = BATTERY_CAPACITY * 0.10


def subset_by_primary_group(
    dataset: PackedEnergyDataset,
    *,
    trajectories_per_group: int,
    seed: int,
) -> PackedEnergyDataset:
    ids = np.unique(dataset.trajectory_ids)
    group_ids: dict[str, list[int]] = {}
    for trajectory_id in ids:
        index = int(np.flatnonzero(dataset.trajectory_ids == trajectory_id)[0])
        group = primary_goal_group(
            str(dataset.goal_types[index]),
            str(dataset.distance_buckets[index]),
        )
        group_ids.setdefault(group, []).append(int(trajectory_id))
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for group, values in sorted(group_ids.items()):
        if len(values) < trajectories_per_group:
            raise ValueError(
                f"short-rollout group {group} has {len(values)} trajectories, "
                f"needs {trajectories_per_group}"
            )
        selected.extend(
            int(value)
            for value in rng.choice(values, size=trajectories_per_group, replace=False)
        )
    selected_ids = set(selected)
    mask = np.asarray(
        [int(value) in selected_ids for value in dataset.trajectory_ids], dtype=bool
    )
    metadata = [
        row for row in dataset.metadata if int(row["trajectory_id"]) in selected_ids
    ]
    return PackedEnergyDataset(
        states=dataset.states[mask],
        targets=dataset.targets[mask],
        step_energy=dataset.step_energy[mask],
        transition_dt=dataset.transition_dt[mask],
        trajectory_ids=dataset.trajectory_ids[mask],
        goal_types=dataset.goal_types[mask],
        distance_buckets=dataset.distance_buckets[mask],
        boundary_contact_trajectories=dataset.boundary_contact_trajectories[mask],
        metadata=metadata,
    )


def rollout_features(
    policy,
    dataset: PackedEnergyDataset,
    indices: np.ndarray | None,
    *,
    horizon: int,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    states = dataset.states if indices is None else dataset.states[indices]
    positions = reconstruct_positions(dataset)
    positions = positions if indices is None else positions[indices]
    builder = ShortFrozenSACRolloutContext(policy, ShortRolloutConfig(horizon=horizon))
    start = time.perf_counter()
    context = builder.build_chunked(states, positions, chunk_size=chunk_size)
    elapsed = time.perf_counter() - start
    return context, elapsed


def full_rollout_oracle(
    policy,
    dataset: PackedEnergyDataset,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    remaining_distance = dataset.states[:, -1].astype(np.float64) * float(
        np.linalg.norm([4000.0, 4000.0, 400.0])
    )
    eligible = np.flatnonzero(remaining_distance >= 5.0)
    if eligible.size < sample_size:
        raise ValueError(
            f"full-rollout oracle needs {sample_size} nonterminal states, "
            f"found {eligible.size}"
        )
    indices = np.sort(rng.choice(eligible, size=sample_size, replace=False))
    positions = reconstruct_positions(dataset)
    metadata = {int(row["trajectory_id"]): row for row in dataset.metadata}
    oracle = ModelBasedEnergyRolloutEstimator(policy, max_policy_steps=4000)
    predictions = []
    truth = []
    steps = []
    elapsed = []
    for index in indices:
        trajectory_id = int(dataset.trajectory_ids[index])
        goal = np.asarray(metadata[trajectory_id]["goal_position"], dtype=np.float32)
        velocity = dataset.states[index, :3].astype(np.float64) * np.asarray(
            [20.0, 20.0, 5.0]
        )
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed > 20.0:
            velocity[:2] *= 20.0 / horizontal_speed
        velocity[2] = np.clip(velocity[2], -5.0, 5.0)
        environment = UAVEnergyDeliverySACEnv(
            phase=SACTrainingPhase.TD_PRETRAINING,
            minimum_task_distance=5.0,
        )
        environment.reset(
            seed=seed + int(index),
            options={
                "start_position": positions[index].astype(np.float32),
                "start_velocity": velocity.astype(np.float32),
                "task_point": goal,
            },
        )
        estimate = oracle.estimate_context(
            environment,
            goal,
            position=positions[index],
            velocity=velocity,
            goal_type=str(dataset.goal_types[index]),
        )
        environment.close()
        predictions.append(estimate.prediction)
        truth.append(float(dataset.targets[index]))
        steps.append(estimate.rollout_steps)
        elapsed.append(estimate.wall_clock_seconds)
    residual = np.asarray(predictions) - np.asarray(truth)
    return {
        "sample_size": sample_size,
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mean_rollout_steps": float(np.mean(steps)),
        "mean_latency_seconds": float(np.mean(elapsed)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Short frozen-SAC rollout risk context ablation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/energy_risk_v5_development/short_rollout",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--navigation-device", default="cuda")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--states-per-trajectory", type=int, default=64)
    parser.add_argument("--trajectories-per-group", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--oracle-samples", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "models").mkdir(exist_ok=True)
    policy = load_frozen_sac(SAC_CHECKPOINT, args.navigation_device)
    point_model = EnergyToGoRegressor.load(POINT_CHECKPOINT, device=args.device)
    train = PackedEnergyDataset.load(SOURCE / "energy_dataset/train")
    validation = PackedEnergyDataset.load(SOURCE / "energy_dataset/validation")
    calibration_full = PackedEnergyDataset.concatenate(
        [
            PackedEnergyDataset.load(SOURCE / "energy_dataset/calibration"),
            PackedEnergyDataset.load(V2 / "calibration_v2/new_trajectories"),
        ]
    )
    diagnostic_full = PackedEnergyDataset.load(V2 / "final_conformal_test_v2/trajectories")
    calibration = subset_by_primary_group(
        calibration_full,
        trajectories_per_group=args.trajectories_per_group,
        seed=830_001,
    )
    diagnostic = subset_by_primary_group(
        diagnostic_full,
        trajectories_per_group=args.trajectories_per_group,
        seed=830_002,
    )
    train_indices = trajectory_balanced_state_indices(
        train.trajectory_ids, max_states_per_trajectory=args.states_per_trajectory
    )
    validation_indices = trajectory_balanced_state_indices(
        validation.trajectory_ids, max_states_per_trajectory=args.states_per_trajectory
    )
    point = {
        "train": chunked_point_predictions(point_model, train.states),
        "validation": chunked_point_predictions(point_model, validation.states),
        "calibration": chunked_point_predictions(point_model, calibration.states),
        "diagnostic": chunked_point_predictions(point_model, diagnostic.states),
    }
    target = {
        "train": future_max_underestimation_target(
            train.targets, point["train"], train.trajectory_ids
        ),
        "validation": future_max_underestimation_target(
            validation.targets, point["validation"], validation.trajectory_ids
        ),
    }
    compact_builder = GoalRiskFeatureBuilder("compact_decision_context")
    compact = {
        "train": compact_builder.build_from_dataset(train)[train_indices],
        "validation": compact_builder.build_from_dataset(validation)[validation_indices],
        "calibration": compact_builder.build_from_dataset(calibration),
        "diagnostic": compact_builder.build_from_dataset(diagnostic),
    }
    rng = np.random.default_rng(830_003)
    remaining = rng.uniform(0.0, BATTERY_CAPACITY, size=diagnostic.targets.shape)
    results: dict[str, object] = {}
    for horizon in args.horizons:
        contexts = {}
        latency = {}
        for name, dataset, indices in (
            ("train", train, train_indices),
            ("validation", validation, validation_indices),
            ("calibration", calibration, None),
            ("diagnostic", diagnostic, None),
        ):
            contexts[name], latency[name] = rollout_features(
                policy,
                dataset,
                indices,
                horizon=horizon,
                chunk_size=args.chunk_size,
            )
        features = {
            name: np.concatenate([compact[name], contexts[name]], axis=1)
            for name in contexts
        }
        model = PositiveResidualQuantileModel(
            input_dim=features["train"].shape[1],
            seed=830_100 + horizon,
            device=args.device,
        )
        history = model.fit(
            features["train"],
            target["train"][train_indices],
            features["validation"],
            target["validation"][validation_indices],
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        model.save(args.output_dir / f"models/k2_short_rollout_h{horizon}.pt")
        calibration_risk = model.predict_level(features["calibration"], 0.95)
        diagnostic_risk = model.predict_level(features["diagnostic"], 0.95)
        fitted = MondrianTrajectoryCalibration.fit(
            calibration.targets,
            point["calibration"],
            calibration_risk,
            calibration.trajectory_ids,
            calibration.goal_types,
            calibration.distance_buckets,
            coverage=0.95,
            mode="additive",
            minimum_group_trajectories=args.trajectories_per_group,
        )
        upper = fitted.apply(
            point["diagnostic"],
            diagnostic_risk,
            diagnostic.goal_types,
            diagnostic.states[:, -1].astype(np.float64) * D_MAX,
        )
        results[f"h{horizon}"] = {
            "metrics": goal_metrics(
                diagnostic,
                point["diagnostic"],
                diagnostic_risk,
                upper,
                remaining_energy=remaining,
            ),
            "training": history.as_dict(),
            "context_generation_seconds": latency,
            "diagnostic_context_microseconds_per_state": (
                1e6 * latency["diagnostic"] / diagnostic.states.shape[0]
            ),
            "calibration": fitted.as_dict(),
        }
        (args.output_dir / "short_rollout_partial_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
    results["full_rollout_oracle"] = full_rollout_oracle(
        policy,
        diagnostic,
        sample_size=args.oracle_samples,
        seed=830_500,
    )
    results["scope"] = {
        "calibration_trajectories": len(calibration.successful_trajectory_ids),
        "diagnostic_trajectories": len(diagnostic.successful_trajectory_ids),
        "full_rollout_is_oracle_only": True,
        "sac_frozen": True,
    }
    (args.output_dir / "short_rollout_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
