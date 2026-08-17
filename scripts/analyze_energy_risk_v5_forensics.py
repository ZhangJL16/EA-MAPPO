from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from envs.UAVEnergyDeliverySAC import UAVEnergyDeliverySACEnv
from experiments.energy_mc.adaptive_analysis import contiguous_trajectory_slices
from experiments.energy_mc.adaptive_uncertainty import (
    HeteroscedasticResidualModel,
    TrajectoryConformalCorrection,
    chunked_point_predictions,
)
from experiments.energy_mc.conformal import PackedMissionDataset
from experiments.energy_mc.conditional_risk import (
    GoalRiskFeatureBuilder,
    HierarchicalScaledConformalCalibration,
    PositiveResidualQuantileModel,
    reconstruct_positions,
)
from experiments.energy_mc.core import PackedEnergyDataset
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.run_energy_uncertainty_v4 import (
    BATTERY_CAPACITY,
    D_MAX,
    mission_correction,
    selected_goal_upper,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V4 = ROOT / "artifacts/uav_energy_uncertainty_v4_20260818_001949_v2"
DEFAULT_POINT = (
    ROOT
    / "artifacts/uav_energy_delivery_mc_formal_20260817_154321/energy_model/best_validation.pt"
)
DEFAULT_MISSION = (
    ROOT
    / "artifacts/uav_energy_adaptive_uncertainty_20260817_214303/models/mission_heteroscedastic_laplace.pt"
)
MAP_EXTENT = np.asarray([4000.0, 4000.0, 400.0], dtype=np.float64)
VELOCITY_SCALE = np.asarray([20.0, 20.0, 5.0], dtype=np.float64)


FEATURE_REGISTRY = {
    "goal_type": True,
    "initial_distance": True,
    "absolute_start_xyz": True,
    "absolute_goal_xyz": True,
    "start_boundary_distances": True,
    "initial_velocity": True,
    "current_velocity": True,
    "current_position": True,
    "current_boundary_distances": True,
    "short_frozen_policy_rollout_context": True,
    "future_flight_steps": False,
    "future_actual_path_length": False,
    "future_path_ratio": False,
    "future_mean_acceleration": False,
    "future_max_acceleration": False,
    "future_boundary_contacts": False,
    "future_consecutive_boundary_contacts": False,
    "true_energy": False,
    "underestimation_magnitude": False,
}


def _boundary_distances(position: np.ndarray) -> dict[str, float]:
    value = np.asarray(position, dtype=np.float64)
    return {
        "distance_xmin": float(value[0]),
        "distance_xmax": float(MAP_EXTENT[0] - value[0]),
        "distance_ymin": float(value[1]),
        "distance_ymax": float(MAP_EXTENT[1] - value[1]),
        "distance_zmin": float(value[2]),
        "distance_zmax": float(MAP_EXTENT[2] - value[2]),
    }


def _acceleration(states: np.ndarray, transition_dt: np.ndarray) -> np.ndarray:
    velocity = np.asarray(states[:, :3], dtype=np.float64) * VELOCITY_SCALE
    if velocity.shape[0] <= 1:
        return np.zeros(1, dtype=np.float64)
    return np.linalg.norm(
        np.diff(velocity, axis=0) / np.maximum(transition_dt[:-1, None], 1e-8),
        axis=1,
    )


def goal_forensics(
    dataset: PackedEnergyDataset,
    point_model: EnergyToGoRegressor,
    risk_model: PositiveResidualQuantileModel,
    calibration: HierarchicalScaledConformalCalibration,
) -> tuple[pd.DataFrame, dict[str, object]]:
    point = chunked_point_predictions(point_model, dataset.states)
    builder = GoalRiskFeatureBuilder("compact_position_boundary")
    positions = reconstruct_positions(dataset)
    features = builder.build(dataset.states, positions)
    risk = risk_model.predict_level(features, 0.95)
    upper = selected_goal_upper(
        risk_model,
        builder,
        calibration,
        dataset,
        point,
        coverage=0.95,
    )
    metadata = {int(row["trajectory_id"]): row for row in dataset.metadata}
    rows: list[dict[str, object]] = []
    for trajectory_id, selected in contiguous_trajectory_slices(dataset.trajectory_ids):
        truth = dataset.targets[selected].astype(np.float64)
        under = truth - upper[selected]
        local = int(np.argmax(under))
        index = selected.start + local
        row = metadata[trajectory_id]
        state = dataset.states[index].astype(np.float64)
        start = np.asarray(row["start_position"], dtype=np.float64)
        goal = np.asarray(row["goal_position"], dtype=np.float64)
        current = positions[index]
        acceleration = _acceleration(dataset.states[selected], dataset.transition_dt[selected])
        record = {
            "trajectory_id": trajectory_id,
            "goal_type": str(row["goal_type"]),
            "distance_bucket": str(row["distance_bucket"]),
            "initial_goal_distance": float(row["initial_goal_distance"]),
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "start_z": float(start[2]),
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "goal_z": float(goal[2]),
            "worst_state_x": float(current[0]),
            "worst_state_y": float(current[1]),
            "worst_state_z": float(current[2]),
            "initial_velocity_x": float(row["initial_velocity"][0]),
            "initial_velocity_y": float(row["initial_velocity"][1]),
            "initial_velocity_z": float(row["initial_velocity"][2]),
            "worst_state_velocity_x": float(state[0] * VELOCITY_SCALE[0]),
            "worst_state_velocity_y": float(state[1] * VELOCITY_SCALE[1]),
            "worst_state_velocity_z": float(state[2] * VELOCITY_SCALE[2]),
            "flight_steps": int(row["steps"]),
            "actual_path_length": float(row["path_length"]),
            "straight_line_distance": float(row["initial_goal_distance"]),
            "path_ratio": float(row["path_length"] / max(row["initial_goal_distance"], 1e-8)),
            "mean_acceleration": float(np.mean(acceleration)),
            "max_acceleration": float(np.max(acceleration)),
            "vertical_displacement": float(abs(goal[2] - start[2])),
            "boundary_contact_steps": int(row["boundary_contact_steps"]),
            "maximum_consecutive_boundary_contacts": int(
                row.get("maximum_consecutive_boundary_contacts", 0)
            ),
            "true_energy_at_worst_state": float(truth[local]),
            "point_prediction_at_worst_state": float(point[index]),
            "risk_prediction_at_worst_state": float(risk[index]),
            "conformal_correction_at_worst_state": float(
                upper[index] - point[index] - risk[index]
            ),
            "final_upper_at_worst_state": float(upper[index]),
            "underestimation_magnitude": float(under[local]),
        }
        record.update({f"start_{key}": value for key, value in _boundary_distances(start).items()})
        record.update(
            {f"worst_state_{key}": value for key, value in _boundary_distances(current).items()}
        )
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("underestimation_magnitude", ascending=False)
    top = frame.head(100).copy()
    summary = {
        "num_trajectories": int(len(frame)),
        "num_undercovered_trajectories": int((frame.underestimation_magnitude > 0.0).sum()),
        "top100_task_fraction": float((top.goal_type == "TASK").mean()),
        "top100_long_distance_fraction": float(
            top.distance_bucket.isin(["2500-4000", ">4000"]).mean()
        ),
        "top100_boundary_contact_fraction": float((top.boundary_contact_steps > 0).mean()),
        "top100_median_path_ratio": float(top.path_ratio.median()),
        "top100_median_max_acceleration": float(top.max_acceleration.median()),
        "maximum_final_bound_underestimation": float(frame.underestimation_magnitude.max()),
    }
    return top, summary


def _mission_geometry(
    dataset: PackedMissionDataset,
    charger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return_distance = dataset.return_after_states[:, -1].astype(np.float64) * D_MAX
    task_goal = charger[None, :] - dataset.return_after_states[:, 3:6] * return_distance[:, None]
    task_distance = dataset.task_states[:, -1].astype(np.float64) * D_MAX
    positions = task_goal - dataset.task_states[:, 3:6] * task_distance[:, None]
    return positions, task_goal, task_distance


def mission_forensics(
    dataset: PackedMissionDataset,
    point_model: EnergyToGoRegressor,
    mission_model: HeteroscedasticResidualModel,
    correction: TrajectoryConformalCorrection,
    charger: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, object]]:
    task_point = chunked_point_predictions(point_model, dataset.task_states)
    return_point = chunked_point_predictions(point_model, dataset.return_after_states)
    point = task_point + return_point
    features = np.concatenate([dataset.task_states, dataset.return_after_states], axis=1)
    risk = mission_model.upper_offset(features, 0.95)
    upper = point + risk + correction.correction
    positions, task_goals, task_distance = _mission_geometry(dataset, charger)
    metadata = {int(row["mission_id"]): row for row in dataset.metadata}
    rows: list[dict[str, object]] = []
    for mission_id, selected in contiguous_trajectory_slices(dataset.mission_ids):
        truth = dataset.true_mission_energy[selected].astype(np.float64)
        under = truth - upper[selected]
        local = int(np.argmax(under))
        index = selected.start + local
        row = metadata[mission_id]
        start = positions[selected.start]
        goal = task_goals[index]
        current = positions[index]
        states = dataset.task_states[selected]
        acceleration = _acceleration(states, np.full(states.shape[0], 0.2))
        task_path_length = float(np.sum(np.linalg.norm(np.diff(positions[selected], axis=0), axis=1)))
        straight = float(row["initial_task_distance"])
        return_distance = float(dataset.return_after_states[index, -1] * D_MAX)
        available_total_path = row.get("actual_path_length")
        record = {
            "mission_id": mission_id,
            "goal_type": "MISSION_TASK_THEN_CHARGER",
            "distance_bucket": str(row["initial_task_distance_bucket"]),
            "initial_goal_distance": straight,
            "start_x": float(start[0]),
            "start_y": float(start[1]),
            "start_z": float(start[2]),
            "goal_x": float(goal[0]),
            "goal_y": float(goal[1]),
            "goal_z": float(goal[2]),
            "worst_state_x": float(current[0]),
            "worst_state_y": float(current[1]),
            "worst_state_z": float(current[2]),
            "initial_velocity_x": float(states[0, 0] * VELOCITY_SCALE[0]),
            "initial_velocity_y": float(states[0, 1] * VELOCITY_SCALE[1]),
            "initial_velocity_z": float(states[0, 2] * VELOCITY_SCALE[2]),
            "worst_state_velocity_x": float(states[local, 0] * VELOCITY_SCALE[0]),
            "worst_state_velocity_y": float(states[local, 1] * VELOCITY_SCALE[1]),
            "worst_state_velocity_z": float(states[local, 2] * VELOCITY_SCALE[2]),
            "flight_steps": int(row.get("total_steps", row["task_steps"] + row["return_steps"])),
            "actual_path_length": (
                float(available_total_path) if available_total_path is not None else np.nan
            ),
            "task_path_length_reconstructed": task_path_length,
            "straight_route_length": straight + return_distance,
            "path_ratio": (
                float(row["path_ratio"]) if "path_ratio" in row else np.nan
            ),
            "mean_acceleration": float(row.get("mean_acceleration", np.mean(acceleration))),
            "max_acceleration": float(row.get("max_acceleration", np.max(acceleration))),
            "vertical_displacement": float(
                row.get("vertical_displacement", abs(goal[2] - start[2]) + abs(charger[2] - goal[2]))
            ),
            "boundary_contact_steps": row.get("boundary_contact_steps"),
            "had_boundary_contact": bool(row["had_boundary_contact"]),
            "maximum_consecutive_boundary_contacts": row.get(
                "maximum_consecutive_boundary_contacts"
            ),
            "true_energy_at_worst_state": float(truth[local]),
            "point_prediction_at_worst_state": float(point[index]),
            "risk_prediction_at_worst_state": float(risk[index]),
            "conformal_correction_at_worst_state": float(correction.correction),
            "final_upper_at_worst_state": float(upper[index]),
            "underestimation_magnitude": float(under[local]),
        }
        record.update({f"start_{key}": value for key, value in _boundary_distances(start).items()})
        record.update(
            {f"worst_state_{key}": value for key, value in _boundary_distances(current).items()}
        )
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("underestimation_magnitude", ascending=False)
    top = frame.head(100).copy()
    summary = {
        "num_missions": int(len(frame)),
        "num_undercovered_missions": int((frame.underestimation_magnitude > 0.0).sum()),
        "top100_long_distance_fraction": float(
            top.distance_bucket.isin(["2500-4000", ">4000"]).mean()
        ),
        "top100_boundary_contact_fraction": float(top.had_boundary_contact.mean()),
        "maximum_final_bound_underestimation": float(frame.underestimation_magnitude.max()),
        "historical_v4_missing_full_path_fields": bool(top.actual_path_length.isna().any()),
    }
    return top, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic Energy Risk v5 development analysis")
    parser.add_argument("--v4-dir", type=Path, default=DEFAULT_V4)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/energy_risk_v5_development/forensics"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point = EnergyToGoRegressor.load(DEFAULT_POINT, device=args.device)
    risk = PositiveResidualQuantileModel.load(
        args.v4_dir / "models/selected_goal_risk_model.pt", device=args.device
    )
    calibration_payload = json.loads(
        (args.v4_dir / "models/selected_goal_calibrations.json").read_text()
    )["0.95"]
    calibration = HierarchicalScaledConformalCalibration(**calibration_payload)
    goal = PackedEnergyDataset.load(args.v4_dir / "fresh_v4/goal_trajectories")
    goal_top, goal_summary = goal_forensics(goal, point, risk, calibration)
    goal_top.to_csv(args.output_dir / "top100_goal_underestimates.csv", index=False)

    mission = PackedMissionDataset.load(args.v4_dir / "fresh_v4/mission_trajectories")
    mission_model = HeteroscedasticResidualModel.load(DEFAULT_MISSION, device=args.device)
    environment = UAVEnergyDeliverySACEnv()
    charger = environment.charger_position.astype(np.float64)
    environment.close()
    mission_top, mission_summary = mission_forensics(
        mission,
        point,
        mission_model,
        mission_correction(0.95),
        charger,
    )
    mission_top.to_csv(args.output_dir / "top100_mission_underestimates.csv", index=False)
    (args.output_dir / "feature_availability.json").write_text(
        json.dumps(
            {
                name: {"AVAILABLE_AT_DECISION_TIME": available}
                for name, available in FEATURE_REGISTRY.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (args.output_dir / "forensic_summary.json").write_text(
        json.dumps(
            {"goal": goal_summary, "mission": mission_summary},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
