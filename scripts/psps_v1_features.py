#!/usr/bin/env python3
"""Frozen compact legal latest and PSPS-v1 feature maps."""
from __future__ import annotations

import hashlib
import numpy as np


CAPACITY = 378.72626091628933
WORKSPACE_DIAGONAL = float(np.linalg.norm([4000.0, 4000.0, 400.0]))
LATEST_NAMES = (
    "battery_fraction", "task_distance_over_workspace_diagonal",
    "charger_distance_over_workspace_diagonal", "task_clock_fraction",
    "prefix_step_fraction", "latest_velocity_toward_task_normalized",
    "latest_horizontal_speed_normalized", "latest_vertical_velocity_normalized",
    "latest_lidar_hit_fraction", "latest_lidar_min_hit_masked_range",
)
TREND_NAMES = (
    "task_distance_net_progress_per_policy_step",
    "task_distance_nonprogress_interval_fraction",
    "longest_nonprogress_run_fraction", "mean_velocity_toward_task_normalized",
    "q10_velocity_toward_task_normalized", "mean_horizontal_speed_normalized",
    "low_horizontal_speed_fraction", "joint_progress_speed_stall_fraction",
    "battery_depletion_per_policy_step_fraction", "mean_lidar_hit_fraction",
)
PSPS_NAMES = LATEST_NAMES + TREND_NAMES


def fold_for_world(identity: str, *, outer: bool) -> int:
    suffix = "psps-v1-outer-fold" if outer else "psps-v1-inner-fold"
    digest = hashlib.sha256(f"{identity}|{suffix}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (6 if outer else 4)


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.tolist():
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def features(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return one 10-D latest and one 20-D PSPS row per stored anchor."""
    nav = np.asarray(arrays["nav_observation"], np.float64)
    battery = np.asarray(arrays["battery"], np.float64)
    charger_distance = np.asarray(arrays["distance_to_charger"], np.float64)
    task_clock = np.asarray(arrays["task_clock"], np.float64)
    step = np.asarray(arrays["step"], np.float64)
    if nav.ndim != 3 or nav.shape[1:] != (64, 2056):
        raise ValueError(f"PSPS-v1 requires [anchors,64,2056] nav history, found {nav.shape}")
    task_distance = np.expm1(
        np.clip(nav[..., 6], 0.0, 1.0) * np.log1p(WORKSPACE_DIAGONAL)
    )
    velocity = nav[..., :3]
    direction = nav[..., 3:6]
    projected = np.sum(velocity * direction, axis=-1)
    speed = np.linalg.norm(velocity[..., :2], axis=-1)
    lidar_ranges = nav[..., 7:1031]
    lidar_hits = nav[..., 1031:2055] > 0.5
    masked = np.where(lidar_hits, lidar_ranges, 1.0)

    latest = np.column_stack((
        battery[:, -1] / CAPACITY,
        task_distance[:, -1] / WORKSPACE_DIAGONAL,
        charger_distance[:, -1] / WORKSPACE_DIAGONAL,
        task_clock[:, -1] / 4000.0,
        step[:, -1] / 768.0,
        projected[:, -1], speed[:, -1], velocity[:, -1, 2],
        lidar_hits[:, -1].mean(axis=1), masked[:, -1].min(axis=1),
    ))
    extras = []
    for index in range(len(nav)):
        elapsed = max(float(step[index, -1] - step[index, 0]), 1.0)
        change = np.diff(task_distance[index])
        nonprogress = change >= -0.001
        low_speed = speed[index] < 0.10
        extras.append((
            (task_distance[index, 0] - task_distance[index, -1]) / elapsed,
            float(nonprogress.mean()),
            _longest_true_run(nonprogress) / 63.0,
            float(projected[index].mean()),
            float(np.quantile(projected[index], 0.10)),
            float(speed[index].mean()),
            float(low_speed.mean()),
            float(np.mean(nonprogress & low_speed[1:])),
            (battery[index, 0] - battery[index, -1]) / (CAPACITY * elapsed),
            float(lidar_hits[index].mean()),
        ))
    psps = np.column_stack((latest, np.asarray(extras, np.float64)))
    if latest.shape[1] != 10 or psps.shape[1] != 20:
        raise AssertionError("PSPS-v1 feature dimension drift")
    if not np.isfinite(latest).all() or not np.isfinite(psps).all():
        raise ValueError("PSPS-v1 feature map produced non-finite values")
    return {"latest": latest, "psps": psps}
