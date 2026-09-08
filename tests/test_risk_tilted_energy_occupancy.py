from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_risk_tilted_energy_occupancy import (
    analyze_trajectory_directory,
)


def _write_trajectory(
    directory: Path,
    *,
    trajectory_id: int,
    step_energy: list[float],
    intervention: float,
) -> dict[str, object]:
    count = len(step_energy)
    nominal = np.zeros((count, 3), dtype=np.float32)
    executed = nominal.copy()
    executed[:, 0] = intervention
    energy = np.asarray(step_energy, dtype=np.float32)
    remaining = np.flip(np.cumsum(np.flip(energy))).copy()
    file_name = f"trajectory_{trajectory_id:06d}.npz"
    np.savez_compressed(
        directory / file_name,
        step_energy=energy,
        nominal_actions=nominal,
        executed_actions=executed,
        energy_to_go=remaining,
    )
    return {
        "trajectory_id": trajectory_id,
        "file": file_name,
        "total_realized_energy": float(np.sum(energy)),
        "num_transitions": count,
        "total_safety_burden": float(count * intervention),
        "initial_distance": float(count),
        "goal_type": "CHARGER",
        "distance_bucket": "synthetic",
    }


def test_risk_tilt_audit_separates_ordinary_and_exponential_coverage(
    tmp_path: Path,
) -> None:
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    records = [
        _write_trajectory(
            trajectory_dir,
            trajectory_id=0,
            step_energy=[1.0, 1.0],
            intervention=0.0,
        ),
        _write_trajectory(
            trajectory_dir,
            trajectory_id=1,
            step_energy=[4.0, 4.0],
            intervention=1.0,
        ),
    ]
    (trajectory_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n",
        encoding="utf-8",
    )

    report = analyze_trajectory_directory(trajectory_dir, betas=[0.0, 2.0])
    ordinary, tilted = report["risk_tilt_rows"]
    assert report["num_trajectories"] == 2
    assert report["num_executed_interface_visits"] == 4
    assert report["ordinary_summary"]["energy_horizon_correlation"] is None
    assert report["horizon_adjusted_summary"]["linear_horizon_r_squared"] >= 0.0
    assert len(
        report["horizon_adjusted_summary"][
            "matched_horizon_intervention_strata"
        ]
    ) == 2
    assert ordinary["terminal_ess"] == pytest.approx(2.0)
    assert ordinary["visit_ess"] == pytest.approx(4.0)
    assert tilted["terminal_ess"] < ordinary["terminal_ess"]
    assert tilted["visit_ess"] < ordinary["visit_ess"]
    assert tilted["terminal_weighted_energy"] > ordinary["terminal_weighted_energy"]
    assert (
        tilted["visit_weighted_intervention_norm"]
        > ordinary["visit_weighted_intervention_norm"]
    )
