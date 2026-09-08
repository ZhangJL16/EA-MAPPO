from __future__ import annotations

import numpy as np

from scripts.analyze_horizon_matched_interface_energy import (
    _moving_block_bootstrap_mean_interval,
    analyze_feature_arrays,
)


def _synthetic_features(num_rows: int = 360) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(9)
    horizon = rng.integers(100, 700, size=num_rows).astype(np.float64)
    distance = rng.uniform(200.0, 2200.0, size=num_rows)
    intervention = rng.uniform(0.0, 0.3, size=num_rows)
    nominal_mean = rng.uniform(0.3, 0.8, size=num_rows)
    total_energy = (
        0.04 * horizon
        + 0.001 * distance
        + 4.0 * intervention
        + rng.normal(0.0, 0.15, size=num_rows)
    )
    goal = np.where(np.arange(num_rows) % 2 == 0, "TASK", "CHARGER")
    bucket = np.where(
        distance < 700.0,
        "100-700",
        np.where(distance < 1500.0, "700-1500", "1500-2500"),
    )
    return {
        "trajectory_id": np.arange(num_rows, dtype=np.float64),
        "total_energy": total_energy,
        "horizon": horizon,
        "total_duration": 0.2 * horizon,
        "initial_distance": distance,
        "path_length": distance * rng.uniform(1.0, 1.3, size=num_rows),
        "path_ratio": rng.uniform(1.0, 1.3, size=num_rows),
        "goal_type": goal,
        "distance_bucket": bucket,
        "nominal_norm_mean": nominal_mean,
        "nominal_norm_q90": nominal_mean + 0.1,
        "nominal_delta_rms": rng.uniform(0.01, 0.08, size=num_rows),
        "intervention_mean": intervention,
        "intervention_q90": 1.2 * intervention,
        "intervention_max": 1.5 * intervention,
        "intervention_rms": 1.1 * intervention,
        "intervention_nonzero_fraction": np.minimum(1.0, 3.0 * intervention),
        "safety_burden_rate": 2.0 * intervention,
        "energy_per_step": total_energy / horizon,
    }


def test_horizon_matched_analysis_detects_incremental_interface_signal() -> None:
    report = analyze_feature_arrays(
        _synthetic_features(),
        bootstrap_repetitions=50,
        permutation_repetitions=10,
        seed=123,
    )
    increment = report["primary_interface_increment"]
    assert increment["relative_mae_reduction"] > 0.0
    assert increment["delta_r_squared"] > 0.0
    assert report["coarsened_matching"]["retained_cells"] > 0
    assert report["feature_contract"]["post_trajectory_mechanism_only"]
    assert len(report["residual_risk_by_intervention_quintile"]) == 5


def test_moving_block_bootstrap_is_deterministic() -> None:
    values = np.linspace(-1.0, 1.0, 50)
    first = _moving_block_bootstrap_mean_interval(
        values,
        block_length=7,
        repetitions=30,
        seed=44,
    )
    second = _moving_block_bootstrap_mean_interval(
        values,
        block_length=7,
        repetitions=30,
        seed=44,
    )
    assert first == second
