from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.energy_transfer.analysis import (
    build_figures,
    horizon_error_report,
    learned_value_decision,
    paired_sortie_comparisons,
    quantile_metrics,
    regression_metrics,
    residual_relationships,
)


def test_regression_and_quantile_metrics_report_safety_errors() -> None:
    target = np.array([1.0, 2.0, 4.0, 8.0])
    prediction = np.array([0.5, 2.5, 3.0, 10.0])
    metrics = regression_metrics(prediction, target)
    assert metrics["mean_signed_error"] == 0.25
    assert metrics["underestimation_rate"] == 0.5
    assert metrics["mean_underestimation_magnitude"] == 0.75
    assert metrics["worst_case_underestimation"] == 1.0
    assert metrics["mean_absolute_relative_error"] > 0.0

    quantiles = np.array(
        [
            [0.9, 1.1, 1.2, 1.3],
            [1.9, 2.1, 2.2, 2.3],
            [3.9, 4.1, 4.2, 4.3],
            [7.9, 8.1, 8.2, 8.3],
        ]
    )
    report = quantile_metrics(quantiles, target, (0.50, 0.90, 0.95, 0.99))
    assert not report["quantile_crossing_occurred"]
    assert report["empirical_coverage"]["0.5"] == 0.0
    assert "0.95" in report["upper_tail_prediction_error"]


def test_paired_sortie_comparison_and_value_decision() -> None:
    target = np.array([1.0, 2.0, 1.5, 2.5, 2.0, 3.0])
    sorties = np.array([0, 0, 1, 1, 2, 2])
    predictions = {
        "B0_distance": target + 1.0,
        "B1_monte_carlo": target + 0.1,
        "B2_scalar_td": target + 0.2,
        "B3_distributional_median": target + 0.3,
    }
    comparison = paired_sortie_comparisons(predictions, target, sorties, seed=7, bootstrap_samples=500)
    assert comparison["B1_monte_carlo"]["bootstrap_95_ci_upper"] < 0.0
    decision = learned_value_decision(comparison)
    assert decision["LEARNED_MODEL_ADDS_VALUE"] == "TRUE"
    assert decision["best_learned_method"] == "B1_monte_carlo"


def test_residual_analysis_and_all_six_figures(tmp_path: Path) -> None:
    target = np.linspace(0.1, 1.0, 20)
    distance = np.linspace(0.2, 3.8, 20)
    predictions = {
        "B0_distance": target + 0.1,
        "B1_monte_carlo": target + 0.05,
        "B2_scalar_td": target + 0.02,
        "B3_distributional_median": target,
    }
    relationships = residual_relationships(
        predictions["B0_distance"] - target,
        {"velocity": np.linspace(0.0, 0.3, 20), "prefix": np.arange(20)},
    )
    assert set(relationships) == {"velocity", "prefix"}
    quantiles = np.column_stack((target, target + 0.1, target + 0.2, target + 0.3))
    starts = np.column_stack((np.linspace(0.2, 3.8, 20), np.linspace(3.8, 0.2, 20), np.ones(20)))
    figures = build_figures(
        tmp_path / "figures",
        predictions=predictions,
        quantile_predictions=quantiles,
        quantile_levels=(0.50, 0.90, 0.95, 0.99),
        target=target,
        distances=distance,
        path_lengths=distance + 0.1,
        return_start_positions=starts,
        world_size=np.array([4.0, 4.0, 2.0]),
    )
    assert len(figures) == 6
    assert all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in figures.values())


def test_horizon_report_separates_terminal_and_long_horizon_error() -> None:
    target = np.array([0.1, 0.2, 0.3, 0.5])
    prediction = np.array([0.1, 0.25, 0.5, 0.9])
    horizons = np.array([1, 2, 3, 8])
    report = horizon_error_report(prediction, target, horizons)
    assert report["bins"]["1"]["mae"] == 0.0
    assert report["bins"]["6-10"]["mae"] == pytest.approx(0.4)
    assert report["absolute_error_vs_horizon_correlation"] > 0.0
