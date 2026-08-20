from __future__ import annotations

from experiments.memory_safety.counterexamples import (
    CounterexampleConfig,
    _association_counterexample,
    _dropout_growth,
    _robust_hocbf_trials,
    _sensor_delay_counterexample,
    _tracking_scaling_trials,
    _valid_observer_trials,
)

import numpy as np


def test_vectorized_counterexample_slices() -> None:
    config = CounterexampleConfig(random_states=1000, abrupt_states=1000)
    rng = np.random.default_rng(config.seed)
    valid = _valid_observer_trials(rng, config)
    assert valid["prediction_underbound_count"] == 0
    assert valid["correction_underbound_count"] == 0
    hocbf = _robust_hocbf_trials(rng, config)
    assert hocbf["violations"] == 0


def test_dropout_radius_and_identity_swap_counterexample() -> None:
    config = CounterexampleConfig(random_states=1000, abrupt_states=1000)
    growth = _dropout_growth(config)
    for earlier, later in zip(("0", "1", "2", "3"), ("1", "2", "3", "5")):
        assert np.all(np.asarray(growth[later]) > np.asarray(growth[earlier]))
    association = _association_counterexample(config)
    assert association["swap_passes_single_track_innovation_gate"] is True


def test_multi_obstacle_tracking_scaling_covers_preregistered_counts() -> None:
    config = CounterexampleConfig(
        random_states=100,
        abrupt_states=100,
        tracking_trials_per_count=2,
        tracking_frames=3,
    )
    result = _tracking_scaling_trials(np.random.default_rng(7), config)
    assert set(result["counts"]) == {"2", "4", "8", "16", "32"}
    previous_memory = 0
    for count in (2, 4, 8, 16, 32):
        row = result["counts"][str(count)]
        assert row["assignments"] == 2 * 3 * count
        assert 0.0 <= row["identity_accuracy"] <= 1.0
        assert 0.0 <= row["ambiguous_track_fraction"] <= 1.0
        assert row["hungarian_latency_p99_ms"] >= 0.0
        assert row["explicit_numeric_memory_bytes"] > previous_memory
        previous_memory = row["explicit_numeric_memory_bytes"]


def test_sensor_delay_requires_timestamp_propagation_or_inflation() -> None:
    result = _sensor_delay_counterexample(CounterexampleConfig())
    assert not result["contained_without_delay_inflation"]
    assert np.all(
        np.asarray(result["minimum_delay_aware_radius"])
        > np.asarray(result["naive_sensor_radius"])
    )
