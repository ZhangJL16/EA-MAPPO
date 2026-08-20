from __future__ import annotations

import numpy as np

from experiments.memory_safety.adaptation_analysis import (
    _adaptation_statistics,
    prefix_split,
)


def test_prefix_padding_is_finite_and_masked() -> None:
    data = {
        "relative_measurement": np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        "ego_position": np.zeros((2, 3, 3), dtype=np.float32),
        "valid": np.ones((2, 3), dtype=bool),
    }
    split = prefix_split(data, 1, 4)
    assert split["relative_measurement"].shape == (2, 4, 3)
    assert np.all(np.isfinite(split["relative_measurement"]))
    assert np.all(~split["valid"][:, :2])
    assert np.all(split["valid"][:, 2:])


def test_adaptation_requires_sustained_threshold_entry() -> None:
    errors = np.array(
        [
            [2.0, 0.5, 1.5, 0.4],
            [2.0, 0.8, 0.7, 0.6],
        ]
    )
    result = _adaptation_statistics(errors, 0.1, 1.0)
    assert result["adapted_within_observed_window_fraction"] == 1.0
    assert result["median_sustained_adaptation_seconds"] == 0.2
