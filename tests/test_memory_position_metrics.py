from __future__ import annotations

import numpy as np

from experiments.memory_safety.posthoc_position_metrics import run


def test_posthoc_position_metrics_use_existing_test_predictions(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    count = 4
    future = np.zeros((count, 21, 3), dtype=np.float32)
    for step in range(21):
        future[:, step, 0] = step * 0.1
    np.savez(
        source / "test.npz",
        relative_measurement=np.zeros((count, 16, 3), dtype=np.float32),
        valid=np.ones((count, 16), dtype=bool),
        ego_position=np.zeros((count, 16, 3), dtype=np.float32),
        future_position=future,
    )
    prediction = np.zeros((count, 6), dtype=np.float32)
    prediction[:, 0] = 1.0
    np.savez(source / "test_predictions.npz", method=prediction)
    (source / "summary.json").write_text("{}", encoding="utf-8")
    result = run(source, tmp_path / "output")
    assert result["methods"]["method"]["1.0"]["position_mae"] < 1e-7
