from __future__ import annotations

import numpy as np
import torch

from experiments.memory_safety.estimation_benchmark import (
    EstimationBenchmarkConfig,
    PhysicsResidualEstimator,
    _features,
    _future_position_at,
    _generate_split,
    _kalman_estimate,
    _mhe_estimate,
    _targets,
)


def test_dataset_split_shapes_and_ego_compensation() -> None:
    config = EstimationBenchmarkConfig(
        train_trajectories=20,
        validation_trajectories=10,
        test_trajectories=10,
        epochs=1,
    )
    split = _generate_split(20, 7, config)
    raw = _features(split, ego_compensated=False)
    ego = _features(split, ego_compensated=True)
    assert raw.shape == (20, config.history_length, 4)
    assert ego.shape == raw.shape
    assert _targets(split).shape == (20, 6)
    assert not np.allclose(raw, ego)
    assert np.all(np.isfinite(raw))
    assert np.all(np.isfinite(ego))
    assert split["future_position"].shape == (20, config.future_steps + 1, 3)
    assert _future_position_at(split, 0.25, config.dt).shape == (20, 3)


def test_classical_estimators_and_contracting_physics_model() -> None:
    config = EstimationBenchmarkConfig(
        train_trajectories=20,
        validation_trajectories=10,
        test_trajectories=10,
        epochs=1,
    )
    split = _generate_split(20, 9, config)
    assert _mhe_estimate(split, config.dt).shape == (20, 6)
    assert _kalman_estimate(split, config.dt, constant_acceleration=False).shape == (20, 6)
    assert _kalman_estimate(split, config.dt, constant_acceleration=True).shape == (20, 6)
    model = PhysicsResidualEstimator(16, contractive=True)
    features = torch.from_numpy(_features(split, ego_compensated=True))
    base = torch.zeros(20, 6)
    output = model(features, base)
    assert output.shape == (20, 6)
    assert torch.all(torch.isfinite(output))
