from __future__ import annotations

import numpy as np
import torch

from experiments.memory_safety.grounded_prediction_diagnostic import (
    FOGMJointModel,
    GenericJointGRU,
    MODEL_FEATURE_DIM,
    build_history_batch,
)


def test_history_batch_never_crosses_trajectory_boundaries() -> None:
    features = np.arange(6 * MODEL_FEATURE_DIM, dtype=np.float32).reshape(6, MODEL_FEATURE_DIM)
    trajectory_ids = np.array([0, 0, 0, 1, 1, 1])
    step = np.array([0, 1, 2, 0, 1, 2])
    history = build_history_batch(features, trajectory_ids, step, np.array([2, 3, 5]), 4)
    assert history.shape == (3, 4, MODEL_FEATURE_DIM)
    assert np.array_equal(history[0, -3:], features[:3])
    assert np.array_equal(history[1, -1], features[3])
    assert np.array_equal(history[2, -3:], features[3:])


def test_generic_and_fogm_models_share_outputs_but_not_routing_structure() -> None:
    history = torch.zeros(4, 16, MODEL_FEATURE_DIM)
    generic = GenericJointGRU(MODEL_FEATURE_DIM, 24)
    fogm = FOGMJointModel(24)
    for model in (generic, fogm):
        route, energy, regression = model(history)
        assert route.shape == (4, 6)
        assert energy.shape == (4, 2)
        assert regression.shape == (4, 7)
    object_channel, route_channel, energy_channel = fogm.typed_channels(history)
    assert object_channel.shape[-1] == 8
    assert route_channel.shape[-1] == 8
    assert energy_channel.shape[-1] == 9
