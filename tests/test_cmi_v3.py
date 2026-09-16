from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.analyze_cmi_v3_dev import conformal_quantile, save_states
from scripts.cmi_v3_models import (
    COMPACT_DIM, FRAME_DIM, FULL_DIM, calibration_worlds, fit_ridge,
    legal_features, load_states, predict_frozen_ensemble, predict_ridge,
    select_alpha,
)
from scripts.validate_paired_advantage_contract import SEED_RANGES, SPLIT_SIZES


def synthetic_legal_arrays(anchors: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(20260912))
    return {
        "nav_observation": rng.normal(size=(anchors, 64, 2056)).astype(np.float32),
        "return_observation": rng.normal(size=(anchors, 64, 2056)).astype(np.float32),
        "battery": rng.uniform(0, 1, size=(anchors, 64)).astype(np.float32),
        "distance_to_charger": rng.uniform(0, 100, size=(anchors, 64)).astype(np.float32),
        "task_progress": rng.integers(0, 5, size=(anchors, 64)).astype(np.float32),
        "task_clock": np.broadcast_to(np.arange(64), (anchors, 64)).astype(np.float32),
        "step": np.broadcast_to(np.arange(64), (anchors, 64)).astype(np.float32),
        "previous_nominal_action": rng.uniform(-1, 1, size=(anchors, 64, 3)).astype(np.float32),
        "previous_executed_action": rng.uniform(-1, 1, size=(anchors, 64, 3)).astype(np.float32),
        "previous_realized_acceleration": rng.normal(size=(anchors, 64, 3)).astype(np.float32),
        "previous_action_valid": np.ones((anchors, 64), np.float32),
        "previous_contact": np.zeros((anchors, 64), np.float32),
    }


def test_legal_feature_dimensions_and_nesting() -> None:
    arrays = synthetic_legal_arrays()
    features = legal_features(arrays)
    assert features["latest"].shape == (3, COMPACT_DIM)
    assert features["full"].shape == (3, FULL_DIM)
    np.testing.assert_array_equal(features["full"][:, :COMPACT_DIM], features["latest"])
    assert FRAME_DIM == 4128
    assert np.isfinite(features["full"]).all()


def test_calibration_world_split_is_exact_and_deterministic() -> None:
    worlds = np.asarray([f"sha256:{index:064x}" for index in range(25) for _ in range(2)])
    first = calibration_worlds(worlds)
    second = calibration_worlds(worlds[::-1])
    assert first == second
    assert len(first) == 5
    assert first.issubset(set(worlds.tolist()))


def test_linear_ridge_recovers_conditional_q_signal() -> None:
    rng = np.random.Generator(np.random.PCG64(7))
    x = rng.normal(size=(80, 12))
    coefficient = rng.normal(size=(12, 2))
    y = x @ coefficient + rng.normal(scale=0.02, size=(80, 2))
    worlds = np.asarray([f"sha256:{index:064x}" for index in range(40) for _ in range(2)])
    weights = np.ones(80)
    alpha = select_alpha(
        x, y, weights, worlds, family="LINEAR_RIDGE", seed=2026091221
    )
    state = fit_ridge(
        x[:60], y[:60], weights[:60], family="LINEAR_RIDGE",
        seed=2026091221, alpha=alpha,
    )
    prediction = predict_ridge(state, x[60:])
    model_mse = float(np.mean(np.square(prediction - y[60:])))
    constant_mse = float(np.mean(np.square(y[:60].mean(axis=0) - y[60:])))
    assert model_mse < 0.02 * constant_mse


def test_frozen_npz_state_round_trip(tmp_path: Path) -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    x = rng.normal(size=(32, 9))
    y = rng.normal(size=(32, 2))
    weights = np.ones(32)
    state = fit_ridge(
        x, y, weights, family="RFF_RIDGE", seed=2026091222, alpha=10.0
    )
    path = tmp_path / "models.npz"
    manifest = save_states(path, [("latest", "RFF_RIDGE", state)])
    loaded = load_states(str(path), manifest)
    np.testing.assert_allclose(predict_ridge(state, x), predict_ridge(loaded[0][2], x))


def test_fixed_split_and_resample_contract_constants() -> None:
    assert SPLIT_SIZES == {"SMOKE_DEBUG": 2, "CMI_V3_DEV": 96, "CMI_V3_CONFIRM": 128}
    assert SEED_RANGES["CMI_V3_DEV"] == {"first": 1_136_000_455, "last": 1_136_000_550}
    assert SEED_RANGES["CMI_V3_CONFIRM"] == {"first": 1_136_000_551, "last": 1_136_000_678}
    assert conformal_quantile(np.arange(9, dtype=np.float64), 0.90) == 8.0
