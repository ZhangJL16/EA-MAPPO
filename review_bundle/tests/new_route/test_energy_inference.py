from __future__ import annotations

import numpy as np
import torch

from safety.energy import MonotoneQuantileCritic, QuantileEnergyPredictor, ScalarEnergyCritic, ScalarEnergyPredictor


def test_saved_scalar_and_quantile_predictors_use_navigation_contract(tmp_path) -> None:
    scalar = ScalarEnergyCritic(80)
    scalar_path = tmp_path / "scalar.pt"
    torch.save(
        {"model_kind": "scalar", "input_dim": 80, "quantile_levels": None, "state_dict": scalar.state_dict()},
        scalar_path,
    )
    scalar_predictor = ScalarEnergyPredictor.load(scalar_path)
    scalar_value = scalar_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32))
    assert np.isfinite(scalar_value) and scalar_value >= 0.0

    quantile = MonotoneQuantileCritic(80)
    quantile_path = tmp_path / "quantile.pt"
    torch.save(
        {
            "model_kind": "quantile",
            "input_dim": 80,
            "quantile_levels": list(quantile.quantile_levels),
            "state_dict": quantile.state_dict(),
        },
        quantile_path,
    )
    quantile_predictor = QuantileEnergyPredictor.load(quantile_path)
    lower = quantile_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32), quantile=0.50)
    upper = quantile_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32), quantile=0.95)
    assert upper >= lower >= 0.0
