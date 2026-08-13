from __future__ import annotations

import numpy as np
import pytest
import torch

from safety.energy import MonotoneQuantileCritic, QuantileEnergyPredictor, ScalarEnergyCritic, ScalarEnergyPredictor


def test_saved_scalar_and_quantile_predictors_use_navigation_contract(tmp_path) -> None:
    scalar = ScalarEnergyCritic(80)
    scalar_path = tmp_path / "scalar.pt"
    torch.save(
        {
            "model_kind": "scalar",
            "input_dim": 80,
            "quantile_levels": None,
            "output_scale": 2.0,
            "state_dict": scalar.state_dict(),
        },
        scalar_path,
    )
    scalar_predictor = ScalarEnergyPredictor.load(scalar_path)
    scalar_value = scalar_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32))
    with torch.no_grad():
        unscaled_scalar = float(scalar(torch.zeros(1, 80)).item())
    assert scalar_value == pytest.approx(2.0 * unscaled_scalar)

    quantile = MonotoneQuantileCritic(80)
    quantile_path = tmp_path / "quantile.pt"
    torch.save(
        {
            "model_kind": "quantile",
            "input_dim": 80,
            "quantile_levels": list(quantile.quantile_levels),
            "output_scale": 3.0,
            "state_dict": quantile.state_dict(),
        },
        quantile_path,
    )
    quantile_predictor = QuantileEnergyPredictor.load(quantile_path)
    lower = quantile_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32), quantile=0.50)
    upper = quantile_predictor.predict(np.zeros(77, dtype=np.float32), np.zeros(3, dtype=np.float32), quantile=0.95)
    with torch.no_grad():
        unscaled = quantile(torch.zeros(1, 80))[0]
    assert lower == pytest.approx(3.0 * float(unscaled[0]))
    assert upper == pytest.approx(3.0 * float(unscaled[2]))
    assert upper >= lower >= 0.0
