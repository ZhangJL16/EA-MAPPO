from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .critics import MonotoneQuantileCritic, ScalarEnergyCritic


class ScalarEnergyPredictor:
    def __init__(self, model: ScalarEnergyCritic, *, output_scale: float = 1.0, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.output_scale = float(output_scale)
        self.device = device

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "ScalarEnergyPredictor":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = ScalarEnergyCritic(int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"])
        return cls(model, output_scale=float(payload.get("output_scale", 1.0)), device=device)

    def predict(self, observation: np.ndarray, action: np.ndarray) -> float:
        features = _features(observation, action, self.device)
        with torch.no_grad():
            return self.output_scale * float(self.model(features).item())


class QuantileEnergyPredictor:
    def __init__(self, model: MonotoneQuantileCritic, *, output_scale: float = 1.0, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.output_scale = float(output_scale)
        self.device = device

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "QuantileEnergyPredictor":
        payload = torch.load(path, map_location=device, weights_only=True)
        levels = tuple(float(value) for value in payload["quantile_levels"])
        model = MonotoneQuantileCritic(int(payload["input_dim"]), levels)
        model.load_state_dict(payload["state_dict"])
        return cls(model, output_scale=float(payload.get("output_scale", 1.0)), device=device)

    def predict(self, observation: np.ndarray, action: np.ndarray, *, quantile: float = 0.95) -> float:
        if quantile not in self.model.quantile_levels:
            raise ValueError(f"quantile {quantile} is not available")
        column = self.model.quantile_levels.index(quantile)
        features = _features(observation, action, self.device)
        with torch.no_grad():
            return self.output_scale * float(self.model(features)[0, column].item())


def _features(observation: np.ndarray, action: np.ndarray, device: str) -> torch.Tensor:
    state = np.asarray(observation, dtype=np.float32)
    control = np.asarray(action, dtype=np.float32)
    if state.shape != (77,) or control.shape != (3,):
        raise ValueError("energy predictor requires the 77-state/3-action navigation contract")
    values = np.concatenate((state, control))[None, :]
    if not np.all(np.isfinite(values)):
        raise ValueError("energy predictor inputs must be finite")
    return torch.as_tensor(values, device=device)
