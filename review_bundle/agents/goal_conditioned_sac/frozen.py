from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3 import SAC


class FrozenGoalConditionedSAC:
    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.checkpoint = Path(checkpoint)
        self.model = SAC.load(self.checkpoint, device=device)

    def action(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    def candidate_actions(
        self,
        observation: np.ndarray,
        *,
        count: int,
        seed: int,
        noise_scale: float = 0.15,
    ) -> np.ndarray:
        if count <= 0 or noise_scale < 0.0:
            raise ValueError("count must be positive and noise_scale nonnegative")
        center = self.action(observation, deterministic=True)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_scale, size=(count, center.size))
        candidates = np.clip(center[None, :] + noise, -1.0, 1.0).astype(np.float32)
        candidates[0] = center
        return candidates
