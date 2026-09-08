"""Frozen-policy return-energy data; no changes to navigation or collision laws."""
from __future__ import annotations

import hashlib
import numpy as np

from experiments.directional_navigation.correction_supervision import CorrectionRecovery


def scene_split(seed: int) -> str:
    """Keep all times from one seeded world together; never split trajectory rows."""
    value = int(hashlib.sha256(f"return-energy-v1:{seed}".encode()).hexdigest()[:8], 16) % 10
    return "train" if value < 7 else "calibration" if value < 9 else "test"


def suffix_labels(energy: np.ndarray, contact: np.ndarray, arrived: bool,
                  complete: bool = True) -> tuple[np.ndarray, np.ndarray]:
    energy = np.asarray(energy, dtype=np.float64)
    contact = np.asarray(contact)
    if (energy.ndim != 1 or energy.shape != contact.shape
            or not np.isfinite(energy).all() or np.any(energy < 0)
            or not np.isin(contact, [0, 1]).all()):
        raise ValueError("aligned nonnegative energy and binary unified contacts required")
    if not complete:
        raise ValueError("budget-censored trajectories have no completed-return labels")
    suffix_energy = np.cumsum(energy[::-1])[::-1].copy()
    safe = bool(arrived) & (np.cumsum(contact[::-1])[::-1] == 0)
    return suffix_energy, safe


class ReturnEnergyRecovery(CorrectionRecovery):
    """Legal fresh scenes, known physical charger, declared moving-start distribution.

    This samples return-leg states, NOT a learned continue/return mission policy.
    Physical geometry is used only by the inherited simulator. The actor still
    receives exactly2056 entries, with its goal now the actual charging station.
    """
    def reset(self, *, seed=None, options=None):
        if options is not None:
            return super().reset(seed=seed, options=options)
        if seed is None:
            seed = self.seed_start + self.seed_stride * self.next_episode_index
            self.next_episode_index += 1
        seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 9208]))
        base = self.base
        charger = base.charger_position.copy()
        for _ in range(1000):
            start = np.array([rng.uniform(base.xy_sampling_margin, base.length - base.xy_sampling_margin),
                              rng.uniform(base.xy_sampling_margin, base.width - base.xy_sampling_margin),
                              rng.uniform(base.task_z_min, base.task_z_max)], dtype=np.float32)
            if np.linalg.norm(start - charger) >= base.minimum_task_distance:
                break
        else:
            raise RuntimeError("no legal return-start distance under declared world bounds")
        velocity = rng.uniform([-8., -8., -2.], [8., 8., 2.]).astype(np.float32)
        return super().reset(seed=seed, options={"start_position": start,
                            "start_velocity": velocity, "task_point": charger})

    def step(self, action):
        energy_before = self.energy
        obs, reward, done, truncated, info = super().step(action)
        info["realized_energy"] = float(self.energy - energy_before)
        info["episode_seed"] = self.episode_seed
        return obs, reward, done, truncated, info
