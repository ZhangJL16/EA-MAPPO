"""Native algorithm baselines and full-state recovery; physics is inherited unchanged."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecEnv

from experiments.directional_navigation.recovery import RecoveryCohort


class ContinuousRecovery(RecoveryCohort):
    """Auto-start another task after a true terminal, unlike the parked collector."""

    def __init__(self, *, seed_start: int, seed_stride: int, **kwargs):
        super().__init__(**kwargs)
        self.seed_start, self.seed_stride = seed_start, seed_stride

    def reset(self, *, seed=None, options=None):
        self.park_after_terminal = False
        return super().reset(seed=seed, options=options)

    def snapshot(self) -> bytes:
        return cloudpickle.dumps((self.__dict__, random.getstate(), np.random.get_state()))

    def restore(self, payload: bytes) -> None:
        state, py_rng, np_rng = cloudpickle.loads(payload)
        self.__dict__.clear()
        self.__dict__.update(state)
        random.setstate(py_rng)
        np.random.set_state(np_rng)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def save_checkpoint(model: PPO | SAC, vec: VecEnv, root: Path, contract: dict,
                    records: dict) -> Path:
    destination = root / f"checkpoint_{model.num_timesteps:09d}"
    destination.mkdir(exist_ok=True)
    model.save(destination / "model.zip")
    if isinstance(model, SAC):
        model.save_replay_buffer(destination / "replay.pkl")
    state = {
        "workers": vec.env_method("snapshot"),
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "records": records,
    }
    torch.save(state, destination / "state.pt")
    atomic_json(destination / "metadata.json", {
        "contract": contract, "transitions": model.num_timesteps,
        "files": {p.name: p.stat().st_size for p in destination.iterdir()
                  if p.name in {"model.zip", "replay.pkl", "state.pt"}},
    })
    atomic_json(root / "latest.json", {"checkpoint": destination.name})
    return destination


def load_checkpoint(algorithm: str, vec: VecEnv, root: Path, contract: dict,
                    device: str) -> tuple[PPO | SAC, dict]:
    destination = root / json.loads((root / "latest.json").read_text())["checkpoint"]
    metadata = json.loads((destination / "metadata.json").read_text())
    if metadata["contract"] != contract:
        raise ValueError("checkpoint contract mismatch")
    for name, size in metadata["files"].items():
        if (destination / name).stat().st_size != size:
            raise ValueError(f"incomplete checkpoint file: {name}")
    cls = SAC if algorithm == "sac" else PPO
    model = cls.load(destination / "model.zip", env=vec, device=device, force_reset=False)
    if isinstance(model, SAC):
        model.load_replay_buffer(destination / "replay.pkl")
    state = torch.load(destination / "state.pt", map_location="cpu", weights_only=False)
    for i, worker in enumerate(state["workers"]):
        vec.env_method("restore", worker, indices=[i])
    vec._reset_seeds()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    return model, state["records"]
