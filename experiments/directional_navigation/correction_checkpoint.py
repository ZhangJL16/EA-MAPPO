"""Load the versioned SAC subclass with the existing full-state snapshot format."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

from .correction_sac import CorrectionSAC


def load_correction_checkpoint(vec: VecEnv, root: Path, contract: dict,
                               device: str) -> tuple[CorrectionSAC, dict]:
    name = json.loads((root / "latest.json").read_text())["checkpoint"]
    if Path(name).name != name or not name.startswith("checkpoint_"):
        raise ValueError("invalid checkpoint pointer")
    destination = root / name
    metadata = json.loads((destination / "metadata.json").read_text())
    if metadata["contract"] != contract:
        raise ValueError("checkpoint contract mismatch")
    if set(metadata["files"]) != {"model.zip", "replay.pkl", "state.pt"}:
        raise ValueError("incomplete checkpoint manifest")
    for filename, size in metadata["files"].items():
        if (destination / filename).stat().st_size != size:
            raise ValueError(f"incomplete checkpoint file: {filename}")
    model = CorrectionSAC.load(destination / "model.zip", env=vec,
                               device=device, force_reset=False)
    model.load_replay_buffer(destination / "replay.pkl")
    state = torch.load(destination / "state.pt", map_location="cpu", weights_only=False)
    if len(state["workers"]) != vec.num_envs:
        raise ValueError("worker count mismatch")
    for i, worker in enumerate(state["workers"]):
        vec.env_method("restore", worker, indices=[i])
    vec._reset_seeds()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    return model, state["records"]
