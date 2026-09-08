"""Bounded supervised bearing probe, NOT a navigation or safety experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from envs.UAVEnergyDeliverySAC import StaticCylinderObstacle, UAVEnergyDeliverySACEnv
from experiments.directional_navigation.features import DirectionalLidarExtractor


def dataset(seed: int, scenes: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    env = UAVEnergyDeliverySACEnv(lidar_enabled=True, num_obstacles=0)
    env.reset(seed=seed)
    origin = np.array([2000.0, 2000.0, 200.0], dtype=np.float32)
    env.agent.pos = origin.copy()
    env.agent.vel = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    goal = origin + np.array([500.0, 0.0, 0.0], dtype=np.float32)
    rows, labels = [], []
    try:
        for _ in range(scenes):
            radius = rng.uniform(50, 120)
            distance = radius + rng.uniform(10, 60)
            angle = rng.uniform(0, np.pi / 2)
            # The complete rotation quartet stays in one data partition.
            for rotation in range(4):
                theta = angle + rotation * np.pi / 2
                direction = np.array([np.cos(theta), np.sin(theta)])
                env.obstacles = [
                    StaticCylinderObstacle(origin[:2] + distance * direction, radius)
                ]
                env.num_obstacles = 1
                env._update_lidar()
                rows.append(env.sac_observation_for_goal(goal))
                labels.append(direction)
    finally:
        env.close()
    return torch.tensor(np.stack(rows)), torch.tensor(
        np.stack(labels), dtype=torch.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=300)
    args = parser.parse_args()
    if args.updates <= 0:
        raise ValueError("updates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    train_x, train_y = dataset(931001, 48)
    test_x, test_y = dataset(932001, 24)
    space_env = UAVEnergyDeliverySACEnv(lidar_enabled=True)
    space = space_env.observation_space
    assert isinstance(space, gym.spaces.Box)
    space_env.close()
    train_x, train_y, test_x, test_y = [
        v.to(device) for v in (train_x, train_y, test_x, test_y)
    ]
    results = {}
    for mode in ("global_tiled", "ordered"):
        torch.manual_seed(933001)
        model = nn.Sequential(
            DirectionalLidarExtractor(space, readout=mode),
            nn.Linear(2080, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Identical batches, initialization and number of trainable parameters.
        rng = np.random.default_rng(934001)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.updates):
            indices = torch.as_tensor(rng.integers(0, len(train_x), 64), device=device)
            loss = (model(train_x[indices]) - train_y[indices]).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite probe loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad():
            prediction = model(test_x)
            mse = (prediction - test_y).square().mean().item()
            cos = nn.functional.cosine_similarity(prediction, test_y, dim=1).clamp(
                -1, 1
            )
            angle = torch.rad2deg(torch.acos(cos)).mean().item()
            features = model[0](test_x[:4])
        results[mode] = {
            "heldout_mse": mse,
            "heldout_mean_angle_degrees": angle,
            "parameters": sum(p.numel() for p in model.parameters()),
            "seconds_per_update": elapsed / args.updates,
            "wall_seconds": elapsed,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda"
                else None
            ),
            "first_quartet_embedding_max_difference": (features - features[:1])
            .abs()
            .max()
            .item(),
        }
        torch.save(model.state_dict(), args.output_dir / f"{mode}.pt")
    payload = {
        "protocol": "bearing_identifiability_probe_v1",
        "navigation_evidence": False,
        "label": "unit vector toward cylinder, not action supervision for navigation",
        "train_scenes": 48,
        "heldout_scenes": 24,
        "rotations_per_scene": 4,
        "updates": args.updates,
        "device": str(device),
        "torch_version": torch.__version__,
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "experiments/directional_navigation/features.py",
            )
        },
        "results": results,
        "preflight_passed": bool(
            results["ordered"]["heldout_mse"] < 0.1
            and results["global_tiled"]["heldout_mse"] > 0.4
        ),
        "limitations": "Parameter counts match; effective function classes differ by design. Not a navigation efficacy or safety result.",
    }
    (args.output_dir / "RESULT.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
