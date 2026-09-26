"""Replay a frozen PPO model in its unshielded training plant.

This diagnostic separates a policy's own return/charge decisions from effects
introduced by the common evaluation safety layer. It does not train or change
the frozen model.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from stable_baselines3 import PPO

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from learning2d.finite_horizon_gym import FiniteHorizonGym
from learning2d.mode_policy import ModeMaskedPolicy  # noqa: F401; model loader


def replay(model_path: Path, map_ids: tuple[int, ...]) -> dict:
    calibration_path = CALIBRATION_OUTPUT / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    model = PPO.load(str(model_path), device="cpu")
    rows = []
    for map_id in map_ids:
        env = FiniteHorizonGym(
            (map_id,), calibration["capacity_synthetic_energy"],
            calibration["full_charge_seconds"], shielded=False,
        )
        try:
            observation, _ = env.reset(seed=map_id, options={"map_id": map_id})
            actions: Counter[int] = Counter()
            shaped_return = 0.0
            decisions = 0
            while not env.env.done:
                action, _ = model.predict(observation, deterministic=True)
                action_id = int(action)
                actions[action_id] += 1
                observation, reward, _, _, _ = env.step(action_id)
                shaped_return += reward
                decisions += 1
                if decisions > 3000:
                    raise RuntimeError("raw replay exceeded the decision limit")
            rows.append({
                "map_id": map_id,
                "policy_decisions": decisions,
                "simulated_seconds": env.env.time_s,
                "completed_targets": env.env.completed_targets,
                "charge_events": env.env.charge_events,
                "collision_count": env.env.state.collision_count,
                "actual_depletion": env.env.failure_reason == "depletion",
                "failure_reason": env.env.failure_reason,
                "shaped_return": shaped_return,
                "action_counts": {str(k): v for k, v in sorted(actions.items())},
            })
        finally:
            env.close()
    return {
        "protocol": "raw_frozen_ppo_replay_v1",
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        "calibration_sha256": sha256(calibration_path.read_bytes()).hexdigest(),
        "map_ids": list(map_ids),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-id", type=int, nargs="+", required=True)
    args = parser.parse_args()
    result = replay(args.model, tuple(args.map_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
