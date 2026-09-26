"""Resumable, bounded startup runner for full-map nonlearning diagnostics.

The default call advances only one policy decision and writes a checkpoint.
Do not treat a startup checkpoint as a completed benchmark result.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from time import perf_counter

from .baseline import FullMapRouteBaseline
from .calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from .environment import DualConstraintEnv


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/dual_constraint_2d_fullmap_baseline_20260926/map000_full"
SOURCE_FILES = (
    "dual_constraint_2d/backup.py",
    "dual_constraint_2d/baseline.py",
    "dual_constraint_2d/baseline_runner.py",
    "dual_constraint_2d/calibration.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/reference.py",
    "dual_constraint_2d/routes.py",
    "dual_constraint_2d/shield.py",
    "dual_constraint_2d/synthetic.py",
    "dual_constraint_2d/targets.py",
    "nav3d/controller.py",
    "nav3d/geometry.py",
    "nav3d/simulation.py",
)


def _write_atomic(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, path)


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _manifest(map_id: int, horizon_s: float, charge_mode: str) -> dict:
    calibration = CALIBRATION_OUTPUT / "calibration.json"
    if not calibration.is_file():
        raise FileNotFoundError("completed synthetic calibration is required")
    return {
        "protocol": "dual_constraint_2d_fullmap_nonlearning_v0",
        "map_id": map_id,
        "horizon_s": horizon_s,
        "charge_mode": charge_mode,
        "calibration_sha256": sha256(calibration.read_bytes()).hexdigest(),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
    }


def _status(env: DualConstraintEnv, events: list[dict]) -> dict:
    return {
        "done": env.done,
        "decision_count": len(events),
        "simulated_seconds": env.time_s,
        "completed_targets": env.completed_targets,
        "collision_count": env.state.collision_count,
        "safety_cost": env.state.safety_cost,
        "energy": env.energy,
        "mode": env.mode,
        "failure_reason": env.failure_reason,
        "charge_events": env.charge_events,
        "return_takeovers": env.return_takeovers,
        "collision_intervention_steps": env.collision_intervention_steps,
        "energy_filter_events": env.energy_filter_events,
        "rejected_departures": env.rejected_departures,
    }


def _checkpoint(output: Path, env: DualConstraintEnv, policy: FullMapRouteBaseline, events: list[dict]) -> None:
    # This pickle is a trusted local checkpoint, never an interchange format.
    _write_atomic(output / "checkpoint.pkl", pickle.dumps(
        {"env": env, "policy": policy, "events": events}, protocol=pickle.HIGHEST_PROTOCOL
    ))
    _write_atomic(
        output / "events.jsonl",
        ("".join(json.dumps(row, sort_keys=True) + "\n" for row in events)).encode("utf-8"),
    )
    _write_atomic(output / "status.json", _json_bytes(_status(env, events)))
    if env.done:
        _write_atomic(output / "summary.json", _json_bytes(_status(env, events)))


def run(
    output: Path = DEFAULT_OUTPUT,
    *,
    map_id: int = 0,
    horizon_s: float = 600.0,
    charge_mode: str = "full",
    max_new_decisions: int = 1,
    checkpoint_every: int = 20,
) -> dict:
    if max_new_decisions <= 0 or checkpoint_every <= 0:
        raise ValueError("max_new_decisions and checkpoint_every must be positive")
    output.mkdir(parents=True, exist_ok=True)
    current_manifest = _manifest(map_id, horizon_s, charge_mode)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != current_manifest:
            raise RuntimeError("baseline manifest or source changed; refusing mixed results")
    else:
        _write_atomic(manifest_path, _json_bytes(current_manifest))
    checkpoint_path = output / "checkpoint.pkl"
    if checkpoint_path.exists():
        saved = pickle.loads(checkpoint_path.read_bytes())
        env = saved["env"]
        policy = saved["policy"]
        events = saved["events"]
    else:
        calibration = json.loads(
            (CALIBRATION_OUTPUT / "calibration.json").read_text(encoding="utf-8")
        )
        env = DualConstraintEnv(
            map_id,
            calibration["capacity_synthetic_energy"],
            calibration["full_charge_seconds"],
            horizon_s=horizon_s,
        )
        policy = FullMapRouteBaseline(env.case, charge_mode=charge_mode)
        events: list[dict] = []
    began = perf_counter()
    new_decisions = 0
    while not env.done and new_decisions < max_new_decisions:
        action = policy.act(env.observe())
        observation, reward, _, info = env.step(action)
        events.append({
            "decision": len(events),
            "simulated_seconds": observation["time_s"],
            "energy": observation["energy"],
            "mode": observation["mode"],
            "target_xy": observation["target_xy"],
            "reward": reward,
            **info,
        })
        new_decisions += 1
        if new_decisions % checkpoint_every == 0 or env.done:
            _checkpoint(output, env, policy, events)
    _checkpoint(output, env, policy, events)
    return {**_status(env, events), "new_decisions": new_decisions, "wall_seconds_this_call": perf_counter() - began}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--map-id", type=int, default=0)
    parser.add_argument("--horizon-s", type=float, default=600.0)
    parser.add_argument("--charge-mode", choices=("full", "reference_minimum"), default="full")
    parser.add_argument("--max-new-decisions", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(run(
        args.output,
        map_id=args.map_id,
        horizon_s=args.horizon_s,
        charge_mode=args.charge_mode,
        max_new_decisions=args.max_new_decisions,
        checkpoint_every=args.checkpoint_every,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
