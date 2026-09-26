"""Resumable exact-environment evaluation of GPU-ranked MPC proposals."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from time import perf_counter, time
import traceback

import torch

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.gym_adapter import DualConstraintGym
from dual_constraint_2d.matrix_runner import _hash_sources
from .rollout_mpc import BatchedRolloutMPC


ROOT = Path(__file__).resolve().parents[1]


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def _manifest(map_id: int, horizon_blocks: int, shortlist: int) -> dict:
    source = _hash_sources()
    for path in sorted((ROOT / "gpu2d").glob("*.py")):
        source[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return {
        "protocol": "gpu_ranked_exact_certified_mpc_v0",
        "map_id": map_id, "horizon_blocks": horizon_blocks,
        "shortlist": shortlist, "device": "cuda",
        "cuda_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "calibration_sha256": sha256(
            (CALIBRATION_OUTPUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": source,
    }


def _summary(wrapper: DualConstraintGym, events: list[dict], policy) -> dict:
    env = wrapper.env
    return {
        "algorithm": f"gpu_mpc_h{policy.horizon_blocks}",
        "map_id": env.case.map_id, "done": env.done,
        "policy_decisions": len(events),
        "plant_decisions": sum(row["plant_decisions"] for row in events),
        "simulated_seconds": env.time_s,
        "completed_targets": env.completed_targets,
        "collision_count": env.state.collision_count,
        "safety_cost": env.state.safety_cost,
        "actual_depletion": env.failure_reason == "depletion",
        "failure_reason": env.failure_reason,
        "charge_events": env.charge_events,
        "return_takeovers": env.return_takeovers,
        "energy_filter_events": env.energy_filter_events,
        "collision_intervention_steps": env.collision_intervention_steps,
        "gpu_candidates": policy.stats.gpu_candidates,
        "exact_candidates": policy.stats.exact_candidates,
        "gpu_wall_seconds": policy.stats.gpu_wall_seconds,
        "exact_wall_seconds": policy.stats.exact_wall_seconds,
        "final_energy": env.energy,
    }


def _save(output: Path, wrapper: DualConstraintGym, policy, events: list[dict]) -> None:
    _atomic_bytes(output / "checkpoint.pkl", pickle.dumps(
        {"wrapper": wrapper, "policy": policy, "events": events},
        protocol=pickle.HIGHEST_PROTOCOL,
    ))
    _atomic_bytes(output / "events.jsonl", (
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in events)
    ).encode())
    summary = _summary(wrapper, events, policy)
    _atomic_json(output / "status.json", summary)
    if summary["done"]:
        _atomic_json(output / "summary.json", summary)


def evaluate(output: Path, *, map_id: int, horizon_blocks: int,
             shortlist: int = 3, deadline_epoch: float | None = None,
             max_new_decisions: int | None = None,
             checkpoint_every: int = 20) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU evaluation requires CUDA")
    if checkpoint_every <= 0 or (max_new_decisions is not None and max_new_decisions <= 0):
        raise ValueError("invalid checkpoint or decision limit")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(map_id, horizon_blocks, shortlist)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("GPU evaluation manifest/source changed")
    else:
        _atomic_json(manifest_path, manifest)
    checkpoint = output / "checkpoint.pkl"
    if checkpoint.exists():
        saved = pickle.loads(checkpoint.read_bytes())
        wrapper, policy, events = saved["wrapper"], saved["policy"], saved["events"]
    else:
        calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text())
        wrapper = DualConstraintGym(
            (map_id,), calibration["capacity_synthetic_energy"],
            calibration["full_charge_seconds"], shielded=True,
        )
        wrapper.reset(seed=map_id, options={"map_id": map_id})
        policy = BatchedRolloutMPC(wrapper.env.case,
                                   horizon_blocks=horizon_blocks,
                                   exact_shortlist=shortlist)
        events = []
    began = perf_counter()
    new_count = 0
    try:
        while not wrapper.env.done:
            if deadline_epoch is not None and time() >= deadline_epoch:
                break
            if max_new_decisions is not None and new_count >= max_new_decisions:
                break
            action = policy.act(wrapper.env)
            _, reward, terminated, truncated, info = wrapper.step(action)
            events.append({"decision": len(events), "action_id": action,
                           "reward_shaped": reward, "terminated": terminated,
                           "truncated": truncated, **info})
            new_count += 1
            if new_count % checkpoint_every == 0 or wrapper.env.done:
                _save(output, wrapper, policy, events)
        _save(output, wrapper, policy, events)
    except Exception:
        _atomic_json(output / "error.json", {"traceback": traceback.format_exc(),
                                             "saved_decisions": len(events)})
        raise
    return {**_summary(wrapper, events, policy),
            "new_decisions": new_count,
            "wall_seconds_this_call": perf_counter() - began}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--horizon-blocks", type=int, choices=(4, 8), required=True)
    parser.add_argument("--shortlist", type=int, default=3)
    parser.add_argument("--deadline-epoch", type=float)
    parser.add_argument("--max-new-decisions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output, map_id=args.map_id,
                              horizon_blocks=args.horizon_blocks,
                              shortlist=args.shortlist,
                              deadline_epoch=args.deadline_epoch,
                              max_new_decisions=args.max_new_decisions,
                              checkpoint_every=args.checkpoint_every), sort_keys=True))


if __name__ == "__main__":
    main()
