"""Resumable paired evaluation of full-map route, MPC and frozen PPO controls."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
from time import perf_counter
import traceback

from stable_baselines3 import PPO

from .calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from .gym_adapter import DualConstraintGym
from .mpc import FullMapMPC


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = ("route_full", "route_partial", "mpc_h1", "mpc_h2", "ppo")
SOURCE_FILES = (
    "dual_constraint_2d/action_adapter.py",
    "dual_constraint_2d/backup.py",
    "dual_constraint_2d/calibration.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/evaluate_matrix.py",
    "dual_constraint_2d/gym_adapter.py",
    "dual_constraint_2d/mpc.py",
    "dual_constraint_2d/reference.py",
    "dual_constraint_2d/routes.py",
    "dual_constraint_2d/shield.py",
    "dual_constraint_2d/synthetic.py",
    "dual_constraint_2d/targets.py",
    "nav3d/controller.py",
    "nav3d/geometry.py",
    "nav3d/simulation.py",
)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def _atomic_json(path: Path, data: dict) -> None:
    _atomic_bytes(path, (json.dumps(data, sort_keys=True, indent=2) + "\n").encode())


class RouteDiscretePolicy:
    def __init__(self, charge_mode: str) -> None:
        if charge_mode not in ("full", "partial"):
            raise ValueError("invalid route charge mode")
        self.charge_mode = charge_mode

    def act(self, env) -> int:
        if env.mode == "return":
            return 9
        if env.mode == "docked":
            if self.charge_mode == "full":
                return 12 if env.energy + 1e-8 < env.charger.capacity else 4
            needed = min(
                env.charger.capacity,
                env.target.reference_round_trip_energy + 0.05 * env.charger.capacity,
            )
            if env.energy + 1e-8 < needed:
                for action_id, fraction in ((10, 0.5), (11, 0.75), (12, 1.0)):
                    if fraction * env.charger.capacity + 1e-8 >= needed:
                        return action_id
        return 4


def _manifest(algorithm: str, map_id: int, model_path: Path | None) -> dict:
    if algorithm not in ALGORITHMS:
        raise ValueError("invalid evaluation algorithm")
    if algorithm == "ppo" and (model_path is None or not model_path.is_file()):
        raise ValueError("PPO evaluation requires a frozen model file")
    if algorithm != "ppo" and model_path is not None:
        raise ValueError("nonlearning algorithm must not receive a model")
    return {
        "protocol": "dual_constraint_2d_static_fullmap_eval_v0",
        "algorithm": algorithm,
        "map_id": map_id,
        "horizon_s": 600.0,
        "model_path": None if model_path is None else str(model_path.resolve()),
        "model_sha256": None if model_path is None else sha256(model_path.read_bytes()).hexdigest(),
        "calibration_sha256": sha256(
            (CALIBRATION_OUTPUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            name: sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES
        },
    }


def _summary(wrapper: DualConstraintGym, events: list[dict], algorithm: str, policy) -> dict:
    env = wrapper.env
    assert env is not None
    result = {
        "algorithm": algorithm,
        "map_id": env.case.map_id,
        "done": env.done,
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
        "rejected_departures": env.rejected_departures,
        "final_energy": env.energy,
    }
    if isinstance(policy, FullMapMPC):
        result["mpc_candidates"] = policy.stats.candidates
        result["mpc_wall_seconds"] = policy.stats.wall_seconds
    return result


def _save(output: Path, wrapper: DualConstraintGym, policy, events: list[dict], algorithm: str) -> None:
    payload = {"wrapper": wrapper, "policy": policy, "events": events}
    _atomic_bytes(output / "checkpoint.pkl", pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    _atomic_bytes(
        output / "events.jsonl",
        ("".join(json.dumps(row, sort_keys=True) + "\n" for row in events)).encode(),
    )
    result = _summary(wrapper, events, algorithm, policy)
    _atomic_json(output / "status.json", result)
    if result["done"]:
        _atomic_json(output / "summary.json", result)


def evaluate(
    output: Path,
    *,
    algorithm: str,
    map_id: int,
    model_path: Path | None = None,
    max_new_decisions: int | None = None,
    checkpoint_every: int = 20,
) -> dict:
    if max_new_decisions is not None and max_new_decisions <= 0:
        raise ValueError("max_new_decisions must be positive")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    output.mkdir(parents=True, exist_ok=True)
    current_manifest = _manifest(algorithm, map_id, model_path)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != current_manifest:
            raise RuntimeError("evaluation manifest or source changed; refusing mixed results")
    else:
        _atomic_json(manifest_path, current_manifest)
    checkpoint_path = output / "checkpoint.pkl"
    if checkpoint_path.exists():
        saved = pickle.loads(checkpoint_path.read_bytes())
        wrapper = saved["wrapper"]
        policy = saved["policy"]
        events = saved["events"]
    else:
        calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text())
        wrapper = DualConstraintGym(
            (map_id,),
            calibration["capacity_synthetic_energy"],
            calibration["full_charge_seconds"],
            shielded=True,
        )
        wrapper.reset(seed=map_id, options={"map_id": map_id})
        if algorithm.startswith("route_"):
            policy = RouteDiscretePolicy(
                "full" if algorithm == "route_full" else "partial"
            )
        elif algorithm.startswith("mpc_"):
            policy = FullMapMPC(
                wrapper.env.case, depth=1 if algorithm == "mpc_h1" else 2
            )
        else:
            policy = None
        events: list[dict] = []
    model = PPO.load(str(model_path), device="cpu") if algorithm == "ppo" else None
    began = perf_counter()
    new_count = 0
    try:
        while not wrapper.env.done and (
            max_new_decisions is None or new_count < max_new_decisions
        ):
            if model is not None:
                action, _ = model.predict(
                    wrapper.adapter.features(wrapper.env.observe()), deterministic=True
                )
                action_id = int(action)
            else:
                action_id = policy.act(wrapper.env)
            _, reward, terminated, truncated, info = wrapper.step(action_id)
            events.append({
                "decision": len(events),
                "action_id": action_id,
                "reward_shaped": reward,
                "terminated": terminated,
                "truncated": truncated,
                **info,
            })
            new_count += 1
            if new_count % checkpoint_every == 0 or wrapper.env.done:
                _save(output, wrapper, policy, events, algorithm)
        _save(output, wrapper, policy, events, algorithm)
    except Exception:
        _atomic_json(output / "error.json", {
            "algorithm": algorithm,
            "map_id": map_id,
            "saved_decisions": len(events),
            "traceback": traceback.format_exc(),
        })
        raise
    return {
        **_summary(wrapper, events, algorithm, policy),
        "new_decisions": new_count,
        "wall_seconds_this_call": perf_counter() - began,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--map-id", type=int, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--max-new-decisions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(
        args.output,
        algorithm=args.algorithm,
        map_id=args.map_id,
        model_path=args.model_path,
        max_new_decisions=args.max_new_decisions,
        checkpoint_every=args.checkpoint_every,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
