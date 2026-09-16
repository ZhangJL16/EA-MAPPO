"""Single fixture, single seed, resumable startup; never creates an instance library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from bundling_calibration_core import (CELLS, HYPOTHESES, METHODS, POSES, Learner,
                                      Plant, audits, catalogue, solve_models)

VERSION = "bundling-calibration-startup-v1"
SEED = 20260916
TRUTH = (.30, .05)  # Evaluator-only. Never passed to an unprivileged learner.


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources():
    names = ("experiments/bundling_calibration_core.py", "scripts/run_bundling_calibration.py")
    return {name: digest((ROOT/name).read_bytes()) for name in names}


def write_new(path, data):
    with path.open("x") as handle:
        handle.write(canonical(data)+"\n")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    predictions = {"version": VERSION, "status": "SEALED_BEFORE_SAMPLING",
                   "hypotheses": HYPOTHESES, "audits": audits()}
    write_new(output/"sealed_predictions.json", predictions)
    manifest = {"version": VERSION, "source_hashes": sources(), "seed": SEED,
                "prediction_hash": digest((output/"sealed_predictions.json").read_bytes()),
                "hypotheses": HYPOTHESES, "cells": CELLS, "methods": METHODS,
                "checkpoints": [128, 4096], "truth": "private evaluator theta_A",
                "authority": "startup only; no library or sweep",
                "learner_isolation": "no evaluator/prediction/file capability in learner API; NOT OS sandbox"}
    write_new(output/"manifest.json", manifest)
    (output/"sealed_predictions.json").chmod(0o444)
    return manifest


def validate(output):
    manifest = json.loads((output/"manifest.json").read_text())
    if manifest["version"] != VERSION or manifest["source_hashes"] != sources():
        raise ValueError("source/version mismatch: amendment required, cannot resume")
    if manifest["prediction_hash"] != digest((output/"sealed_predictions.json").read_bytes()):
        raise ValueError("prediction seal mismatch")
    return manifest


def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value


def atomic_checkpoint(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        handle.write(canonical(data)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def run_one(output, capacity, bundling, method, checkpoint):
    key = f"B{capacity}_{'on' if bundling else 'off'}_{method}"
    saved = output/f"{key}.checkpoint.json"
    plant = Plant(capacity, bundling)
    routes = catalogue(capacity, bundling)
    learner = None if method == "oracle_allocation" else Learner(routes, method)
    streams = {c: random.Random(SEED+i*1000003) for i, c in enumerate(("A", "B", "dock"))}
    model = next(m for m in solve_models(routes) if m["theta"] == TRUTH)
    state = {"time": 0, "plant": plant.state, "policy": None, "rng": {}, "pending": None,
             "queries": {"A": 0, "B": 0, "dock": 0}, "counts": [0]*len(routes),
             "pseudo_regret": 0., "reward": 0., "trace": [], "key": key, "checkpoints": []}
    if saved.exists():
        state = json.loads(saved.read_text())
        plant.state, plant.time = tuples(state["plant"]), state["time"]
        for c in streams:
            streams[c].setstate(tuples(state["rng"][c]))
        if learner:
            for name, value in state["policy"].items():
                setattr(learner, name, value)
    if plant.time >= checkpoint:
        return {"key": key, "time": plant.time, "already_complete": True}
    while plant.time < checkpoint:
        if state["pending"] is None:
            assert plant.at_dock
            if learner:
                index, exploration, reason = learner.select(plant.time)
            else:
                # Privileged logarithmic allocation schedule, NOT execution oracle.
                deficient = [i for i, a in enumerate(model["allocation"])
                             if state["counts"][i] < a*__import__("math").log(plant.time+3)]
                index, exploration, reason = (deficient[0] if deficient else model["opt"]), False, "privileged_allocation"
            state["pending"] = {"index": index, "offset": 0, "feedback": [],
                                "exploration": exploration, "reason": reason}
        pending = state["pending"]
        route = routes[pending["index"]]
        action = route.actions[pending["offset"]]
        before = plant.state
        channel = plant.step(action)
        mean = .05 if channel == "dock" else TRUTH[0 if channel == "A" else 1] if channel else 0.
        reward = int(streams[channel].random() < mean) if channel else 0
        if channel:
            state["queries"][channel] += 1
            pending["feedback"].append((channel, reward))
        state["reward"] += reward
        state["pseudo_regret"] += model["gain"]-mean
        state["trace"].append({"t": plant.time, "path": route.name, "action": action,
                               "pose": POSES[plant.state[0]], "before": before, "after": plant.state,
                               "channel": channel, "reward": reward, "selection": pending["reason"]})
        pending["offset"] += 1
        if pending["offset"] == route.length:
            assert plant.at_dock
            state["counts"][pending["index"]] += 1
            if learner:
                learner.observe(pending["index"], pending["feedback"], pending["exploration"])
            state["pending"] = None
    state["time"], state["plant"] = plant.time, plant.state
    state["rng"] = {c: rng.getstate() for c, rng in streams.items()}
    if learner:
        state["policy"] = {k: getattr(learner, k) for k in ("Q", "likelihood", "plays", "totals")}
    state["checkpoints"].append({"t": plant.time, "pseudo_regret": state["pseudo_regret"],
                                 "reward": state["reward"], "counts": state["counts"][:],
                                 "queries": state["queries"].copy()})
    atomic_checkpoint(saved, state)
    return {"key": key, "time": plant.time, "trace_steps": len(state["trace"]),
            "resumable": True, "incomplete_path": state["pending"] is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "startup", "resume"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().parent != (ROOT/"artifacts").resolve():
        raise SystemExit("isolated artifact child required")
    if args.mode == "prepare":
        prepare(args.output)
        print("prediction/gain/alternative audits sealed; zero learning draws")
        return
    validate(args.output)
    checkpoint = 128 if args.mode == "startup" else 4096
    # No concurrent writers or automatic retry. Stale lock needs explicit attention.
    lock = args.output/"RUNNING.lock"
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    started = time.monotonic()
    try:
        health = [run_one(args.output, b, on, method, checkpoint)
                  for b, on in CELLS for method in METHODS]
        receipt = {"version": VERSION, "verification": "UNVERIFIED; startup health only",
                   "scientific_analysis": False, "CONFIRM_access": 0, "library_worlds": 0,
                   "seed_count": 1, "checkpoint": checkpoint, "health": health,
                   "seconds": time.monotonic()-started}
        atomic_checkpoint(args.output/f"health_{checkpoint}.json", receipt)
        print(canonical(receipt))  # No scientific outcome printed by startup audit.
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
