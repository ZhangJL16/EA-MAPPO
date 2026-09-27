"""Audit and collect the frozen ARM CPU shield-feedback diagnostics.

This reads archived final-checkpoint results only; it runs no simulator.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PANELS = {
    "vectorized": ARTIFACTS / "dual_constraint_2d_shield_feedback_arm_cpu_16workers_20260926",
    "independent_seeds": ARTIFACTS / "dual_constraint_2d_shield_feedback_cpu_replicates_20260926",
}
SEEDS = {"vectorized": (101,), "independent_seeds": tuple(range(102, 107))}
MAPS = tuple(range(32, 40))
CONDITIONS = ("control", "feedback")
MODES = ("shielded", "raw")
ACTION_ID = re.compile(rb'^\{"action_id": (\d+),')


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def training_dir(panel: str, seed: int, condition: str) -> Path:
    base = PANELS[panel]
    return (base / condition / "training" if panel == "vectorized"
            else base / f"seed_{seed}" / condition / "training")


def evaluation_dir(panel: str, seed: int, condition: str, mode: str, map_id: int) -> Path:
    base = PANELS[panel]
    prefix = base / condition if panel == "vectorized" else base / f"seed_{seed}" / condition
    return prefix / "validation" / mode / f"map_{map_id:03d}"


def action_counts(path: Path) -> dict[str, int]:
    counts: Counter[int] = Counter()
    with path.open("rb") as stream:
        for line in stream:
            match = ACTION_ID.match(line)
            if match is None:
                raise ValueError(f"unexpected event row in {path}")
            counts[int(match.group(1))] += 1
    return {str(k): v for k, v in sorted(counts.items())}


def collect() -> dict:
    records = []
    training = []
    for panel, seeds in SEEDS.items():
        for seed in seeds:
            for condition in CONDITIONS:
                train = training_dir(panel, seed, condition)
                manifest_path, status_path = train / "manifest.json", train / "status.json"
                manifest, status = read_json(manifest_path), read_json(status_path)
                if (manifest["seed"] != seed or manifest["condition"] != condition
                        or manifest["device"] != "cpu" or not status["complete"]
                        or status["timesteps"] != 25600
                        or manifest["total_timesteps"] != 25600):
                    raise ValueError(f"training not complete or mismatched: {train}")
                for relative, expected in manifest["source_sha256"].items():
                    if digest(ROOT / relative) != expected:
                        raise ValueError(f"training source hash mismatch: {relative}")
                model_path = train / status["latest_model"]
                model_sha = digest(model_path)
                training.append({
                    "panel": panel, "seed": seed, "condition": condition,
                    "episodes": status["episodes"], "timesteps": status["timesteps"],
                    "model_sha256": model_sha,
                })
                for mode in MODES:
                    for map_id in MAPS:
                        output = evaluation_dir(panel, seed, condition, mode, map_id)
                        provenance = read_json(output / "provenance.json")
                        summary = read_json(output / "summary.json")
                        if (provenance["condition"] != condition
                                or provenance["mode"] != mode
                                or provenance["map_id"] != map_id
                                or provenance["model_sha256"] != model_sha
                                or provenance["training_manifest_sha256"] != digest(manifest_path)
                                or provenance["training_status_sha256"] != digest(status_path)):
                            raise ValueError(f"evaluation provenance mismatch: {output}")
                        for relative, expected in provenance["source_sha256"].items():
                            if digest(ROOT / relative) != expected:
                                raise ValueError(f"evaluation source hash mismatch: {relative}")
                        if mode == "raw":
                            if summary["model_sha256"] != model_sha or len(summary["rows"]) != 1:
                                raise ValueError(f"raw summary mismatch: {output}")
                            row = summary["rows"][0]
                            actions = {str(k): v for k, v in row["action_counts"].items()}
                        else:
                            row = summary
                            if not row["done"]:
                                raise ValueError(f"shielded evaluation incomplete: {output}")
                            actions = action_counts(output / "events.jsonl")
                        if (row["map_id"] != map_id
                                or sum(actions.values()) != row["policy_decisions"]):
                            raise ValueError(f"action/event count mismatch: {output}")
                        records.append({
                            "panel": panel, "seed": seed, "condition": condition,
                            "mode": mode, "map_id": map_id,
                            "completed_targets": row["completed_targets"],
                            "actual_depletion": bool(row["actual_depletion"]),
                            "collision_count": row["collision_count"],
                            "charge_events": row["charge_events"],
                            "return_takeovers": row.get("return_takeovers", 0),
                            "rejected_departures": row.get("rejected_departures", 0),
                            "simulated_seconds": row["simulated_seconds"],
                            "failure_reason": row["failure_reason"],
                            "policy_decisions": row["policy_decisions"],
                            "voluntary_returns": actions.get("9", 0),
                            "action_counts": actions,
                        })
    expected = sum(len(seeds) for seeds in SEEDS.values()) * len(CONDITIONS) * len(MODES) * len(MAPS)
    if len(records) != expected or len({(r["panel"], r["seed"], r["condition"], r["mode"], r["map_id"]) for r in records}) != expected:
        raise ValueError("missing or duplicate validation jobs")
    return {"protocol": "2d_shield_feedback_cpu_archived_collection_v1",
            "validation_jobs": expected, "training_jobs": len(training),
            "training": training, "records": records}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
