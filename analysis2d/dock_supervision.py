"""Audit and pair the completed 2D PPO dock-supervision validation matrix."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / "artifacts" / "ppo_dock_supervision_validation_20260926"
OLD = ROOT / "artifacts" / "dual_constraint_2d_static_matrix_v2_horizon_fix_20260926" / "validation"
GPU = ROOT / "artifacts" / "gpu_mpc_parallel_v6_8workers_20260926" / "validation"
SEEDS = (101, 202, 303)
MAPS = tuple(range(32, 40))


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_job(directory: Path) -> tuple[dict, dict, list[dict]]:
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    if not summary["done"] or summary["failure_reason"] is not None:
        raise RuntimeError(f"incomplete or failed job: {directory}")
    if abs(summary["simulated_seconds"] - 600.0) > 1e-7:
        raise RuntimeError(f"job lacks full simulated horizon: {directory}")
    if len(events) != summary["policy_decisions"]:
        raise RuntimeError(f"event count mismatch: {directory}")
    if not events or [row["decision"] for row in events] != list(range(len(events))):
        raise RuntimeError(f"noncontiguous event sequence: {directory}")
    if abs(events[-1]["time_s"] - summary["simulated_seconds"]) > 1e-7:
        raise RuntimeError(f"event tail differs from summary: {directory}")
    return summary, manifest, events


def analyze(new_root: Path = NEW, old_root: Path = OLD,
            gpu_root: Path = GPU) -> dict:
    status = json.loads((new_root / "status.json").read_text())
    manifest = json.loads((new_root / "manifest.json").read_text())
    if not status.get("complete") or status.get("completed_jobs") != 24:
        raise RuntimeError("dock-supervision matrix is not complete")
    if manifest.get("validation_maps") != list(MAPS) or manifest.get("ppo_seeds") != list(SEEDS):
        raise RuntimeError("unexpected validation split")
    if (new_root / "orchestrator_error.json").exists():
        raise RuntimeError("orchestrator error artifact exists")
    if list(new_root.glob("**/error.json")):
        raise RuntimeError("job error artifact exists")

    rows: list[dict] = []
    for seed in SEEDS:
        for map_id in MAPS:
            raw_path = old_root / f"ppo_seed_{seed}" / f"map_{map_id:03d}"
            new_path = new_root / f"seed_{seed}" / f"map_{map_id:03d}"
            raw, raw_manifest, raw_events = _load_job(raw_path)
            new, new_manifest, new_events = _load_job(new_path)
            if raw["map_id"] != new["map_id"] or new["map_id"] != map_id:
                raise RuntimeError("map mismatch in paired jobs")
            if raw_manifest["model_sha256"] != new_manifest["model_sha256"]:
                raise RuntimeError("model mismatch in paired jobs")
            if raw_manifest["calibration_sha256"] != new_manifest["calibration_sha256"]:
                raise RuntimeError("calibration mismatch in paired jobs")
            shared = set(raw_manifest["source_sha256"]) & set(new_manifest["source_sha256"])
            if any(raw_manifest["source_sha256"][name] != new_manifest["source_sha256"][name]
                   for name in shared):
                raise RuntimeError("environment source mismatch in paired jobs")
            overrides = sum(event["proposed_action_id"] != event["executed_action_id"]
                            for event in new_events)
            if overrides != new["supervisor_overrides"]:
                raise RuntimeError("override log and summary disagree")
            rows.append({
                "seed": seed, "map_id": map_id,
                "raw_completed": raw["completed_targets"],
                "supervised_completed": new["completed_targets"],
                "completed_difference": new["completed_targets"] - raw["completed_targets"],
                "raw_charge_events": raw["charge_events"],
                "supervised_charge_events": new["charge_events"],
                "raw_departure_rejections": sum(
                    event["event"] == "departure_rejected" for event in raw_events
                ),
                "supervised_departure_rejections": new["departure_rejections"],
                "supervisor_overrides": overrides,
                "proposed_flight_return_actions": sum(
                    event["mode_before"] == "flight" and event["proposed_action_id"] == 9
                    for event in new_events
                ),
                "flight_decisions": sum(event["mode_before"] == "flight" for event in new_events),
                "proposed_dock_charge_actions": new["proposed_dock_charge_actions"],
                "raw_return_takeovers": raw["return_takeovers"],
                "supervised_return_takeovers": new["return_takeovers"],
                "raw_energy_filter_events": raw["energy_filter_events"],
                "supervised_energy_filter_events": new["energy_filter_events"],
                "raw_collision_count": raw["collision_count"],
                "supervised_collision_count": new["collision_count"],
                "raw_actual_depletion": raw["actual_depletion"],
                "supervised_actual_depletion": new["actual_depletion"],
                "raw_policy_decisions": raw["policy_decisions"],
                "supervised_policy_decisions": new["policy_decisions"],
                "new_summary_sha256": _sha(new_path / "summary.json"),
                "new_events_sha256": _sha(new_path / "events.jsonl"),
            })

    map_means = np.asarray([
        np.mean([row["completed_difference"] for row in rows if row["map_id"] == map_id])
        for map_id in MAPS
    ], dtype=float)
    delta_mean = float(map_means.mean())
    margin = float(t.ppf(0.975, df=len(MAPS)-1) * map_means.std(ddof=1) / np.sqrt(len(MAPS)))
    by_seed = {}
    for seed in SEEDS:
        subset = [row for row in rows if row["seed"] == seed]
        by_seed[str(seed)] = {
            "raw_completed_total": sum(row["raw_completed"] for row in subset),
            "supervised_completed_total": sum(row["supervised_completed"] for row in subset),
            "supervised_charge_events": sum(row["supervised_charge_events"] for row in subset),
            "supervisor_overrides": sum(row["supervisor_overrides"] for row in subset),
            "proposed_flight_return_actions": sum(
                row["proposed_flight_return_actions"] for row in subset
            ),
            "flight_decisions": sum(row["flight_decisions"] for row in subset),
            "supervised_departure_rejections": sum(
                row["supervised_departure_rejections"] for row in subset
            ),
        }
    raw_total = sum(row["raw_completed"] for row in rows)
    new_total = sum(row["supervised_completed"] for row in rows)
    context = {}
    for label, base, method in (
        ("route_full", old_root, "route_full"),
        ("gpu_h4", gpu_root, "gpu_mpc_h4"),
    ):
        values = [json.loads((base / method / f"map_{map_id:03d}" / "summary.json").read_text())
                  ["completed_targets"] for map_id in MAPS]
        context[label] = {"completed_total": sum(values), "completed_mean": float(np.mean(values))}
    return {
        "protocol": "ppo_dock_supervision_pair_audit_v0",
        "matrix_manifest_sha256": _sha(new_root / "manifest.json"),
        "matrix_status_sha256": _sha(new_root / "status.json"),
        "paired_jobs": len(rows),
        "map_ids": list(MAPS), "seed_ids": list(SEEDS),
        "raw_completed_total": raw_total,
        "supervised_completed_total": new_total,
        "raw_completed_mean_per_job": raw_total / len(rows),
        "supervised_completed_mean_per_job": new_total / len(rows),
        "relative_completed_increase": (new_total - raw_total) / raw_total,
        "mean_paired_difference_by_map": delta_mean,
        "paired_difference_t95_by_map": [delta_mean - margin, delta_mean + margin],
        "per_map_mean_differences": {str(map_id): float(value)
                                     for map_id, value in zip(MAPS, map_means)},
        "pair_wins_ties_losses": [
            sum(row["completed_difference"] > 0 for row in rows),
            sum(row["completed_difference"] == 0 for row in rows),
            sum(row["completed_difference"] < 0 for row in rows),
        ],
        "raw_departure_rejections": sum(row["raw_departure_rejections"] for row in rows),
        "supervised_departure_rejections": sum(
            row["supervised_departure_rejections"] for row in rows
        ),
        "raw_charge_events": sum(row["raw_charge_events"] for row in rows),
        "supervised_charge_events": sum(row["supervised_charge_events"] for row in rows),
        "supervisor_overrides": sum(row["supervisor_overrides"] for row in rows),
        "proposed_dock_charge_actions": sum(
            row["proposed_dock_charge_actions"] for row in rows
        ),
        "proposed_flight_return_actions": sum(
            row["proposed_flight_return_actions"] for row in rows
        ),
        "raw_collision_count": sum(row["raw_collision_count"] for row in rows),
        "supervised_collision_count": sum(row["supervised_collision_count"] for row in rows),
        "raw_actual_depletion": sum(row["raw_actual_depletion"] for row in rows),
        "supervised_actual_depletion": sum(row["supervised_actual_depletion"] for row in rows),
        "by_seed": by_seed, "context_only": context,
        "paired_rows": rows,
        "interpretation_limit": (
            "Dock-supervised PPO is a hybrid intervention, not pure PPO."
            " Map-level t interval has only eight map clusters and is descriptive."
            " This validation-only comparison cannot establish a learning advantage."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=NEW / "PAIR_AUDIT.json")
    args = parser.parse_args()
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "paired_jobs", "raw_completed_total", "supervised_completed_total",
        "paired_difference_t95_by_map", "pair_wins_ties_losses",
        "raw_departure_rejections", "supervised_departure_rejections",
        "raw_charge_events", "supervised_charge_events",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
