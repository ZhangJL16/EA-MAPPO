"""Resumable, policy-independent synthetic battery calibration.

The first phase runs fixed station-target-station reference flights.  Only
after all rows exist does it set capacity = twice the median successful energy
and full-charge time = median successful round-trip time.  No learned policy
or method comparison is consulted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import os

import numpy as np

from .reference import ReferenceLeg, simulate_reference_leg
from .routes import NoPlanarRoute, PlanarRouter
from .synthetic import SYNTHETIC_V0, SyntheticEnergy, SyntheticMap, make_synthetic_map


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/dual_constraint_2d_calibration_20260926"
MAP_IDS = tuple(range(8))
CANDIDATES_PER_MAP = 8
ENERGY = SyntheticEnergy()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_hashes() -> dict[str, str]:
    files = ("synthetic.py", "routes.py", "reference.py", "calibration.py")
    return {name: sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in files}


def manifest() -> dict:
    return {
        "protocol": "synthetic_v0_policy_independent_round_trip_20260926",
        "map_ids": list(MAP_IDS),
        "candidates_per_map": CANDIDATES_PER_MAP,
        "scenario": asdict(SYNTHETIC_V0),
        "energy": asdict(ENERGY),
        "reference_max_steps": 3000,
        "reference_goal_radius_m": 1.5,
        "reference_arrival_speed_mps": 0.1,
        "capacity_rule": "2 * median(successful reference round-trip energy)",
        "full_charge_time_rule": "median(successful reference round-trip seconds)",
        "source_sha256": _source_hashes(),
    }


def draw_free_candidate(case: SyntheticMap, candidate_id: int) -> tuple[tuple[float, float], int]:
    """Rejection sampling is uniform over the eligible clear planar area."""
    if not isinstance(candidate_id, int) or candidate_id < 0:
        raise ValueError("candidate ID must be a nonnegative integer")
    rng = np.random.default_rng(5_000_000_007 + 1_000_003 * case.map_id + 7919 * candidate_id)
    edge = case.config.body_radius_m + case.config.planning_margin_m
    for attempt in range(1, 1001):
        point = rng.uniform(edge, case.config.side_m - edge, size=2)
        if case.is_clear(point):
            return (float(point[0]), float(point[1])), attempt
    raise RuntimeError("failed to sample a clear planar point")


def _leg_payload(leg: ReferenceLeg) -> dict:
    result = asdict(leg)
    result["final_position_xy"] = [float(x) for x in leg.final_position_xy]
    return result


def run_job(case: SyntheticMap, router: PlanarRouter, candidate_id: int) -> dict:
    point, attempts = draw_free_candidate(case, candidate_id)
    result = {
        "job_id": f"m{case.map_id:03d}_c{candidate_id:03d}",
        "map_id": case.map_id,
        "candidate_id": candidate_id,
        "target_xy": list(point),
        "proposal_attempts": attempts,
    }
    try:
        outbound = simulate_reference_leg(case, case.station_xy, point, router=router, energy_model=ENERGY)
        result["outbound"] = _leg_payload(outbound)
        if not outbound.arrived or outbound.collision_count or outbound.infeasible_steps:
            result["outcome"] = "outbound_reference_failure"
            return result
        back = simulate_reference_leg(case, outbound.final_position_xy, case.station_xy, router=router, energy_model=ENERGY)
        result["return"] = _leg_payload(back)
        if not back.arrived or back.collision_count or back.infeasible_steps:
            result["outcome"] = "return_reference_failure"
            return result
    except NoPlanarRoute as error:
        result["outcome"] = "no_planar_route"
        result["detail"] = str(error)
        return result
    result["outcome"] = "reference_round_trip_success"
    result["round_trip_energy"] = outbound.energy + back.energy
    result["round_trip_seconds"] = outbound.seconds + back.seconds
    return result


def _expected_jobs() -> list[tuple[int, int]]:
    return [(map_id, candidate_id) for map_id in MAP_IDS for candidate_id in range(CANDIDATES_PER_MAP)]


def run(output: Path = DEFAULT_OUTPUT, *, max_new_jobs: int | None = None) -> dict:
    if max_new_jobs is not None and max_new_jobs < 0:
        raise ValueError("max_new_jobs must be nonnegative")
    output.mkdir(parents=True, exist_ok=True)
    current = manifest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != current:
            raise RuntimeError("calibration manifest or source hash changed; do not mix results")
    else:
        _write_json_atomic(manifest_path, current)
    results_dir = output / "results"
    new_jobs = 0
    cached_map: int | None = None
    case = None
    router = None
    for map_id, candidate_id in _expected_jobs():
        target = results_dir / f"m{map_id:03d}_c{candidate_id:03d}.json"
        if target.exists():
            row = json.loads(target.read_text(encoding="utf-8"))
            if row.get("map_id") != map_id or row.get("candidate_id") != candidate_id:
                raise RuntimeError(f"invalid saved calibration row: {target}")
            continue
        if max_new_jobs is not None and new_jobs >= max_new_jobs:
            break
        if cached_map != map_id:
            case = make_synthetic_map(map_id)
            router = PlanarRouter(case)
            cached_map = map_id
        _write_json_atomic(target, run_job(case, router, candidate_id))
        new_jobs += 1
    rows = []
    for map_id, candidate_id in _expected_jobs():
        target = results_dir / f"m{map_id:03d}_c{candidate_id:03d}.json"
        if target.exists():
            rows.append(json.loads(target.read_text(encoding="utf-8")))
    status = {"completed_jobs": len(rows), "total_jobs": len(_expected_jobs()), "new_jobs": new_jobs}
    if len(rows) == len(_expected_jobs()):
        successful = [row for row in rows if row["outcome"] == "reference_round_trip_success"]
        if not successful:
            raise RuntimeError("no valid reference round trip; capacity cannot be calibrated")
        capacity = 2.0 * median(row["round_trip_energy"] for row in successful)
        charge_seconds = median(row["round_trip_seconds"] for row in successful)
        summary = {
            **status,
            "successful_reference_round_trips": len(successful),
            "capacity_synthetic_energy": capacity,
            "full_charge_seconds": charge_seconds,
            "full_battery_infeasible_count": sum(
                row["outcome"] != "reference_round_trip_success"
                or row.get("round_trip_energy", float("inf")) > capacity
                for row in rows
            ),
            "proposal_attempts_total": sum(row["proposal_attempts"] for row in rows),
        }
        _write_json_atomic(output / "calibration.json", summary)
        return summary
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-jobs", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.output, max_new_jobs=args.max_new_jobs), sort_keys=True))


if __name__ == "__main__":
    main()
