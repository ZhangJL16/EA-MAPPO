"""Per-map SOC confirmation using the frozen preflight route estimator."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from nav3d.planner import NoRouteError

from .navigation import DeliveryRoutePlanner
from .return_audit import ROOT, _atomic_json, _hash
from .return_matrix_v4 import PROTOCOL as MATRIX_PROTOCOL, source_hash as matrix_source_hash
from .return_soc_audit import (PREFLIGHT, SOC_GRID, fit_excluding_map,
                               route_features)
from .scenario import M100_1KM, make_map


PROTOCOL = "m100_1km_return_multimap_soc_v1"
CALIBRATION_MAPS = (112, 113, 114)
CONFIRMATION_MAPS = tuple(range(115, 128))
SOURCE_FILES = ("delivery_1km/return_multimap_soc.py",
                "delivery_1km/return_soc_audit.py",
                "RETURN_SOC_STRESS_PROTOCOL_20260926.md",
                "RETURN_BACKUP_CBF_CALIBRATION_20260926.md")


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0")
        digest.update((ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def _blank() -> dict:
    return {"reachable_states": 0, "predicted_safe_states": 0,
            "actual_safe_states": 0, "false_safe_states": 0,
            "false_safe_energy_only": 0, "false_safe_physical_only": 0,
            "false_safe_both": 0, "maximum_underprediction_wh": 0.0}


def _analyze_route(args: tuple[Path, Path, dict, list[dict], Path]) -> dict:
    matrix, preflight, job, training_rows, output = args
    route_output = matrix / "routes" / job["job_id"]
    manifest_path = route_output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads((route_output / "checkpoint.json").read_text(encoding="utf-8"))
    return_summary_path = route_output / "summary.json"
    return_summary = json.loads(return_summary_path.read_text(encoding="utf-8"))
    if (checkpoint["completed"] != manifest["states"]
            or return_summary["manifest_sha256"] != _hash(manifest_path)
            or return_summary["flight_state_count"] != manifest["states"]
            or manifest["job"]["job_id"] != job["job_id"]):
        raise RuntimeError(f"incomplete or inconsistent return route: {job['job_id']}")
    map_id = job["map_id"]
    coefficients = fit_excluding_map(training_rows, map_id)
    case = make_map(map_id)
    planner = DeliveryRoutePlanner(case.world, body_radius=M100_1KM.body_radius_m)
    groups = {(soc, payload, profile["name"]): _blank()
              for soc in SOC_GRID for payload in manifest["payloads_kg"]
              for profile in manifest["profiles"]}
    predictions = []
    for index in range(1, manifest["states"] + 1):
        row = json.loads((route_output / "results" / f"step_{index:05d}.json").read_text(encoding="utf-8"))
        try:
            route = planner.plan(row["position_m"], case.station_m)
        except NoRouteError:
            if row["outcome"] != "no_route":
                raise RuntimeError(f"return planner changed at {job['job_id']} step {index}")
            features = None
            predicted = {payload: None for payload in manifest["payloads_kg"]}
        else:
            if row["outcome"] == "no_route" or not np.isclose(
                route.geometric_length, row["return_route_length_m"], atol=1e-6,
            ):
                raise RuntimeError(f"return route changed at {job['job_id']} step {index}")
            features = route_features(route)
            predicted = {payload: max(0.0, float(np.asarray(features) @ coefficients[payload]))
                         for payload in manifest["payloads_kg"]}
        predictions.append({"step_index": index, "route_features": features,
                            "predicted_return_wh_by_payload": {str(k): v for k, v in predicted.items()}})
        for sample in row["samples"]:
            payload = sample["payload_kg"]
            physical_valid = (row["outcome"] == "arrived" and row.get("return_collisions", 0) == 0
                              and row.get("return_controller_infeasible_steps", 0) == 0
                              and sample["outbound_capability_violations"] == 0
                              and sample["return_capability_violations"] == 0)
            for soc in SOC_GRID:
                current_wh = soc * sample["usable_capacity_wh"] - sample["outbound_wh"]
                reachable = current_wh > 0
                predicted_safe = reachable and predicted[payload] is not None and predicted[payload] <= current_wh
                energy_safe = (reachable and sample["return_wh"] is not None
                               and sample["return_wh"] <= current_wh)
                actual_safe = physical_valid and energy_safe
                group = groups[(soc, payload, sample["profile"])]
                group["reachable_states"] += int(reachable)
                group["predicted_safe_states"] += int(predicted_safe)
                group["actual_safe_states"] += int(actual_safe)
                if predicted_safe and not actual_safe:
                    group["false_safe_states"] += 1
                    cause = ("false_safe_both" if not physical_valid and not energy_safe
                             else "false_safe_physical_only" if not physical_valid
                             else "false_safe_energy_only")
                    group[cause] += 1
                if reachable and sample["return_wh"] is not None and predicted[payload] is not None:
                    group["maximum_underprediction_wh"] = max(
                        group["maximum_underprediction_wh"],
                        sample["return_wh"] - predicted[payload], 0.0,
                    )
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "predictions.json", predictions)
    summary = {"job_id": job["job_id"], "map_id": map_id,
               "cohort": "calibration" if map_id in CALIBRATION_MAPS else "confirmation",
               "state_count": manifest["states"],
               "return_manifest_sha256": _hash(manifest_path),
               "return_summary_sha256": _hash(return_summary_path),
               "groups": [{"starting_soc": soc, "payload_kg": payload, "profile": profile, **counts}
                          for (soc, payload, profile), counts in groups.items()]}
    _atomic_json(output / "summary.json", summary)
    return summary


def run(matrix: Path, output: Path, *, preflight: Path = PREFLIGHT, workers: int = 16) -> dict:
    if not 1 <= workers <= 16 or output.exists():
        raise ValueError("workers must be 1..16 and output must be new")
    matrix_manifest_path = matrix / "manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    matrix_checkpoint = json.loads((matrix / "checkpoint.json").read_text(encoding="utf-8"))
    if (matrix_manifest["protocol"] != MATRIX_PROTOCOL
            or matrix_manifest["source_sha256"] != matrix_source_hash()
            or matrix_checkpoint["completed_jobs"] != len(matrix_manifest["jobs"])
            or set(matrix_manifest["map_ids"]) != set(CALIBRATION_MAPS + CONFIRMATION_MAPS)):
        raise RuntimeError("the complete frozen 16-map v4 matrix is required")
    preflight_manifest_path = preflight / "manifest.json"
    preflight_manifest = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    if preflight_manifest["features"]["route"] != [
        "intercept", "preflight_route_length_m", "planned_ascent_m",
        "planned_descent_m", "planned_segment_count",
    ]:
        raise RuntimeError("preflight route features changed")
    rows_path = preflight / "rows.json"
    training_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    (output / "routes").mkdir()
    analysis_manifest = {
        "protocol": PROTOCOL, "source_sha256": source_hash(),
        "matrix_manifest_sha256": _hash(matrix_manifest_path),
        "preflight_manifest_sha256": _hash(preflight_manifest_path),
        "preflight_rows_sha256": _hash(rows_path),
        "training_map_ids": sorted({row["map_id"] for row in training_rows}),
        "calibration_map_ids": CALIBRATION_MAPS,
        "confirmation_map_ids": CONFIRMATION_MAPS,
        "soc_grid": SOC_GRID,
        "interpretation": "retrospective SOC labels with preflight route estimator; map is the independent unit",
    }
    _atomic_json(output / "manifest.json", analysis_manifest)
    args = [(matrix, preflight, job, training_rows, output / "routes" / job["job_id"])
            for job in matrix_manifest["jobs"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        route_summaries = list(pool.map(_analyze_route, args))
    map_groups = defaultdict(lambda: defaultdict(_blank))
    for route in route_summaries:
        for group in route["groups"]:
            key = (group["starting_soc"], group["payload_kg"], group["profile"])
            target = map_groups[route["map_id"]][key]
            for field, value in group.items():
                if field in target:
                    target[field] = max(target[field], value) if field == "maximum_underprediction_wh" else target[field] + value
    maps = [{"map_id": map_id,
             "cohort": "calibration" if map_id in CALIBRATION_MAPS else "confirmation",
             "groups": [{"starting_soc": soc, "payload_kg": payload, "profile": profile, **counts}
                        for (soc, payload, profile), counts in groups.items()]}
            for map_id, groups in sorted(map_groups.items())]
    summary = {"protocol": PROTOCOL, "map_count": len(maps),
               "route_count": len(route_summaries),
               "state_count": sum(route["state_count"] for route in route_summaries),
               "calibration_maps": len(CALIBRATION_MAPS),
               "independent_confirmation_maps": len(CONFIRMATION_MAPS),
               "maps": maps}
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    result = run(args.matrix, args.output, preflight=args.preflight, workers=args.workers)
    print(json.dumps({key: result[key] for key in ("map_count", "route_count", "state_count")}))


if __name__ == "__main__":
    main()
