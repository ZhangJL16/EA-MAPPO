"""Map-grouped audit of simple preflight time and energy estimates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .energy_audit import ROOT
from .navigation import DeliveryRoutePlanner
from .scenario import M100_1KM, make_map


NAVIGATION = ROOT / "artifacts/delivery_1km_nav_v1_20260923"
ENERGY = ROOT / "artifacts/delivery_1km_energy_oat_20260925"
PROTOCOL = "m100_1km_preflight_leave_one_map_out_v1"
FEATURES = ("direct", "route")


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_rows(navigation: Path = NAVIGATION, energy: Path = ENERGY) -> tuple[dict, list[dict]]:
    nav_manifest_path = navigation / "manifest.json"
    nav_manifest = json.loads(nav_manifest_path.read_text(encoding="utf-8"))
    energy_manifest_path = energy / "manifest.json"
    energy_manifest = json.loads(energy_manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads((energy / "checkpoint.json").read_text(encoding="utf-8"))
    if energy_manifest["navigation_manifest_sha256"] != _hash(nav_manifest_path):
        raise RuntimeError("energy and navigation batches do not match")
    if checkpoint["completed"] != len(nav_manifest["jobs"]) or checkpoint["total"] != len(nav_manifest["jobs"]):
        raise RuntimeError("energy audit is incomplete")
    energy_jobs = {job["job_id"]: job for job in energy_manifest["jobs"]}
    if set(energy_jobs) != {job["job_id"] for job in nav_manifest["jobs"]}:
        raise RuntimeError("energy and navigation job sets differ")
    rows = []
    for job in nav_manifest["jobs"]:
        job_id = job["job_id"]
        original = navigation / "results" / f"{job_id}.json"
        energy_result = energy / "results" / f"{job_id}.json"
        if _hash(original) != energy_jobs[job_id]["result_sha256"]:
            raise RuntimeError(f"navigation result changed for {job_id}")
        result = json.loads(original.read_text(encoding="utf-8"))
        measurements = json.loads(energy_result.read_text(encoding="utf-8"))
        if result["outcome"] != "arrived" or measurements["job_id"] != job_id:
            raise RuntimeError(f"incomplete flight or energy label for {job_id}")
        case = make_map(job["map_id"])
        if not np.allclose(case.station_m, job["station_m"]):
            raise RuntimeError(f"map station differs for {job_id}")
        planner = DeliveryRoutePlanner(case.world, body_radius=M100_1KM.body_radius_m)
        route = planner.plan(job["start_m"], job["goal_m"])
        if not np.isclose(route.geometric_length, result["route_length_m"], atol=1e-6):
            raise RuntimeError(f"preflight route differs from executed route for {job_id}")
        rises = [float(b[2] - a[2]) for a, b in zip(route.waypoints, route.waypoints[1:])]
        direct_distance = float(np.linalg.norm(np.asarray(job["goal_m"]) - np.asarray(job["start_m"])))
        base_energy = {
            str(sample["payload_kg"]): sample["energy_wh"]
            for sample in measurements["samples"] if sample["profile"] == "base"
        }
        if set(base_energy) != {str(mass) for mass in M100_1KM.cargo_masses_kg}:
            raise RuntimeError(f"missing baseline energy labels for {job_id}")
        rows.append({
            "job_id": job_id,
            "map_id": job["map_id"],
            "stratum": job["stratum"],
            "direct_features": [1.0, direct_distance, abs(float(job["goal_m"][2] - job["start_m"][2]))],
            "route_features": [
                1.0, float(route.geometric_length),
                sum(max(0.0, rise) for rise in rises),
                sum(max(0.0, -rise) for rise in rises),
                len(route.waypoints) - 1,
            ],
            "flight_seconds": result["seconds"],
            "base_energy_wh_by_payload": base_energy,
            "navigation_result_sha256": _hash(original),
            "energy_result_sha256": _hash(energy_result),
        })
    manifest = {
        "protocol": PROTOCOL,
        "source_sha256": _hash(Path(__file__)),
        "navigation_manifest_sha256": _hash(nav_manifest_path),
        "energy_manifest_sha256": _hash(energy_manifest_path),
        "map_ids": sorted({row["map_id"] for row in rows}),
        "jobs": len(rows),
        "features": {
            "direct": ["intercept", "euclidean_distance_m", "absolute_endpoint_height_change_m"],
            "route": ["intercept", "preflight_route_length_m", "planned_ascent_m", "planned_descent_m", "planned_segment_count"],
        },
        "fit": "ordinary_least_squares; predictions_clipped_at_zero; leave_one_map_out",
    }
    return manifest, rows


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = actual - predicted
    scale = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "count": int(len(actual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / scale) if scale > 0 else None,
        "underprediction_rate": float(np.mean(residual > 0)),
        "positive_residual_p90": float(np.quantile(np.maximum(residual, 0), .90)),
        "positive_residual_p95": float(np.quantile(np.maximum(residual, 0), .95)),
        "positive_residual_p99": float(np.quantile(np.maximum(residual, 0), .99)),
        "largest_underprediction": float(max(0, np.max(residual))),
    }


def cross_validate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no flight rows")
    map_ids = np.array([row["map_id"] for row in rows])
    groups = sorted(set(map_ids.tolist()))
    if len(groups) < 2:
        raise ValueError("cross-validation needs at least two maps")
    target_names = ["flight_seconds"] + [f"energy_wh_payload_{mass}" for mass in M100_1KM.cargo_masses_kg]
    targets = {"flight_seconds": np.array([row["flight_seconds"] for row in rows])}
    for mass in M100_1KM.cargo_masses_kg:
        targets[f"energy_wh_payload_{mass}"] = np.array([
            row["base_energy_wh_by_payload"][str(mass)] for row in rows
        ])
    predictions: dict[str, dict[str, list[float]]] = {}
    summaries: dict[str, dict[str, dict]] = {}
    for feature_name in FEATURES:
        matrix = np.array([row[f"{feature_name}_features"] for row in rows], dtype=float)
        predictions[feature_name] = {}
        summaries[feature_name] = {}
        for target_name in target_names:
            target = targets[target_name]
            predicted = np.zeros(len(rows), dtype=float)
            for heldout in groups:
                train = map_ids != heldout
                test = ~train
                coefficients = np.linalg.lstsq(matrix[train], target[train], rcond=None)[0]
                predicted[test] = np.maximum(0, matrix[test] @ coefficients)
            predictions[feature_name][target_name] = predicted.tolist()
            summaries[feature_name][target_name] = _metrics(target, predicted)
    return {"groups": groups, "summaries": summaries, "predictions": predictions}


def run(output: Path, navigation: Path = NAVIGATION, energy: Path = ENERGY) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest, rows = build_rows(navigation, energy)
    cv = cross_validate(rows)
    output.mkdir(parents=True)
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "rows.json", rows)
    _write_json(output / "summary.json", {"protocol": PROTOCOL, "map_ids": cv["groups"], "metrics": cv["summaries"]})
    _write_json(output / "predictions.json", [
        {
            "job_id": row["job_id"], "map_id": row["map_id"],
            "actual": {"flight_seconds": row["flight_seconds"], **{
                f"energy_wh_payload_{mass}": row["base_energy_wh_by_payload"][str(mass)]
                for mass in M100_1KM.cargo_masses_kg
            }},
            "predicted": {
                feature_name: {target_name: values[index]
                               for target_name, values in cv["predictions"][feature_name].items()}
                for feature_name in FEATURES
            },
        }
        for index, row in enumerate(rows)
    ])
    return cv["summaries"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/delivery_1km_preflight_cv_20260925")
    args = parser.parse_args()
    metrics = run(args.output)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
