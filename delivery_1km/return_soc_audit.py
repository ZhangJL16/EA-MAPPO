"""Internal SOC stress audit using legal preflight route-summary features."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from nav3d.planner import NoRouteError

from .navigation import DeliveryRoutePlanner
from .return_audit import ROOT, _atomic_json, _hash
from .return_audit_summary import summarize
from .scenario import M100_1KM, make_map


PROTOCOL = "m100_1km_return_soc_internal_stress_v1"
SOC_GRID = (0.05, 0.10, 0.15, 0.20, 0.40, 0.60, 0.80, 1.00)
PREFLIGHT = ROOT / "artifacts/delivery_1km_preflight_cv_20260925"
PROTOCOL_FILE = ROOT / "RETURN_SOC_STRESS_PROTOCOL_20260926.md"


def source_hash() -> str:
    digest = sha256()
    for path in (Path(__file__), PROTOCOL_FILE):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fit_excluding_map(rows: list[dict], map_id: int) -> dict[float, np.ndarray]:
    training = [row for row in rows if row["map_id"] != map_id]
    if len(training) < 2 or len({row["map_id"] for row in training}) < 2:
        raise ValueError("route estimator needs at least two other maps")
    matrix = np.asarray([row["route_features"] for row in training], dtype=float)
    if matrix.shape[1] != 5:
        raise ValueError("unexpected route-summary feature count")
    return {
        payload: np.linalg.lstsq(
            matrix,
            np.asarray([row["base_energy_wh_by_payload"][str(payload)] for row in training]),
            rcond=None,
        )[0]
        for payload in M100_1KM.cargo_masses_kg
    }


def route_features(route) -> list[float]:
    rises = [float(b[2] - a[2]) for a, b in zip(route.waypoints, route.waypoints[1:])]
    return [1.0, float(route.geometric_length),
            sum(max(0.0, rise) for rise in rises),
            sum(max(0.0, -rise) for rise in rises),
            len(route.waypoints) - 1]


def classify(current_wh: float, predicted_return_wh: float | None, actual_return_wh: float | None,
             physical_evidence_valid: bool) -> tuple[bool, bool, bool]:
    reachable = current_wh > 0
    predicted_safe = reachable and predicted_return_wh is not None and predicted_return_wh <= current_wh
    actually_safe = (reachable and physical_evidence_valid and actual_return_wh is not None
                     and actual_return_wh <= current_wh)
    return reachable, predicted_safe, actually_safe


def run(return_output: Path, output: Path, preflight: Path = PREFLIGHT) -> dict:
    if output.exists():
        raise FileExistsError(output)
    verified = summarize(return_output)
    return_manifest_path = return_output / "manifest.json"
    return_manifest = json.loads(return_manifest_path.read_text(encoding="utf-8"))
    map_id = return_manifest["job"]["map_id"]
    preflight_manifest_path = preflight / "manifest.json"
    preflight_manifest = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    if preflight_manifest["features"]["route"] != [
        "intercept", "preflight_route_length_m", "planned_ascent_m",
        "planned_descent_m", "planned_segment_count",
    ]:
        raise RuntimeError("historical route feature contract changed")
    rows_path = preflight / "rows.json"
    training_rows = json.loads(rows_path.read_text(encoding="utf-8"))
    coefficients = fit_excluding_map(training_rows, map_id)
    case = make_map(map_id)
    planner = DeliveryRoutePlanner(case.world, body_radius=M100_1KM.body_radius_m)
    groups = {(soc, payload, profile["name"]): {
        "reachable_states": 0, "predicted_safe_states": 0,
        "actual_safe_states": 0, "false_safe_states": 0,
        "first_false_safe_step": None, "maximum_underprediction_wh": 0.0,
    } for soc in SOC_GRID for payload in return_manifest["payloads_kg"]
              for profile in return_manifest["profiles"]}
    predictions = []
    for index in range(1, return_manifest["states"] + 1):
        row = json.loads((return_output / "results" / f"step_{index:05d}.json").read_text(encoding="utf-8"))
        try:
            route = planner.plan(row["position_m"], case.station_m)
        except NoRouteError:
            if row["outcome"] != "no_route":
                raise RuntimeError(f"return route changed for step {index}")
            features = None
            predicted = {payload: None for payload in return_manifest["payloads_kg"]}
        else:
            if row["outcome"] == "no_route" or not np.isclose(
                route.geometric_length, row["return_route_length_m"], atol=1e-6,
            ):
                raise RuntimeError(f"return route changed for step {index}")
            features = route_features(route)
            predicted = {payload: max(0.0, float(np.asarray(features) @ coefficients[payload]))
                         for payload in return_manifest["payloads_kg"]}
        predictions.append({"step_index": index, "route_features": features,
                            "predicted_return_wh_by_payload": {str(k): v for k, v in predicted.items()}})
        for sample in row["samples"]:
            payload = sample["payload_kg"]
            profile = sample["profile"]
            physical_valid = (row["outcome"] == "arrived" and row.get("return_collisions", 0) == 0
                              and row.get("return_controller_infeasible_steps", 0) == 0
                              and sample["outbound_capability_violations"] == 0
                              and sample["return_capability_violations"] == 0)
            actual_return_wh = sample["return_wh"]
            for soc in SOC_GRID:
                current_wh = soc * sample["usable_capacity_wh"] - sample["outbound_wh"]
                reachable, predicted_safe, actually_safe = classify(
                    current_wh, predicted[payload], actual_return_wh, physical_valid,
                )
                group = groups[(soc, payload, profile)]
                group["reachable_states"] += int(reachable)
                group["predicted_safe_states"] += int(predicted_safe)
                group["actual_safe_states"] += int(actually_safe)
                if predicted_safe and not actually_safe:
                    group["false_safe_states"] += 1
                    if group["first_false_safe_step"] is None:
                        group["first_false_safe_step"] = index
                if reachable and actual_return_wh is not None and predicted[payload] is not None:
                    group["maximum_underprediction_wh"] = max(
                        group["maximum_underprediction_wh"], actual_return_wh - predicted[payload], 0.0,
                    )

    matrix = []
    for (soc, payload, profile), values in groups.items():
        matrix.append({"starting_soc": soc, "payload_kg": payload, "profile": profile, **values,
                       "false_safe_fraction_of_predicted_safe": (
                           values["false_safe_states"] / values["predicted_safe_states"]
                           if values["predicted_safe_states"] else None)})
    manifest = {
        "protocol": PROTOCOL, "source_sha256": source_hash(),
        "return_manifest_sha256": _hash(return_manifest_path),
        "return_summary_sha256": _hash(return_output / "summary.json"),
        "preflight_manifest_sha256": _hash(preflight_manifest_path),
        "preflight_rows_sha256": _hash(rows_path),
        "held_out_map_id": map_id, "training_map_ids": sorted({row["map_id"] for row in training_rows if row["map_id"] != map_id}),
        "training_routes": sum(row["map_id"] != map_id for row in training_rows),
        "soc_grid": SOC_GRID, "payloads_kg": return_manifest["payloads_kg"],
        "profiles": [profile["name"] for profile in return_manifest["profiles"]],
        "coefficients_by_payload": {str(payload): value.tolist() for payload, value in coefficients.items()},
        "diagnostic_only": "one development flight; old rest-start predictor applied to moving-state returns",
    }
    summary = {"protocol": PROTOCOL, "job_id": verified["job_id"],
               "independent_navigation_flights": 1, "state_count": len(predictions),
               "group_count": len(matrix), "groups": matrix}
    output.mkdir(parents=True)
    _atomic_json(output / "manifest.json", manifest)
    _atomic_json(output / "predictions.json", predictions)
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    args = parser.parse_args()
    summary = run(args.returns, args.output, args.preflight)
    print(json.dumps({"state_count": summary["state_count"], "group_count": summary["group_count"]}))


if __name__ == "__main__":
    main()
