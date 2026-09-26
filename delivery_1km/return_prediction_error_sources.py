"""Secondary, read-only explanation audit for the completed v4 return matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .return_audit import ROOT, _atomic_json, _hash
from .return_matrix_v4 import PROTOCOL as MATRIX_PROTOCOL, source_hash as matrix_source_hash
from .return_multimap_soc import (CALIBRATION_MAPS, CONFIRMATION_MAPS,
                                   PROTOCOL as SOC_PROTOCOL, source_hash as soc_source_hash)
from .return_soc_audit import SOC_GRID


PROTOCOL = "m100_1km_return_prediction_error_sources_v1"
PROTOCOL_FILE = ROOT / "RETURN_PREDICTION_ERROR_DECOMPOSITION_20260926.md"


def source_hash() -> str:
    digest = sha256()
    for path in (Path(__file__), PROTOCOL_FILE):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def false_safe_category(*, predicted_wh: float | None, base_wh: float | None,
                        actual_wh: float | None, remaining_wh: float,
                        physical_valid: bool) -> str | None:
    """Classify a predicted-safe label; `base` is tested at the same remaining Wh."""
    if (remaining_wh <= 0 or predicted_wh is None or predicted_wh > remaining_wh):
        return None
    energy_failed = actual_wh is None or actual_wh > remaining_wh
    if not energy_failed and physical_valid:
        return None
    if not energy_failed:
        return "physical_only"
    if not physical_valid:
        return "physical_and_energy"
    if base_wh is not None and base_wh > remaining_wh:
        return "energy_base_error_sufficient"
    if base_wh is not None and actual_wh is not None and base_wh <= remaining_wh:
        return "energy_profile_shift_needed"
    return "energy_unresolved"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=float)
    return {"count": len(values), "mean": float(array.mean()),
            "p05": float(np.quantile(array, .05)),
            "p50": float(np.quantile(array, .50)),
            "p95": float(np.quantile(array, .95)),
            "min": float(array.min()), "max": float(array.max())}


def _speed_band(speed: float) -> str:
    return "lt2" if speed < 2 else "2to5" if speed < 5 else "ge5"


def _physical_valid(row: dict, sample: dict) -> bool:
    return (row["outcome"] == "arrived" and row.get("return_collisions", 0) == 0
            and row.get("return_controller_infeasible_steps", 0) == 0
            and sample["outbound_capability_violations"] == 0
            and sample["return_capability_violations"] == 0)


def run(matrix: Path, soc: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    matrix_manifest_path = matrix / "manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    matrix_checkpoint = json.loads((matrix / "checkpoint.json").read_text(encoding="utf-8"))
    soc_manifest_path = soc / "manifest.json"
    soc_manifest = json.loads(soc_manifest_path.read_text(encoding="utf-8"))
    soc_summary_path = soc / "summary.json"
    soc_summary = json.loads(soc_summary_path.read_text(encoding="utf-8"))
    expected_maps = set(CALIBRATION_MAPS + CONFIRMATION_MAPS)
    if (matrix_manifest["protocol"] != MATRIX_PROTOCOL
            or matrix_manifest["source_sha256"] != matrix_source_hash()
            or matrix_checkpoint["completed_jobs"] != len(matrix_manifest["jobs"])
            or set(matrix_manifest["map_ids"]) != expected_maps
            or soc_manifest["protocol"] != SOC_PROTOCOL
            or soc_manifest["source_sha256"] != soc_source_hash()
            or soc_manifest["matrix_manifest_sha256"] != _hash(matrix_manifest_path)
            or soc_summary["route_count"] != len(matrix_manifest["jobs"])
            or soc_summary["map_count"] != len(expected_maps)):
        raise RuntimeError("complete, matching matrix and SOC inputs are required")

    by_map = defaultdict(lambda: {"base_residual": defaultdict(list),
                                  "base_residual_by_speed": defaultdict(list),
                                  "profile_shift": [], "false_safe": defaultdict(lambda: defaultdict(int)),
                                  "predicted_safe": defaultdict(int),
                                  "false_safe_by_profile": defaultdict(lambda: defaultdict(int)),
                                  "predicted_safe_by_profile": defaultdict(int), "states": 0})
    for job in matrix_manifest["jobs"]:
        map_id, job_id = job["map_id"], job["job_id"]
        if map_id not in CONFIRMATION_MAPS:
            continue
        route_output = matrix / "routes" / job_id
        route_manifest = json.loads((route_output / "manifest.json").read_text(encoding="utf-8"))
        route_checkpoint = json.loads((route_output / "checkpoint.json").read_text(encoding="utf-8"))
        route_summary_path = route_output / "summary.json"
        soc_route_output = soc / "routes" / job_id
        soc_route_summary = json.loads((soc_route_output / "summary.json").read_text(encoding="utf-8"))
        predictions = json.loads((soc_route_output / "predictions.json").read_text(encoding="utf-8"))
        if (route_checkpoint["completed"] != route_manifest["states"]
                or soc_route_summary["return_manifest_sha256"] != _hash(route_output / "manifest.json")
                or soc_route_summary["return_summary_sha256"] != _hash(route_summary_path)
                or len(predictions) != route_manifest["states"]):
            raise RuntimeError(f"route inputs changed or are incomplete: {job_id}")
        target = by_map[map_id]
        for index, prediction in enumerate(predictions, start=1):
            if prediction["step_index"] != index:
                raise RuntimeError(f"prediction index changed: {job_id} step {index}")
            row = json.loads((route_output / "results" / f"step_{index:05d}.json").read_text(encoding="utf-8"))
            if row["step_index"] != index:
                raise RuntimeError(f"return index changed: {job_id} step {index}")
            target["states"] += 1
            speed = float(np.linalg.norm(row["velocity_mps"]))
            base = {sample["payload_kg"]: sample for sample in row["samples"]
                    if sample["profile"] == "base"}
            if len(base) != len(route_manifest["payloads_kg"]):
                raise RuntimeError(f"missing base energy label: {job_id} step {index}")
            for payload, sample in base.items():
                predicted = prediction["predicted_return_wh_by_payload"][str(payload)]
                if predicted is not None and sample["return_wh"] is not None and _physical_valid(row, sample):
                    residual = sample["return_wh"] - predicted
                    target["base_residual"][str(payload)].append(residual)
                    target["base_residual_by_speed"][(str(payload), _speed_band(speed))].append(residual)
            for sample in row["samples"]:
                payload = sample["payload_kg"]
                predicted = prediction["predicted_return_wh_by_payload"][str(payload)]
                base_wh = base[payload]["return_wh"]
                if (sample["profile"] != "base" and sample["return_wh"] is not None
                        and base_wh is not None and _physical_valid(row, sample)
                        and _physical_valid(row, base[payload])):
                    target["profile_shift"].append(sample["return_wh"] - base_wh)
                for starting_soc in SOC_GRID:
                    key = (str(starting_soc), "base" if sample["profile"] == "base" else "perturbed")
                    remaining_wh = starting_soc * sample["usable_capacity_wh"] - sample["outbound_wh"]
                    if remaining_wh > 0 and predicted is not None and predicted <= remaining_wh:
                        target["predicted_safe"][key] += 1
                        target["predicted_safe_by_profile"][(str(starting_soc), sample["profile"])] += 1
                    category = false_safe_category(
                        predicted_wh=predicted, base_wh=base_wh,
                        actual_wh=sample["return_wh"], remaining_wh=remaining_wh,
                        physical_valid=_physical_valid(row, sample))
                    if category is not None:
                        target["false_safe"][key][category] += 1
                        target["false_safe_by_profile"][(str(starting_soc), sample["profile"])][category] += 1

    maps = []
    soc_maps = {item["map_id"]: item for item in soc_summary["maps"]}
    for map_id in sorted(by_map):
        data = by_map[map_id]
        frozen = defaultdict(lambda: {"predicted_safe": 0, "false_safe": 0})
        for group in soc_maps[map_id]["groups"]:
            key = (str(group["starting_soc"]),
                   "base" if group["profile"] == "base" else "perturbed")
            frozen[key]["predicted_safe"] += group["predicted_safe_states"]
            frozen[key]["false_safe"] += group["false_safe_states"]
        for key, expected in frozen.items():
            if (data["predicted_safe"][key] != expected["predicted_safe"]
                    or sum(data["false_safe"][key].values()) != expected["false_safe"]):
                raise RuntimeError(f"false-safe accounting differs from frozen SOC analysis: map {map_id}, {key}")
        frozen_profile = defaultdict(lambda: {"predicted_safe": 0, "false_safe": 0})
        for group in soc_maps[map_id]["groups"]:
            key = (str(group["starting_soc"]), group["profile"])
            frozen_profile[key]["predicted_safe"] += group["predicted_safe_states"]
            frozen_profile[key]["false_safe"] += group["false_safe_states"]
        for key, expected in frozen_profile.items():
            if (data["predicted_safe_by_profile"][key] != expected["predicted_safe"]
                    or sum(data["false_safe_by_profile"][key].values()) != expected["false_safe"]):
                raise RuntimeError(f"profile accounting differs from frozen SOC analysis: map {map_id}, {key}")
        maps.append({"map_id": map_id, "route_count": 2, "state_count": data["states"],
                     "base_residual_wh_by_payload": {
                         mass: _stats(values) for mass, values in data["base_residual"].items()},
                     "base_residual_wh_by_payload_and_speed": [
                         {"payload_kg": mass, "speed_band_mps": band, **_stats(values)}
                         for (mass, band), values in sorted(data["base_residual_by_speed"].items())],
                     "profile_minus_base_return_wh": _stats(data["profile_shift"]),
                     "false_safe_by_soc_and_profile_class": [
                         {"starting_soc": float(soc_value), "profile_class": kind,
                          "predicted_safe_labels": data["predicted_safe"][(soc_value, kind)],
                          **dict(data["false_safe"][(soc_value, kind)])}
                         for soc_value in map(str, SOC_GRID) for kind in ("base", "perturbed")],
                     "false_safe_by_soc_and_profile": [
                         {"starting_soc": float(soc_value), "profile": profile,
                          "predicted_safe_labels": data["predicted_safe_by_profile"][(soc_value, profile)],
                          **dict(data["false_safe_by_profile"][(soc_value, profile)])}
                         for soc_value, profile in sorted(frozen_profile)],
                     })
    if len(maps) != len(CONFIRMATION_MAPS):
        raise RuntimeError("missing independent confirmation maps")
    result = {"protocol": PROTOCOL, "source_sha256": source_hash(),
              "matrix_manifest_sha256": _hash(matrix_manifest_path),
              "soc_manifest_sha256": _hash(soc_manifest_path),
              "soc_summary_sha256": _hash(soc_summary_path),
              "independent_confirmation_maps": len(maps), "maps": maps,
              "interpretation": "descriptive accounting; correlated states and profile labels are not independent flights"}
    output.mkdir(parents=True)
    _atomic_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--soc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.matrix, args.soc, args.output)
    print(json.dumps({"independent_confirmation_maps": result["independent_confirmation_maps"]}))


if __name__ == "__main__":
    main()
