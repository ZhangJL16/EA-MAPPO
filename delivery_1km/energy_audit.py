"""Resumable model-sensitivity accounting on the frozen 96 navigation traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .energy import account_flight_trace
from .energy_profiles import ENERGY_PROFILES
from .scenario import M100_1KM


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = ROOT / "artifacts/delivery_1km_nav_v1_20260923"
PROTOCOL = "m100_1km_energy_model_oat_20260925"


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_hash() -> str:
    digest = sha256()
    for relative in (
        "delivery_1km/scenario.py", "delivery_1km/energy.py",
        "delivery_1km/energy_profiles.py", "delivery_1km/energy_audit.py",
    ):
        path = ROOT / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_manifest(navigation: Path = NAVIGATION) -> dict:
    nav_manifest_path = navigation / "manifest.json"
    nav_manifest = json.loads(nav_manifest_path.read_text(encoding="utf-8"))
    nav_checkpoint = json.loads((navigation / "checkpoint.json").read_text(encoding="utf-8"))
    if nav_checkpoint["completed"] != nav_checkpoint["total"] or nav_checkpoint["total"] != len(nav_manifest["jobs"]):
        raise RuntimeError("navigation qualification is incomplete")
    jobs = []
    for job in nav_manifest["jobs"]:
        result_path = navigation / "results" / f"{job['job_id']}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["job_id"] != job["job_id"] or result["outcome"] != "arrived":
            raise RuntimeError("navigation job is missing or did not arrive")
        trace_path = navigation / result["trace"]
        jobs.append({
            "job_id": job["job_id"], "map_id": job["map_id"], "stratum": job["stratum"],
            "trace": result["trace"], "trace_sha256": _sha256(trace_path),
            "result_sha256": _sha256(result_path),
            "flight_seconds": result["seconds"],
            "collision_count": result["collision_count"],
        })
    return {
        "protocol": PROTOCOL,
        "source_sha256": source_hash(),
        "navigation_manifest_sha256": _sha256(nav_manifest_path),
        "navigation_source_sha256": nav_manifest["source_sha256"],
        "checkpoint_jobs": 2,
        "payloads_kg": list(M100_1KM.cargo_masses_kg),
        "profiles": [asdict(profile) for profile in ENERGY_PROFILES],
        "jobs": jobs,
    }


def execute_job(navigation: Path, manifest: dict, job: dict) -> dict:
    trace_path = navigation / job["trace"]
    result_path = navigation / "results" / f"{job['job_id']}.json"
    if _sha256(trace_path) != job["trace_sha256"] or _sha256(result_path) != job["result_sha256"]:
        raise RuntimeError("frozen navigation trace or result changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        velocities = trace["velocities"]
        accelerations = trace["accelerations"]
    if len(accelerations) * M100_1KM.physics_dt_s != job["flight_seconds"]:
        if not np.isclose(len(accelerations) * M100_1KM.physics_dt_s, job["flight_seconds"], atol=1e-8):
            raise RuntimeError("flight duration differs from its trace")
    samples = []
    for payload in manifest["payloads_kg"]:
        for profile in ENERGY_PROFILES:
            measurement = account_flight_trace(velocities, accelerations, payload, profile)
            samples.append({
                "payload_kg": payload,
                "profile": profile.name,
                "energy_wh": measurement.energy_wh,
                "max_power_w": measurement.max_power_w,
                "max_thrust_n": measurement.max_thrust_n,
                "max_tilt_deg": measurement.max_tilt_deg,
                "infeasible_steps": measurement.infeasible_steps,
                "first_infeasible_step": measurement.first_infeasible_step,
                "violated": measurement.violated,
            })
    return {
        "job_id": job["job_id"], "map_id": job["map_id"], "stratum": job["stratum"],
        "flight_seconds": job["flight_seconds"], "collision_count": job["collision_count"],
        "samples": samples,
    }


def run(output: Path, *, navigation: Path = NAVIGATION, resume: bool = False,
        stop_after_checkpoint: bool = False) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_hash() or manifest["navigation_manifest_sha256"] != _sha256(navigation / "manifest.json"):
            raise RuntimeError("source or navigation qualification changed")
        if manifest["profiles"] != [asdict(profile) for profile in ENERGY_PROFILES]:
            raise RuntimeError("energy sensitivity profiles changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        manifest = build_manifest(navigation)
        _atomic_json(output / "manifest.json", manifest)
    completed = 0
    for job in manifest["jobs"]:
        result_path = output / "results" / f"{job['job_id']}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["job_id"] != job["job_id"] or len(saved["samples"]) != len(manifest["profiles"]) * len(manifest["payloads_kg"]):
                raise RuntimeError("saved energy result does not match its job")
            completed += 1
            continue
        _atomic_json(result_path, execute_job(navigation, manifest, job))
        completed += 1
        if completed % manifest["checkpoint_jobs"] == 0 or completed == len(manifest["jobs"]):
            checkpoint = {
                "protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"],
                "completed": completed, "total": len(manifest["jobs"]),
                "last_job_id": job["job_id"],
            }
            _atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    return json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--navigation", type=Path, default=NAVIGATION)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, navigation=arguments.navigation, resume=arguments.resume,
        stop_after_checkpoint=arguments.stop_after_checkpoint)


if __name__ == "__main__":
    main()
