"""Bounded, resumable multi-map return audit using frozen navigation jobs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

from .confirmation_qualify import MAP_IDS
from .return_audit import _atomic_json, _hash
from .return_audit_v2 import build_manifest as build_return_manifest, source_hash as return_source_hash
from .return_audit_summary import summarize


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = ROOT / "artifacts/delivery_1km_nav_confirm_112_127_20260926"
PROTOCOL_FILE = ROOT / "RETURN_MULTIMAP_PROTOCOL_20260926.md"
AMENDMENT_FILE = ROOT / "RETURN_MULTIMAP_EXECUTION_AMENDMENT_20260926.md"
PROTOCOL = "m100_1km_return_multimap_v1"
DISK_BUDGET_BYTES = 4 * 1024**3


def source_hash() -> str:
    digest = sha256()
    for path in (Path(__file__), PROTOCOL_FILE, AMENDMENT_FILE):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest(navigation: Path, map_ids: tuple[int, ...]) -> dict:
    if not map_ids or len(set(map_ids)) != len(map_ids) or any(map_id not in MAP_IDS for map_id in map_ids):
        raise ValueError("map IDs must be a nonempty unique subset of 112..127")
    nav_manifest_path = navigation / "manifest.json"
    nav_manifest = json.loads(nav_manifest_path.read_text(encoding="utf-8"))
    nav_checkpoint = json.loads((navigation / "checkpoint.json").read_text(encoding="utf-8"))
    if nav_checkpoint["completed"] != len(nav_manifest["jobs"]):
        raise RuntimeError("confirmation navigation is incomplete")
    selected = [job for job in nav_manifest["jobs"] if job["map_id"] in map_ids]
    if len(selected) != 2 * len(map_ids):
        raise RuntimeError("selected maps lack paired ground and roof routes")
    jobs = []
    for job in selected:
        result_path = navigation / "results" / f"{job['job_id']}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        jobs.append({"job_id": job["job_id"], "map_id": job["map_id"],
                     "stratum": job["stratum"], "navigation_outcome": result["outcome"],
                     "navigation_result_sha256": _hash(result_path)})
    return {"protocol": PROTOCOL, "source_sha256": source_hash(),
            "return_source_sha256": return_source_hash(),
            "navigation_manifest_sha256": _hash(nav_manifest_path),
            "map_ids": map_ids, "jobs": jobs, "disk_budget_bytes": DISK_BUDGET_BYTES,
            "checkpoint_jobs": 1}


def _size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run(output: Path, *, navigation: Path = NAVIGATION,
        map_ids: tuple[int, ...] = MAP_IDS, workers: int = 16,
        worker_memory_gib: int = 2, resume: bool = False) -> dict:
    if not 1 <= workers <= 16 or worker_memory_gib != 2:
        raise ValueError("this matrix is capped at 16 workers and 2 GiB per worker")
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["source_sha256"] != source_hash()
                or manifest["return_source_sha256"] != return_source_hash()
                or manifest["navigation_manifest_sha256"] != _hash(navigation / "manifest.json")
                or manifest["map_ids"] != list(map_ids)):
            raise RuntimeError("return matrix inputs changed")
    else:
        manifest = build_manifest(navigation, map_ids)
        output.mkdir(parents=True, exist_ok=False)
        (output / "routes").mkdir()
        _atomic_json(output / "manifest.json", manifest)

    for completed, job in enumerate(manifest["jobs"], start=1):
        job_id = job["job_id"]
        if _hash(navigation / "results" / f"{job_id}.json") != job["navigation_result_sha256"]:
            raise RuntimeError(f"navigation result changed: {job_id}")
        route_output = output / "routes" / job_id
        if job["navigation_outcome"] == "arrived":
            if not route_output.exists():
                route_output.mkdir()
                (route_output / "results").mkdir()
                (route_output / "traces").mkdir()
                _atomic_json(route_output / "manifest.json", build_return_manifest(navigation, job_id))
            checkpoint_path = route_output / "checkpoint.json"
            checkpoint = (json.loads(checkpoint_path.read_text(encoding="utf-8"))
                          if checkpoint_path.exists() else None)
            if checkpoint is None or checkpoint["completed"] < checkpoint["total"]:
                command = [sys.executable, str(ROOT / "run_return_audit_parallel.py"),
                           "--output", str(route_output), "--navigation", str(navigation),
                           "--workers", str(workers), "--worker-memory-gib", str(worker_memory_gib)]
                subprocess.run(command, check=True)
            summary = summarize(route_output)
            _atomic_json(route_output / "summary.json", summary)
            status = "return_audit_complete"
        else:
            status = "navigation_failure"
        matrix_checkpoint = {"protocol": PROTOCOL, "source_sha256": manifest["source_sha256"],
                             "completed_jobs": completed, "total_jobs": len(manifest["jobs"]),
                             "last_job_id": job_id, "last_status": status,
                             "output_bytes": _size_bytes(output)}
        _atomic_json(output / "checkpoint.json", matrix_checkpoint)
        print(json.dumps(matrix_checkpoint, ensure_ascii=False), flush=True)
        if matrix_checkpoint["output_bytes"] > manifest["disk_budget_bytes"]:
            raise RuntimeError("return matrix disk budget exceeded after a complete route")
    return matrix_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, default=NAVIGATION)
    parser.add_argument("--map-ids", type=int, nargs="+", default=MAP_IDS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-memory-gib", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, navigation=args.navigation, map_ids=tuple(args.map_ids),
        workers=args.workers, worker_memory_gib=args.worker_memory_gib, resume=args.resume)


if __name__ == "__main__":
    main()
