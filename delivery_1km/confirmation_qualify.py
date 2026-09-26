"""Frozen unseen-map cohort built from the existing 1 km navigator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .qualify import _atomic_json, build_manifest as base_manifest, run as run_navigation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m100_1km_return_multimap_navigation_20260926"
MAP_IDS = tuple(range(112, 128))
STRATA = ("station_to_ground", "station_to_roof")
PROTOCOL_FILE = ROOT / "RETURN_MULTIMAP_PROTOCOL_20260926.md"


def cohort_hash() -> str:
    digest = sha256()
    for path in (Path(__file__), PROTOCOL_FILE):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest() -> dict:
    manifest = base_manifest(map_count=MAP_IDS[-1] - 100 + 1, per_stratum=1)
    jobs = [job for job in manifest["jobs"] if job["map_id"] in MAP_IDS and job["stratum"] in STRATA]
    if len(jobs) != len(MAP_IDS) * len(STRATA):
        raise RuntimeError("confirmation cohort has missing navigation jobs")
    manifest.update({"jobs": jobs, "map_count": len(MAP_IDS), "per_stratum": 1,
                     "map_ids": MAP_IDS, "strata": STRATA,
                     "cohort_protocol": PROTOCOL, "cohort_source_sha256": cohort_hash(),
                     "checkpoint_jobs": 2})
    return manifest


def run(output: Path, *, resume: bool = False, stop_after_checkpoint: bool = False) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["cohort_source_sha256"] != cohort_hash()
                or manifest["map_ids"] != list(MAP_IDS)
                or manifest["strata"] != list(STRATA)):
            raise RuntimeError("confirmation cohort changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        (output / "traces").mkdir()
        _atomic_json(output / "manifest.json", build_manifest())
    return run_navigation(output, resume=True, stop_after_checkpoint=stop_after_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    run(args.output, resume=args.resume, stop_after_checkpoint=args.stop_after_checkpoint)


if __name__ == "__main__":
    main()
