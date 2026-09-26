"""Eight-hour, balanced multi-horizon GPU-MPC research campaign.

The full CPU-exact static matrix runs independently.  This companion campaign
tests GPU-ranked 2 s and 4 s planning horizons on paired unseen maps while
retaining exact CPU shielding.  It stops at its wall-clock deadline and can be
resumed only under the original manifest and deadline.
"""

from __future__ import annotations

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from time import time
import traceback

import torch

from dual_constraint_2d.matrix_runner import _hash_sources


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "gpu_mpc_8h_campaign_20260926"
SECONDS = 8 * 60 * 60


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _jobs() -> list[dict]:
    jobs = []
    for stage, split, map_ids in (
        ("core_validation", "validation", range(32, 36)),
        ("core_test", "test", range(40, 48)),
        ("expansion_validation", "validation", range(36, 40)),
        ("expansion_test", "test", range(48, 56)),
    ):
        for map_id in map_ids:
            for horizon in (4, 8):
                jobs.append({"stage": stage, "split": split,
                             "map_id": map_id, "horizon_blocks": horizon})
    return jobs


def _source_hashes() -> dict[str, str]:
    result = _hash_sources()
    for path in sorted((ROOT / "gpu2d").glob("*.py")):
        result[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return result


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for the GPU campaign")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        expected = {key: manifest[key] for key in manifest if key not in ("start_epoch", "deadline_epoch")}
        current = {
            "protocol": "gpu_ranked_mpc_eight_hour_campaign_v0",
            "duration_seconds": SECONDS,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "jobs": _jobs(),
            "source_sha256": _source_hashes(),
        }
        if expected != current:
            raise RuntimeError("campaign source, GPU or job manifest changed")
    else:
        started = time()
        manifest = {
            "protocol": "gpu_ranked_mpc_eight_hour_campaign_v0",
            "duration_seconds": SECONDS,
            "start_epoch": started,
            "deadline_epoch": started + SECONDS,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "jobs": _jobs(),
            "source_sha256": _source_hashes(),
        }
        _atomic_json(path, manifest)
    try:
        for index, job in enumerate(manifest["jobs"]):
            directory = (output / job["split"] /
                         f"gpu_mpc_h{job['horizon_blocks']}" /
                         f"map_{job['map_id']:03d}")
            if (directory / "summary.json").exists():
                continue
            if time() >= manifest["deadline_epoch"]:
                break
            directory.mkdir(parents=True, exist_ok=True)
            _atomic_json(output / "status.json", {
                "phase": "running", "job_index": index,
                "job": job, "deadline_epoch": manifest["deadline_epoch"],
                "completed_jobs": sum(
                    (output / x["split"] / f"gpu_mpc_h{x['horizon_blocks']}" /
                     f"map_{x['map_id']:03d}" / "summary.json").exists()
                    for x in manifest["jobs"]
                ),
                "planned_jobs": len(manifest["jobs"]),
            })
            command = [
                sys.executable, "-m", "gpu2d.evaluate",
                "--output", str(directory), "--map-id", str(job["map_id"]),
                "--horizon-blocks", str(job["horizon_blocks"]),
                "--deadline-epoch", str(manifest["deadline_epoch"]),
            ]
            environment = os.environ.copy()
            environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            with (directory / "run.log").open("a", encoding="utf-8") as handle:
                handle.write("COMMAND " + json.dumps(command) + "\n")
                handle.flush()
                subprocess.run(command, cwd=ROOT, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT, check=True)
            if not (directory / "summary.json").exists():
                # The child hit the shared deadline and saved a resumable state.
                break
        completed = sum(
            (output / x["split"] / f"gpu_mpc_h{x['horizon_blocks']}" /
             f"map_{x['map_id']:03d}" / "summary.json").exists()
            for x in manifest["jobs"]
        )
        result = {
            "phase": "complete" if completed == len(manifest["jobs"]) else "time_budget_exhausted",
            "completed_jobs": completed, "planned_jobs": len(manifest["jobs"]),
            "elapsed_seconds": time() - manifest["start_epoch"],
            "deadline_epoch": manifest["deadline_epoch"],
        }
        from .analyze import analyze
        result["paired_report"] = analyze(output)
        _atomic_json(output / "status.json", result)
        return result
    except Exception:
        _atomic_json(output / "error.json", {"traceback": traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("GPU campaign already running") from exc
        print(json.dumps(run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
