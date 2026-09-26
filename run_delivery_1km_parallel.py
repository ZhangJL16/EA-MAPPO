"""Resume the frozen 1 km qualification with independent route workers."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import resource

from delivery_1km.confirmation_qualify import cohort_hash
from delivery_1km.qualify import _atomic_json, execute_job, source_hash
from nav3d_v2.qualify import source_hash as frozen_navigation_hash


def _run_job(args: tuple[Path, dict, dict]) -> dict:
    return execute_job(*args)


def _limit_address_space(gib: int) -> None:
    limit = gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-memory-gib", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.worker_memory_gib <= 8:
        parser.error("workers must be 1..16 and per-worker memory must be 1..8 GiB")

    output = args.output
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != source_hash() or manifest["frozen_navigation_sha256"] != frozen_navigation_hash():
        raise RuntimeError("qualification source differs from the manifest")
    if "cohort_source_sha256" in manifest and manifest["cohort_source_sha256"] != cohort_hash():
        raise RuntimeError("confirmation cohort differs from the manifest")

    existing = set()
    for job in manifest["jobs"]:
        result_path = output / "results" / f"{job['job_id']}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["job_id"] != job["job_id"] or ("trace" in saved and not (output / saved["trace"]).exists()):
                raise RuntimeError(f"invalid saved result: {job['job_id']}")
            existing.add(job["job_id"])

    jobs = manifest["jobs"]
    next_complete = 0

    def advance() -> None:
        nonlocal next_complete
        while next_complete < len(jobs) and jobs[next_complete]["job_id"] in existing:
            next_complete += 1
            if next_complete % manifest["checkpoint_jobs"] == 0 or next_complete == len(jobs):
                checkpoint = {
                    "protocol": manifest["protocol"],
                    "source_sha256": manifest["source_sha256"],
                    "completed": next_complete,
                    "total": len(jobs),
                    "last_job_id": jobs[next_complete - 1]["job_id"],
                }
                _atomic_json(output / "checkpoint.json", checkpoint)
                print(json.dumps(checkpoint, ensure_ascii=False), flush=True)

    advance()
    remaining = iter(job for job in jobs if job["job_id"] not in existing)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_limit_address_space,
                             initargs=(args.worker_memory_gib,)) as pool:
        pending = {}

        def fill() -> None:
            while len(pending) < 2 * args.workers:
                try:
                    job = next(remaining)
                except StopIteration:
                    break
                pending[pool.submit(_run_job, (output, manifest, job))] = job

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                _atomic_json(output / "results" / f"{job['job_id']}.json", future.result())
                existing.add(job["job_id"])
            advance()
            fill()


if __name__ == "__main__":
    main()
