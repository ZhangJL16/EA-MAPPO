"""Bounded parallel continuation of a frozen return-audit manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
import resource

from delivery_1km.energy_profiles import ENERGY_PROFILES
from delivery_1km.return_audit import NAVIGATION, _atomic_json, _hash
from delivery_1km import return_audit, return_audit_v2, return_audit_v3, return_audit_v4


def _limit_address_space(gib: int) -> None:
    limit = gib * 1024**3
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def _validate_result(output: Path, manifest: dict, index: int) -> bool:
    path = output / "results" / f"step_{index:05d}.json"
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    if (row["step_index"] != index
            or len(row["samples"]) != len(manifest["profiles"]) * len(manifest["payloads_kg"])):
        raise RuntimeError(f"invalid saved result for state {index}")
    if "return_trace" in row and _hash(output / row["return_trace"]) != row["return_trace_sha256"]:
        raise RuntimeError(f"return trace changed for state {index}")
    return True


def run(output: Path, navigation: Path, workers: int, worker_memory_gib: int) -> dict:
    if not 1 <= workers <= 16 or not 1 <= worker_memory_gib <= 8:
        raise ValueError("workers must be 1..16 and per-worker memory must be 1..8 GiB")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    engine = {return_audit.PROTOCOL: return_audit,
              return_audit_v2.PROTOCOL: return_audit_v2,
              return_audit_v3.PROTOCOL: return_audit_v3,
              return_audit_v4.PROTOCOL: return_audit_v4}.get(manifest["protocol"])
    if engine is None:
        raise RuntimeError("unknown return audit protocol")
    if (manifest["source_sha256"] != engine.source_hash()
            or manifest["navigation_manifest_sha256"] != _hash(navigation / "manifest.json")
            or manifest["profiles"] != [vars(profile) for profile in ENERGY_PROFILES]):
        raise RuntimeError("audit source, navigation, or energy profiles changed")
    job = manifest["job"]
    if (_hash(navigation / "results" / f"{job['job_id']}.json") != manifest["navigation_result_sha256"]
            or _hash(navigation / manifest["trace"]) != manifest["navigation_trace_sha256"]):
        raise RuntimeError("frozen navigation input changed")
    order = manifest["state_indices"]
    if len(order) != manifest["states"] or set(order) != set(range(1, manifest["states"] + 1)):
        raise RuntimeError("audit state schedule is incomplete")

    existing = {index for index in order if _validate_result(output, manifest, index)}
    next_complete = 0

    def advance() -> None:
        nonlocal next_complete
        while next_complete < len(order) and order[next_complete] in existing:
            next_complete += 1
            if next_complete % manifest["checkpoint_states"] == 0 or next_complete == len(order):
                checkpoint = {
                    "protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"],
                    "completed": next_complete, "total": len(order),
                    "last_step_index": order[next_complete - 1],
                }
                _atomic_json(output / "checkpoint.json", checkpoint)
                print(json.dumps(checkpoint, ensure_ascii=False), flush=True)

    advance()
    remaining = iter(index for index in order if index not in existing)
    with ProcessPoolExecutor(max_workers=workers, initializer=_limit_address_space,
                             initargs=(worker_memory_gib,)) as pool:
        pending = {}

        def fill() -> None:
            while len(pending) < 2 * workers:
                try:
                    index = next(remaining)
                except StopIteration:
                    break
                pending[pool.submit(engine.audit_state, navigation, output, manifest, index)] = index

        fill()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                row = future.result()
                _atomic_json(output / "results" / f"step_{index:05d}.json", row)
                existing.add(index)
            advance()
            fill()
    return json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, default=NAVIGATION)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker-memory-gib", type=int, default=2)
    args = parser.parse_args()
    run(args.output, args.navigation, args.workers, args.worker_memory_gib)


if __name__ == "__main__":
    main()
