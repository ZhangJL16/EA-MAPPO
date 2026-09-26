"""Full strict-evidence matrix with calibrated return QP retry."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

from .confirmation_qualify import MAP_IDS
from . import return_audit_v4, return_matrix


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m100_1km_return_multimap_v4_backup_cbf_verified_qp_retry"
SOURCE_FILES = ("delivery_1km/return_matrix.py", "delivery_1km/return_matrix_v4.py",
                "RETURN_MULTIMAP_PROTOCOL_20260926.md",
                "RETURN_MULTIMAP_EXECUTION_AMENDMENT_20260926.md",
                "RETURN_BACKUP_CBF_CALIBRATION_20260926.md",
                "RETURN_BACKUP_QP_RETRY_20260926.md")


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0")
        digest.update((ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def run(output: Path, *, navigation: Path = return_matrix.NAVIGATION,
        map_ids: tuple[int, ...] = MAP_IDS, workers: int = 16,
        worker_memory_gib: int = 2, resume: bool = False) -> dict:
    original = (return_matrix.PROTOCOL, return_matrix.source_hash,
                return_matrix.return_source_hash, return_matrix.build_return_manifest)
    try:
        return_matrix.PROTOCOL = PROTOCOL
        return_matrix.source_hash = source_hash
        return_matrix.return_source_hash = return_audit_v4.source_hash
        return_matrix.build_return_manifest = return_audit_v4.build_manifest
        return return_matrix.run(output, navigation=navigation, map_ids=map_ids,
                                 workers=workers, worker_memory_gib=worker_memory_gib,
                                 resume=resume)
    finally:
        (return_matrix.PROTOCOL, return_matrix.source_hash,
         return_matrix.return_source_hash, return_matrix.build_return_manifest) = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, default=return_matrix.NAVIGATION)
    parser.add_argument("--map-ids", type=int, nargs="+", default=MAP_IDS)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--worker-memory-gib", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.output, navigation=args.navigation, map_ids=tuple(args.map_ids),
        workers=args.workers, worker_memory_gib=args.worker_memory_gib, resume=args.resume)


if __name__ == "__main__":
    main()
