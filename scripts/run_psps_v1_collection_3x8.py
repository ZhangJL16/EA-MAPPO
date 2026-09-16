#!/usr/bin/env python3
"""Run the hash-authorized PSPS-v1 DEV 3x8 execution overlay."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_psps_v1_collection as base
from scripts.amend_psps_v1_parallelism import validate_amendment

WORLD_PROCESSES = 3
WORKERS_PER_WORLD = 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--split", choices=("PSPS_DEV",), required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--execution-amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.resume:
        raise ValueError("3x8 execution overlay is resume-only")
    result = validate_amendment(
        args.execution_amendment.resolve(), args.contract.resolve(),
        args.freeze_receipt.resolve(), args.output_dir.resolve(),
    )
    if not result["valid"]:
        raise ValueError("execution amendment invalid: " + "; ".join(result["errors"]))
    base.WORLD_PROCESSES = WORLD_PROCESSES
    base.WORKERS_PER_WORLD = WORKERS_PER_WORLD
    base_args = argparse.Namespace(
        contract=args.contract, split=args.split,
        freeze_receipt=args.freeze_receipt, output_dir=args.output_dir,
        device=args.device, resume=True, stop_after_worlds=0,
    )
    return base.run(base_args)


if __name__ == "__main__":
    raise SystemExit(main())
