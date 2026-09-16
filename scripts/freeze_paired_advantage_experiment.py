#!/usr/bin/env python3
"""Freeze fresh paired-advantage worlds and DEV execution sources."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.freeze_cmi_v4 as base
from scripts.validate_paired_advantage_contract import (
    LIVE_SOURCE_PATHS, PRE_H_SCHEMA, SCHEMA, SEED_RANGES, SPLIT_SIZES,
    expected_pre_h_without_hash,
)


def main() -> int:
    base.LIVE_SOURCE_PATHS = LIVE_SOURCE_PATHS
    base.PRE_H_SCHEMA = PRE_H_SCHEMA
    base.SCHEMA = SCHEMA
    base.SEED_RANGES = SEED_RANGES
    base.SPLIT_SIZES = SPLIT_SIZES
    base.expected_pre_h_without_hash = expected_pre_h_without_hash
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
