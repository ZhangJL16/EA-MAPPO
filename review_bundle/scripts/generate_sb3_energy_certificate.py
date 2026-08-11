#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_random_persistent_uav_env
from scripts.sb3_recovery_teacher import write_certificate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically regenerate a multiplier-bound recovery atlas")
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--flight-energy-multiplier", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    environment = make_random_persistent_uav_env(
        args.scenario,
        timing_mode="functional",
        flight_energy_multiplier=args.flight_energy_multiplier,
    )
    try:
        descriptor = write_certificate_artifact(args.output, environment)
    finally:
        environment.close()
    print(descriptor)


if __name__ == "__main__":
    main()
