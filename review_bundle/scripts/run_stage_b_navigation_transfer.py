from __future__ import annotations

import argparse

from experiments.energy_transfer.stage_b_navigation_transfer import StageBNavigationConfig, run_navigation_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen 4x4 SAC policy in the open 16x16 target world")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sorties", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--scenario", default="target_open_16x16.json")
    parser.add_argument("--operational-energy-capacity", type=float, default=10.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_navigation_transfer(StageBNavigationConfig(**vars(args)))


if __name__ == "__main__":
    main()
