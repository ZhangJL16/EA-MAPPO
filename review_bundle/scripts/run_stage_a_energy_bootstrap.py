from __future__ import annotations

import argparse

from experiments.energy_transfer import StageABootstrapConfig, run_stage_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and train Stage A 4x4 energy bootstrap models")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sorties", type=int, default=60)
    parser.add_argument("--task-prefix-min-steps", type=int, default=0)
    parser.add_argument("--task-prefix-max-steps", type=int, default=80)
    parser.add_argument("--max-return-steps", type=int, default=800)
    parser.add_argument("--operational-energy-capacity", type=float, default=3.0)
    parser.add_argument("--training-updates", type=int, default=800)
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-kind", choices=("smoke", "validation", "formal"), default="smoke")
    args = parser.parse_args()
    run_stage_a(StageABootstrapConfig(**vars(args)))


if __name__ == "__main__":
    main()
