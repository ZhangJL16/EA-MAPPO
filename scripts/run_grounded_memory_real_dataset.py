from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from experiments.memory_safety.grounded_real_dataset import (
    GroundedDatasetConfig,
    collect_grounded_real_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--minimum-controlled-transitions", type=int, default=100_000)
    parser.add_argument("--max-scenarios", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--sensor-period-steps", type=int, default=2)
    parser.add_argument("--sensor-error-bound", type=float, default=0.10)
    parser.add_argument("--dropout-probability", type=float, default=0.05)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--contraction-stride", type=int, default=10)
    parser.add_argument("--obstacle-radius", type=float, default=20.0)
    parser.add_argument(
        "--sac-checkpoint",
        default=(
            "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
            "phase1_navigation/checkpoint_transition_500000.zip"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GroundedDatasetConfig(
        seed=args.seed,
        minimum_controlled_transitions=args.minimum_controlled_transitions,
        max_scenarios=args.max_scenarios,
        max_steps=args.max_steps,
        sensor_period_steps=args.sensor_period_steps,
        sensor_error_bound=args.sensor_error_bound,
        dropout_probability=args.dropout_probability,
        history_seconds=args.history_seconds,
        contraction_stride=args.contraction_stride,
        obstacle_radius=args.obstacle_radius,
        sac_checkpoint=args.sac_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        require_hard_filter_valid=True,
    )
    print(json.dumps(asdict(config), indent=2), flush=True)
    collect_grounded_real_dataset(config)


if __name__ == "__main__":
    main()
