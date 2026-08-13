from __future__ import annotations

import argparse

from experiments.energy_transfer.retraining import StageARetrainConfig, run_stage_a_retraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain Stage A B2/B3 from immutable raw trajectories and split")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-updates", type=int, default=800)
    parser.add_argument("--selected-mc-pretrain-updates", type=int, default=600)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_stage_a_retraining(StageARetrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
