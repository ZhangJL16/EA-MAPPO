from __future__ import annotations

import argparse

from experiments.memory_safety.grounded_prediction_diagnostic import (
    PredictionDiagnosticConfig,
    run_prediction_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--route-closed-loop-scenarios", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_prediction_diagnostic(
        PredictionDiagnosticConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            route_closed_loop_scenarios=args.route_closed_loop_scenarios,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()

