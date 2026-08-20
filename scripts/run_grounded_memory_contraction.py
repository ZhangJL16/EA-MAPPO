from __future__ import annotations

import argparse

from experiments.memory_safety.grounded_contraction_diagnostic import (
    ContractionDiagnosticConfig,
    run_contraction_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-epochs", type=int, default=12)
    parser.add_argument("--maximum-training-snapshots", type=int, default=8000)
    parser.add_argument("--maximum-evaluation-snapshots", type=int, default=2000)
    args = parser.parse_args()
    run_contraction_diagnostic(
        ContractionDiagnosticConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            training_epochs=args.training_epochs,
            maximum_training_snapshots=args.maximum_training_snapshots,
            maximum_evaluation_snapshots=args.maximum_evaluation_snapshots,
        )
    )


if __name__ == "__main__":
    main()

