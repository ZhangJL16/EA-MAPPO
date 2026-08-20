from __future__ import annotations

import argparse
import json

from experiments.memory_safety.grounded_routing_diagnostic import (
    GroundedDiagnosticConfig,
    run_grounded_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--adversarial-message-draws", type=int, default=2000)
    args = parser.parse_args()
    result = run_grounded_diagnostic(
        GroundedDiagnosticConfig(
            seed=args.seed,
            epochs=args.epochs,
            adversarial_message_draws=args.adversarial_message_draws,
            output_dir=args.output_dir,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
