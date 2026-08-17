from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.energy_mc.statistical_gate import (
    binomial_coverage_evidence,
    sample_size_power_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit finite-sample Energy Risk gates")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "nominal_coverage_target": 0.95,
        "raw_empirical_gate_recommended_as_sole_gate": False,
        "recommended_gate": (
            "predefined Mondrian finite-sample construction plus Holm-corrected "
            "one-sided exact-binomial fresh-test stress tests"
        ),
        "v4_goal_worst": binomial_coverage_evidence(215, 231).as_dict(),
        "v4_mission_worst": binomial_coverage_evidence(375, 400).as_dict(),
        "power_table": sample_size_power_table(),
        "guarantee_scope": "predefined_group_conditional_marginal",
        "arbitrary_conditional_coverage_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
