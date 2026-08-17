from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.energy_mc.final_risk import attribute_switch


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reserve versus uncertainty switch causes")
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    methods: dict[str, object] = {}
    for path in args.events:
        rows = []
        for event in load_jsonl(path):
            point = float(event["mission_energy_prediction"])
            upper = float(event["mission_energy_upper95"])
            uncertainty = max(upper - point, 0.0)
            reserve = float(event["reserve"])
            attribution = attribute_switch(
                remaining_energy=float(event["remaining_energy"]),
                point_estimate=point,
                uncertainty_margin=uncertainty,
                reserve=reserve,
            )
            rows.append(
                {
                    "global_step": int(event["global_step"]),
                    "battery_cycle_id": int(event["battery_cycle_id"]),
                    "point_estimate": point,
                    "uncertainty_margin": uncertainty,
                    "reserve": reserve,
                    "reserve_to_uncertainty_ratio": reserve / max(uncertainty, 1e-12),
                    **attribution.as_dict(),
                }
            )
        counts = Counter(str(row["cause"]) for row in rows)
        methods[path.parent.name] = {
            "events_path": str(path),
            "num_switches": len(rows),
            "cause_counts": dict(counts),
            "cause_fractions": {
                name: count / max(len(rows), 1) for name, count in counts.items()
            },
            "mean_reserve": float(np.mean([row["reserve"] for row in rows])),
            "mean_uncertainty_margin": float(
                np.mean([row["uncertainty_margin"] for row in rows])
            ),
            "mean_reserve_to_uncertainty_ratio": float(
                np.mean([row["reserve_to_uncertainty_ratio"] for row in rows])
            ),
            "events": rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"methods": methods}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
