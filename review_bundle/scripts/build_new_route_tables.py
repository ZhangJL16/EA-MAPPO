#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", nargs="?", default="artifacts/new_route/e1_energy_aggregate.json")
    parser.add_argument("--output", default="artifacts/new_route/e1_energy_table.md")
    args = parser.parse_args()
    payload = json.loads(Path(args.aggregate).read_text(encoding="utf-8"))
    methods = ("distance_baseline", "monte_carlo", "scalar_td", "distributional_median")
    lines = ["| Method | Seed | MAE | RMSE | Underestimation |", "| --- | ---: | ---: | ---: | ---: |"]
    for seed, result in enumerate(payload["results"]):
        for method in methods:
            metric = result[method]
            lines.append(f"| {method} | {seed} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['underestimation_rate']:.6f} |")
    Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
