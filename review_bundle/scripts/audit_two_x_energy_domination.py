from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_random_persistent_uav_env


DEFAULT_SCENARIOS = (
    "random_persistent_open.json",
    "random_persistent_obstacle.json",
    "random_persistent_energy_tight.json",
)


def audit_scenario(scenario: str) -> dict[str, object]:
    environment = make_random_persistent_uav_env(
        scenario,
        seed=0,
        timing_mode="functional",
        flight_energy_multiplier=2.0,
    )
    tracking = np.asarray(environment.runtime.config.tracking_error_bound, dtype=np.float64)
    cells = {cell.cell_id: cell for cell in environment.atlas.manifest.cells}
    minimum_one_step_slack = float("inf")
    minimum_recursion_slack = float("inf")
    worst_one_step_cell = None
    worst_recursion_cell = None
    nonterminal_count = 0
    terminal_count = 0
    violations: list[dict[str, object]] = []
    for cell in cells.values():
        if cell.level == 0:
            terminal_count += 1
            if cell.one_step_energy_upper != 0.0 or cell.energy_upper != 0.0:
                violations.append({"cell_id": cell.cell_id, "kind": "terminal_nonzero"})
            continue
        nonterminal_count += 1
        velocity = np.maximum(
            np.abs(np.asarray(cell.state_bounds.velocity.low, dtype=np.float64)),
            np.abs(np.asarray(cell.state_bounds.velocity.high, dtype=np.float64)),
        )
        command = np.maximum(
            np.abs(np.asarray(cell.action_low, dtype=np.float64)),
            np.abs(np.asarray(cell.action_high, dtype=np.float64)),
        )
        report = environment.plant.energy_model.audit_certified_upper(
            cell.one_step_energy_upper,
            velocity,
            command + tracking,
            environment.runtime.config.dt,
        )
        if report.slack < minimum_one_step_slack:
            minimum_one_step_slack = report.slack
            worst_one_step_cell = cell.cell_id
        if not report.verified:
            violations.append({
                "cell_id": cell.cell_id,
                "kind": "one_step_underbound",
                "certified_upper": report.certified_upper,
                "realized_box_upper": report.realized_box_upper,
                "slack": report.slack,
            })
        successor = cells[cell.successor_target_cell]
        recursion_slack = float(
            cell.energy_upper
            - cell.one_step_energy_upper
            - successor.energy_upper
        )
        if recursion_slack < minimum_recursion_slack:
            minimum_recursion_slack = recursion_slack
            worst_recursion_cell = cell.cell_id
        if successor.level >= cell.level or recursion_slack < 0.0 or cell.e3_residual < 0.0:
            violations.append({
                "cell_id": cell.cell_id,
                "kind": "cumulative_recursion",
                "successor": successor.cell_id,
                "rank": cell.level,
                "successor_rank": successor.level,
                "recursion_slack": recursion_slack,
                "stored_residual": cell.e3_residual,
            })
    environment.close()
    return {
        "scenario": scenario,
        "flight_energy_multiplier": 2.0,
        "energy_version": environment.runtime.calibration.energy.version,
        "energy_contract_hash": environment.runtime.calibration.energy.contract_hash,
        "atlas_hash": environment.atlas.atlas_hash,
        "nonterminal_cells": nonterminal_count,
        "terminal_cells": terminal_count,
        "minimum_one_step_slack": minimum_one_step_slack,
        "worst_one_step_cell": worst_one_step_cell,
        "minimum_recursion_slack": minimum_recursion_slack,
        "worst_recursion_cell": worst_recursion_cell,
        "violations": violations[:20],
        "violation_count": len(violations),
        "verified": len(violations) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit full-box 2x plant-energy domination")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scenarios = tuple(args.scenarios) if args.scenarios else DEFAULT_SCENARIOS
    results = [audit_scenario(scenario) for scenario in scenarios]
    payload = {
        "schema": "ea-mappo-two-x-energy-domination-v1",
        "method": "analytic separable endpoint maximum over every certified cell; no random sampling",
        "scenarios": results,
        "verified": all(result["verified"] for result in results),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
