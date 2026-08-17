from __future__ import annotations

import argparse
import json
from pathlib import Path

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv


def audit_guard_resets(
    run_directory: str | Path,
    *,
    battery_capacity: float,
) -> dict[str, object]:
    run = Path(run_directory)
    records = [
        json.loads(line)
        for line in (run / "phase2" / "battery_cycles.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    guards = [row for row in records if bool(row.get("emergency_time_limit"))]
    events = []
    for index, guard in enumerate(guards):
        environment = UAVEnergyDeliverySACEnv(
            phase=SACTrainingPhase.ENERGY_MANAGED,
            operational_energy_capacity=battery_capacity,
        )
        _, reset_info = environment.reset(seed=900_000 + index)
        events.append(
            {
                "guard_event_index": index,
                "guard_transition": int(guard["cycle_end_transition"]),
                "formal_global_transition": int((index + 1) * 20_000),
                "energy_before_guard": float(guard["remaining_energy_at_cycle_end"]),
                "remaining_fraction_before_guard": float(
                    guard["remaining_energy_at_cycle_end"] / battery_capacity
                ),
                "battery_cycle_id_before": int(guard["battery_cycle_id"]),
                "energy_immediately_after_reset": float(reset_info["remaining_energy"]),
                "remaining_fraction_after_reset": float(
                    reset_info["remaining_energy_fraction"]
                ),
                "battery_cycle_id_after": int(reset_info["battery_cycle_id"]),
                "after_reset_evidence": "runtime_replay_of_exact_environment_reset_semantics",
            }
        )
        environment.close()
    artificial = bool(
        events
        and all(
            abs(row["energy_immediately_after_reset"] - battery_capacity) < 1e-9
            and row["remaining_fraction_after_reset"] == 1.0
            for row in events
        )
    )
    return {
        "source_run": str(run.resolve()),
        "battery_capacity": float(battery_capacity),
        "guard_event_count": len(events),
        "FORMAL_PHASE2_HAS_ARTIFICIAL_BATTERY_RESETS": artificial,
        "historical_completed_recharge_cycles": sum(
            bool(row.get("return_success")) for row in records
        ),
        "historical_guard_truncated_partial_segments": len(guards),
        "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit historical Phase2 guard resets")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--battery-capacity", required=True, type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_guard_resets(
        args.run_directory,
        battery_capacity=args.battery_capacity,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
