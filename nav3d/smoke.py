"""Tiny, deterministic execution smoke for the analytic navigator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .controller import CBFConfig
from .geometry import BoxObstacle, CylinderObstacle, World3D
from .simulation import simulate_flight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are refused")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    world = World3D(
        [35, 30, 16],
        (
            BoxObstacle(np.array([10, 10, 0]), np.array([18, 20, 8]), "block"),
            CylinderObstacle(np.array([25, 8]), 2, 0, 12, "tower"),
        ),
    )
    config = CBFConfig(
        dt=0.05,
        body_radius=0.4,
        clearance_margin=0.2,
        max_horizontal_speed=7,
        max_vertical_speed=4,
        max_horizontal_acceleration=4,
        max_vertical_acceleration=3,
        activation_distance=15,
    )
    start, goal = [3, 15, 2], [31, 15, 2]
    flight = simulate_flight(world, start, goal, config=config, max_steps=500)
    summary = {
        "scenario": "mixed_height_box_and_cylinder",
        "arrived": flight.arrived,
        "reason": flight.reason,
        "elapsed_seconds": flight.elapsed_seconds,
        "steps": flight.steps,
        "route_waypoints": [point.tolist() for point in flight.route.waypoints],
        "route_geometric_length": flight.route.geometric_length,
        "route_plan_seconds": flight.route.planning_seconds,
        "route_checked_edges": flight.route.checked_edges,
        "collision_count": flight.final_state.collision_count,
        "safety_cost": flight.final_state.safety_cost,
        "controller_infeasible_steps": flight.controller_infeasible_steps,
        "intervention_steps": flight.intervention_steps,
        "final_position": flight.final_state.position.tolist(),
        "minimum_recorded_center_clearance": min(world.clearance(p, config.body_radius) for p in flight.positions),
        "control_p95_seconds": float(np.quantile(flight.control_seconds, 0.95)) if flight.steps else 0.0,
        "qp_p95_seconds": float(np.quantile(flight.qp_seconds, 0.95)) if flight.steps else 0.0,
        "neural_updates": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    np.savez_compressed(
        args.output / "trace.npz",
        positions=flight.positions,
        velocities=flight.velocities,
        accelerations=flight.accelerations,
        control_seconds=flight.control_seconds,
        qp_seconds=flight.qp_seconds,
    )
    print(json.dumps(summary, ensure_ascii=False))
    if not flight.arrived or flight.final_state.collision_count or flight.controller_infeasible_steps:
        raise SystemExit("navigation smoke did not meet arrival/safety checks")


if __name__ == "__main__":
    main()
