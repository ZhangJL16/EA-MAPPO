from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from review_bundle.safety.energy.gradients import (
    mc_energy_physical_gradient,
    physical_energy_state,
)
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from review_bundle.safety.collision.hocbf import energy_to_go_action_gradient


WORLD_SIZE = np.array([4000.0, 4000.0, 400.0], dtype=np.float64)
HORIZONTAL_VELOCITY_LIMIT = 20.0
VERTICAL_VELOCITY_LIMIT = 5.0
DISTANCE_SCALE = float(np.linalg.norm(WORLD_SIZE))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite-difference audit of MC energy gradients")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--position-epsilon", type=float, default=1.0)
    parser.add_argument("--velocity-epsilon", type=float, default=0.1)
    parser.add_argument("--action-epsilon", type=float, default=0.1)
    parser.add_argument("--safety-dt", type=float, default=0.05)
    args = parser.parse_args()
    if (
        args.states <= 0
        or args.position_epsilon <= 0.0
        or args.velocity_epsilon <= 0.0
        or args.action_epsilon <= 0.0
        or args.safety_dt <= 0.0
    ):
        raise ValueError("state count and finite-difference epsilons must be positive")

    estimator = EnergyToGoRegressor.load(Path(args.model), device=args.device)
    rng = np.random.default_rng(args.seed)
    relative_errors = []
    absolute_errors = []
    gradient_norms = []
    action_relative_errors = []
    action_absolute_errors = []
    action_gradient_norms = []
    records = []
    for index in range(args.states):
        while True:
            position = rng.uniform([100.0, 100.0, 20.0], [3900.0, 3900.0, 380.0])
            goal = rng.uniform([100.0, 100.0, 20.0], [3900.0, 3900.0, 380.0])
            if np.linalg.norm(goal - position) > 10.0:
                break
        velocity = rng.uniform(
            [-HORIZONTAL_VELOCITY_LIMIT, -HORIZONTAL_VELOCITY_LIMIT, -VERTICAL_VELOCITY_LIMIT],
            [HORIZONTAL_VELOCITY_LIMIT, HORIZONTAL_VELOCITY_LIMIT, VERTICAL_VELOCITY_LIMIT],
        )
        analytic = mc_energy_physical_gradient(
            estimator,
            position,
            velocity,
            goal,
            horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
            vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
            distance_scale=DISTANCE_SCALE,
        )
        numeric_position = np.zeros(3)
        numeric_velocity = np.zeros(3)
        for axis in range(3):
            position_delta = np.zeros(3)
            position_delta[axis] = args.position_epsilon
            plus = estimator.predict(
                physical_energy_state(
                    position + position_delta,
                    velocity,
                    goal,
                    horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
                    vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
                    distance_scale=DISTANCE_SCALE,
                )
            )
            minus = estimator.predict(
                physical_energy_state(
                    position - position_delta,
                    velocity,
                    goal,
                    horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
                    vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
                    distance_scale=DISTANCE_SCALE,
                )
            )
            numeric_position[axis] = (plus - minus) / (2.0 * args.position_epsilon)

            velocity_delta = np.zeros(3)
            velocity_delta[axis] = args.velocity_epsilon
            plus = estimator.predict(
                physical_energy_state(
                    position,
                    velocity + velocity_delta,
                    goal,
                    horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
                    vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
                    distance_scale=DISTANCE_SCALE,
                )
            )
            minus = estimator.predict(
                physical_energy_state(
                    position,
                    velocity - velocity_delta,
                    goal,
                    horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
                    vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
                    distance_scale=DISTANCE_SCALE,
                )
            )
            numeric_velocity[axis] = (plus - minus) / (2.0 * args.velocity_epsilon)

        analytic_vector = np.concatenate((analytic.grad_position, analytic.grad_velocity))
        numeric_vector = np.concatenate((numeric_position, numeric_velocity))
        absolute = float(np.linalg.norm(analytic_vector - numeric_vector))
        scale = max(float(np.linalg.norm(analytic_vector)), float(np.linalg.norm(numeric_vector)), 1e-8)
        relative = absolute / scale
        relative_errors.append(relative)
        absolute_errors.append(absolute)
        gradient_norms.append(float(np.linalg.norm(analytic_vector)))

        horizontal_action = rng.normal(size=2)
        horizontal_action *= rng.uniform(0.0, 5.0) / max(
            np.linalg.norm(horizontal_action),
            1e-12,
        )
        nominal_action = np.array(
            [horizontal_action[0], horizontal_action[1], rng.uniform(-3.0, 3.0)]
        )
        next_position = (
            position
            + velocity * args.safety_dt
            + 0.5 * nominal_action * args.safety_dt**2
        )
        next_velocity = velocity + nominal_action * args.safety_dt
        next_gradient = mc_energy_physical_gradient(
            estimator,
            next_position,
            next_velocity,
            goal,
            horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
            vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
            distance_scale=DISTANCE_SCALE,
        )
        analytic_action = energy_to_go_action_gradient(
            next_gradient.grad_position,
            next_gradient.grad_velocity,
            args.safety_dt,
        )
        numeric_action = np.zeros(3)
        for axis in range(3):
            perturbation = np.zeros(3)
            perturbation[axis] = args.action_epsilon

            def predicted_after_action(action: np.ndarray) -> float:
                candidate_position = (
                    position
                    + velocity * args.safety_dt
                    + 0.5 * action * args.safety_dt**2
                )
                candidate_velocity = velocity + action * args.safety_dt
                return estimator.predict(
                    physical_energy_state(
                        candidate_position,
                        candidate_velocity,
                        goal,
                        horizontal_velocity_limit=HORIZONTAL_VELOCITY_LIMIT,
                        vertical_velocity_limit=VERTICAL_VELOCITY_LIMIT,
                        distance_scale=DISTANCE_SCALE,
                    )
                )

            numeric_action[axis] = (
                predicted_after_action(nominal_action + perturbation)
                - predicted_after_action(nominal_action - perturbation)
            ) / (2.0 * args.action_epsilon)
        action_absolute = float(np.linalg.norm(analytic_action - numeric_action))
        action_scale = max(
            float(np.linalg.norm(analytic_action)),
            float(np.linalg.norm(numeric_action)),
            1e-8,
        )
        action_relative = action_absolute / action_scale
        action_relative_errors.append(action_relative)
        action_absolute_errors.append(action_absolute)
        action_gradient_norms.append(float(np.linalg.norm(analytic_action)))
        records.append(
            {
                "index": index,
                "relative_error": relative,
                "absolute_error": absolute,
                "analytic_norm": gradient_norms[-1],
                "action_relative_error": action_relative,
                "action_absolute_error": action_absolute,
                "action_analytic_norm": action_gradient_norms[-1],
            }
        )

    relative = np.asarray(relative_errors)
    absolute = np.asarray(absolute_errors)
    norms = np.asarray(gradient_norms)
    action_relative = np.asarray(action_relative_errors)
    action_absolute = np.asarray(action_absolute_errors)
    action_norms = np.asarray(action_gradient_norms)
    write_json(
        Path(args.output),
        {
            "model": str(Path(args.model).resolve()),
            "states": args.states,
            "seed": args.seed,
            "position_epsilon": args.position_epsilon,
            "velocity_epsilon": args.velocity_epsilon,
            "action_epsilon": args.action_epsilon,
            "safety_dt": args.safety_dt,
            "relative_error": {
                "mean": float(np.mean(relative)),
                "median": float(np.median(relative)),
                "p95": float(np.quantile(relative, 0.95)),
                "p99": float(np.quantile(relative, 0.99)),
                "max": float(np.max(relative)),
                "fraction_below_1e-3": float(np.mean(relative < 1e-3)),
                "fraction_below_1e-2": float(np.mean(relative < 1e-2)),
            },
            "absolute_error": {
                "mean": float(np.mean(absolute)),
                "p95": float(np.quantile(absolute, 0.95)),
                "max": float(np.max(absolute)),
            },
            "gradient_norm": {
                "mean": float(np.mean(norms)),
                "median": float(np.median(norms)),
                "p95": float(np.quantile(norms, 0.95)),
                "max": float(np.max(norms)),
                "near_zero_fraction": float(np.mean(norms < 1e-8)),
            },
            "action_gradient_relative_error": {
                "mean": float(np.mean(action_relative)),
                "median": float(np.median(action_relative)),
                "p95": float(np.quantile(action_relative, 0.95)),
                "p99": float(np.quantile(action_relative, 0.99)),
                "max": float(np.max(action_relative)),
                "fraction_below_1e-3": float(np.mean(action_relative < 1e-3)),
                "fraction_below_1e-2": float(np.mean(action_relative < 1e-2)),
            },
            "action_gradient_absolute_error": {
                "mean": float(np.mean(action_absolute)),
                "p95": float(np.quantile(action_absolute, 0.95)),
                "max": float(np.max(action_absolute)),
            },
            "action_gradient_norm": {
                "mean": float(np.mean(action_norms)),
                "median": float(np.median(action_norms)),
                "p95": float(np.quantile(action_norms, 0.95)),
                "max": float(np.max(action_norms)),
                "near_zero_fraction": float(np.mean(action_norms < 1e-8)),
            },
            "records": records,
            "calibration_gradient_included": False,
            "formal_result": False,
        },
    )


if __name__ == "__main__":
    main()
