"""Diagnostics for recoverable UAV safety under intermittent perception."""

from .diagnostics import (
    CounterexampleRecord,
    DynamicObstacleSpec,
    HardScenario,
    PerceptionMode,
    RecoverabilityDiagnosticConfig,
    compute_pointwise_margin,
    make_hand_scenarios,
    make_random_hard_scenarios,
    run_scenario,
    search_counterexamples,
)

__all__ = [
    "CounterexampleRecord",
    "DynamicObstacleSpec",
    "HardScenario",
    "PerceptionMode",
    "RecoverabilityDiagnosticConfig",
    "compute_pointwise_margin",
    "make_hand_scenarios",
    "make_random_hard_scenarios",
    "run_scenario",
    "search_counterexamples",
]
