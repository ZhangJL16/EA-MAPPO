"""Energy bootstrap and target-domain adaptation experiments."""

from .stage_a_bootstrap import StageABootstrapConfig, collect_bootstrap_trajectory, run_stage_a
from .control import ManagedGoalDecision, decide_managed_goal
from .stage_b_navigation_transfer import StageBNavigationConfig, run_navigation_transfer

__all__ = [
    "ManagedGoalDecision",
    "StageABootstrapConfig",
    "StageBNavigationConfig",
    "collect_bootstrap_trajectory",
    "decide_managed_goal",
    "run_stage_a",
    "run_navigation_transfer",
]
