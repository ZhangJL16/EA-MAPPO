"""R6 end-to-end learned navigation components.

This package is intentionally separate from the frozen R3 SAC/JSEB path.
"""

from .model import R6ModelConfig, RecurrentSetActorCritic
from .ppo import ConstraintState, PPOConfig, RolloutBatch, compute_gae, ppo_update

__all__ = [
    "ConstraintState",
    "PPOConfig",
    "R6ModelConfig",
    "RecurrentSetActorCritic",
    "RolloutBatch",
    "compute_gae",
    "ppo_update",
]
