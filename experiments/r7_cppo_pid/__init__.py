"""R7 structured CPPO-PID learned-navigation experiment.

R7 deliberately reuses the successful R3 perception/environment contract and
changes the optimization method only.  It is isolated from the frozen R3 SAC
artifacts and from the failed R6 recurrent PPO experiment.
"""

from .algorithm import (
    CPPOPIDConfig,
    PIDLagrangian,
    PIDLagrangianConfig,
    RolloutBatch,
    compute_gae,
    cppo_pid_update,
)
from .model import R7ActorCritic, R7ModelConfig

__all__ = [
    "CPPOPIDConfig",
    "PIDLagrangian",
    "PIDLagrangianConfig",
    "R7ActorCritic",
    "R7ModelConfig",
    "RolloutBatch",
    "compute_gae",
    "cppo_pid_update",
]
