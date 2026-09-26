"""Use the 600-second operating window as a true finite-horizon terminal.

The parent Gym adapter marks every non-failure endpoint as a truncation.  That
is correct for an externally imposed sampling limit, but the fixed operating
window is part of this study's objective and belongs in the MDP termination.
The remaining-time feature is already present in the observation.
"""

from __future__ import annotations

from dual_constraint_2d.gym_adapter import DualConstraintGym


class FiniteHorizonGym(DualConstraintGym):
    def step(self, action: int):
        observation, reward, terminated, truncated, info = super().step(action)
        if truncated:
            if self.env is None or not self.env.done or info.get("failure_reason") is not None:
                raise RuntimeError("unexpected non-horizon truncation")
            return observation, reward, True, False, {**info, "finite_horizon_terminal": True}
        return observation, reward, terminated, False, info
