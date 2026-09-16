"""Gate-P development plant with a single physically shared C-prefix."""
from __future__ import annotations

from experiments.directional_navigation.dvoi_h_branch import DVOIHBranch


class DVOIPBranch(DVOIHBranch):
    """Extends the administrative clock to cover a 256-step post-anchor prefix."""

    MAX_PREBRANCH_PREFIX_STEPS = 1024

    def begin_p_prefix(self) -> None:
        self.branch_mode = "P_PREFIX"
        self.branch_anchor_tasks = int(self.base.tasks_completed)
        self.branch_anchor_contacts = int(self.collision_count)
        self.branch_anchor_energy = float(self.base.agent.energy)
        self.branch_anchor_step = int(self.total_steps)
        self.return_steps = 0
