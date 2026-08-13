from __future__ import annotations

import numpy as np
import pytest

from safety.energy import EnergyReturnEpisode, EnergyTransition


def transition(*, terminal: bool, cost: float = 1.0) -> EnergyTransition:
    return EnergyTransition(np.zeros(2), np.zeros(1), cost, np.ones(2), terminal)


def test_completed_episode_requires_final_charger_hit() -> None:
    episode = EnergyReturnEpisode((transition(terminal=False), transition(terminal=True)), True, "hash", {})
    assert episode.completed
    with pytest.raises(ValueError, match="completed"):
        EnergyReturnEpisode((transition(terminal=False),), True, "hash", {})


def test_charger_hit_cannot_precede_last_transition() -> None:
    with pytest.raises(ValueError, match="last"):
        EnergyReturnEpisode((transition(terminal=True), transition(terminal=False)), False, "hash", {})
