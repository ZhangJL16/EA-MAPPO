from __future__ import annotations

import numpy as np

from experiments.memory_safety.grounded_routing_diagnostic import (
    GroundedObjectSafetyProbe,
    _routing_diagnostic,
)
from review_bundle.safety.grounded_memory import (
    CertifiedSafetyMemory,
    GroundedMemoryGraph,
    LearnedModuleMessages,
)


def test_learned_messages_cannot_change_certified_constraints() -> None:
    rows = np.vstack((np.eye(3), -np.eye(3)))
    memory = CertifiedSafetyMemory(
        rows,
        np.full(6, -2.0),
        np.full(6, 0.2),
        "unit_test",
    )
    graph = GroundedMemoryGraph()
    first = graph.execute(
        np.array([3.0, 0.0, 0.0]),
        memory,
        LearnedModuleMessages(np.zeros(3), np.zeros(3), np.zeros(3)),
    )
    adversarial = graph.execute(
        np.array([3.0, 0.0, 0.0]),
        memory,
        LearnedModuleMessages(
            np.full(3, 1e9),
            np.array([-1e9, 1e9, -1e9]),
            np.array([1e9, -1e9, 1e9]),
        ),
    )
    assert first.certified_constraint_digest == adversarial.certified_constraint_digest
    assert first.projection.feasible
    assert adversarial.projection.feasible
    assert np.all(rows @ adversarial.projection.acceleration >= memory.robust_lower_bounds - 1e-6)


def test_uncertainty_contraction_does_not_increase_pointwise_intervention() -> None:
    result = _routing_diagnostic(100, 9)
    assert result["all_actions_feasible"]
    assert result["certified_constraints_message_invariant"]
    assert result["nested_set_intervention_nonincrease"]


def test_grounded_probe_has_typed_heads() -> None:
    model = GroundedObjectSafetyProbe(8)
    import torch

    motion, safety = model(torch.zeros(4, 16, 4))
    assert motion.shape == (4, 6)
    assert safety.shape == (4, 2)
