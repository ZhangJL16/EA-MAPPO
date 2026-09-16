"""Focused checks of the new mathematical audit, not learner experiments."""
import importlib.util
from fractions import Fraction as F
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]/"scripts/audit_resource_separation_bridge.py"
spec = importlib.util.spec_from_file_location("resource_bridge_audit", SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_normalized_posterior_and_predictive_law():
    belief, outcomes, _ = audit.information_state((F(1, 2), F(9, 20), F(1, 20)))
    counts = (2, 1, 1, 0)
    p = belief(counts)
    assert sum(p) == 1 and min(p) > 0
    probabilities = [F(0)]*3
    predictive = F(0)
    for updated, row in outcomes(counts, (0, 1)):
        predictive += sum(x*y for x, y in zip(p, row))
        for i in range(3):
            probabilities[i] += row[i]
        assert sum(belief(updated)) == 1
    assert predictive == 1 and tuple(probabilities) == (F(1),)*3


def test_safe_catalogue_exact_embedding():
    low = tuple(r for r in audit.ROUTES if r[1] <= 3)
    high = tuple(r for r in audit.ROUTES if r[1] <= 4)
    assert set(low) <= set(high)
    assert {r[0] for r in high if r not in low} == {"AB", "BA"}
    for name, length, channels in audit.ROUTES:
        assert len(channels) == len(set(channels))
        if name not in ("dock", "empty"):
            assert length == len(channels)+2


def test_static_control_exact_primal_dual():
    assert audit.static_terminal_certificate() == (F(32, 115), F(21, 115), F(32, 115))


def test_low_anchor_and_saved_high_witness():
    _, lower, _ = audit.primitive_low(12, (F(1, 2), F(9, 20), F(1, 20)))
    risk = audit.committed_terminal(12, (F(1, 3),)*3, 4)
    assert lower == F(2173, 8000)
    assert risk == (F(349679, 1600000), F(173067, 800000), F(41029, 800000))
