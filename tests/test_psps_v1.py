import numpy as np

from scripts.analyze_psps_v1_dev import choose_threshold, fit_logistic, predict
from scripts.psps_v1_features import LATEST_NAMES, PSPS_NAMES, features
from scripts.validate_psps_v1_contract import static_spec


def _arrays() -> dict[str, np.ndarray]:
    nav = np.zeros((2, 64, 2056), np.float32)
    nav[..., 3] = 1.0
    nav[..., 6] = np.linspace(0.8, 0.2, 64)
    nav[..., 7:1031] = 1.0
    return {
        "nav_observation": nav,
        "battery": np.tile(np.linspace(378.0, 350.0, 64), (2, 1)),
        "distance_to_charger": np.tile(np.linspace(1000.0, 1200.0, 64), (2, 1)),
        "task_clock": np.tile(np.arange(64) * 4, (2, 1)),
        "step": np.vstack((np.arange(64) * 4 + 4, np.arange(64) * 4 + 516)),
    }


def test_psps_is_exactly_compact_and_latest_is_prefix() -> None:
    value = features(_arrays())
    assert len(LATEST_NAMES) == 10
    assert len(PSPS_NAMES) == 20
    assert value["latest"].shape == (2, 10)
    assert value["psps"].shape == (2, 20)
    assert np.array_equal(value["latest"], value["psps"][:, :10])
    assert np.isfinite(value["psps"]).all()


def test_logistic_probability_and_conservative_threshold_tie() -> None:
    x = np.asarray([[-1.0], [-0.5], [0.5], [1.0]])
    success = np.asarray([1.0, 3.0, 29.0, 31.0])
    state = fit_logistic(x, success, np.full(4, 32.0), 1.0)
    probability = predict(state, x)
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.all(np.diff(probability) > 0.0)
    # Identical action utilities make every threshold tie; choose 0.95.
    assert choose_threshold(probability, np.zeros((4, 2))) == 0.95


def test_gate_order_and_permanent_denials_are_frozen() -> None:
    spec = static_spec()
    assert spec["gate_order"] == ["O", "D"]
    assert spec["gate_d"]["interpreted_only_if_gate_o_passes"] is True
    assert spec["old_pai_confirm_access_forbidden_forever"] is True
    assert spec["feature_channels"]["generic_history_encoder"] is False
    assert spec["method_train_authorized"] is False
