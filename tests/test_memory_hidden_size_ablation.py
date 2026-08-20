from __future__ import annotations

import pytest

from experiments.memory_safety.hidden_size_ablation import HiddenSizeAblationConfig


def test_hidden_size_ablation_is_preregistered_and_bounded() -> None:
    config = HiddenSizeAblationConfig(epochs=1)
    assert config.hidden_sizes == (16, 32, 64, 128)
    with pytest.raises(ValueError, match="pre-registered"):
        HiddenSizeAblationConfig(hidden_sizes=(16, 32), epochs=1)
