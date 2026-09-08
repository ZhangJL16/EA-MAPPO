from __future__ import annotations

import numpy as np

from experiments.directional_navigation.anchor_return import (
    AnchorReturnRecovery,
    obstacle_layout_sha256,
)


def test_anchor_reset_preserves_state_and_starts_new_return_leg() -> None:
    environment = AnchorReturnRecovery(horizon=32, obstacles=2, hocbf=False)
    try:
        _, _ = environment.reset(seed=17)
        layout = environment.base.static_obstacle_layout()
        position = environment.base.agent.pos.copy()
        velocity = np.asarray([0.4, -0.3, 0.1], dtype=np.float32)
        charger = environment.base.charger_position.copy()
        active_goal = environment.base.current_task_point.copy()
        anchor = {
            "anchor_id": "unit_anchor",
            "anchor_index": 0,
            "scene_index": 0,
            "validation_start_position": position,
            "position": position,
            "velocity": velocity,
            "active_goal": active_goal,
            "charger_goal": charger,
            "obstacle_layout": layout,
            "obstacle_layout_sha256": obstacle_layout_sha256(layout),
        }
        observation, info = environment.reset_from_anchor(anchor, 17)
        assert observation.shape == (2056,)
        assert info["anchor_id"] == "unit_anchor"
        np.testing.assert_array_equal(environment.base.agent.pos, position)
        np.testing.assert_array_equal(environment.base.agent.vel, velocity)
        np.testing.assert_array_equal(environment.base.active_goal, charger)
        assert environment.steps == 0
        assert environment.collision_count == 0
        assert not environment.finished
    finally:
        environment.close()


def test_hocbf_interface_is_explicit() -> None:
    raw = AnchorReturnRecovery(horizon=16, obstacles=1, hocbf=False)
    shielded = AnchorReturnRecovery(horizon=16, obstacles=1, hocbf=True)
    try:
        assert raw.base.safety_filter is None
        assert shielded.base.safety_filter is not None
    finally:
        raw.close()
        shielded.close()
