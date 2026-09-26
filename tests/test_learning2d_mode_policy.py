"""Focused checks for mode-conditioned categorical action support."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces
from learning2d.mode_policy import ModeMaskedPolicy, valid_action_mask


def _features(*, docked: bool, energy: float) -> torch.Tensor:
    obs = torch.zeros((1, 40), dtype=torch.float32)
    obs[0, 8] = energy
    obs[0, 10 if docked else 11] = 1.0
    return obs


def test_valid_action_mask_modes_and_charge_thresholds():
    flight = valid_action_mask(_features(docked=False, energy=0.2))[0].tolist()
    assert flight == [True] * 10 + [False] * 3
    dock_low = valid_action_mask(_features(docked=True, energy=0.2))[0].tolist()
    assert [index for index, valid in enumerate(dock_low) if valid] == [4, 10, 11, 12]
    dock_mid = valid_action_mask(_features(docked=True, energy=0.75))[0].tolist()
    assert [index for index, valid in enumerate(dock_mid) if valid] == [4, 12]
    dock_full = valid_action_mask(_features(docked=True, energy=1.0))[0].tolist()
    assert [index for index, valid in enumerate(dock_full) if valid] == [4]


def test_policy_sampling_prediction_and_training_distribution_agree():
    policy = ModeMaskedPolicy(
        spaces.Box(-5, 5, shape=(40,), dtype=np.float32),
        spaces.Discrete(13),
        lr_schedule=lambda _: 0.0003,
        net_arch=[16, 16],
    )
    observation = torch.cat((
        _features(docked=True, energy=0.2),
        _features(docked=True, energy=1.0),
        _features(docked=False, energy=0.2),
    ))
    support = valid_action_mask(observation)
    actions, values, rollout_logp = policy(observation)
    assert bool(support.gather(1, actions[:, None]).all())
    eval_values, eval_logp, entropy = policy.evaluate_actions(observation, actions)
    assert torch.allclose(values, eval_values)
    assert torch.allclose(rollout_logp, eval_logp)
    assert torch.isfinite(entropy).all()
    assert int(policy.get_distribution(observation[1:2]).get_actions(True).item()) == 4
    for _ in range(20):
        chosen = policy.get_distribution(observation).get_actions()
        assert bool(support.gather(1, chosen[:, None]).all())


def test_policy_rejects_unclassified_mode():
    try:
        valid_action_mask(torch.zeros((1, 40)))
    except ValueError as error:
        assert "docked or in flight" in str(error)
    else:
        raise AssertionError("missing mode was silently accepted")
