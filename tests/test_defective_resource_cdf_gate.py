from __future__ import annotations

import numpy as np
import torch

from scripts.run_defective_resource_cdf_gate import (
    DefectiveResourceCDF,
    defective_cdf_numpy,
    defective_cdf_torch,
    defective_resource_loss,
    fit_calibration,
    parse_args,
    select_best_epoch,
    variant_spec,
)


class DummyEncoder(torch.nn.Module):
    features_dim = 2

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, :2]


def test_defective_cdf_is_monotone_and_has_failure_mass_limit() -> None:
    failure_logit = torch.logit(torch.tensor([0.25]))
    location = torch.tensor([np.log(0.2)], dtype=torch.float32)
    scale = torch.tensor(0.15)
    budgets = torch.tensor([0.05, 0.1, 0.2, 0.4, 1e6])
    probability = defective_cdf_torch(
        failure_logit, location, scale, budgets
    )
    assert torch.all(probability[:, 1:] >= probability[:, :-1])
    torch.testing.assert_close(probability[0, 2], torch.tensor(0.375))
    torch.testing.assert_close(probability[0, -1], torch.tensor(0.75))


def test_defective_loss_is_finite_and_backpropagates() -> None:
    failure_logit = torch.tensor([-1.0, 1.0, -0.5], requires_grad=True)
    location = torch.tensor([-2.0, -1.5, -1.0], requires_grad=True)
    raw_scale = torch.tensor(0.2, requires_grad=True)
    energy = torch.tensor([0.12, float("nan"), 0.30])
    budgets = torch.tensor([0.08, 0.2, 0.36])
    labels = torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    loss, pieces = defective_resource_loss(
        failure_logit,
        location,
        raw_scale,
        energy,
        labels,
        torch.ones(3),
        budgets,
        cdf_weight=2.0,
        huber_beta=0.25,
    )
    assert torch.isfinite(loss)
    assert all(np.isfinite(value) for value in pieces.values())
    loss.backward()
    assert failure_logit.grad is not None
    assert location.grad is not None
    assert raw_scale.grad is not None


def test_loss_handles_all_failure_minibatch() -> None:
    failure_logit = torch.zeros(2, requires_grad=True)
    location = torch.zeros(2, requires_grad=True)
    scale = torch.tensor(0.2, requires_grad=True)
    energy = torch.full((2,), float("nan"))
    labels = torch.zeros((2, 2))
    loss, pieces = defective_resource_loss(
        failure_logit,
        location,
        scale,
        energy,
        labels,
        torch.ones(2),
        torch.tensor([0.1, 0.2]),
        cdf_weight=2.0,
        huber_beta=0.25,
    )
    assert pieces["energy_huber"] == 0.0
    loss.backward()
    assert torch.all(torch.isfinite(failure_logit.grad))


def test_compact_model_has_positive_global_scale_and_two_outputs() -> None:
    model = DefectiveResourceCDF(
        DummyEncoder(),
        variant="compact_hurdle",
        hidden_dim=8,
        compact_dim=18,
        context_dim=4,
        minimum_scale=0.03,
        initial_scale=0.25,
        initial_log_energy=-2.0,
    )
    batch = 3
    output = model(
        torch.zeros(batch, 5),
        torch.zeros(batch, 5),
        torch.zeros(batch, 4),
        torch.zeros(batch, 18),
    )
    assert output[0].shape == (batch,)
    assert output[1].shape == (batch,)
    assert np.isclose(float(output[2].detach()), 0.25)


def test_fusion_uses_encoder_and_cdf_loss_while_ablation_does_not() -> None:
    primary = variant_spec("fusion_hurdle", 2.0)
    ablation = variant_spec("fusion_no_cdf", 2.0)
    assert primary == {"use_encoder": True, "use_compact": True, "cdf_weight": 2.0}
    assert ablation["cdf_weight"] == 0.0


def test_calibration_recovers_better_synthetic_location_shift() -> None:
    failure_logit = np.full(20, -5.0)
    location = np.full(20, np.log(0.30))
    budgets = np.asarray([0.1, 0.2, 0.3, 0.4])
    labels = np.tile(np.asarray([0.0, 1.0, 1.0, 1.0]), (20, 1))
    tasks = np.arange(20)
    before = defective_cdf_numpy(
        failure_logit, location, 0.15, budgets
    )
    before_brier = float(np.mean((before - labels) ** 2))
    parameters, _ = fit_calibration(
        failure_logit,
        location,
        0.15,
        budgets,
        labels,
        tasks,
    )
    after = defective_cdf_numpy(
        failure_logit,
        location,
        0.15,
        budgets,
        failure_temperature=parameters["failure_temperature"],
        location_shift=parameters["location_shift"],
        scale_multiplier=parameters["scale_multiplier"],
    )
    assert float(np.mean((after - labels) ** 2)) < before_brier
    assert parameters["location_shift"] < 0.0


def test_best_epoch_uses_earliest_minimum() -> None:
    history = [
        {"epoch": 5, "validation_brier": 0.12},
        {"epoch": 10, "validation_brier": 0.10},
        {"epoch": 15, "validation_brier": 0.10},
    ]
    assert select_best_epoch(history) == 10


def test_default_protocol_is_locked() -> None:
    args = parse_args(
        [
            "--source-dir",
            "/tmp/source",
            "--artifact",
            "/tmp/artifact",
            "--checkpoint",
            "/tmp/checkpoint",
            "--output-dir",
            "/tmp/output",
        ]
    )
    assert args.max_epochs == 400
    assert args.minimum_epochs == 5
    assert args.cdf_loss_weight == 2.0
    assert args.variants == list(
        ("compact_hurdle", "encoder_hurdle", "fusion_no_cdf", "fusion_hurdle")
    )
