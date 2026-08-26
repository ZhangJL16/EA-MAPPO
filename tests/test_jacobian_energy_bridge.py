from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.jacobian_energy_bridge.callbacks import SafetyBridgeCollectionCallback
from experiments.jacobian_energy_bridge.dataset import (
    SafetyBridgeTrajectoryWriter,
    concatenate_bridge_datasets,
    load_bridge_dataset,
)
from experiments.jacobian_energy_bridge.energy_model import (
    ActionConditionedEnergyCritic,
    CalibratedCompactEnergyEstimator,
    FlexibleEnergyRegressor,
)
from review_bundle.safety.energy.mc_regression import (
    HierarchicalConformalCalibration,
    MissionConformalCalibration,
)
from experiments.jacobian_energy_bridge.losses import (
    local_projected_action,
    masked_energy_bridge_loss,
    shield_consistency_loss,
)
from experiments.jacobian_energy_bridge.safety_buffer import (
    SafetyBridgeReplay,
    projection_context,
)
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.run_jacobian_safety_energy_1m import (
    FORMAL_TOTAL_TRANSITIONS,
    collect_mission_budget,
    disjoint_bridge_and_energy_data,
    formal_config,
    load_intermediate_phase1,
    make_bridge_replay,
    parse_args,
    prepare_output_directory,
    split_by_mission_units,
    split_goal_trajectories,
)
from scripts.evaluate_jseb_checkpoints import (
    checkpoint_transition,
    discover_checkpoints,
    migrate_legacy_jseb_command,
)
from scripts.train_uav_energy_delivery_sac import HeuristicGoalPolicy


def _geometry(jacobian: np.ndarray | None = None) -> dict[str, object]:
    matrix = np.eye(3, dtype=np.float32) if jacobian is None else jacobian
    return {
        "jacobian_total": matrix.tolist(),
        "jacobian_barrier": matrix.tolist(),
        "jacobian_physical_total": matrix.tolist(),
        "jacobian_physical_barrier": matrix.tolist(),
        "singular_values": np.linalg.svd(matrix, compute_uv=False).tolist(),
        "rank": int(np.linalg.matrix_rank(matrix)),
        "action_authority": float(np.trace(matrix) / 3.0),
        "blocked_fraction": float(1.0 - np.trace(matrix) / 3.0),
        "normal_action_fraction": 0.25,
        "active_barrier_constraints": int(np.linalg.matrix_rank(np.eye(3) - matrix)),
        "active_physical_constraints": 0,
        "active_constraint_indices": [],
        "nominal_safe": False,
        "minimum_nominal_slack": -0.5,
        "minimum_executed_slack": 0.2,
        "valid": True,
        "active_set_stable": True,
        "coordinate_map_stable": True,
        "reason": "test",
    }


def test_legacy_jseb_phase_budgets_migrate_to_unified_energy_budget() -> None:
    migrated, audit = migrate_legacy_jseb_command(
        [
            "--output-dir",
            "/tmp/output",
            "--phase2a-transitions",
            "100000",
            "--phase2b-transitions",
            "300000",
            "--phase2c-transitions",
            "100000",
            "--phase-end-eval-only",
        ]
    )
    assert "--phase2a-transitions" not in migrated
    assert migrated[-2:] == ["--phase2-energy-transitions", "500000"]
    assert audit["legacy_phase2_budget_migrated"] is True
    assert audit["unified_phase2_energy_transitions"] == 500000


def test_legacy_jseb_phase_budget_migration_rejects_partial_set() -> None:
    with pytest.raises(ValueError, match="incomplete legacy phase2 budget set"):
        migrate_legacy_jseb_command(
            ["--phase2a-transitions", "100000", "--phase2b-transitions", "300000"]
        )


def test_calibrated_compact_estimator_checkpoint_round_trip(tmp_path: Path) -> None:
    point = FlexibleEnergyRegressor(7, hidden_dim=8, energy_scale=20.0)
    goal = HierarchicalConformalCalibration.fit(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0, 1, 2]),
        np.asarray(["TASK", "TASK", "CHARGER"]),
        np.asarray(["100-500", "500-1500", "100-500"]),
        coverage=0.90,
    )
    mission = MissionConformalCalibration.fit(
        np.asarray([1.0, 2.0]),
        np.asarray(["100-500", "500-1500"]),
        coverage=0.90,
    )
    estimator = CalibratedCompactEnergyEstimator(
        point,
        goal,
        mission,
        device="cpu",
    )
    checkpoint = tmp_path / "estimator.pt"
    estimator.save(checkpoint)
    restored = CalibratedCompactEnergyEstimator.load(checkpoint, device="cpu")
    state = np.zeros(7, dtype=np.float32)
    assert restored.point_model.predict(state, device="cpu") == pytest.approx(
        estimator.point_model.predict(state, device="cpu")
    )
    assert restored.coverage == pytest.approx(0.90)
    assert restored.mission_coverage == pytest.approx(0.90)


def _info(*, terminal: bool = False, jacobian: np.ndarray | None = None) -> dict[str, object]:
    return {
        "anchor_sac_observation": np.linspace(-1.0, 1.0, 7, dtype=np.float32),
        "anchor_compact_energy_state": np.linspace(-0.5, 0.5, 7, dtype=np.float32),
        "nominal_action": np.array([-0.4, 0.2, 0.1], dtype=np.float32),
        "anchor_executed_action": np.array([0.0, 0.2, 0.1], dtype=np.float32),
        "projection_geometry": _geometry(jacobian),
        "hocbf_intervention_norm": 0.4,
        "hocbf_emergency_brake": False,
        "hocbf_fallback_used": False,
        "realized_acceleration": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "realized_energy_cost": 0.25,
        "transition_dt": 0.2,
        "goal_type": "TASK",
        "goal_initial_distance": 1000.0,
        "anchor_position": np.array([100.0, 200.0, 30.0], dtype=np.float32),
        "anchor_goal": np.array([1000.0, 200.0, 30.0], dtype=np.float32),
        "task_completed_now": terminal,
        "charger_reached_now": False,
        "energy_exhausted": False,
        "task_stuck": False,
        "navigation_failure": False,
        "switched_now": False,
    }


def test_environment_exposes_actor_and_compact_energy_anchors() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.NAVIGATION,
        lidar_enabled=True,
        lidar_horizontal_sectors=4,
        lidar_vertical_sectors=2,
        num_obstacles=0,
        cbf_enabled=True,
    )
    environment.reset(seed=4)
    _, _, _, _, info = environment.step(np.zeros(3, dtype=np.float32))
    assert np.asarray(info["anchor_sac_observation"]).shape == (23,)
    assert np.asarray(info["anchor_compact_energy_state"]).shape == (7,)
    assert np.asarray(info["projection_geometry"]["jacobian_total"]).shape == (3, 3)
    environment.close()


def test_projection_context_is_compact_and_finite() -> None:
    context = projection_context(
        _geometry(np.diag([0.0, 1.0, 1.0])),
        intervention_norm=0.5,
        emergency=False,
    )
    assert context.shape == (11,)
    assert np.all(np.isfinite(context))
    assert np.isclose(context[0], 2.0 / 3.0)


def test_local_projection_has_exact_anchor_and_jacobian_gradient() -> None:
    actor = torch.tensor([[0.1, 0.4, -0.2]], requires_grad=True)
    nominal = torch.tensor([[-0.5, 0.2, 0.3]])
    executed = torch.tensor([[0.0, 0.2, 0.3]])
    jacobian = torch.diag(torch.tensor([0.0, 1.0, 1.0])).unsqueeze(0)
    anchor = local_projected_action(nominal, nominal, executed, jacobian)
    torch.testing.assert_close(anchor, executed)
    projected = local_projected_action(actor, nominal, executed, jacobian)
    projected.sum().backward()
    torch.testing.assert_close(actor.grad, torch.tensor([[0.0, 1.0, 1.0]]))


def test_shield_loss_moves_actor_normal_component_toward_projection() -> None:
    actor = torch.tensor([[-0.3, 0.4, 0.1]], requires_grad=True)
    nominal = torch.tensor([[-0.5, 0.2, 0.3]])
    executed = torch.tensor([[0.0, 0.2, 0.3]])
    jacobian = torch.diag(torch.tensor([0.0, 1.0, 1.0])).unsqueeze(0)
    loss, _, mask = shield_consistency_loss(
        actor,
        nominal,
        executed,
        jacobian,
        torch.tensor([True]),
        trust_region_radius=1.0,
    )
    loss.backward()
    assert mask.item()
    assert actor.grad is not None
    assert actor.grad[0, 0] < 0.0
    assert torch.allclose(actor.grad[0, 1:], torch.zeros(2))


def test_energy_gradient_is_transformed_by_projection_jacobian() -> None:
    critic = ActionConditionedEnergyCritic(energy_scale=10.0, hidden_dim=16)
    for parameter in critic.parameters():
        torch.nn.init.constant_(parameter, 0.1)
    critic.freeze()
    actor = torch.tensor([[0.2, 0.3, 0.4]], requires_grad=True)
    nominal = torch.zeros_like(actor)
    executed = torch.zeros_like(actor)
    jacobian = torch.diag(torch.tensor([0.0, 1.0, 1.0])).unsqueeze(0)
    projected = local_projected_action(actor, nominal, executed, jacobian)
    energy = critic.energy(
        torch.zeros((1, 7)),
        torch.zeros((1, 11)),
        projected,
    )
    loss = masked_energy_bridge_loss(energy, torch.tensor([True]), energy_scale=10.0)
    loss.backward()
    assert actor.grad is not None
    assert abs(float(actor.grad[0, 0])) < 1e-8
    assert float(torch.linalg.vector_norm(actor.grad[0, 1:])) > 0.0


def test_replay_and_completed_trajectory_dataset(tmp_path: Path) -> None:
    replay = SafetyBridgeReplay(capacity=8, observation_dim=7, seed=3)
    replay.add_from_info(_info())
    assert len(replay) == 1
    batch = replay.sample(1, "cpu")
    assert batch.observations.shape == (1, 7)
    writer = SafetyBridgeTrajectoryWriter(tmp_path / "dataset", num_envs=1)
    writer.observe(0, _info(), False)
    writer.observe(0, _info(terminal=True), True)
    dataset = load_bridge_dataset(tmp_path / "dataset")
    assert len(dataset) == 2
    np.testing.assert_allclose(dataset.energy_to_go, [0.5, 0.25])
    assert dataset.unique_trajectory_ids == {0}


def test_replay_sample_age_is_exact_after_ring_buffer_wrap() -> None:
    replay = SafetyBridgeReplay(capacity=4, observation_dim=7, seed=17)
    for insertion_id in range(6):
        info = _info()
        observation = np.asarray(info["anchor_sac_observation"], dtype=np.float32).copy()
        observation[0] = float(insertion_id)
        info["anchor_sac_observation"] = observation
        replay.add_from_info(info)
    batch = replay.sample(256, "cpu")
    sampled_insertion_ids = torch.round(batch.observations[:, 0]).to(dtype=torch.int64)
    torch.testing.assert_close(batch.sample_ages, 5 - sampled_insertion_ids)
    assert batch.sample_ages.dtype == torch.int64
    metadata = replay.metadata()
    assert metadata["sample_age_min"] == 0
    assert metadata["sample_age_p50"] == pytest.approx(1.5)
    assert metadata["sample_age_p90"] == pytest.approx(2.7)
    assert metadata["sample_age_p95"] == pytest.approx(2.85)
    assert metadata["sample_age_max"] == 3


def test_invalid_jacobian_is_retained_for_diagnostics_but_masked() -> None:
    info = _info()
    info["projection_geometry"] = {**info["projection_geometry"], "valid": False}
    replay = SafetyBridgeReplay(capacity=4, observation_dim=7, seed=1)
    replay.add_from_info(info)
    batch = replay.sample(1, "cpu")
    assert not batch.valid_masks.item()


def test_jacobian_disabled_preserves_hocbf_closed_loop_action() -> None:
    common = {
        "phase": SACTrainingPhase.NAVIGATION,
        "lidar_enabled": True,
        "lidar_horizontal_sectors": 8,
        "lidar_vertical_sectors": 2,
        "num_obstacles": 4,
        "cbf_enabled": True,
        "hocbf_top_k": 4,
    }
    enabled = UAVEnergyDeliverySACEnv(
        **common,
        projection_geometry_enabled=True,
    )
    disabled = UAVEnergyDeliverySACEnv(
        **common,
        projection_geometry_enabled=False,
    )
    options = {
        "start_position": np.array([1800.0, 1800.0, 200.0], dtype=np.float32),
        "start_velocity": np.array([2.0, -1.0, 0.5], dtype=np.float32),
        "task_point": np.array([2200.0, 2200.0, 240.0], dtype=np.float32),
    }
    observation_enabled, _ = enabled.reset(seed=201, options=options)
    layout = enabled.static_obstacle_layout()
    observation_disabled, _ = disabled.reset(
        seed=201,
        options={**options, "static_obstacles": layout},
    )
    np.testing.assert_allclose(observation_enabled, observation_disabled)
    action = np.array([0.8, -0.4, 0.2], dtype=np.float32)
    result_enabled = enabled.step(action)
    result_disabled = disabled.step(action)
    np.testing.assert_allclose(enabled.agent.pos, disabled.agent.pos, atol=1e-6)
    np.testing.assert_allclose(enabled.agent.vel, disabled.agent.vel, atol=1e-6)
    np.testing.assert_allclose(
        result_enabled[4]["executed_action"],
        result_disabled[4]["executed_action"],
        atol=1e-6,
    )
    assert result_enabled[1:4] == result_disabled[1:4]
    assert result_enabled[4]["projection_geometry"]["valid"]
    assert not result_disabled[4]["projection_geometry"]["valid"]
    assert (
        result_disabled[4]["projection_geometry"]["reason"]
        == "projection_geometry_disabled"
    )
    enabled.close()
    disabled.close()


def test_custom_sac_executes_nonzero_energy_bridge_gradient(tmp_path: Path) -> None:
    def factory() -> UAVEnergyDeliverySACEnv:
        return UAVEnergyDeliverySACEnv(
            phase=SACTrainingPhase.NAVIGATION,
            lidar_enabled=False,
            cbf_enabled=False,
            projection_geometry_enabled=True,
            phase1_episode_max_policy_steps=100,
        )

    vector_environment = DummyVecEnv([factory])
    replay = SafetyBridgeReplay(capacity=64, observation_dim=7, seed=13)
    writer = SafetyBridgeTrajectoryWriter(tmp_path / "trajectory", num_envs=1)
    callback = SafetyBridgeCollectionCallback(
        replay=replay,
        trajectory_writer=writer,
        log_frequency_transitions=100,
    )
    model = JacobianBridgeSAC(
        "MlpPolicy",
        vector_environment,
        device="cpu",
        seed=13,
        learning_starts=0,
        buffer_size=64,
        batch_size=4,
        train_freq=(1, "step"),
        gradient_steps=1,
        shield_loss_weight=0.1,
        energy_loss_weight=0.2,
        bridge_batch_size=4,
        bridge_learning_starts=1,
        bridge_trust_region=10.0,
        verbose=0,
    )
    model.set_bridge_replay(replay)
    energy_critic = ActionConditionedEnergyCritic(
        hidden_dim=16,
        energy_scale=1.0,
    )
    for parameter in energy_critic.parameters():
        torch.nn.init.constant_(parameter, 0.05)
    model.set_energy_bridge(
        energy_critic,
        energy_scale=1.0,
        phase_start_transition=0,
        warmup_transitions=0,
        ramp_transitions=1,
    )
    model.learn(total_timesteps=12, callback=callback, progress_bar=False)
    metrics = model.bridge_training_metrics()
    assert metrics["bridge_gradient_steps"] > 0
    assert metrics["energy_nonzero_steps"] > 0
    assert metrics["mean_energy_loss"] > 0.0
    assert metrics["mean_pretrust_valid_fraction"] >= metrics["mean_posttrust_valid_fraction"]
    assert metrics["mean_sample_age_p95"] >= metrics["mean_sample_age_p50"] >= 0.0
    assert model.last_bridge_metrics["action_delta_p95"] >= model.last_bridge_metrics[
        "action_delta_p50"
    ]
    assert model.last_bridge_metrics["sample_age_p95"] >= model.last_bridge_metrics[
        "sample_age_p50"
    ]
    assert model.num_timesteps == 12
    vector_environment.close()


def test_bridge_training_summary_loads_legacy_counter_schema() -> None:
    summary = JacobianBridgeSAC.summarize_bridge_training(
        {
            "actor_gradient_steps": 10,
            "bridge_gradient_steps": 5,
            "shield_loss_sum": 1.0,
            "shield_nonzero_steps": 4,
            "energy_loss_sum": 0.0,
            "energy_nonzero_steps": 0,
            "valid_fraction_sum": 2.5,
            "energy_weight_sum": 0.0,
        }
    )
    assert summary["mean_valid_fraction"] == pytest.approx(0.5)
    assert summary["mean_pretrust_valid_fraction"] == 0.0
    assert summary["mean_action_delta_p95"] == 0.0
    assert summary["mean_sample_age_p95"] == 0.0


def test_formal_runner_has_exact_static_one_million_transition_contract(
    tmp_path: Path,
) -> None:
    args = parse_args(["--output-dir", str(tmp_path / "formal")])
    total = sum(
        (
            args.phase1_transitions,
            args.phase2_energy_transitions,
        )
    )
    assert total == FORMAL_TOTAL_TRANSITIONS == 1_000_000
    assert args.phase1_transitions == 500_000
    assert args.phase2_energy_transitions == 500_000
    assert args.phase2_transition_budget == 0
    assert args.num_obstacles == 24
    assert (args.lidar_horizontal_sectors, args.lidar_vertical_sectors) == (128, 8)
    assert args.projection_geometry_enabled
    assert args.phase_end_eval_only


def test_formal_runner_rejects_periodic_navigation_evaluation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--output-dir",
                str(tmp_path / "formal"),
                "--no-phase-end-eval-only",
            ]
        )


def test_smoke_uses_frozen_policy_energy_collection_only(tmp_path: Path) -> None:
    args = parse_args(
        ["--output-dir", str(tmp_path / "smoke"), "--smoke", "--allow-dirty"]
    )
    assert args.phase2_energy_transitions == 2_000
    assert args.phase2_transition_budget == 0


def test_battery_calibration_is_deferred_until_after_energy_learning(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["--output-dir", str(tmp_path / "formal")]
    )
    config = formal_config(args)
    stage_order = config["stage_order"]
    assert stage_order.index("phase2_frozen_energy_learning") < stage_order.index(
        "battery_calibration"
    )
    assert config["battery_calibration_protocol"] == {
        "ordered_after_energy_learning": True,
        "training_budget_contribution": 0,
        "uses_td_predictions": False,
        "purpose": "set synthetic-unit battery capacity for downstream switching evaluation",
    }


def test_pilot_keeps_50k_learning_budget_but_bounds_diagnostic_rollouts(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["--output-dir", str(tmp_path / "pilot"), "--pilot", "--allow-dirty"]
    )
    assert (
        args.phase1_transitions
        + args.phase2_energy_transitions
    ) == 50_000
    assert args.eval_navigation_tasks == 10
    assert args.navigation_eval_max_steps == 1_200
    assert args.battery_calibration_tasks == 5
    assert args.battery_validation_runs == 1


def test_formal_mission_split_rejects_single_trajectory_fallback(tmp_path: Path) -> None:
    writer = SafetyBridgeTrajectoryWriter(tmp_path / "dataset", num_envs=1)
    for _ in range(4):
        writer.observe(0, _info(terminal=True), True)
    dataset = load_bridge_dataset(tmp_path / "dataset")
    with pytest.raises(RuntimeError, match="complete TASK-to-CHARGER missions"):
        split_by_mission_units(dataset, [], seed=1)
    splits = split_goal_trajectories(dataset, seed=1)
    assert not (splits["train"].unique_trajectory_ids & splits["test"].unique_trajectory_ids)
    assert not (
        splits["calibration"].unique_trajectory_ids
        & splits["test"].unique_trajectory_ids
    )
    assert not (
        splits["validation"].unique_trajectory_ids
        & splits["calibration"].unique_trajectory_ids
    )


def test_concatenated_real_goal_datasets_reindex_trajectory_ids(tmp_path: Path) -> None:
    writer = SafetyBridgeTrajectoryWriter(tmp_path / "dataset", num_envs=1)
    for _ in range(3):
        writer.observe(0, _info(terminal=True), True)
    dataset = load_bridge_dataset(tmp_path / "dataset")
    combined = concatenate_bridge_datasets(dataset, dataset)
    assert len(combined) == 2 * len(dataset)
    assert combined.unique_trajectory_ids == set(range(6))


def test_budget_stop_without_completed_goal_returns_explicit_empty_dataset(
    tmp_path: Path,
) -> None:
    args = parse_args(
        ["--output-dir", str(tmp_path / "unused"), "--smoke", "--allow-dirty"]
    )
    replay = make_bridge_replay(args)
    dataset, missions, summary = collect_mission_budget(
        HeuristicGoalPolicy(),
        args,
        transition_budget=1,
        output=tmp_path / "collection",
        seed=91,
        replay=replay,
    )
    assert dataset is None
    assert missions == []
    assert summary["actual_training_transitions"] == 1
    assert not summary["completed_trajectory_dataset_available"]


def test_precreated_launcher_output_contract(tmp_path: Path) -> None:
    output = tmp_path / "launch"
    output.mkdir()
    (output / "exact_command.txt").write_text("command\n", encoding="utf-8")
    (output / "git_sha.txt").write_text("sha\n", encoding="utf-8")
    assert prepare_output_directory(output) == output
    (output / "stale_result.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not fresh"):
        prepare_output_directory(output)


def test_formal_launcher_has_valid_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", "scripts/launch_jacobian_safety_energy_1m.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_sweep_discovers_transition_order(tmp_path: Path) -> None:
    root = tmp_path / "artifact" / "phase1_navigation"
    root.mkdir(parents=True)
    for transition in (150_000, 50_000, 100_000):
        (root / f"checkpoint_transition_{transition:06d}.zip").touch()
    discovered = discover_checkpoints(tmp_path / "artifact")
    assert [transition for transition, _ in discovered] == [50_000, 100_000, 150_000]
    assert checkpoint_transition(discovered[-1][1]) == 150_000


def test_intermediate_jseb_parser_requires_exact_500k_downstream_budget() -> None:
    args = parse_args(
        [
            "--output-dir",
            "/tmp/jseb-intermediate",
            "--intermediate-checkpoint-energy-ablation",
            "--resume-phase1-checkpoint",
            "/tmp/checkpoint_transition_100000.zip",
            "--source-phase1-transition",
            "100000",
            "--phase1-transitions",
            "100000",
            "--phase2-energy-transitions",
            "500000",
        ]
    )
    assert args.source_phase1_transition == 100_000
    assert args.phase2_energy_transitions == 500_000
    assert args.phase2_transition_budget == 0
    assert args.evaluation_num_envs == 6


def test_intermediate_jseb_loads_100k_checkpoint_without_phase1_dataset(
    tmp_path: Path,
) -> None:
    environment = UAVEnergyDeliverySACEnv(
        lidar_enabled=True,
        lidar_horizontal_sectors=8,
        lidar_vertical_sectors=2,
        num_obstacles=24,
        cbf_enabled=True,
        hocbf_top_k=16,
    )
    model = JacobianBridgeSAC(
        "MlpPolicy",
        environment,
        device="cpu",
        buffer_size=32,
        learning_starts=32,
    )
    model.num_timesteps = 100_000
    checkpoint = tmp_path / "checkpoint_transition_100000.zip"
    model.save(checkpoint)
    output = tmp_path / "output"
    (output / "phase1_navigation").mkdir(parents=True)
    args = parse_args(
        [
            "--output-dir",
            str(output),
            "--smoke",
            "--device",
            "cpu",
            "--lidar-horizontal-sectors",
            "8",
            "--lidar-vertical-sectors",
            "2",
            "--intermediate-checkpoint-energy-ablation",
            "--resume-phase1-checkpoint",
            str(checkpoint),
            "--source-phase1-transition",
            "100000",
        ]
    )
    resumed, audit = load_intermediate_phase1(args, output)
    assert resumed.num_timesteps == 100_000
    assert audit["phase1_trajectory_dataset_reused"] is False
    assert audit["navigation_energy_ready"] is None
    assert "EXPLORATORY" in audit["claim_status"]
    environment.close()


def test_intermediate_jseb_fresh_bridge_split_has_no_trajectory_overlap(
    tmp_path: Path,
) -> None:
    writer = SafetyBridgeTrajectoryWriter(tmp_path / "dataset", num_envs=1)
    mission_units = []
    for mission_id in range(5):
        ids = []
        for _ in range(2):
            trajectory_id = writer.completed_trajectories
            writer.observe(0, _info(terminal=True), True)
            ids.append(trajectory_id)
        mission_units.append(
            {
                "mission_id": mission_id,
                "task_trajectory_id": ids[0],
                "return_trajectory_id": ids[1],
                "direct_charger_trajectory_id": None,
            }
        )
    dataset = load_bridge_dataset(tmp_path / "dataset")
    bridge, energy, energy_units, audit = disjoint_bridge_and_energy_data(
        dataset,
        mission_units,
        seed=7,
    )
    assert bridge.unique_trajectory_ids.isdisjoint(energy.unique_trajectory_ids)
    assert len(energy_units) >= 3
    assert audit["existing_phase1_500k_data_reused"] is False
