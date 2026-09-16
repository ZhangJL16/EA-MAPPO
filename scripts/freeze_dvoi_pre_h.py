#!/usr/bin/env python3
"""Blindly allocate fresh worlds and materialize the frozen pre-H contract."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
import torch

from experiments.directional_navigation.dvoi_h_branch import DVOIHBranch
from scripts.anchor_identification_contract import (
    fetch_rekor_public_key,
    generate_private_key,
    public_key_from_private,
)
from scripts.validate_identification_contract import (
    SPLIT_HASH_FIELDS,
    SPLITS,
    normalized_splits,
    object_hash,
    population_definition,
    record_hash,
    sampling_contract,
    seal_access_registry,
    split_hash,
)


CONTRACT_ID = "dvoi-h-gate-20260911-v3"
SEED_BASE = 1_133_000_001
ALLOCATION_RNG_SEED = 20_260_911
SPLIT_COUNTS = {
    "SMOKE_DEBUG": 2,
    "H_DEV": 48,
    "H_CONFIRM": 64,
    "P_DEV": 48,
    "P_CONFIRM": 64,
    "R_DEV": 0,
    "R_CONFIRM": 0,
    "E_DEV": 0,
    "E_CONFIRM": 0,
    "METHOD_TRAIN": 96,
    "METHOD_DEV": 48,
    "RISK_REFERENCE": 0,
    "CALIBRATION": 64,
    "FINAL_TEST": 64,
}


def file_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return "sha256:" + digest


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def add_inline(registry: dict[str, Any], value: Any) -> str:
    digest = object_hash(value)
    registry[digest] = {"inline": value}
    return digest


def source_hashes(paths: list[str]) -> dict[str, str]:
    return {path: file_digest(ROOT / path) for path in paths}


def semantic_artifacts(
    registry: dict[str, Any],
    *,
    rekor_public_key: str,
    signer_public_key: str,
) -> dict[str, str]:
    values: dict[str, Any] = {
        "world_generator_content_hash": {
            "schema_version": "dvoi-world-generator-v1",
            "generator": "seeded LockedRecoveryPlant static cylinders + keyed task schedule",
            "num_obstacles": 48,
            "seed_base": SEED_BASE,
            "allocation_rng_seed": ALLOCATION_RNG_SEED,
            "sources": source_hashes(
                [
                    "envs/UAVEnergyDeliverySAC.py",
                    "experiments/directional_navigation/dvoi_h_branch.py",
                    "experiments/directional_navigation/recovery.py",
                    "scripts/freeze_dvoi_pre_h.py",
                ]
            ),
        },
        "world_distribution_spec_hash": {
            "schema_version": "dvoi-world-distribution-v1",
            "population": "finite preallocated fresh seeded worlds",
            "num_obstacles": 48,
            "worlds_without_replacement": sum(SPLIT_COUNTS.values()),
            "split_counts": SPLIT_COUNTS,
            "assignment": "seed list shuffled once before physical realization and before rollout",
        },
        "simulator_version_hash": {
            "schema_version": "dvoi-simulator-version-v1",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "sources": source_hashes(
                [
                    "envs/UAVEnergyDeliverySAC.py",
                    "experiments/directional_navigation/correction_supervision.py",
                    "experiments/directional_navigation/recovery.py",
                ]
            ),
        },
        "environment_contract_hash": {
            "schema_version": "dvoi-h-environment-contract-v2",
            "collision_protocol": file_digest(ROOT / "docs/LOCKED_COLLISION_RECOVERY_PROTOCOL.md"),
            "contact_terminates": False,
            "velocity_zeroed_at_repair": True,
            "raw_boundary_penalty": -1.2,
            "raw_obstacle_penalty": -1.2,
            "repeat_scale_if_previous_step_contact": 0.35,
            "unified_contact_cost_per_policy_step": True,
            "num_obstacles": 48,
            "battery_capacity": 378.72626091628933,
            "return_deadline_steps": 4000,
            "branch_guard_steps": 12000,
            "maximum_prebranch_prefix_steps": 768,
            "base_administrative_clock_steps": 12769,
            "base_administrative_clocks_precede_branch_guard": False,
            "source": file_digest(ROOT / "experiments/directional_navigation/dvoi_h_branch.py"),
        },
        "history_generation_policy_hash": {
            "schema_version": "dvoi-h-history-policy-v1",
            "policy": "frozen deterministic SAC task navigation",
            "return_decisions_before_anchor": "none",
            "start_soc": 1.0,
            "model_lineage": json.loads(
                (ROOT / "artifacts/new_navigation_energy_global_scale_repair_20260908_v1/repair.json").read_text()
            ),
        },
        "eligibility_rule_hash": {
            "schema_version": "dvoi-h-eligibility-v1",
            "scheduled_policy_steps": [256, 768],
            "requires_task_mode": True,
            "requires_not_at_charger": True,
            "minimum_battery_fraction": 0.25,
            "future_outcomes_used": False,
        },
        "anchor_sampling_rule_hash": {
            "schema_version": "dvoi-h-anchor-sampling-v1",
            "candidate_times": [256, 768],
            "include_every_eligible_candidate": True,
            "max_anchors_per_world": 2,
        },
        "within_world_anchor_weighting_hash": {
            "schema_version": "dvoi-within-world-weight-v1",
            "rule": "equal total weight per world; equal weight among eligible anchors in that world",
        },
        "cross_world_weighting_rule_hash": {
            "schema_version": "dvoi-cross-world-weight-v1",
            "rule": "equal weight per assigned physical world",
        },
        "analysis_unit_and_cluster_rule_hash": {
            "schema_version": "dvoi-cluster-rule-v1",
            "independence_unit": "physical world",
            "anchors": "clustered within physical world",
            "resampling_unit": "whole physical world",
        },
        "c_option_semantics_hash": {
            "schema_version": "dvoi-c-option-v2",
            "action": "continue frozen task navigation until exactly one additional task completes, then commit return",
            "internal_recheck": "physical collision recovery every step; no contact termination",
            "return_deadline_steps": 4000,
            "branch_guard_steps": 12000,
            "base_task_and_episode_clocks": "strictly beyond maximum anchor prefix plus branch guard",
        },
        "r_option_semantics_hash": {
            "schema_version": "dvoi-r-option-v1",
            "action": "commit immediately and navigate to known charger",
            "return_deadline_steps": 4000,
            "contact_terminates": False,
        },
        "downstream_policy_semantics_hash": {
            "schema_version": "dvoi-downstream-policy-v1",
            "low_level_policy": "same frozen deterministic SAC in C and R",
            "C_goal": "current task then charger",
            "R_goal": "charger immediately",
        },
        "utility_contract_hash": {
            "schema_version": "dvoi-utility-v1",
            "formula": "task_increment - 2*operational_failure - 0.25*I(any unified contact)",
            "task_increment_cap": 1,
            "operational_failure": "branch does not return before registered terminal/guard",
            "contact_indicator": "collision_count > 0",
            "units": "registered task-equivalent utility",
        },
        "horizon_and_terminal_rule_id": {
            "schema_version": "dvoi-horizon-terminal-v2",
            "C": "one task then return; terminal on return, depletion, 4000-step return deadline, or 12000 branch steps",
            "R": "return; terminal on return, depletion, 4000-step return deadline, or 12000 branch steps",
            "contact_terminal": False,
            "unregistered_terminal_policy": "collector fails closed and run is invalid",
            "base_administrative_clock_steps": 12769,
        },
        "censoring_rule_id": {
            "schema_version": "dvoi-censoring-v1",
            "administrative_censoring": "none below frozen branch guard",
            "branch_guard": "operational_failure and separately reported",
            "missing_anchor": "predecision ineligible, not imputed",
        },
        "legal_history_schema_hash": {
            "schema_version": "dvoi-legal-history-v1",
            "frames": 64,
            "stride_policy_steps": 4,
            "fields": [
                "nav_observation",
                "return_observation",
                "battery",
                "distance_to_charger",
                "task_progress",
                "task_clock",
                "previous_nominal_action",
                "previous_executed_action",
                "previous_realized_acceleration",
                "previous_action_valid",
                "previous_contact",
            ],
            "forbidden_actor_fields": [
                "world_identity",
                "world_seed",
                "obstacle_centers",
                "obstacle_radii",
                "full_map",
                "future_outcome",
                "branch_label",
            ],
        },
        "g_hist_kernel_id_and_hash": {
            "schema_version": "dvoi-g-hist-latest-frame-v1",
            "mapping": "retain only frame 64 (current legal observation/scalars); discard frames 1..63",
            "deterministic": True,
        },
        "exact_tie_rule_hash": {
            "schema_version": "dvoi-tie-rule-v1",
            "argmax_exact_tie_action": "R",
            "operational_indifference_action": "R",
        },
        "full_q_estimator_hash": {
            "schema_version": "dvoi-full-q-estimator-v2",
            "analysis_source": file_digest(ROOT / "scripts/analyze_dvoi_h.py"),
            "model": "torch GRU(input_size=80,hidden_size=64,batch_first=true), final hidden state, Linear(64,2)",
            "action_order": ["R", "C"],
            "projection_seed": 2026091101,
            "projection_rng": "numpy Generator(PCG64); nav matrix then independent return matrix; integers(0,2,int8)*2-1; float32/sqrt(2056)",
            "projection_output_per_nav_or_return": 32,
            "feature_order": "projected nav[32], projected return[32], then legal-history scalar/action fields in schema order; total 80",
            "standardization": "training-fold mean and population std over sample and 64 time axes; float64 reduction to float32; std floor 1e-6",
            "initialization_seed": "2026091103 + 10*fold; final all-HDEV fit uses 2026091203",
            "device": "CPU float32 with torch deterministic algorithms",
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 200,
            "batch_and_order": "one deterministic full batch; frozen split order; no shuffle",
            "loss": "mean over R/C squared errors, equal total weight per world and equal anchor weight within world",
        },
        "garbled_q_estimator_hash": {
            "schema_version": "dvoi-garbled-q-estimator-v2",
            "analysis_source": file_digest(ROOT / "scripts/analyze_dvoi_h.py"),
            "model": "latest frame only; Linear(80,128), ReLU, Linear(128,64), ReLU, Linear(64,2)",
            "action_order": ["R", "C"],
            "projection_and_standardization": "identical fold-specific tensors to full-Q before discarding frames 1..63",
            "initialization_seed": "2026091104 + 10*fold; final all-HDEV fit uses 2026091204",
            "optimizer_batch_order_and_loss": "identical to full-Q",
        },
        "history_representation_hash": {
            "schema_version": "dvoi-history-representation-v2",
            "input": "only dvoi-legal-history-v1 fields",
            "standardization": "fit on training folds only",
            "random_projection": "two sequential PCG64 Rademacher matrices exactly specified in estimator artifacts, generated independently of outcomes",
            "sequence_length": 64,
            "feature_dimension_after_projection": 80,
        },
        "nuisance_models_hash": {
            "schema_version": "dvoi-nuisance-v1",
            "models": "none beyond the two frozen conditional-Q regressors",
        },
        "cross_fit_folds_hash": {
            "schema_version": "dvoi-crossfit-v2",
            "folds": 5,
            "assignment": "big-endian integer of all 32 sha256 bytes for UTF8(world_identity || ':h-crossfit-v1'), modulo 5",
            "grouping": "all anchors/arms from one physical world remain together",
        },
        "hyperparameter_selection_rule_hash": {
            "schema_version": "dvoi-hyperparameter-rule-v2",
            "selection": "no outcome-adaptive architecture or hyperparameter search",
            "fixed_epochs": 200,
            "development_use": "fit frozen estimators; region is identity predicate ALL_ELIGIBLE_REGISTERED_HISTORIES",
            "candidate_regions": ["ALL_ELIGIBLE_REGISTERED_HISTORIES"],
            "region_predicate": "included == true under the frozen eligibility rule",
            "region_selection": "single candidate; no outcomes or fitted values used",
        },
        "calibration_rule_hash": {
            "schema_version": "dvoi-q-calibration-v2",
            "rule": "store paired R/C residuals from five-fold held-out predictions for both full and garbled Q, grouped by world and anchor step; no confirmation refit",
        },
        "inference_and_bootstrap_hash": {
            "schema_version": "dvoi-h-inference-v2",
            "confidence": 0.95,
            "resampling": "10000 deterministic whole-world standard-normal multiplier replicates; PCG64 seed 2026091102; same multiplier couples all anchors/endpoints in a world",
            "bootstrap_seed": 2026091102,
            "selection": "H-DEV selects region; H-CONFIRM interval conditional on frozen region/model",
            "region": "ALL_ELIGIBLE_REGISTERED_HISTORIES fixed without outcome-adaptive selection",
            "point_estimate": "equal mean of within-world anchor means",
            "simultaneous_interval": "studentized max absolute multiplier statistic jointly over DVOI and flip prevalence; zero-SE endpoint is degenerate at point estimate",
            "verdict_boundary": "strict LCB>threshold for GO, strict UCB<threshold for NO_GO",
        },
        "q_nuisance_uncertainty_propagation_hash": {
            "schema_version": "dvoi-q-uncertainty-v2",
            "rule": "each bootstrap replicate resamples HDEV residual worlds with replacement, matches anchor step, adds the coupled full/garbled paired R/C residual vector to each HCONF world before recomputing max, argmax, DVOI, and flips",
            "rng": "same PCG64(2026091102) stream after multiplier draws; sorted HDEV world and frozen anchor order",
        },
        "learned_selection_and_near_tie_nonregularity_hash": {
            "schema_version": "dvoi-nonregularity-v1",
            "rule": "confirmation region/model frozen; max/argmax recomputed in every whole-world perturbation; ties and |advantage|<delta_A map to R/indeterminate exactly as frozen",
        },
        "multiplicity_rule_hash": {
            "schema_version": "dvoi-multiplicity-v1",
            "rule": "H-DEV region search only; H-CONFIRM reports simultaneous value/flip intervals using max-t bootstrap across both endpoints",
        },
        "support_overlap_rule_hash": {
            "schema_version": "dvoi-support-v2",
            "minimum_worlds_in_region": 20,
            "minimum_action_observations_per_world": 1,
            "representation": "final-all-HDEV standardized latest 80-dimensional projected legal frame",
            "distance": "Euclidean nearest HDEV anchor from a different world",
            "maximum_confirmation_projection_distance": "1.1 times numpy-linear H-DEV leave-one-world-out 99th percentile",
            "confirmation_pass": "at least 20 eligible worlds and every HCONF anchor distance at or below threshold",
            "failure": "INCONCLUSIVE",
        },
        "sample_size_semantics": {
            "schema_version": "dvoi-fixed-n-rationale-v1",
            "H_DEV_worlds": 48,
            "H_CONFIRM_worlds": 64,
            "design": "fixed-N development and independent confirmation; no optional stopping",
            "target_power": 0.8,
            "type_i_error": 0.05,
        },
        "split_assignment_rule_hash": {
            "schema_version": "dvoi-split-allocation-v1",
            "seed_base": SEED_BASE,
            "count": sum(SPLIT_COUNTS.values()),
            "shuffle": "numpy PCG64 permutation",
            "shuffle_seed": ALLOCATION_RNG_SEED,
            "split_order": list(SPLITS),
            "split_counts": SPLIT_COUNTS,
            "outcomes_or_geometry_used": False,
        },
        "debug_seed_namespace_hash": {
            "schema_version": "dvoi-debug-namespace-v1",
            "split": "SMOKE_DEBUG",
            "permanently_nonformal": True,
        },
        "alpha_generation_rule_hash": {
            "schema_version": "dvoi-alpha-generation-v1",
            "source": "EXTERNAL",
            "rule": "future operational tolerance artifact must directly state alpha_op and alpha_contact before method freeze",
        },
    }
    hashes = {name: add_inline(registry, value) for name, value in values.items()}

    rekor_key_hash = add_inline(registry, rekor_public_key)
    signer_key_hash = add_inline(registry, signer_public_key)
    service = {
        "schema_version": "dvoi-rekor-anchor-service-contract-v1",
        "service_identity": "rekor.sigstore.dev",
        "service_url": "https://rekor.sigstore.dev",
        "checkpoint_origin_prefix": "rekor.sigstore.dev - ",
        "rekor_public_key_hash": rekor_key_hash,
        "anchor_signer_public_key_hash": signer_key_hash,
    }
    policy = {
        "schema_version": "dvoi-rekor-anchor-verification-policy-v1",
        "mode": "REKOR_V1_HASHEDREKORD_INCLUSION",
        "entry_kind": "hashedrekord",
        "entry_api_version": "0.0.1",
        "artifact_hash_algorithm": "sha256",
        "require_project_signature": True,
        "require_signed_entry_timestamp": True,
        "require_inclusion_proof": True,
        "require_signed_checkpoint": True,
    }
    hashes["anchor_service_contract_hash"] = add_inline(registry, service)
    hashes["anchor_verification_policy_hash"] = add_inline(registry, policy)
    return hashes


def build_contract(output_dir: Path, private_key: Path) -> dict[str, Any]:
    template = json.loads(
        (
            ROOT
            / "docs/decision-relevant-probing-closure-20260911/IDENTIFICATION_CONTRACT_TEMPLATE.json"
        ).read_text()
    )
    registry: dict[str, Any] = {}
    template["content_registry"] = registry
    rekor_key = fetch_rekor_public_key("https://rekor.sigstore.dev")
    signer_key = public_key_from_private(private_key)
    hashes = semantic_artifacts(registry, rekor_public_key=rekor_key, signer_public_key=signer_key)

    all_seeds = np.arange(SEED_BASE, SEED_BASE + sum(SPLIT_COUNTS.values()), dtype=np.int64)
    np.random.default_rng(ALLOCATION_RNG_SEED).shuffle(all_seeds)
    seed_universe = {
        "schema_version": "dvoi-seed-universe-v1",
        "seeds": sorted(int(value) for value in all_seeds),
        "reuse_of_prior_experiment_seeds": False,
    }
    seed_universe_hash = add_inline(registry, seed_universe)

    split_seeds: dict[str, list[int]] = {}
    offset = 0
    for split in SPLITS:
        count = SPLIT_COUNTS[split]
        split_seeds[split] = [int(value) for value in all_seeds[offset : offset + count]]
        offset += count

    worlds_dir = output_dir / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=False)
    master_splits = {split: [] for split in SPLITS}
    world_registry: dict[str, Any] = {}
    env = DVOIHBranch(obstacles=48, guard=12_000)
    try:
        for split in SPLITS:
            for seed in split_seeds[split]:
                env.reset_world(seed)
                realization = env.physical_world_record()
                identity = object_hash(realization)
                relative = Path("worlds") / f"{identity.removeprefix('sha256:')}.json"
                target = output_dir / relative
                target.write_bytes(canonical_bytes(realization))
                target.chmod(0o600)
                if file_digest(target) != identity:
                    raise RuntimeError("canonical world file does not match its realization identity")
                registry[identity] = {"path": relative.as_posix()}
                seed_artifact = {"schema_version": "dvoi-world-seed-v1", "world_seed": seed}
                seed_hash = add_inline(registry, seed_artifact)
                provenance = {
                    "world_generator_content_hash": hashes["world_generator_content_hash"],
                    "world_distribution_spec_hash": hashes["world_distribution_spec_hash"],
                    "simulator_version_hash": hashes["simulator_version_hash"],
                    "environment_contract_hash": hashes["environment_contract_hash"],
                    "world_seed_or_parameters_hash": seed_hash,
                }
                world_registry[identity] = {
                    "physical_world_realization_hash": identity,
                    **provenance,
                    "generation_provenance_hash": object_hash(provenance),
                }
                master_splits[split].append(identity)
    finally:
        env.close()

    pre = template["pre_h"]
    pre.update(
        {
            "status": "FROZEN",
            "contract_id": CONTRACT_ID,
            "world_generator_content_hash": hashes["world_generator_content_hash"],
            "world_distribution_spec_hash": hashes["world_distribution_spec_hash"],
            "simulator_version_hash": hashes["simulator_version_hash"],
            "environment_contract_hash": hashes["environment_contract_hash"],
            "seed_universe_hash": seed_universe_hash,
            "history_generation_policy_hash": hashes["history_generation_policy_hash"],
            "eligibility_rule_id": "scheduled-legal-h-anchor-v1",
            "eligibility_rule_hash": hashes["eligibility_rule_hash"],
            "eligibility_uses_only_predecision_legal_information": True,
            "anchor_sampling_rule_hash": hashes["anchor_sampling_rule_hash"],
            "max_anchors_per_world": 2,
            "within_world_anchor_weighting_hash": hashes["within_world_anchor_weighting_hash"],
            "cross_world_weighting_rule_hash": hashes["cross_world_weighting_rule_hash"],
            "analysis_unit_and_cluster_rule_hash": hashes["analysis_unit_and_cluster_rule_hash"],
            "c_option_semantics_hash": hashes["c_option_semantics_hash"],
            "r_option_semantics_hash": hashes["r_option_semantics_hash"],
            "downstream_policy_semantics_hash": hashes["downstream_policy_semantics_hash"],
            "utility_id": "task-minus-operational-and-contact-cost-v1",
            "utility_contract_hash": hashes["utility_contract_hash"],
            "horizon_and_terminal_rule_id": hashes["horizon_and_terminal_rule_id"],
            "censoring_rule_id": hashes["censoring_rule_id"],
            "legal_history_schema_hash": hashes["legal_history_schema_hash"],
            "g_hist_kernel_id_and_hash": hashes["g_hist_kernel_id_and_hash"],
            "h_garbling_rng_and_coupling_rule_hash": "null_deterministic",
            "exact_tie_rule_hash": hashes["exact_tie_rule_hash"],
            "primary_q_identification_route": "CROSS_FITTED_CONDITIONAL_Q",
            "sensitivity_q_identification_route": None,
            "full_q_estimator_hash": hashes["full_q_estimator_hash"],
            "garbled_q_estimator_hash": hashes["garbled_q_estimator_hash"],
            "history_representation_hash": hashes["history_representation_hash"],
            "nuisance_models_hash": hashes["nuisance_models_hash"],
            "cross_fit_folds_hash": hashes["cross_fit_folds_hash"],
            "hyperparameter_selection_rule_hash": hashes["hyperparameter_selection_rule_hash"],
            "calibration_rule_hash": hashes["calibration_rule_hash"],
            "inference_and_bootstrap_hash": hashes["inference_and_bootstrap_hash"],
            "q_nuisance_uncertainty_propagation_hash": hashes["q_nuisance_uncertainty_propagation_hash"],
            "learned_selection_and_near_tie_nonregularity_hash": hashes[
                "learned_selection_and_near_tie_nonregularity_hash"
            ],
            "inference_sequential_valid_if_applicable": False,
            "multiplicity_rule_hash": hashes["multiplicity_rule_hash"],
            "support_overlap_rule_id": "whole-world-overlap-v1",
            "support_overlap_rule_hash": hashes["support_overlap_rule_hash"],
            "operational_indeterminate_fallback": "RETURN",
            "minimum_advantage_delta_a": 0.25,
            "minimum_pointwise_regret_delta_l_h": 0.25,
            "minimum_mean_dvoi_delta_d_h": 0.05,
            "minimum_flip_prevalence_p_h": 0.05,
            "sample_size_rule_id": "FIXED_N",
            "power_or_precision_target": {
                "mode": "POWER",
                "target_power": 0.8,
                "type_i_error": 0.05,
            },
            "planned_and_max_worlds": {
                "H_DEV": {"planned": SPLIT_COUNTS["H_DEV"], "maximum": SPLIT_COUNTS["H_DEV"]},
                "H_CONFIRM": {
                    "planned": SPLIT_COUNTS["H_CONFIRM"],
                    "maximum": SPLIT_COUNTS["H_CONFIRM"],
                },
            },
            "batch_size": 1,
            "maximum_expansions": 0,
            "stopping_rule_id": "FIXED_N",
            "p_resource_allocation_trigger": {"mode": "ALWAYS_RUN_P"},
            "split_assignment_rule_hash": hashes["split_assignment_rule_hash"],
            "debug_seed_namespace_hash": hashes["debug_seed_namespace_hash"],
            "smoke_debug_worlds_never_reassignable": True,
            "alpha_generation_rule_hash": hashes["alpha_generation_rule_hash"],
            "alpha_source_selection_rule": {"source": "EXTERNAL"},
            "anchor_service_contract_hash": hashes["anchor_service_contract_hash"],
            "anchor_verification_policy_hash": hashes["anchor_verification_policy_hash"],
            "record_freeze_event_id": "freeze-pre-h-v3",
            "master_splits": master_splits,
        }
    )
    pre["sample_size_and_stopping_contract_hash"] = object_hash(sampling_contract(pre))
    for split, field in SPLIT_HASH_FIELDS.items():
        identities = master_splits[split]
        pre[field] = split_hash(identities) if identities else None
    pre["master_world_manifest_hash"] = object_hash(normalized_splits(master_splits))
    pre["smoke_debug_disjointness_audit_hash"] = object_hash(
        {
            "smoke_debug": sorted(master_splits["SMOKE_DEBUG"]),
            "formal": sorted(
                identity
                for split, identities in master_splits.items()
                if split != "SMOKE_DEBUG"
                for identity in identities
            ),
        }
    )
    pre["target_population_definition_hash"] = object_hash(population_definition(pre))
    pre["target_population_id"] = pre["target_population_definition_hash"]
    pre["record_sha256"] = record_hash(pre)

    template["world_registry"] = world_registry
    template["contract_status"] = "ACTIVE_STAGED"
    template["access_registry"] = {
        "mode": "HASH_CHAINED_APPEND_ONLY_V1",
        "chain_genesis": "sha256:" + "0" * 64,
        "head_event_hash": "sha256:" + "0" * 64,
        "event_count": 0,
        "events": [
            {
                "sequence": 1,
                "event_id": pre["record_freeze_event_id"],
                "event_type": "FREEZE",
                "stage": "PRE_H",
                "record_hash": pre["record_sha256"],
            }
        ],
    }
    seal_access_registry(template)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    private_key = args.private_key.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    if not private_key.exists():
        generate_private_key(private_key)
    if private_key.stat().st_mode & 0o077:
        raise PermissionError("project signing key must not be group/world accessible")
    try:
        contract = build_contract(output_dir, private_key)
        atomic_json(output_dir / "identification_contract.json", contract)
        atomic_json(
            output_dir / "freeze_summary.json",
            {
                "schema_version": "dvoi-pre-h-freeze-summary-v1",
                "contract_id": CONTRACT_ID,
                "pre_h_record_hash": contract["pre_h"]["record_sha256"],
                "access_chain_head": contract["access_registry"]["head_event_hash"],
                "split_counts": SPLIT_COUNTS,
                "formal_data_accessed": False,
            },
        )
    except Exception:
        atomic_json(output_dir / "ERROR.json", {"error": __import__("traceback").format_exc()})
        raise
    print(output_dir / "identification_contract.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
