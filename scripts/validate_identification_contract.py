"""Validate staged readiness of a canonical DVOI identification contract.

The validator reads JSON, never Markdown.  It checks schema completeness,
content hashes, split identity disjointness, parent-record links, P-population
inheritance, and freeze-before-access order.  It does not run a simulator or
promote an experiment.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "dvoi-identification-contract-v4"
ACCESS_CHAIN_MODE = "HASH_CHAINED_APPEND_ONLY_V1"
ACCESS_CHAIN_GENESIS = "sha256:" + "0" * 64
ACCESS_ANCHOR_SCHEMA = "dvoi-rekor-access-anchor-receipt-v1"
ANCHOR_SERVICE_SCHEMA = "dvoi-rekor-anchor-service-contract-v1"
ANCHOR_POLICY_SCHEMA = "dvoi-rekor-anchor-verification-policy-v1"
STAGES = (
    "PRE_H_READY",
    "H_CONFIRM_READY",
    "P_ENTRY_READY",
    "P_RESOURCE_TRIGGER_SATISFIED",
    "P_DEV_READY",
    "P_CONFIRM_READY",
    "METHOD_TRAIN_READY",
    "CALIBRATION_READY",
    "FINAL_TEST_READY",
)
RECORD_FOR_STAGE = {
    "PRE_H": "pre_h",
    "H_CONFIRM": "h_confirmation",
    "P_ENTRY": "p_entry",
    "P_CONFIRM": "p_confirmation",
    "METHOD_TRAIN": "method_train",
    "CALIBRATION": "calibration",
    "FINAL_TEST": "final_test",
}
SPLITS = (
    "SMOKE_DEBUG",
    "H_DEV",
    "H_CONFIRM",
    "P_DEV",
    "P_CONFIRM",
    "R_DEV",
    "R_CONFIRM",
    "E_DEV",
    "E_CONFIRM",
    "METHOD_TRAIN",
    "METHOD_DEV",
    "RISK_REFERENCE",
    "CALIBRATION",
    "FINAL_TEST",
)
REQUIRED_NONEMPTY_SPLITS = {
    "SMOKE_DEBUG",
    "H_DEV",
    "H_CONFIRM",
    "P_DEV",
    "P_CONFIRM",
    "METHOD_TRAIN",
    "METHOD_DEV",
    "CALIBRATION",
    "FINAL_TEST",
}
SPLIT_HASH_FIELDS = {
    "SMOKE_DEBUG": "smoke_debug_world_manifest_hash",
    "H_DEV": "h_dev_world_manifest_hash",
    "H_CONFIRM": "h_confirm_world_manifest_hash",
    "P_DEV": "p_dev_world_manifest_hash",
    "P_CONFIRM": "p_confirm_world_manifest_hash",
    "R_DEV": "r_dev_world_manifest_hash",
    "R_CONFIRM": "r_confirm_world_manifest_hash",
    "E_DEV": "e_dev_world_manifest_hash",
    "E_CONFIRM": "e_confirm_world_manifest_hash",
    "METHOD_TRAIN": "method_train_world_manifest_hash",
    "METHOD_DEV": "method_dev_world_manifest_hash",
    "RISK_REFERENCE": "risk_reference_world_manifest_hash",
    "CALIBRATION": "calibration_world_manifest_hash",
    "FINAL_TEST": "final_test_world_manifest_hash",
}
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
VERDICTS = ("GO", "NO_GO", "INCONCLUSIVE")
ALLOWED_P_CLAIMS = {"decision_relevant_acquisition", "committed_reversal"}
VERDICT_FIELDS = {"H_VALUE", "H_FLIP"}
Q_IDENTIFICATION_ROUTES = {
    "EXACT_MODEL_Q",
    "FINITE_CELL_Q",
    "CROSS_FITTED_CONDITIONAL_Q",
}
INDETERMINATE_FALLBACKS = {"RETURN", "ABSTAIN_AND_RETURN"}
SAMPLE_SIZE_RULES = {
    "FIXED_N",
    "EFFECT_BLIND_PRECISION",
    "GROUP_SEQUENTIAL",
    "CONFIDENCE_SEQUENCE",
}
STOPPING_RULES = {"FIXED_N", "EFFECT_BLIND_PRECISION", "GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"}
ALPHA_SOURCES = {"EXTERNAL", "RISK_REFERENCE"}
WORLD_PROVENANCE_FIELDS = (
    "world_generator_content_hash",
    "world_distribution_spec_hash",
    "simulator_version_hash",
    "environment_contract_hash",
    "world_seed_or_parameters_hash",
)


PRE_H_FIELDS = (
    "contract_id",
    "target_population_id",
    "target_population_definition_hash",
    "world_generator_content_hash",
    "world_distribution_spec_hash",
    "simulator_version_hash",
    "environment_contract_hash",
    "seed_universe_hash",
    "history_generation_policy_hash",
    "eligibility_rule_id",
    "eligibility_rule_hash",
    "eligibility_uses_only_predecision_legal_information",
    "anchor_sampling_rule_hash",
    "max_anchors_per_world",
    "within_world_anchor_weighting_hash",
    "cross_world_weighting_rule_hash",
    "analysis_unit_and_cluster_rule_hash",
    "c_option_semantics_hash",
    "r_option_semantics_hash",
    "downstream_policy_semantics_hash",
    "utility_id",
    "utility_contract_hash",
    "horizon_and_terminal_rule_id",
    "censoring_rule_id",
    "legal_history_schema_hash",
    "g_hist_kernel_id_and_hash",
    "h_garbling_rng_and_coupling_rule_hash",
    "exact_tie_rule_hash",
    "primary_q_identification_route",
    "full_q_estimator_hash",
    "garbled_q_estimator_hash",
    "history_representation_hash",
    "nuisance_models_hash",
    "cross_fit_folds_hash",
    "hyperparameter_selection_rule_hash",
    "calibration_rule_hash",
    "inference_and_bootstrap_hash",
    "q_nuisance_uncertainty_propagation_hash",
    "learned_selection_and_near_tie_nonregularity_hash",
    "inference_sequential_valid_if_applicable",
    "multiplicity_rule_hash",
    "support_overlap_rule_id",
    "support_overlap_rule_hash",
    "operational_indeterminate_fallback",
    "minimum_advantage_delta_a",
    "minimum_pointwise_regret_delta_l_h",
    "minimum_mean_dvoi_delta_d_h",
    "minimum_flip_prevalence_p_h",
    "sample_size_rule_id",
    "power_or_precision_target",
    "planned_and_max_worlds",
    "batch_size",
    "maximum_expansions",
    "stopping_rule_id",
    "sample_size_and_stopping_contract_hash",
    "p_resource_allocation_trigger",
    "split_assignment_rule_hash",
    "debug_seed_namespace_hash",
    "smoke_debug_disjointness_audit_hash",
    "smoke_debug_worlds_never_reassignable",
    "alpha_generation_rule_hash",
    "alpha_source_selection_rule",
    "anchor_service_contract_hash",
    "anchor_verification_policy_hash",
    "record_freeze_event_id",
    "master_splits",
)
PRE_H_HASH_FIELDS = (
    "target_population_definition_hash",
    "world_generator_content_hash",
    "world_distribution_spec_hash",
    "simulator_version_hash",
    "environment_contract_hash",
    "seed_universe_hash",
    "history_generation_policy_hash",
    "eligibility_rule_hash",
    "anchor_sampling_rule_hash",
    "within_world_anchor_weighting_hash",
    "cross_world_weighting_rule_hash",
    "analysis_unit_and_cluster_rule_hash",
    "c_option_semantics_hash",
    "r_option_semantics_hash",
    "downstream_policy_semantics_hash",
    "horizon_and_terminal_rule_id",
    "censoring_rule_id",
    "legal_history_schema_hash",
    "g_hist_kernel_id_and_hash",
    "h_garbling_rng_and_coupling_rule_hash",
    "exact_tie_rule_hash",
    "full_q_estimator_hash",
    "garbled_q_estimator_hash",
    "history_representation_hash",
    "nuisance_models_hash",
    "cross_fit_folds_hash",
    "hyperparameter_selection_rule_hash",
    "calibration_rule_hash",
    "inference_and_bootstrap_hash",
    "q_nuisance_uncertainty_propagation_hash",
    "learned_selection_and_near_tie_nonregularity_hash",
    "multiplicity_rule_hash",
    "split_assignment_rule_hash",
    "debug_seed_namespace_hash",
    "smoke_debug_disjointness_audit_hash",
    "alpha_generation_rule_hash",
    "anchor_service_contract_hash",
    "anchor_verification_policy_hash",
    "utility_contract_hash",
    "support_overlap_rule_hash",
    "sample_size_and_stopping_contract_hash",
)
PRE_H_DERIVED_HASH_FIELDS = {
    "target_population_definition_hash",
    "smoke_debug_disjointness_audit_hash",
    "sample_size_and_stopping_contract_hash",
}


def object_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def record_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return object_hash(payload)


def access_event_hash(event: dict[str, Any]) -> str:
    return object_hash({key: value for key, value in event.items() if key != "event_hash"})


def seal_access_registry(contract: dict[str, Any]) -> str:
    """Hash-chain an in-memory registry before sending its head to an external anchor."""
    registry = contract["access_registry"]
    registry["mode"] = ACCESS_CHAIN_MODE
    registry["chain_genesis"] = ACCESS_CHAIN_GENESIS
    previous = ACCESS_CHAIN_GENESIS
    for event in registry["events"]:
        event["previous_event_hash"] = previous
        event["event_hash"] = access_event_hash(event)
        previous = event["event_hash"]
    registry["head_event_hash"] = previous
    registry["event_count"] = len(registry["events"])
    return previous


def access_anchor_statement(contract: dict[str, Any]) -> dict[str, Any]:
    """Canonical statement signed by the project key and submitted to Rekor."""
    registry = contract.get("access_registry", {})
    pre_h = contract.get("pre_h", {})
    return {
        "schema_version": "dvoi-access-anchor-statement-v1",
        "contract_id": pre_h.get("contract_id"),
        "pre_h_record_hash": record_hash(pre_h) if isinstance(pre_h, dict) else None,
        "event_count": registry.get("event_count"),
        "head_event_hash": registry.get("head_event_hash"),
    }


@lru_cache(maxsize=512)
def _openssl_public_key_der(public_key_pem: bytes) -> bytes | None:
    completed = subprocess.run(
        ["openssl", "pkey", "-pubin", "-outform", "DER"],
        input=public_key_pem,
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


@lru_cache(maxsize=2048)
def _openssl_verify_signature(
    public_key_pem: bytes, message: bytes, signature: bytes
) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="dvoi-anchor-verify-") as tmp:
            directory = Path(tmp)
            key_path = directory / "public.pem"
            message_path = directory / "message.bin"
            signature_path = directory / "signature.der"
            key_path.write_bytes(public_key_pem)
            message_path.write_bytes(message)
            signature_path.write_bytes(signature)
            completed = subprocess.run(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-verify",
                    str(key_path),
                    "-signature",
                    str(signature_path),
                    str(message_path),
                ],
                capture_output=True,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def _b64decode(value: Any) -> bytes | None:
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError):
        return None


def _verify_merkle_inclusion(
    body: bytes, log_index: int, tree_size: int, hashes: list[str], root_hex: str
) -> bool:
    if log_index < 0 or tree_size <= 0 or log_index >= tree_size:
        return False
    node = hashlib.sha256(b"\x00" + body).digest()
    fn = log_index
    sn = tree_size - 1
    try:
        for item in hashes:
            sibling = bytes.fromhex(item)
            if len(sibling) != 32:
                return False
            if fn % 2 == 1 or fn == sn:
                node = hashlib.sha256(b"\x01" + sibling + node).digest()
                while fn % 2 == 0 and fn != 0:
                    fn //= 2
                    sn //= 2
            else:
                node = hashlib.sha256(b"\x01" + node + sibling).digest()
            fn //= 2
            sn //= 2
    except (TypeError, ValueError):
        return False
    return fn == 0 and sn == 0 and node.hex() == root_hex


def normalized_splits(splits: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: sorted(values) if isinstance(values, list) and all(isinstance(value, str) for value in values) else []
        for name, values in sorted(splits.items())
    }


def split_hash(values: list[str]) -> str:
    return object_hash(sorted(values))


def population_definition(pre_h: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "world_generator_content_hash",
        "world_distribution_spec_hash",
        "simulator_version_hash",
        "environment_contract_hash",
        "seed_universe_hash",
        "history_generation_policy_hash",
        "eligibility_rule_hash",
        "anchor_sampling_rule_hash",
        "max_anchors_per_world",
        "within_world_anchor_weighting_hash",
        "cross_world_weighting_rule_hash",
        "analysis_unit_and_cluster_rule_hash",
    )
    return {field: pre_h.get(field) for field in fields}


def p_population_definition(p_entry: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "p_parent_population_contract_hash",
        "p_history_generation_policy_hash",
        "p_eligibility_rule_hash",
        "p_anchor_sampling_rule_hash",
        "p_max_anchors_per_world",
        "p_within_world_anchor_weighting_hash",
        "p_cross_world_weighting_rule_hash",
        "p_analysis_unit_and_cluster_rule_hash",
    )
    return {field: p_entry.get(field) for field in fields}


def is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def sampling_contract(record: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        "sample_size_rule_id": record.get(f"{prefix}sample_size_rule_id"),
        "power_or_precision_target": record.get(f"{prefix}power_or_precision_target"),
        "planned_and_max_worlds": record.get(f"{prefix}planned_and_max_worlds"),
        "batch_size": record.get(f"{prefix}batch_size"),
        "maximum_expansions": record.get(f"{prefix}maximum_expansions"),
        "stopping_rule_id": record.get(f"{prefix}stopping_rule_id"),
        "inference_sequential_valid_if_applicable": record.get(
            f"{prefix}inference_sequential_valid_if_applicable"
        ),
    }


def validate_power_or_precision(value: Any, name: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{name} must be a structured object"]
    mode = value.get("mode")
    if mode == "POWER" and set(value) == {"mode", "target_power", "type_i_error"}:
        if all(is_number(value[key]) and 0 < value[key] < 1 for key in ("target_power", "type_i_error")):
            return []
    elif mode == "PRECISION" and set(value) == {"mode", "max_ci_half_width", "coverage"}:
        if (
            is_number(value["max_ci_half_width"])
            and value["max_ci_half_width"] > 0
            and is_number(value["coverage"])
            and 0 < value["coverage"] < 1
        ):
            return []
    return [f"{name} must be a valid POWER or PRECISION contract"]


def validate_sampling_contract(
    record: dict[str, Any],
    name: str,
    prefix: str,
    scopes: tuple[str, ...],
    master_splits: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    sample_rule = record.get(f"{prefix}sample_size_rule_id")
    stopping_rule = record.get(f"{prefix}stopping_rule_id")
    sequential = record.get(f"{prefix}inference_sequential_valid_if_applicable")
    batch_size = record.get(f"{prefix}batch_size")
    maximum_expansions = record.get(f"{prefix}maximum_expansions")
    plans = record.get(f"{prefix}planned_and_max_worlds")
    if sample_rule not in SAMPLE_SIZE_RULES:
        errors.append(f"{name}.{prefix}sample_size_rule_id has an invalid enum value")
    if stopping_rule not in STOPPING_RULES:
        errors.append(f"{name}.{prefix}stopping_rule_id has an invalid enum value")
    if not isinstance(sequential, bool):
        errors.append(f"{name}.{prefix}inference_sequential_valid_if_applicable must be boolean")
    if not is_positive_int(batch_size):
        errors.append(f"{name}.{prefix}batch_size must be a positive integer")
    if not is_nonnegative_int(maximum_expansions):
        errors.append(f"{name}.{prefix}maximum_expansions must be a nonnegative integer")
    errors.extend(
        validate_power_or_precision(
            record.get(f"{prefix}power_or_precision_target"),
            f"{name}.{prefix}power_or_precision_target",
        )
    )
    if not isinstance(plans, dict) or set(plans) != set(scopes):
        errors.append(
            f"{name}.{prefix}planned_and_max_worlds must contain exactly {sorted(scopes)}"
        )
    elif is_positive_int(batch_size) and is_nonnegative_int(maximum_expansions):
        for scope in scopes:
            plan = plans.get(scope)
            if not isinstance(plan, dict) or set(plan) != {"planned", "maximum"}:
                errors.append(f"{name}.{prefix}planned_and_max_worlds.{scope} is invalid")
                continue
            planned, maximum = plan.get("planned"), plan.get("maximum")
            if not is_positive_int(planned) or not is_positive_int(maximum) or planned > maximum:
                errors.append(f"{name}.{prefix}planned_and_max_worlds.{scope} has invalid bounds")
                continue
            if maximum != planned + batch_size * maximum_expansions:
                errors.append(
                    f"{name}.{prefix}planned_and_max_worlds.{scope}.maximum is inconsistent "
                    "with planned + batch_size * maximum_expansions"
                )
            identities = master_splits.get(scope) if isinstance(master_splits, dict) else None
            if not isinstance(identities, list) or len(identities) != maximum:
                errors.append(f"{name}.{prefix}planned_and_max_worlds.{scope} does not match master allocation")
    if sample_rule == "FIXED_N":
        if stopping_rule != "FIXED_N" or maximum_expansions != 0:
            errors.append(f"{name} FIXED_N requires FIXED_N stopping and zero expansions")
    elif sample_rule == "EFFECT_BLIND_PRECISION":
        target = record.get(f"{prefix}power_or_precision_target")
        if stopping_rule != "EFFECT_BLIND_PRECISION":
            errors.append(f"{name} effect-blind precision requires matching stopping")
        if not isinstance(target, dict) or target.get("mode") != "PRECISION":
            errors.append(f"{name} effect-blind precision requires a PRECISION target")
        if is_nonnegative_int(maximum_expansions) and maximum_expansions > 0 and sequential is not True:
            errors.append(
                f"{name} repeated precision stopping requires sequentially valid inference"
            )
    elif sample_rule in {"GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"}:
        if stopping_rule != sample_rule or sequential is not True:
            errors.append(f"{name} sequential sampling requires matching sequentially valid inference")
    expected_hash = object_hash(sampling_contract(record, prefix))
    if record.get(f"{prefix}sample_size_and_stopping_contract_hash") != expected_hash:
        errors.append(f"{name}.{prefix}sample_size_and_stopping_contract_hash does not bind sampling fields")
    return errors


def validate_resource_trigger_rule(value: Any) -> list[str]:
    if value == {"mode": "ALWAYS_RUN_P"}:
        return []
    if not isinstance(value, dict) or set(value) != {"mode", "required_verdicts"}:
        return ["pre_h.p_resource_allocation_trigger must be a structured trigger"]
    required = value.get("required_verdicts")
    if value.get("mode") != "H_VERDICT" or not isinstance(required, dict) or not required:
        return ["pre_h.p_resource_allocation_trigger has an invalid H_VERDICT rule"]
    if not set(required).issubset(VERDICT_FIELDS) or any(
        not isinstance(verdicts, list)
        or not verdicts
        or any(verdict not in VERDICTS for verdict in verdicts)
        for verdicts in required.values()
    ):
        return ["pre_h.p_resource_allocation_trigger has invalid required verdicts"]
    return []


def inline_registered_artifact(contract: dict[str, Any], digest: Any) -> Any:
    registry = contract.get("content_registry")
    entry = registry.get(digest) if isinstance(registry, dict) else None
    return entry.get("inline") if isinstance(entry, dict) and set(entry) == {"inline"} else None


def validate_interval(value: Any, name: str, *, probability: bool = False) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"estimate", "lcb", "ucb"}:
        return [f"{name} must contain exactly estimate/lcb/ucb"]
    estimate, lcb, ucb = value.get("estimate"), value.get("lcb"), value.get("ucb")
    if not all(is_number(item) for item in (estimate, lcb, ucb)):
        return [f"{name} values must be finite numbers"]
    if not lcb <= estimate <= ucb:
        return [f"{name} must satisfy lcb <= estimate <= ucb"]
    if probability and not 0 <= lcb <= estimate <= ucb <= 1:
        return [f"{name} probability interval must lie in [0,1]"]
    return []


def _derived_verdict(interval: dict[str, Any], threshold: Any, valid: bool) -> str:
    if not valid or not is_number(threshold):
        return "INCONCLUSIVE"
    if interval["lcb"] > threshold:
        return "GO"
    if interval["ucb"] < threshold:
        return "NO_GO"
    return "INCONCLUSIVE"


def validate_gate_result_artifact(
    contract: dict[str, Any],
    digest: Any,
    gate: str,
    parent_hash: Any,
    access_prefix_hash: str,
    access_prefix: list[dict[str, Any]],
    sampling_record: dict[str, Any],
    sampling_prefix: str,
    sampling_scope: str,
    value_threshold: Any,
    flip_threshold: Any,
    registered_hashes: set[str],
) -> tuple[dict[str, str] | None, list[str]]:
    prefix = gate.lower()
    verdict_fields = {f"{gate}_VALUE", f"{gate}_FLIP"}
    parent_field = f"parent_{prefix}_confirmation_contract_hash"
    access_field = f"{prefix}_confirm_access_prefix_hash"
    fields = {
        "schema_version",
        parent_field,
        access_field,
        "analysis_artifact_hash",
        "sample_size_contract_hash",
        "worlds_analyzed",
        "support",
        "stopping",
        "inference",
        "value_interval",
        "flip_interval",
    }
    artifact = inline_registered_artifact(contract, digest)
    if not isinstance(artifact, dict) or set(artifact) != fields:
        return None, [f"{gate} confirmation result must be a canonical inline artifact"]
    errors: list[str] = []
    if artifact.get("schema_version") != f"dvoi-{prefix}-confirmation-result-v2":
        errors.append(f"{gate} confirmation result has an invalid schema version")
    if artifact.get(parent_field) != parent_hash:
        errors.append(f"{gate} confirmation result parent hash is invalid")
    if artifact.get(access_field) != access_prefix_hash:
        errors.append(f"{gate} confirmation result access-prefix hash is invalid")
    analysis_hash = artifact.get("analysis_artifact_hash")
    if analysis_hash not in registered_hashes:
        errors.append(f"{gate} confirmation result analysis artifact is not content-backed")
    expected_sampling_hash = sampling_record.get(
        f"{sampling_prefix}sample_size_and_stopping_contract_hash"
    )
    if artifact.get("sample_size_contract_hash") != expected_sampling_hash:
        errors.append(f"{gate} confirmation result does not bind the frozen sampling contract")
    worlds_analyzed = artifact.get("worlds_analyzed")
    accessed_worlds = {
        event.get("world_identity") for event in access_prefix if event.get("world_identity")
    }
    if not is_positive_int(worlds_analyzed) or worlds_analyzed != len(accessed_worlds):
        errors.append(f"{gate} confirmation result world count differs from the access prefix")
    support = artifact.get("support")
    if (
        not isinstance(support, dict)
        or set(support) != {"passed", "evidence_hash"}
        or not isinstance(support.get("passed"), bool)
        or support.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"{gate} confirmation result support record is invalid")
    inference = artifact.get("inference")
    inference_fields = {
        "coverage_valid",
        "nuisance_uncertainty_propagated",
        "nonregularity_covered",
        "multiplicity_controlled",
        "sequential_valid",
        "evidence_hash",
    }
    if (
        not isinstance(inference, dict)
        or set(inference) != inference_fields
        or any(
            not isinstance(inference.get(field), bool)
            for field in inference_fields - {"evidence_hash"}
        )
        or inference.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"{gate} confirmation result inference record is invalid")
        inference = {}
    value_interval = artifact.get("value_interval")
    flip_interval = artifact.get("flip_interval")
    interval_errors = validate_interval(value_interval, f"{gate} value interval")
    interval_errors.extend(
        validate_interval(flip_interval, f"{gate} flip interval", probability=True)
    )
    errors.extend(interval_errors)

    stopping = artifact.get("stopping")
    if (
        not isinstance(stopping, dict)
        or set(stopping) != {"rule_id", "reason", "evidence_hash"}
        or stopping.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"{gate} confirmation result stopping record is invalid")
        stopping = {}
    rule_id = sampling_record.get(f"{sampling_prefix}stopping_rule_id")
    plans = sampling_record.get(f"{sampling_prefix}planned_and_max_worlds")
    plan = plans.get(sampling_scope) if isinstance(plans, dict) else None
    planned = plan.get("planned") if isinstance(plan, dict) else None
    maximum = plan.get("maximum") if isinstance(plan, dict) else None
    batch_size = sampling_record.get(f"{sampling_prefix}batch_size")
    expansion_aligned = (
        is_positive_int(worlds_analyzed)
        and is_positive_int(planned)
        and is_positive_int(maximum)
        and is_positive_int(batch_size)
        and planned <= worlds_analyzed <= maximum
        and (worlds_analyzed - planned) % batch_size == 0
    )
    stop_valid = expansion_aligned and stopping.get("rule_id") == rule_id
    reason = stopping.get("reason")
    if rule_id == "FIXED_N":
        stop_valid = stop_valid and worlds_analyzed == planned and reason == "PLANNED_N_REACHED"
    elif rule_id == "EFFECT_BLIND_PRECISION":
        target = sampling_record.get(f"{sampling_prefix}power_or_precision_target")
        half_width = target.get("max_ci_half_width") if isinstance(target, dict) else None
        precision_met = (
            not interval_errors
            and is_number(half_width)
            and (value_interval["ucb"] - value_interval["lcb"]) / 2 <= half_width
            and (flip_interval["ucb"] - flip_interval["lcb"]) / 2 <= half_width
        )
        stop_valid = stop_valid and (
            (precision_met and reason == "PRECISION_REACHED")
            or (worlds_analyzed == maximum and reason == "MAXIMUM_REACHED")
        )
    elif rule_id in {"GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"}:
        stop_valid = stop_valid and reason == "SEQUENTIAL_RULE_REACHED"
    else:
        stop_valid = False
    if not stop_valid:
        errors.append(f"{gate} confirmation result does not satisfy the frozen stopping rule")

    requires_sequential = (
        rule_id in {"GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"}
        or (
            rule_id == "EFFECT_BLIND_PRECISION"
            and is_positive_int(planned)
            and is_positive_int(worlds_analyzed)
            and worlds_analyzed > planned
        )
    )
    inferential_validity = bool(inference) and all(
        inference.get(field) is True
        for field in (
            "coverage_valid",
            "nuisance_uncertainty_propagated",
            "nonregularity_covered",
            "multiplicity_controlled",
        )
    )
    if requires_sequential:
        inferential_validity = inferential_validity and inference.get("sequential_valid") is True
    support_valid = isinstance(support, dict) and support.get("passed") is True
    validity = (
        not interval_errors
        and support_valid
        and stop_valid
        and inferential_validity
    )
    verdicts = None
    if not interval_errors:
        verdicts = {
            f"{gate}_VALUE": _derived_verdict(value_interval, value_threshold, validity),
            f"{gate}_FLIP": _derived_verdict(flip_interval, flip_threshold, validity),
        }
        if set(verdicts) != verdict_fields:
            errors.append(f"{gate} derived verdict fields are invalid")
    if digest != object_hash(artifact) or digest not in registered_hashes:
        errors.append(f"{gate} confirmation result artifact hash is invalid")
    return verdicts, errors


def validate_component_result_artifact(
    contract: dict[str, Any],
    digest: Any,
    gate: str,
    parent_hash: Any,
    access_prefix: list[dict[str, Any]],
    confirmation: dict[str, Any],
    registered_hashes: set[str],
) -> tuple[str | None, list[str]]:
    fields = {
        "schema_version",
        "gate",
        "parent_confirmation_contract_hash",
        "confirm_access_prefix_hash",
        "analysis_artifact_hash",
        "sample_size_contract_hash",
        "worlds_analyzed",
        "support",
        "stopping",
        "inference",
        "effect_interval",
    }
    artifact = inline_registered_artifact(contract, digest)
    if not isinstance(artifact, dict) or set(artifact) != fields:
        return None, [f"Gate {gate} result must be a canonical inline analysis artifact"]
    errors: list[str] = []
    if artifact.get("schema_version") != "dvoi-component-confirmation-result-v1":
        errors.append(f"Gate {gate} result has an invalid schema version")
    if artifact.get("gate") != gate or artifact.get("parent_confirmation_contract_hash") != parent_hash:
        errors.append(f"Gate {gate} result parent/gate binding is invalid")
    if artifact.get("confirm_access_prefix_hash") != object_hash(access_prefix):
        errors.append(f"Gate {gate} result confirmation access-prefix hash is invalid")
    for field in ("analysis_artifact_hash",):
        if artifact.get(field) not in registered_hashes:
            errors.append(f"Gate {gate} result {field} is not content-backed")
    if artifact.get("sample_size_contract_hash") != confirmation.get(
        "sample_size_and_stopping_contract_hash"
    ):
        errors.append(f"Gate {gate} result does not bind the frozen sampling contract")
    worlds_analyzed = artifact.get("worlds_analyzed")
    accessed_worlds = {
        event.get("world_identity") for event in access_prefix if event.get("world_identity")
    }
    if not is_positive_int(worlds_analyzed) or worlds_analyzed != len(accessed_worlds):
        errors.append(f"Gate {gate} result world count differs from the access prefix")
    support = artifact.get("support")
    if (
        not isinstance(support, dict)
        or set(support) != {"passed", "evidence_hash"}
        or not isinstance(support.get("passed"), bool)
        or support.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"Gate {gate} result support record is invalid")
        support = {}
    inference = artifact.get("inference")
    inference_fields = {
        "coverage_valid",
        "nuisance_uncertainty_propagated",
        "nonregularity_covered",
        "multiplicity_controlled",
        "sequential_valid",
        "evidence_hash",
    }
    if (
        not isinstance(inference, dict)
        or set(inference) != inference_fields
        or any(
            not isinstance(inference.get(field), bool)
            for field in inference_fields - {"evidence_hash"}
        )
        or inference.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"Gate {gate} result inference record is invalid")
        inference = {}
    interval = artifact.get("effect_interval")
    interval_errors = validate_interval(interval, f"Gate {gate} effect interval")
    errors.extend(interval_errors)
    stopping = artifact.get("stopping")
    if (
        not isinstance(stopping, dict)
        or set(stopping) != {"rule_id", "reason", "evidence_hash"}
        or stopping.get("evidence_hash") not in registered_hashes
    ):
        errors.append(f"Gate {gate} result stopping record is invalid")
        stopping = {}
    plans = confirmation.get("planned_and_max_worlds")
    split = f"{gate}_CONFIRM"
    plan = plans.get(split) if isinstance(plans, dict) else None
    planned = plan.get("planned") if isinstance(plan, dict) else None
    maximum = plan.get("maximum") if isinstance(plan, dict) else None
    batch_size = confirmation.get("batch_size")
    rule_id = confirmation.get("stopping_rule_id")
    aligned = (
        is_positive_int(worlds_analyzed)
        and is_positive_int(planned)
        and is_positive_int(maximum)
        and is_positive_int(batch_size)
        and planned <= worlds_analyzed <= maximum
        and (worlds_analyzed - planned) % batch_size == 0
    )
    stop_valid = aligned and stopping.get("rule_id") == rule_id
    reason = stopping.get("reason")
    if rule_id == "FIXED_N":
        stop_valid = stop_valid and worlds_analyzed == planned and reason == "PLANNED_N_REACHED"
    elif rule_id == "EFFECT_BLIND_PRECISION":
        target = confirmation.get("power_or_precision_target")
        width = target.get("max_ci_half_width") if isinstance(target, dict) else None
        precision_met = (
            not interval_errors
            and is_number(width)
            and (interval["ucb"] - interval["lcb"]) / 2 <= width
        )
        stop_valid = stop_valid and (
            (precision_met and reason == "PRECISION_REACHED")
            or (worlds_analyzed == maximum and reason == "MAXIMUM_REACHED")
        )
    elif rule_id in {"GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"}:
        stop_valid = stop_valid and reason == "SEQUENTIAL_RULE_REACHED"
    else:
        stop_valid = False
    if not stop_valid:
        errors.append(f"Gate {gate} result does not satisfy the frozen stopping rule")
    requires_sequential = rule_id in {"GROUP_SEQUENTIAL", "CONFIDENCE_SEQUENCE"} or (
        rule_id == "EFFECT_BLIND_PRECISION"
        and is_positive_int(planned)
        and is_positive_int(worlds_analyzed)
        and worlds_analyzed > planned
    )
    inference_valid = bool(inference) and all(
        inference.get(field) is True
        for field in (
            "coverage_valid",
            "nuisance_uncertainty_propagated",
            "nonregularity_covered",
            "multiplicity_controlled",
        )
    )
    if requires_sequential:
        inference_valid = inference_valid and inference.get("sequential_valid") is True
    validity = (
        not interval_errors
        and support.get("passed") is True
        and stop_valid
        and inference_valid
    )
    verdict = None if interval_errors else _derived_verdict(
        interval, confirmation.get("minimum_effect"), validity
    )
    if digest != object_hash(artifact) or digest not in registered_hashes:
        errors.append(f"Gate {gate} result artifact hash is invalid")
    return verdict, errors


def validate_component_gate(
    gate: str,
    status: Any,
    evidence: Any,
    pre_h: dict[str, Any],
    events: list[dict[str, Any]],
    registered_hashes: set[str],
    content_registry: Any,
    method_freeze_sequence: int | None,
) -> list[str]:
    name = gate.lower()
    if status == "EXCLUDED_FROM_HEADLINE":
        return [] if evidence is None else [f"method_train.gate_{name}_evidence must be null when excluded"]
    if status != "GO":
        return [f"method_train.gate_{name}_status has an invalid enum value"]
    fields = {
        "dev_artifact_hash",
        "dev_access_prefix_hash",
        "confirmation_contract_hash",
        "confirmation_freeze_event_id",
        "confirm_access_prefix_hash",
        "result_artifact_hash",
        "observed_verdict",
    }
    if not isinstance(evidence, dict) or set(evidence) != fields:
        return [f"method_train.gate_{name}_evidence has an invalid GO schema"]
    errors: list[str] = []
    dev_split, confirm_split = f"{gate}_DEV", f"{gate}_CONFIRM"
    splits = pre_h.get("master_splits", {})
    if not isinstance(splits, dict):
        splits = {}
    if not isinstance(splits.get(dev_split), list) or not splits.get(dev_split):
        errors.append(f"{dev_split} must be nonempty when Gate {gate} is GO")
    if not isinstance(splits.get(confirm_split), list) or not splits.get(confirm_split):
        errors.append(f"{confirm_split} must be nonempty when Gate {gate} is GO")
    for field in ("dev_artifact_hash", "confirmation_contract_hash", "result_artifact_hash"):
        digest = evidence.get(field)
        if not isinstance(digest, str) or digest not in registered_hashes:
            errors.append(f"method_train.gate_{name}_evidence.{field} is not content-backed")
    confirmation_hash = evidence.get("confirmation_contract_hash")
    registry_entry = (
        content_registry.get(confirmation_hash) if isinstance(content_registry, dict) else None
    )
    confirmation = registry_entry.get("inline") if isinstance(registry_entry, dict) else None
    confirmation_fields = {
        "schema_version",
        "status",
        "gate",
        "estimand_hash",
        "region_hash",
        "estimator_hash",
        "support_rule_hash",
        "inference_hash",
        "minimum_effect",
        "sample_size_rule_id",
        "power_or_precision_target",
        "planned_and_max_worlds",
        "batch_size",
        "maximum_expansions",
        "stopping_rule_id",
        "inference_sequential_valid_if_applicable",
        "sample_size_and_stopping_contract_hash",
        "confirm_world_manifest_hash",
    }
    if not isinstance(confirmation, dict) or set(confirmation) != confirmation_fields:
        errors.append(f"Gate {gate} confirmation contract must be a canonical inline child record")
    else:
        if (
            confirmation.get("schema_version") != "dvoi-component-confirmation-contract-v1"
            or confirmation.get("status") != "FROZEN"
            or confirmation.get("gate") != gate
        ):
            errors.append(f"Gate {gate} confirmation child status/gate is invalid")
        for field in (
            "estimand_hash",
            "region_hash",
            "estimator_hash",
            "support_rule_hash",
            "inference_hash",
        ):
            if confirmation.get(field) not in registered_hashes:
                errors.append(f"Gate {gate} confirmation child {field} is not content-backed")
        if confirmation.get("confirm_world_manifest_hash") != pre_h.get(
            SPLIT_HASH_FIELDS[confirm_split]
        ):
            errors.append(f"Gate {gate} confirmation child manifest differs from master")
        if not is_number(confirmation.get("minimum_effect")) or confirmation.get("minimum_effect", -1) < 0:
            errors.append(f"Gate {gate} confirmation minimum_effect must be nonnegative")
        errors.extend(
            validate_sampling_contract(
                confirmation,
                f"Gate {gate} confirmation",
                "",
                (confirm_split,),
                splits,
            )
        )
    gate_freeze = matching_freeze_event(
        events,
        confirm_split,
        evidence.get("confirmation_contract_hash"),
        evidence.get("confirmation_freeze_event_id"),
    )
    if gate_freeze is None or not isinstance(gate_freeze.get("sequence"), int):
        errors.append(f"exactly one matching {confirm_split} FREEZE event is required")
    else:
        gate_freeze_sequence = gate_freeze["sequence"]
        dev_prefix = canonical_access_prefix(events, dev_split, gate_freeze_sequence)
        if not dev_prefix:
            errors.append(f"{dev_split} must be accessed before {confirm_split} freeze")
        if evidence.get("dev_access_prefix_hash") != object_hash(dev_prefix):
            errors.append(f"Gate {gate} DEV access-prefix hash is invalid")
        if any(
            event.get("event_type") == "ACCESS"
            and event.get("split") == dev_split
            and isinstance(event.get("sequence"), int)
            and event["sequence"] >= gate_freeze_sequence
            for event in events
        ):
            errors.append(f"all {dev_split} access must precede {confirm_split} freeze")
        if method_freeze_sequence is not None:
            confirm_prefix = canonical_access_prefix(events, confirm_split, method_freeze_sequence)
            if not confirm_prefix:
                errors.append(f"{confirm_split} must be accessed before METHOD_TRAIN freeze")
            if any(event["sequence"] <= gate_freeze_sequence for event in confirm_prefix):
                errors.append(f"all {confirm_split} access must follow its freeze")
            if evidence.get("confirm_access_prefix_hash") != object_hash(confirm_prefix):
                errors.append(f"Gate {gate} CONFIRM access-prefix hash is invalid")
            if any(
                event.get("event_type") == "ACCESS"
                and event.get("split") == confirm_split
                and isinstance(event.get("sequence"), int)
                and event["sequence"] >= method_freeze_sequence
                for event in events
            ):
                errors.append(f"all {confirm_split} access must precede METHOD_TRAIN freeze")
    derived_verdict = None
    if (
        isinstance(confirmation, dict)
        and gate_freeze is not None
        and method_freeze_sequence is not None
    ):
        confirm_prefix = canonical_access_prefix(events, confirm_split, method_freeze_sequence)
        derived_verdict, result_errors = validate_component_result_artifact(
            {"content_registry": content_registry},
            evidence.get("result_artifact_hash"),
            gate,
            evidence.get("confirmation_contract_hash"),
            confirm_prefix,
            confirmation,
            registered_hashes,
        )
        errors.extend(result_errors)
    if evidence.get("observed_verdict") != derived_verdict:
        errors.append(f"Gate {gate} observed verdict differs from validator-derived result")
    if derived_verdict != "GO":
        errors.append(f"Gate {gate} validator-derived verdict must be GO for inclusion")
    return errors


def expected_p_decision_table(p_confirmation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    flip_for_acquisition = p_confirmation.get("p_flip_required_for_acquisition_claim") is True
    flip_for_training = p_confirmation.get("p_flip_required_for_method_training") is True
    for p_value in VERDICTS:
        for p_flip in VERDICTS:
            acquisition = p_value == "GO" and (not flip_for_acquisition or p_flip == "GO")
            reversal = p_flip == "GO"
            training = p_value == "GO" and (not flip_for_training or p_flip == "GO")
            claims = []
            if acquisition:
                claims.append("decision_relevant_acquisition")
            if reversal:
                claims.append("committed_reversal")
            table[f"{p_value}|{p_flip}"] = {
                "continue_method_training": training,
                "allowed_headline_claims": claims,
            }
    return table


def is_placeholder(value: Any) -> bool:
    if value is None or value == "":
        return True
    if isinstance(value, str):
        return value.startswith("TBD_") or value.startswith("MUST_EQUAL")
    return False


def require_fields(
    record: Any, name: str, fields: tuple[str, ...], errors: list[str]
) -> dict[str, Any]:
    if not isinstance(record, dict):
        errors.append(f"{name} must be an object")
        return {}
    if record.get("status") != "FROZEN":
        errors.append(f"{name}.status must be FROZEN")
    for field in fields:
        if field not in record or is_placeholder(record.get(field)):
            errors.append(f"{name}.{field} is missing or unresolved")
    expected = record_hash(record)
    if record.get("record_sha256") != expected:
        errors.append(f"{name}.record_sha256 does not match canonical content")
    return record


def require_hashes(
    record: dict[str, Any], name: str, fields: tuple[str, ...], errors: list[str]
) -> None:
    for field in fields:
        value = record.get(field)
        if field == "h_garbling_rng_and_coupling_rule_hash" and value == "null_deterministic":
            continue
        if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
            errors.append(f"{name}.{field} must be a sha256:<64 lowercase hex> hash")


def validate_content_registry(
    contract: dict[str, Any], base_dir: Path | None
) -> tuple[set[str], list[str]]:
    registry = contract.get("content_registry")
    if not isinstance(registry, dict):
        return set(), ["content_registry must be an object"]
    valid: set[str] = set()
    errors: list[str] = []
    for digest, entry in registry.items():
        if not isinstance(digest, str) or HASH_RE.fullmatch(digest) is None:
            errors.append(f"content_registry key {digest!r} is not a canonical hash")
            continue
        if not isinstance(entry, dict):
            errors.append(f"content_registry[{digest}] must be an object")
            continue
        if set(entry) == {"inline"}:
            observed = object_hash(entry["inline"])
        elif set(entry) == {"path"}:
            if base_dir is None:
                errors.append(f"content_registry[{digest}] path cannot be checked without a base directory")
                continue
            relative = Path(entry["path"]) if isinstance(entry["path"], str) else Path()
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"content_registry[{digest}] path must be relative and may not contain ..")
                continue
            target = base_dir / relative
            if not target.is_file():
                errors.append(f"content_registry[{digest}] path does not exist: {relative}")
                continue
            observed = file_hash(target)
        else:
            errors.append(f"content_registry[{digest}] must contain exactly inline or path")
            continue
        if observed != digest:
            errors.append(f"content_registry[{digest}] content hash mismatch")
            continue
        valid.add(digest)
    return valid, errors


def require_registered_hashes(
    record: dict[str, Any],
    name: str,
    fields: tuple[str, ...],
    registered: set[str],
    errors: list[str],
) -> None:
    for field in fields:
        value = record.get(field)
        if value == "null_deterministic":
            continue
        if isinstance(value, str) and HASH_RE.fullmatch(value) and value not in registered:
            errors.append(f"{name}.{field} is not backed by content_registry")


def validate_registry(
    contract: dict[str, Any], registered_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    registry = contract.get("access_registry")
    if not isinstance(registry, dict):
        return [], ["access_registry must be an object"]
    if registry.get("mode") != ACCESS_CHAIN_MODE:
        errors.append(f"access_registry.mode must be {ACCESS_CHAIN_MODE}")
    if registry.get("chain_genesis") != ACCESS_CHAIN_GENESIS:
        errors.append("access_registry.chain_genesis is invalid")
    events = registry.get("events")
    if not isinstance(events, list):
        return [], errors + ["access_registry.events must be a list"]
    previous = 0
    previous_hash = ACCESS_CHAIN_GENESIS
    event_ids: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"access_registry.events[{index}] must be an object")
            continue
        sequence = event.get("sequence")
        if not is_positive_int(sequence) or sequence <= previous:
            errors.append("access_registry event sequences must be strictly increasing")
        else:
            previous = sequence
        if event.get("event_type") not in {"FREEZE", "ACCESS"}:
            errors.append(f"access_registry.events[{index}] has invalid event_type")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            errors.append(f"access_registry.events[{index}].event_id must be nonempty")
        elif event_id in event_ids:
            errors.append(f"access_registry event_id {event_id!r} is duplicated")
        else:
            event_ids.add(event_id)
        if event.get("event_type") == "ACCESS":
            artifact_hash = event.get("access_artifact_hash")
            if not isinstance(artifact_hash, str) or HASH_RE.fullmatch(artifact_hash) is None:
                errors.append(f"access_registry.events[{index}].access_artifact_hash is invalid")
            elif artifact_hash not in registered_hashes:
                errors.append(f"access_registry.events[{index}].access_artifact_hash is not registered")
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"access_registry.events[{index}] breaks the previous-hash chain")
        expected_event_hash = access_event_hash(event)
        if event.get("event_hash") != expected_event_hash:
            errors.append(f"access_registry.events[{index}].event_hash does not match canonical content")
        previous_hash = expected_event_hash
    if registry.get("event_count") != len(events):
        errors.append("access_registry.event_count does not match events")
    if registry.get("head_event_hash") != previous_hash:
        errors.append("access_registry.head_event_hash does not match the hash-chain head")
    return events, errors


def validate_access_anchor(
    contract: dict[str, Any], receipt: Any, registered_hashes: set[str]
) -> list[str]:
    """Verify a Rekor v1 receipt against pre-H-pinned keys and policy."""
    receipt_fields = {
        "schema_version",
        "anchor_service_contract_hash",
        "anchor_verification_policy_hash",
        "statement",
        "entry_uuid",
        "log_entry",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_fields:
        return ["an authenticated contract-external access-anchor receipt is required"]
    errors: list[str] = []
    if receipt.get("schema_version") != ACCESS_ANCHOR_SCHEMA:
        errors.append("external access-anchor receipt has an invalid schema version")
    pre_h = contract.get("pre_h", {})
    if not isinstance(pre_h, dict):
        return ["pre_h must exist before the access anchor can be verified"]
    service_hash = pre_h.get("anchor_service_contract_hash")
    policy_hash = pre_h.get("anchor_verification_policy_hash")
    if receipt.get("anchor_service_contract_hash") != service_hash:
        errors.append("anchor receipt service contract differs from the pre-H trust root")
    if receipt.get("anchor_verification_policy_hash") != policy_hash:
        errors.append("anchor receipt verification policy differs from pre-H")
    if service_hash not in registered_hashes or policy_hash not in registered_hashes:
        errors.append("pre-H anchor trust contract/policy is not content-backed")
    service = inline_registered_artifact(contract, service_hash)
    policy = inline_registered_artifact(contract, policy_hash)
    service_fields = {
        "schema_version",
        "service_identity",
        "service_url",
        "checkpoint_origin_prefix",
        "rekor_public_key_hash",
        "anchor_signer_public_key_hash",
    }
    policy_fields = {
        "schema_version",
        "mode",
        "entry_kind",
        "entry_api_version",
        "artifact_hash_algorithm",
        "require_project_signature",
        "require_signed_entry_timestamp",
        "require_inclusion_proof",
        "require_signed_checkpoint",
    }
    if not isinstance(service, dict) or set(service) != service_fields:
        errors.append("pre-H anchor service contract has an invalid canonical schema")
        return errors
    if not isinstance(policy, dict) or set(policy) != policy_fields:
        errors.append("pre-H anchor verification policy has an invalid canonical schema")
        return errors
    if service.get("schema_version") != ANCHOR_SERVICE_SCHEMA:
        errors.append("pre-H anchor service contract has an invalid schema version")
    if (
        not isinstance(service.get("service_identity"), str)
        or not service["service_identity"]
        or not isinstance(service.get("service_url"), str)
        or not service["service_url"].startswith("https://")
        or service.get("checkpoint_origin_prefix")
        != f"{service.get('service_identity')} - "
    ):
        errors.append("pre-H anchor service identity/URL/checkpoint origin is invalid")
    expected_policy = {
        "schema_version": ANCHOR_POLICY_SCHEMA,
        "mode": "REKOR_V1_HASHEDREKORD_INCLUSION",
        "entry_kind": "hashedrekord",
        "entry_api_version": "0.0.1",
        "artifact_hash_algorithm": "sha256",
        "require_project_signature": True,
        "require_signed_entry_timestamp": True,
        "require_inclusion_proof": True,
        "require_signed_checkpoint": True,
    }
    if policy != expected_policy:
        errors.append("pre-H anchor verification policy is not the required strict policy")

    rekor_key = inline_registered_artifact(contract, service.get("rekor_public_key_hash"))
    signer_key = inline_registered_artifact(contract, service.get("anchor_signer_public_key_hash"))
    if (
        service.get("rekor_public_key_hash") not in registered_hashes
        or not isinstance(rekor_key, str)
    ):
        errors.append("pre-H Rekor public key is not content-backed")
    if (
        service.get("anchor_signer_public_key_hash") not in registered_hashes
        or not isinstance(signer_key, str)
    ):
        errors.append("pre-H anchor signer public key is not content-backed")
    if errors:
        return errors

    statement = receipt.get("statement")
    if statement != access_anchor_statement(contract):
        errors.append("external access-anchor statement does not match the current registry head")
    entry_uuid = receipt.get("entry_uuid")
    if not isinstance(entry_uuid, str) or re.fullmatch(r"[0-9a-f]{64}([0-9a-f]{16})?", entry_uuid) is None:
        errors.append("external access-anchor entry_uuid is invalid")
    entry = receipt.get("log_entry")
    if not isinstance(entry, dict) or set(entry) != {
        "body",
        "integratedTime",
        "logID",
        "logIndex",
        "verification",
    }:
        errors.append("external access-anchor log entry has an invalid schema")
        return errors
    body_bytes = _b64decode(entry.get("body"))
    if body_bytes is None:
        errors.append("external access-anchor body is not canonical base64")
        return errors
    try:
        body = json.loads(body_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError):
        errors.append("external access-anchor body is not JSON")
        return errors
    if canonical_json_bytes(body) != body_bytes:
        errors.append("external access-anchor body is not canonical JSON")
    body_fields = {"apiVersion", "kind", "spec"}
    if not isinstance(body, dict) or set(body) != body_fields:
        errors.append("external access-anchor body has an invalid hashedrekord schema")
        return errors
    spec = body.get("spec")
    if (
        body.get("apiVersion") != policy.get("entry_api_version")
        or body.get("kind") != policy.get("entry_kind")
        or not isinstance(spec, dict)
        or set(spec) != {"data", "signature"}
    ):
        errors.append("external access-anchor body violates the frozen entry policy")
        return errors
    data = spec.get("data")
    signature_record = spec.get("signature")
    hash_record = data.get("hash") if isinstance(data, dict) and set(data) == {"hash"} else None
    embedded_key = (
        signature_record.get("publicKey")
        if isinstance(signature_record, dict) and set(signature_record) == {"content", "publicKey"}
        else None
    )
    embedded_key_bytes = (
        _b64decode(embedded_key.get("content"))
        if isinstance(embedded_key, dict) and set(embedded_key) == {"content"}
        else None
    )
    project_signature = (
        _b64decode(signature_record.get("content"))
        if isinstance(signature_record, dict)
        else None
    )
    statement_bytes = canonical_json_bytes(statement)
    expected_digest = hashlib.sha256(statement_bytes).hexdigest()
    if hash_record != {"algorithm": "sha256", "value": expected_digest}:
        errors.append("Rekor hashedrekord does not bind the canonical anchor statement")
    if embedded_key_bytes != signer_key.encode("utf-8"):
        errors.append("Rekor hashedrekord signer differs from the pre-H signer trust root")
    if project_signature is None or not _openssl_verify_signature(
        signer_key.encode("utf-8"), statement_bytes, project_signature
    ):
        errors.append("Rekor hashedrekord project signature is invalid")

    rekor_key_bytes = rekor_key.encode("utf-8")
    rekor_der = _openssl_public_key_der(rekor_key_bytes)
    if rekor_der is None:
        errors.append("pre-H Rekor public key cannot be parsed")
        return errors
    expected_log_id = hashlib.sha256(rekor_der).hexdigest()
    if entry.get("logID") != expected_log_id:
        errors.append("Rekor logID differs from the pre-H Rekor public key")
    if not is_nonnegative_int(entry.get("logIndex")) or not is_positive_int(entry.get("integratedTime")):
        errors.append("Rekor logIndex/integratedTime is invalid")
    verification = entry.get("verification")
    if not isinstance(verification, dict) or set(verification) != {
        "inclusionProof",
        "signedEntryTimestamp",
    }:
        errors.append("Rekor verification object is incomplete")
        return errors
    set_signature = _b64decode(verification.get("signedEntryTimestamp"))
    set_payload = {
        key: entry[key] for key in ("body", "integratedTime", "logID", "logIndex")
    }
    if set_signature is None or not _openssl_verify_signature(
        rekor_key_bytes, canonical_json_bytes(set_payload), set_signature
    ):
        errors.append("Rekor signed entry timestamp is invalid")

    proof = verification.get("inclusionProof")
    proof_fields = {"checkpoint", "hashes", "logIndex", "rootHash", "treeSize"}
    if not isinstance(proof, dict) or set(proof) != proof_fields:
        errors.append("Rekor inclusion proof has an invalid schema")
        return errors
    # Rekor's sharded v1 API exposes a global entry logIndex while the proof's
    # logIndex is local to the checkpoint tree named in the signed note.  They
    # are intentionally not required to be equal; the SET authenticates the
    # former, and the RFC6962 path + signed checkpoint authenticate the latter.
    hashes = proof.get("hashes")
    if (
        not isinstance(hashes, list)
        or not all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes)
        or not is_positive_int(proof.get("treeSize"))
        or not isinstance(proof.get("rootHash"), str)
        or re.fullmatch(r"[0-9a-f]{64}", proof["rootHash"]) is None
        or not _verify_merkle_inclusion(
            body_bytes,
            proof.get("logIndex") if is_nonnegative_int(proof.get("logIndex")) else -1,
            proof.get("treeSize") if is_positive_int(proof.get("treeSize")) else 0,
            hashes if isinstance(hashes, list) else [],
            proof.get("rootHash") if isinstance(proof.get("rootHash"), str) else "",
        )
    ):
        errors.append("Rekor Merkle inclusion proof is invalid")

    checkpoint = proof.get("checkpoint")
    try:
        checkpoint_head, signature_block = checkpoint.rsplit("\n\n", 1)
        checkpoint_note = (checkpoint_head + "\n").encode("utf-8")
        checkpoint_lines = checkpoint_head.splitlines()
        signature_lines = signature_block.splitlines()
        origin, size_text, root_b64 = checkpoint_lines
        signature_marker, signature_name, signature_b64 = signature_lines[0].split()
        checkpoint_signature = base64.b64decode(signature_b64, validate=True)
        checkpoint_root = base64.b64decode(root_b64, validate=True).hex()
        checkpoint_size = int(size_text)
    except (AttributeError, ValueError, TypeError):
        errors.append("Rekor signed checkpoint is malformed")
        return errors
    if (
        len(signature_lines) != 1
        or not origin.startswith(service["checkpoint_origin_prefix"])
        or not origin.removeprefix(service["checkpoint_origin_prefix"]).isdigit()
        or signature_marker != "—"
        or signature_name != service["service_identity"]
        or checkpoint_size != proof.get("treeSize")
        or checkpoint_root != proof.get("rootHash")
    ):
        errors.append("Rekor checkpoint does not bind the inclusion-proof root")
    key_hint = hashlib.sha256(rekor_der).digest()[:4]
    if (
        len(checkpoint_signature) <= 4
        or checkpoint_signature[:4] != key_hint
        or not _openssl_verify_signature(
            rekor_key_bytes, checkpoint_note, checkpoint_signature[4:]
        )
    ):
        errors.append("Rekor signed checkpoint signature is invalid")
    return errors


def matching_freeze_event(
    events: list[dict[str, Any]], stage: str, record_digest: Any, event_id: Any
) -> dict[str, Any] | None:
    matches = [
        event
        for event in events
        if event.get("event_type") == "FREEZE"
        and event.get("stage") == stage
        and event.get("record_hash") == record_digest
        and event.get("event_id") == event_id
    ]
    return matches[0] if len(matches) == 1 else None


def canonical_access_prefix(
    events: list[dict[str, Any]], split: str, before_sequence: int
) -> list[dict[str, Any]]:
    return [
        {
            "sequence": event["sequence"],
            "event_id": event["event_id"],
            "split": event["split"],
            "world_identity": event["world_identity"],
            "access_artifact_hash": event["access_artifact_hash"],
        }
        for event in events
        if event.get("event_type") == "ACCESS"
        and event.get("split") == split
        and isinstance(event.get("sequence"), int)
        and event["sequence"] < before_sequence
        and all(
            key in event
            for key in ("event_id", "split", "world_identity", "access_artifact_hash")
        )
    ]


def validate_world_registry(
    contract: dict[str, Any], pre_h: dict[str, Any], registered_hashes: set[str]
) -> list[str]:
    errors: list[str] = []
    registry = contract.get("world_registry")
    if not isinstance(registry, dict):
        return ["world_registry must be an object"]
    master_splits = pre_h.get("master_splits")
    if not isinstance(master_splits, dict):
        master_splits = {}
    assigned = {
        identity
        for identities in master_splits.values()
        if isinstance(identities, list)
        for identity in identities
        if isinstance(identity, str)
    }
    if set(registry) != assigned:
        errors.append("world_registry keys must equal the assigned canonical world identities")
    expected_keys = {
        "physical_world_realization_hash",
        *WORLD_PROVENANCE_FIELDS,
        "generation_provenance_hash",
    }
    for identity in assigned:
        entry = registry.get(identity)
        if not isinstance(entry, dict) or set(entry) != expected_keys:
            errors.append(f"world_registry[{identity!r}] has an invalid schema")
            continue
        if HASH_RE.fullmatch(identity) is None or entry.get("physical_world_realization_hash") != identity:
            errors.append(f"world_registry[{identity!r}] identity must equal its realization hash")
        if identity not in registered_hashes:
            errors.append(f"world_registry[{identity!r}] realization is not content-backed")
        for field in WORLD_PROVENANCE_FIELDS[:-1]:
            if entry.get(field) != pre_h.get(field):
                errors.append(f"world_registry[{identity!r}].{field} differs from pre_h")
        seed_hash = entry.get("world_seed_or_parameters_hash")
        if not isinstance(seed_hash, str) or seed_hash not in registered_hashes:
            errors.append(f"world_registry[{identity!r}] seed/parameters hash is not content-backed")
        provenance = {field: entry.get(field) for field in WORLD_PROVENANCE_FIELDS}
        if entry.get("generation_provenance_hash") != object_hash(provenance):
            errors.append(f"world_registry[{identity!r}] generation provenance hash is invalid")
    return errors


def freeze_order_errors(
    events: list[dict[str, Any]],
    freeze_stage: str,
    record_digest: str,
    freeze_event_id: Any,
    protected_splits: set[str],
    require_prior_access: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    freezes = [
        event
        for event in events
        if event.get("event_type") == "FREEZE"
        and event.get("stage") == freeze_stage
        and event.get("record_hash") == record_digest
        and event.get("event_id") == freeze_event_id
    ]
    if len(freezes) != 1:
        return [f"exactly one matching {freeze_stage} FREEZE event is required"]
    freeze_sequence = freezes[0].get("sequence")
    if not isinstance(freeze_sequence, int):
        return [f"{freeze_stage} FREEZE event has invalid sequence"]
    for event in events:
        if (
            event.get("event_type") == "ACCESS"
            and event.get("split") in protected_splits
            and isinstance(event.get("sequence"), int)
            and event["sequence"] <= freeze_sequence
        ):
            errors.append(f"{event['split']} was accessed before {freeze_stage} freeze")
    if require_prior_access:
        for split in require_prior_access:
            accesses = [
                event
                for event in events
                if event.get("event_type") == "ACCESS" and event.get("split") == split
            ]
            if not accesses:
                errors.append(f"{split} must be accessed before {freeze_stage} freeze")
            elif any(
                isinstance(event.get("sequence"), int)
                and event["sequence"] >= freeze_sequence
                for event in accesses
            ):
                errors.append(f"all {split} access must precede {freeze_stage} freeze")
    return errors


def validate_splits(pre_h: dict[str, Any], events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    splits = pre_h.get("master_splits")
    if not isinstance(splits, dict):
        return ["pre_h.master_splits must be an object"]
    if set(splits) != set(SPLITS):
        errors.append("pre_h.master_splits must contain exactly the registered split names")
    owner: dict[str, str] = {}
    alpha_selection = pre_h.get("alpha_source_selection_rule")
    alpha_source = alpha_selection.get("source") if isinstance(alpha_selection, dict) else None
    required_nonempty = set(REQUIRED_NONEMPTY_SPLITS)
    if alpha_source == "RISK_REFERENCE":
        required_nonempty.add("RISK_REFERENCE")
    for split in SPLITS:
        identities = splits.get(split)
        if not isinstance(identities, list):
            errors.append(f"master split {split} must be a list")
            continue
        if split in required_nonempty and not identities:
            errors.append(f"master split {split} must not be empty")
        if split == "RISK_REFERENCE" and alpha_source == "EXTERNAL" and identities:
            errors.append("master split RISK_REFERENCE must be empty for EXTERNAL alpha")
        string_identities = [identity for identity in identities if isinstance(identity, str)]
        if len(string_identities) != len(set(string_identities)):
            errors.append(f"master split {split} contains duplicate identities")
        for identity in identities:
            if not isinstance(identity, str) or HASH_RE.fullmatch(identity) is None:
                errors.append(f"master split {split} contains a noncanonical world identity")
                continue
            if identity in owner:
                errors.append(f"world identity {identity} appears in {owner[identity]} and {split}")
            owner[identity] = split
        hash_field = SPLIT_HASH_FIELDS[split]
        recorded = pre_h.get(hash_field)
        optional_empty = {"R_DEV", "R_CONFIRM", "E_DEV", "E_CONFIRM"}
        if alpha_source == "EXTERNAL":
            optional_empty.add("RISK_REFERENCE")
        if identities or split not in optional_empty:
            if recorded != split_hash(identities):
                errors.append(f"pre_h.{hash_field} does not match {split} identities")
        elif recorded is not None:
            errors.append(f"pre_h.{hash_field} must be null when optional split {split} is empty")
    if pre_h.get("master_world_manifest_hash") != object_hash(normalized_splits(splits)):
        errors.append("pre_h.master_world_manifest_hash does not match master_splits")
    smoke_audit = object_hash(
        {
            "smoke_debug": sorted(splits.get("SMOKE_DEBUG", [])),
            "formal": sorted(
                identity
                for split, identities in splits.items()
                if split != "SMOKE_DEBUG" and isinstance(identities, list)
                for identity in identities
            ),
        }
    )
    if pre_h.get("smoke_debug_disjointness_audit_hash") != smoke_audit:
        errors.append("pre_h.smoke_debug_disjointness_audit_hash does not match actual identities")
    if pre_h.get("smoke_debug_worlds_never_reassignable") is not True:
        errors.append("pre_h.smoke_debug_worlds_never_reassignable must be true")
    for event in events:
        if event.get("event_type") != "ACCESS":
            continue
        split = event.get("split")
        identity = event.get("world_identity")
        if split not in splits or identity not in splits.get(split, []):
            errors.append(f"ACCESS event references unassigned identity {identity!r} in {split!r}")
    return errors


def validate_contract(
    contract: dict[str, Any],
    base_dir: Path | None = None,
    access_anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    common_errors: list[str] = []
    if contract.get("schema_version") != SCHEMA_VERSION:
        common_errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if contract.get("contract_status") != "ACTIVE_STAGED":
        common_errors.append("contract_status must be ACTIVE_STAGED")
    registered_hashes, content_errors = validate_content_registry(contract, base_dir)
    common_errors.extend(content_errors)
    events, registry_errors = validate_registry(contract, registered_hashes)
    common_errors.extend(registry_errors)
    common_errors.extend(validate_access_anchor(contract, access_anchor, registered_hashes))

    computed: dict[str, str | None] = {}
    for key in RECORD_FOR_STAGE.values():
        record = contract.get(key)
        computed[key] = record_hash(record) if isinstance(record, dict) else None

    errors_by_stage: dict[str, list[str]] = {}

    pre_errors = list(common_errors)
    pre = require_fields(contract.get("pre_h"), "pre_h", PRE_H_FIELDS, pre_errors)
    require_hashes(pre, "pre_h", PRE_H_HASH_FIELDS, pre_errors)
    require_registered_hashes(
        pre,
        "pre_h",
        tuple(field for field in PRE_H_HASH_FIELDS if field not in PRE_H_DERIVED_HASH_FIELDS),
        registered_hashes,
        pre_errors,
    )
    if pre.get("eligibility_uses_only_predecision_legal_information") is not True:
        pre_errors.append("pre_h eligibility must use only predecision legal information")
    if not isinstance(pre.get("max_anchors_per_world"), int) or pre.get("max_anchors_per_world", 0) <= 0:
        pre_errors.append("pre_h.max_anchors_per_world must be a positive integer")
    if isinstance(pre.get("max_anchors_per_world"), bool):
        pre_errors.append("pre_h.max_anchors_per_world may not be boolean")
    for field in (
        "minimum_advantage_delta_a",
        "minimum_pointwise_regret_delta_l_h",
        "minimum_mean_dvoi_delta_d_h",
    ):
        value = pre.get(field)
        if not is_number(value) or value < 0:
            pre_errors.append(f"pre_h.{field} must be a nonnegative number")
    prevalence = pre.get("minimum_flip_prevalence_p_h")
    if not is_number(prevalence) or not 0 <= prevalence <= 1:
        pre_errors.append("pre_h.minimum_flip_prevalence_p_h must be in [0,1]")
    if pre.get("primary_q_identification_route") not in Q_IDENTIFICATION_ROUTES:
        pre_errors.append("pre_h.primary_q_identification_route has an invalid enum value")
    if pre.get("operational_indeterminate_fallback") not in INDETERMINATE_FALLBACKS:
        pre_errors.append("pre_h.operational_indeterminate_fallback has an invalid enum value")
    pre_errors.extend(validate_resource_trigger_rule(pre.get("p_resource_allocation_trigger")))
    alpha_selection = pre.get("alpha_source_selection_rule")
    if (
        not isinstance(alpha_selection, dict)
        or set(alpha_selection) != {"source"}
        or alpha_selection.get("source") not in ALPHA_SOURCES
    ):
        pre_errors.append("pre_h.alpha_source_selection_rule must freeze exactly one valid source")
    pre_errors.extend(
        validate_sampling_contract(
            pre,
            "pre_h",
            "",
            ("H_DEV", "H_CONFIRM"),
            pre.get("master_splits", {}),
        )
    )
    if pre.get("target_population_definition_hash") != object_hash(population_definition(pre)):
        pre_errors.append("pre_h.target_population_definition_hash does not bind the induced history law")
    if pre.get("target_population_id") != pre.get("target_population_definition_hash"):
        pre_errors.append("pre_h.target_population_id must equal its content-addressed definition hash")
    pre_errors.extend(validate_splits(pre, events))
    pre_errors.extend(validate_world_registry(contract, pre, registered_hashes))
    pre_errors.extend(
        freeze_order_errors(
            events,
            "PRE_H",
            record_hash(pre),
            pre.get("record_freeze_event_id"),
            set(SPLITS),
        )
        if pre
        else []
    )
    errors_by_stage["PRE_H_READY"] = pre_errors

    h_errors = [] if not pre_errors else ["dependency PRE_H_READY failed"]
    h_fields = (
        "gate_h_confirmation_contract_id",
        "parent_pre_h_contract_hash",
        "h_dev_artifact_hash",
        "h_dev_access_log_hash",
        "b_h_region_hash",
        "b_h_region_selection_audit_hash",
        "h_fitted_estimator_state_hash",
        "h_fitted_nuisance_state_hash",
        "h_support_calibration_artifact_hash",
        "exact_tie_rule_hash",
        "h_confirm_world_manifest_hash",
        "record_freeze_event_id",
    )
    h = require_fields(contract.get("h_confirmation"), "h_confirmation", h_fields, h_errors)
    require_hashes(h, "h_confirmation", h_fields[1:-1], h_errors)
    require_registered_hashes(
        h,
        "h_confirmation",
        (
            "h_dev_artifact_hash",
            "b_h_region_hash",
            "b_h_region_selection_audit_hash",
            "h_fitted_estimator_state_hash",
            "h_fitted_nuisance_state_hash",
            "h_support_calibration_artifact_hash",
        ),
        registered_hashes,
        h_errors,
    )
    if h.get("parent_pre_h_contract_hash") != record_hash(pre):
        h_errors.append("h_confirmation parent hash does not match pre_h")
    if h.get("exact_tie_rule_hash") != pre.get("exact_tie_rule_hash"):
        h_errors.append("h_confirmation exact tie rule differs from pre_h")
    if h.get("h_confirm_world_manifest_hash") != pre.get("h_confirm_world_manifest_hash"):
        h_errors.append("h_confirmation manifest differs from master preallocation")
    h_freeze = matching_freeze_event(
        events,
        "H_CONFIRM",
        record_hash(h),
        h.get("record_freeze_event_id"),
    )
    if h_freeze is not None and isinstance(h_freeze.get("sequence"), int):
        expected_access_hash = object_hash(
            canonical_access_prefix(events, "H_DEV", h_freeze["sequence"])
        )
        if h.get("h_dev_access_log_hash") != expected_access_hash:
            h_errors.append("h_confirmation.h_dev_access_log_hash does not bind the freeze-time H_DEV prefix")
    h_errors.extend(
        freeze_order_errors(
            events,
            "H_CONFIRM",
            record_hash(h),
            h.get("record_freeze_event_id"),
            {"H_CONFIRM"},
            require_prior_access={"H_DEV"},
        )
        if h
        else []
    )
    errors_by_stage["H_CONFIRM_READY"] = h_errors

    pdev_errors = [] if not pre_errors else ["dependency PRE_H_READY failed"]
    p_entry_fields = (
        "p_entry_contract_id",
        "parent_pre_h_contract_hash",
        "p_population_mode",
        "p_target_population_id",
        "p_target_population_definition_hash",
        "p_parent_population_contract_hash",
        "p_history_generation_policy_hash",
        "p_eligibility_rule_hash",
        "p_eligibility_uses_only_pre_c_legal_information",
        "p_anchor_sampling_rule_hash",
        "p_max_anchors_per_world",
        "p_within_world_anchor_weighting_hash",
        "p_cross_world_weighting_rule_hash",
        "p_analysis_unit_and_cluster_rule_hash",
        "p_dev_world_manifest_hash",
        "p_resource_trigger_evidence",
        "record_freeze_event_id",
    )
    p_entry = require_fields(contract.get("p_entry"), "p_entry", p_entry_fields, pdev_errors)
    require_hashes(
        p_entry,
        "p_entry",
        (
            "parent_pre_h_contract_hash",
            "p_parent_population_contract_hash",
            "p_target_population_definition_hash",
            "p_history_generation_policy_hash",
            "p_eligibility_rule_hash",
            "p_anchor_sampling_rule_hash",
            "p_within_world_anchor_weighting_hash",
            "p_cross_world_weighting_rule_hash",
            "p_analysis_unit_and_cluster_rule_hash",
            "p_dev_world_manifest_hash",
        ),
        pdev_errors,
    )
    require_registered_hashes(
        p_entry,
        "p_entry",
        (
            "p_history_generation_policy_hash",
            "p_eligibility_rule_hash",
            "p_anchor_sampling_rule_hash",
            "p_within_world_anchor_weighting_hash",
            "p_cross_world_weighting_rule_hash",
            "p_analysis_unit_and_cluster_rule_hash",
        ),
        registered_hashes,
        pdev_errors,
    )
    if p_entry.get("parent_pre_h_contract_hash") != record_hash(pre):
        pdev_errors.append("p_entry parent hash does not match pre_h")
    if p_entry.get("p_parent_population_contract_hash") != record_hash(pre):
        pdev_errors.append("p_entry population parent hash does not match pre_h")
    if p_entry.get("p_eligibility_uses_only_pre_c_legal_information") is not True:
        pdev_errors.append("p_entry eligibility must use only pre-C legal information")
    if not isinstance(p_entry.get("p_max_anchors_per_world"), int) or p_entry.get("p_max_anchors_per_world", 0) <= 0:
        pdev_errors.append("p_entry.p_max_anchors_per_world must be a positive integer")
    if p_entry.get("p_dev_world_manifest_hash") != pre.get("p_dev_world_manifest_hash"):
        pdev_errors.append("p_entry P_DEV manifest differs from master preallocation")
    mode = p_entry.get("p_population_mode")
    if mode not in {"INHERIT_PRE_H", "P_SPECIFIC"}:
        pdev_errors.append("p_entry.p_population_mode must be INHERIT_PRE_H or P_SPECIFIC")
    if mode == "INHERIT_PRE_H":
        inheritance = {
            "p_target_population_id": "target_population_id",
            "p_history_generation_policy_hash": "history_generation_policy_hash",
            "p_eligibility_rule_hash": "eligibility_rule_hash",
            "p_anchor_sampling_rule_hash": "anchor_sampling_rule_hash",
            "p_max_anchors_per_world": "max_anchors_per_world",
            "p_within_world_anchor_weighting_hash": "within_world_anchor_weighting_hash",
            "p_cross_world_weighting_rule_hash": "cross_world_weighting_rule_hash",
            "p_analysis_unit_and_cluster_rule_hash": "analysis_unit_and_cluster_rule_hash",
        }
        for child_field, parent_field in inheritance.items():
            if p_entry.get(child_field) != pre.get(parent_field):
                pdev_errors.append(f"p_entry.{child_field} does not inherit pre_h.{parent_field}")
        if p_entry.get("p_target_population_definition_hash") != pre.get("target_population_definition_hash"):
            pdev_errors.append("p_entry target definition hash does not inherit pre_h")
    elif mode == "P_SPECIFIC":
        expected_p_population_hash = object_hash(p_population_definition(p_entry))
        if p_entry.get("p_target_population_definition_hash") != expected_p_population_hash:
            pdev_errors.append("p_entry target definition hash does not bind its P-specific history law")
        if p_entry.get("p_target_population_id") != expected_p_population_hash:
            pdev_errors.append("p_entry target population ID must equal its content-addressed definition hash")
    pdev_errors.extend(
        freeze_order_errors(
            events,
            "P_ENTRY",
            record_hash(p_entry),
            p_entry.get("record_freeze_event_id"),
            {"P_DEV"},
        )
        if p_entry
        else []
    )
    pentry_errors = pdev_errors
    errors_by_stage["P_ENTRY_READY"] = pentry_errors

    trigger_errors = [] if not pentry_errors else ["dependency P_ENTRY_READY failed"]
    trigger_rule = pre.get("p_resource_allocation_trigger")
    trigger_evidence = p_entry.get("p_resource_trigger_evidence")
    if trigger_rule == {"mode": "ALWAYS_RUN_P"}:
        if trigger_evidence != {"mode": "ALWAYS_RUN_P"}:
            trigger_errors.append("p_entry trigger evidence must inherit ALWAYS_RUN_P")
    elif isinstance(trigger_rule, dict) and trigger_rule.get("mode") == "H_VERDICT":
        if not isinstance(trigger_evidence, dict) or set(trigger_evidence) != {
            "mode",
            "h_result_artifact_hash",
        }:
            trigger_errors.append("p_entry H_VERDICT trigger evidence has an invalid schema")
        else:
            result_hash = trigger_evidence.get("h_result_artifact_hash")
            if trigger_evidence.get("mode") != "H_VERDICT":
                trigger_errors.append("p_entry trigger evidence mode must be H_VERDICT")
            if h_errors:
                trigger_errors.append("dependency H_CONFIRM_READY failed for H_VERDICT trigger")
            p_entry_freeze = matching_freeze_event(
                events,
                "P_ENTRY",
                record_hash(p_entry),
                p_entry.get("record_freeze_event_id"),
            )
            if p_entry_freeze is not None and isinstance(p_entry_freeze.get("sequence"), int):
                h_confirm_accesses = canonical_access_prefix(
                    events, "H_CONFIRM", p_entry_freeze["sequence"]
                )
                if not h_confirm_accesses:
                    trigger_errors.append("H_CONFIRM must be accessed before an H_VERDICT P trigger")
                h_access_hash = object_hash(h_confirm_accesses)
                observed, result_errors = validate_gate_result_artifact(
                    contract,
                    result_hash,
                    "H",
                    record_hash(h),
                    h_access_hash,
                    h_confirm_accesses,
                    pre,
                    "",
                    "H_CONFIRM",
                    pre.get("minimum_mean_dvoi_delta_d_h"),
                    pre.get("minimum_flip_prevalence_p_h"),
                    registered_hashes,
                )
                trigger_errors.extend(result_errors)
                if observed is not None:
                    required = trigger_rule.get("required_verdicts", {})
                    if any(
                        observed.get(gate) not in allowed
                        for gate, allowed in required.items()
                    ):
                        trigger_errors.append("frozen P resource trigger is not satisfied")
                if any(
                    event.get("event_type") == "ACCESS"
                    and event.get("split") == "H_CONFIRM"
                    and isinstance(event.get("sequence"), int)
                    and event["sequence"] >= p_entry_freeze["sequence"]
                    for event in events
                ):
                    trigger_errors.append("all H_CONFIRM access must precede P_ENTRY freeze")
    errors_by_stage["P_RESOURCE_TRIGGER_SATISFIED"] = trigger_errors

    pdev_errors = []
    if pentry_errors:
        pdev_errors.append("dependency P_ENTRY_READY failed")
    if trigger_errors:
        pdev_errors.append("dependency P_RESOURCE_TRIGGER_SATISFIED failed")
    errors_by_stage["P_DEV_READY"] = pdev_errors

    pconfirm_errors = [] if not pdev_errors else ["dependency P_DEV_READY failed"]
    pconfirm_fields = (
        "gate_p_contract_id",
        "parent_p_entry_contract_hash",
        "p_dev_artifact_hash",
        "p_dev_access_log_hash",
        "g_post_kernel_hash",
        "g_post_rng_and_coupling_rule_hash",
        "c_prefix_semantics_hash",
        "post_c_handoff_rule_hash",
        "b_p_region_hash",
        "pi_down_architecture_hash",
        "pi_down_parameter_hash",
        "pi_down_training_data_hash",
        "real_garbled_interface_hash",
        "mask_token_and_missingness_semantics_hash",
        "channel_assignment_rule_hash",
        "p_full_and_garbled_estimation_stack_hash",
        "p_support_and_abstention_rule_hash",
        "p_ood_control_suite_hash",
        "r_comparator_channel_invariant_for_primary",
        "r_reference_value_contract_hash",
        "p_minimum_advantage_delta_a",
        "p_minimum_pointwise_value_shift_delta_l",
        "p_minimum_mean_acquisition_delta_d",
        "p_minimum_flip_prevalence",
        "p_sample_size_rule_id",
        "p_power_or_precision_target",
        "p_planned_and_max_worlds",
        "p_batch_size",
        "p_maximum_expansions",
        "p_stopping_rule_id",
        "p_sample_size_and_stopping_contract_hash",
        "p_inference_and_multiplicity_hash",
        "p_q_nuisance_uncertainty_propagation_hash",
        "p_learned_selection_and_near_tie_nonregularity_hash",
        "p_inference_sequential_valid_if_applicable",
        "p_method_promotion_rule_hash",
        "p_training_trigger_rule_hash",
        "p_claim_promotion_rule_hash",
        "p_value_required_for_acquisition_claim",
        "p_flip_required_for_flip_headline",
        "p_flip_required_for_acquisition_claim",
        "p_value_required_for_method_training",
        "p_flip_required_for_method_training",
        "inconclusive_blocks_method_training",
        "inconclusive_blocks_corresponding_claim",
        "p_verdict_decision_table",
        "p_confirm_world_manifest_hash",
        "record_freeze_event_id",
    )
    pconfirm = require_fields(contract.get("p_confirmation"), "p_confirmation", pconfirm_fields, pconfirm_errors)
    p_hash_fields = tuple(
        field
        for field in pconfirm_fields
        if field.endswith("_hash") and field not in {"parent_p_entry_contract_hash"}
    ) + ("parent_p_entry_contract_hash",)
    require_hashes(pconfirm, "p_confirmation", p_hash_fields, pconfirm_errors)
    require_registered_hashes(
        pconfirm,
        "p_confirmation",
        tuple(
            field
            for field in p_hash_fields
            if field
            not in {
                "parent_p_entry_contract_hash",
                "p_dev_access_log_hash",
                "p_method_promotion_rule_hash",
                "p_training_trigger_rule_hash",
                "p_claim_promotion_rule_hash",
                "p_confirm_world_manifest_hash",
                "p_sample_size_and_stopping_contract_hash",
            }
        ),
        registered_hashes,
        pconfirm_errors,
    )
    if pconfirm.get("parent_p_entry_contract_hash") != record_hash(p_entry):
        pconfirm_errors.append("p_confirmation parent hash does not match p_entry")
    if pconfirm.get("p_confirm_world_manifest_hash") != pre.get("p_confirm_world_manifest_hash"):
        pconfirm_errors.append("p_confirmation manifest differs from master preallocation")
    for field in (
        "p_minimum_advantage_delta_a",
        "p_minimum_pointwise_value_shift_delta_l",
        "p_minimum_mean_acquisition_delta_d",
    ):
        value = pconfirm.get(field)
        if not is_number(value) or value < 0:
            pconfirm_errors.append(f"p_confirmation.{field} must be a nonnegative number")
    p_prevalence = pconfirm.get("p_minimum_flip_prevalence")
    if not is_number(p_prevalence) or not 0 <= p_prevalence <= 1:
        pconfirm_errors.append("p_confirmation.p_minimum_flip_prevalence must be in [0,1]")
    pconfirm_errors.extend(
        validate_sampling_contract(
            pconfirm,
            "p_confirmation",
            "p_",
            ("P_CONFIRM",),
            pre.get("master_splits", {}),
        )
    )
    for field in (
        "r_comparator_channel_invariant_for_primary",
        "p_value_required_for_acquisition_claim",
        "p_flip_required_for_flip_headline",
        "p_value_required_for_method_training",
        "inconclusive_blocks_method_training",
        "inconclusive_blocks_corresponding_claim",
    ):
        if pconfirm.get(field) is not True:
            pconfirm_errors.append(f"p_confirmation.{field} must be true")
    if not isinstance(pconfirm.get("p_flip_required_for_acquisition_claim"), bool):
        pconfirm_errors.append("p_confirmation.p_flip_required_for_acquisition_claim must be boolean")
    if not isinstance(pconfirm.get("p_flip_required_for_method_training"), bool):
        pconfirm_errors.append("p_confirmation.p_flip_required_for_method_training must be boolean")
    decision_table = pconfirm.get("p_verdict_decision_table")
    expected_table = expected_p_decision_table(pconfirm)
    if decision_table != expected_table:
        pconfirm_errors.append(
            "p_confirmation.p_verdict_decision_table must cover all 9 verdict "
            "combinations and match its frozen flags"
        )
    else:
        if any(
            not isinstance(row.get("continue_method_training"), bool)
            or not isinstance(row.get("allowed_headline_claims"), list)
            or not set(row["allowed_headline_claims"]).issubset(ALLOWED_P_CLAIMS)
            for row in decision_table.values()
        ):
            pconfirm_errors.append("p_confirmation decision-table rows have invalid types or claims")
    promotion_flags = {
        field: pconfirm.get(field)
        for field in (
            "p_value_required_for_acquisition_claim",
            "p_flip_required_for_flip_headline",
            "p_flip_required_for_acquisition_claim",
            "p_value_required_for_method_training",
            "p_flip_required_for_method_training",
            "inconclusive_blocks_method_training",
            "inconclusive_blocks_corresponding_claim",
        )
    }
    if pconfirm.get("p_training_trigger_rule_hash") != object_hash(
        {key: row["continue_method_training"] for key, row in expected_table.items()}
    ):
        pconfirm_errors.append("p_confirmation.p_training_trigger_rule_hash does not match decision table")
    if pconfirm.get("p_claim_promotion_rule_hash") != object_hash(
        {key: row["allowed_headline_claims"] for key, row in expected_table.items()}
    ):
        pconfirm_errors.append("p_confirmation.p_claim_promotion_rule_hash does not match decision table")
    if pconfirm.get("p_method_promotion_rule_hash") != object_hash(
        {"flags": promotion_flags, "decision_table": expected_table}
    ):
        pconfirm_errors.append("p_confirmation.p_method_promotion_rule_hash does not bind flags and decision table")
    p_freeze = matching_freeze_event(
        events,
        "P_CONFIRM",
        record_hash(pconfirm),
        pconfirm.get("record_freeze_event_id"),
    )
    if p_freeze is not None and isinstance(p_freeze.get("sequence"), int):
        expected_access_hash = object_hash(
            canonical_access_prefix(events, "P_DEV", p_freeze["sequence"])
        )
        if pconfirm.get("p_dev_access_log_hash") != expected_access_hash:
            pconfirm_errors.append("p_confirmation.p_dev_access_log_hash does not bind the freeze-time P_DEV prefix")
    pconfirm_errors.extend(
        freeze_order_errors(
            events,
            "P_CONFIRM",
            record_hash(pconfirm),
            pconfirm.get("record_freeze_event_id"),
            {"P_CONFIRM"},
            require_prior_access={"P_DEV"},
        )
        if pconfirm
        else []
    )
    errors_by_stage["P_CONFIRM_READY"] = pconfirm_errors

    method_errors = [] if not pconfirm_errors else ["dependency P_CONFIRM_READY failed"]
    method_fields = (
        "method_contract_id",
        "parent_p_confirmation_contract_hash",
        "p_confirmation_result_artifact_hash",
        "p_observed_verdicts",
        "p_observed_verdicts_hash",
        "p_training_trigger_decision",
        "p_claim_permissions_hash",
        "numeric_alpha",
        "alpha_source",
        "alpha_derivation_artifact_hash",
        "architecture_hash",
        "baseline_freeze_hash",
        "training_protocol_hash",
        "gate_r_status",
        "gate_e_status",
        "method_train_world_manifest_hash",
        "method_dev_world_manifest_hash",
        "record_freeze_event_id",
    )
    method = require_fields(contract.get("method_train"), "method_train", method_fields, method_errors)
    require_hashes(
        method,
        "method_train",
        (
            "parent_p_confirmation_contract_hash",
            "p_confirmation_result_artifact_hash",
            "p_observed_verdicts_hash",
            "p_claim_permissions_hash",
            "alpha_derivation_artifact_hash",
            "architecture_hash",
            "baseline_freeze_hash",
            "training_protocol_hash",
            "method_train_world_manifest_hash",
            "method_dev_world_manifest_hash",
        ),
        method_errors,
    )
    require_registered_hashes(
        method,
        "method_train",
        (
            "p_confirmation_result_artifact_hash",
            "alpha_derivation_artifact_hash",
            "architecture_hash",
            "baseline_freeze_hash",
            "training_protocol_hash",
        ),
        registered_hashes,
        method_errors,
    )
    if method.get("parent_p_confirmation_contract_hash") != record_hash(pconfirm):
        method_errors.append("method_train parent hash does not match p_confirmation")
    method_freeze = matching_freeze_event(
        events,
        "METHOD_TRAIN",
        record_hash(method),
        method.get("record_freeze_event_id"),
    )
    method_freeze_sequence = (
        method_freeze.get("sequence")
        if method_freeze is not None and isinstance(method_freeze.get("sequence"), int)
        else None
    )
    result_verdicts: dict[str, str] | None = None
    if method_freeze_sequence is not None:
        p_confirm_prefix = canonical_access_prefix(
            events, "P_CONFIRM", method_freeze_sequence
        )
        result_verdicts, result_errors = validate_gate_result_artifact(
            contract,
            method.get("p_confirmation_result_artifact_hash"),
            "P",
            record_hash(pconfirm),
            object_hash(p_confirm_prefix),
            p_confirm_prefix,
            pconfirm,
            "p_",
            "P_CONFIRM",
            pconfirm.get("p_minimum_mean_acquisition_delta_d"),
            pconfirm.get("p_minimum_flip_prevalence"),
            registered_hashes,
        )
        method_errors.extend(result_errors)
    observed = method.get("p_observed_verdicts")
    if not isinstance(observed, dict) or set(observed) != {"P_VALUE", "P_FLIP"} or any(
        value not in VERDICTS for value in observed.values()
    ):
        method_errors.append("method_train.p_observed_verdicts must contain valid P_VALUE/P_FLIP verdicts")
    else:
        if method.get("p_observed_verdicts_hash") != object_hash(observed):
            method_errors.append("method_train.p_observed_verdicts_hash does not match verdicts")
        if result_verdicts != observed:
            method_errors.append("method_train P verdict projection differs from confirmation result artifact")
        row = expected_table[f"{observed['P_VALUE']}|{observed['P_FLIP']}"]
        if method.get("p_training_trigger_decision") != row["continue_method_training"]:
            method_errors.append("method_train trigger decision does not follow frozen P decision table")
        elif method.get("p_training_trigger_decision") is not True:
            method_errors.append("frozen P decision table does not authorize method training")
        if method.get("p_claim_permissions_hash") != object_hash(row["allowed_headline_claims"]):
            method_errors.append("method_train claim permissions do not follow frozen P decision table")
    for nullable_field in (
        "external_tolerance_artifact_hash",
        "risk_reference_artifact_hash",
        "gate_r_evidence",
        "gate_e_evidence",
    ):
        if nullable_field not in method:
            method_errors.append(f"method_train.{nullable_field} must be present")
    alpha = method.get("numeric_alpha")
    if not isinstance(alpha, dict) or set(alpha) != {"alpha_op", "alpha_contact"} or any(
        not is_number(value) or value < 0 or value > 1
        for value in alpha.values()
    ):
        method_errors.append("method_train.numeric_alpha must contain exactly alpha_op/alpha_contact in [0,1]")
    if method.get("alpha_source") not in ALPHA_SOURCES:
        method_errors.append("method_train.alpha_source must be EXTERNAL or RISK_REFERENCE")
    selected_source = pre.get("alpha_source_selection_rule", {}).get("source") if isinstance(
        pre.get("alpha_source_selection_rule"), dict
    ) else None
    if method.get("alpha_source") != selected_source:
        method_errors.append("method_train.alpha_source differs from the pre-H source selection")
    source_hash: Any = None
    if method.get("alpha_source") == "EXTERNAL":
        source_hash = method.get("external_tolerance_artifact_hash")
        if not isinstance(source_hash, str) or source_hash not in registered_hashes:
            method_errors.append("external alpha requires a content-backed tolerance artifact")
        if (
            isinstance(alpha, dict)
            and set(alpha) == {"alpha_op", "alpha_contact"}
            and source_hash != object_hash({"source": "EXTERNAL", "numeric_alpha": alpha})
        ):
            method_errors.append("external tolerance artifact must directly bind numeric alpha")
        if method.get("risk_reference_artifact_hash") is not None:
            method_errors.append("external alpha must not carry a risk-reference artifact")
    elif method.get("alpha_source") == "RISK_REFERENCE":
        source_hash = method.get("risk_reference_artifact_hash")
        if not isinstance(source_hash, str) or source_hash not in registered_hashes:
            method_errors.append("risk-reference alpha requires a content-backed risk artifact")
        if method.get("external_tolerance_artifact_hash") is not None:
            method_errors.append("risk-reference alpha must not carry an external tolerance artifact")
    if isinstance(alpha, dict) and set(alpha) == {"alpha_op", "alpha_contact"}:
        alpha_derivation = {
            "alpha_generation_rule_hash": pre.get("alpha_generation_rule_hash"),
            "alpha_source": method.get("alpha_source"),
            "source_artifact_hash": source_hash,
            "numeric_alpha": alpha,
        }
        expected_alpha_hash = object_hash(alpha_derivation)
        if method.get("alpha_derivation_artifact_hash") != expected_alpha_hash:
            method_errors.append("alpha derivation artifact does not bind source, frozen rule, and numeric alpha")
        elif expected_alpha_hash not in registered_hashes:
            method_errors.append("alpha derivation artifact is not content-backed")
    if method.get("method_train_world_manifest_hash") != pre.get("method_train_world_manifest_hash"):
        method_errors.append("method_train METHOD_TRAIN manifest differs from master")
    if method.get("method_dev_world_manifest_hash") != pre.get("method_dev_world_manifest_hash"):
        method_errors.append("method_train METHOD_DEV manifest differs from master")
    method_errors.extend(
        validate_component_gate(
            "R",
            method.get("gate_r_status"),
            method.get("gate_r_evidence"),
            pre,
            events,
            registered_hashes,
            contract.get("content_registry"),
            method_freeze_sequence,
        )
    )
    method_errors.extend(
        validate_component_gate(
            "E",
            method.get("gate_e_status"),
            method.get("gate_e_evidence"),
            pre,
            events,
            registered_hashes,
            contract.get("content_registry"),
            method_freeze_sequence,
        )
    )
    required_prior_access = {"P_CONFIRM"}
    if method.get("alpha_source") == "RISK_REFERENCE":
        required_prior_access.add("RISK_REFERENCE")
    method_errors.extend(
        freeze_order_errors(
            events,
            "METHOD_TRAIN",
            record_hash(method),
            method.get("record_freeze_event_id"),
            {"METHOD_TRAIN", "METHOD_DEV"},
            require_prior_access=required_prior_access,
        )
        if method
        else []
    )
    errors_by_stage["METHOD_TRAIN_READY"] = method_errors

    calibration_errors = [] if not method_errors else ["dependency METHOD_TRAIN_READY failed"]
    calibration_fields = (
        "calibration_contract_id",
        "parent_method_contract_hash",
        "trained_policy_artifact_hash",
        "calibration_rule_hash",
        "calibration_world_manifest_hash",
        "record_freeze_event_id",
    )
    calibration = require_fields(contract.get("calibration"), "calibration", calibration_fields, calibration_errors)
    require_hashes(calibration, "calibration", calibration_fields[1:-1], calibration_errors)
    require_registered_hashes(
        calibration,
        "calibration",
        ("trained_policy_artifact_hash", "calibration_rule_hash"),
        registered_hashes,
        calibration_errors,
    )
    if calibration.get("parent_method_contract_hash") != record_hash(method):
        calibration_errors.append("calibration parent hash does not match method_train")
    if calibration.get("calibration_world_manifest_hash") != pre.get("calibration_world_manifest_hash"):
        calibration_errors.append("calibration manifest differs from master")
    calibration_errors.extend(
        freeze_order_errors(
            events,
            "CALIBRATION",
            record_hash(calibration),
            calibration.get("record_freeze_event_id"),
            {"CALIBRATION"},
            require_prior_access={"METHOD_TRAIN", "METHOD_DEV"},
        )
        if calibration
        else []
    )
    errors_by_stage["CALIBRATION_READY"] = calibration_errors

    final_errors = [] if not calibration_errors else ["dependency CALIBRATION_READY failed"]
    final_fields = (
        "final_test_contract_id",
        "parent_calibration_contract_hash",
        "selected_operating_points_hash",
        "locked_reporting_rule_hash",
        "final_test_world_manifest_hash",
        "record_freeze_event_id",
    )
    final = require_fields(contract.get("final_test"), "final_test", final_fields, final_errors)
    require_hashes(final, "final_test", final_fields[1:-1], final_errors)
    require_registered_hashes(
        final,
        "final_test",
        ("selected_operating_points_hash", "locked_reporting_rule_hash"),
        registered_hashes,
        final_errors,
    )
    if final.get("parent_calibration_contract_hash") != record_hash(calibration):
        final_errors.append("final_test parent hash does not match calibration")
    if final.get("final_test_world_manifest_hash") != pre.get("final_test_world_manifest_hash"):
        final_errors.append("final_test manifest differs from master")
    final_errors.extend(
        freeze_order_errors(
            events,
            "FINAL_TEST",
            record_hash(final),
            final.get("record_freeze_event_id"),
            {"FINAL_TEST"},
            require_prior_access={"CALIBRATION"},
        )
        if final
        else []
    )
    errors_by_stage["FINAL_TEST_READY"] = final_errors

    readiness = {stage: not errors_by_stage[stage] for stage in STAGES}
    splits = pre.get("master_splits") if isinstance(pre.get("master_splits"), dict) else {}
    world_usage = {}
    for split in SPLITS:
        assigned = splits.get(split, []) if isinstance(splits.get(split, []), list) else []
        accessed = sorted(
            {
                event.get("world_identity")
                for event in events
                if event.get("event_type") == "ACCESS"
                and event.get("split") == split
                and isinstance(event.get("world_identity"), str)
            }
        )
        world_usage[split] = {
            "assigned": len(assigned),
            "accessed": accessed,
            "unused": sorted(set(assigned) - set(accessed)),
        }
    return {
        "schema_version": contract.get("schema_version"),
        "readiness": readiness,
        "errors": errors_by_stage,
        "computed_record_hashes": computed,
        "world_usage": world_usage,
        "overall_ready": all(readiness.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument(
        "--require-stage",
        choices=("ALL",) + STAGES,
        default="ALL",
        help="Exit successfully only when this stage (and its dependencies) is ready.",
    )
    parser.add_argument(
        "--access-anchor-receipt",
        type=Path,
        help=(
            "Contract-external JSON receipt from an immutable anchor service; "
            "required for a ready stage."
        ),
    )
    args = parser.parse_args()
    try:
        raw = json.loads(args.contract.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if not isinstance(raw, dict):
        parser.error("contract root must be a JSON object")
    anchor_receipt = None
    if args.access_anchor_receipt is not None:
        try:
            anchor_receipt = json.loads(args.access_anchor_receipt.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    result = validate_contract(raw, args.contract.parent, anchor_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    ready = result["overall_ready"] if args.require_stage == "ALL" else result["readiness"][args.require_stage]
    if not ready:
        sys.exit(1)


if __name__ == "__main__":
    main()
