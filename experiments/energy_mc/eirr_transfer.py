from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TargetOperatorRadius:
    transient_operator_radius_inf: float
    terminal_vector_radius_inf: float
    target_cell_radii: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ResolventPluginCertificate:
    learned_resolvent_norm_inf: float
    stability_product: float
    learned_value_norm_inf: float
    value_error_bound_inf: float
    learned_value: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InterfaceTransferLowerBound:
    rare_event_probability: float
    source_interface_probability: float
    number_of_source_transitions: int
    source_law_total_variation_upper: float
    target_mgf_separation: float
    minimax_absolute_error_lower: float
    minimax_decision_error_lower: float
    constant_error_regime: bool
    maximum_transitions_for_constant_error: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def bernstein_interface_radii(
    *,
    sample_variances: np.ndarray,
    sample_counts: np.ndarray,
    outcome_upper_bound: float,
    failure_probability: float,
) -> np.ndarray:
    """Two-sided union-Bernstein radii for interface/outcome cells.

    ``sample_variances`` has shape ``(state, executed_action, outcome)`` and
    uses denominator ``count - 1``. The constants implement a two-sided
    empirical Bernstein bound.
    The theorem protocol uses a fixed number of predictable samples from each
    executed interface; arbitrary replay dependence is not certified here.
    """

    variances = np.asarray(sample_variances, dtype=np.float64)
    counts = np.asarray(sample_counts, dtype=np.int64)
    upper = float(outcome_upper_bound)
    alpha = float(failure_probability)
    if variances.ndim != 3 or min(variances.shape) <= 0:
        raise ValueError("sample variances must be a nonempty 3D tensor")
    if counts.shape != variances.shape[:2]:
        raise ValueError("sample counts must align with state/action interfaces")
    if not np.all(np.isfinite(variances)) or np.any(variances < 0.0):
        raise ValueError("sample variances must be finite and nonnegative")
    if np.any(counts < 2):
        raise ValueError("every interface sample count must be at least two")
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("outcome upper bound must be finite and positive")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("failure probability must lie strictly between zero and one")

    cells = int(np.prod(variances.shape))
    log_term = float(np.log(4.0 * cells / alpha))
    expanded_counts = counts[:, :, None].astype(np.float64)
    return np.sqrt(2.0 * variances * log_term / expanded_counts) + (
        8.0 * upper * log_term / (3.0 * (expanded_counts - 1.0))
    )


def compose_target_operator_radii(
    *,
    interface_cell_radii: np.ndarray,
    target_executed_action_weights: np.ndarray,
) -> TargetOperatorRadius:
    """Push primitive-cell radii through a queryable policy/filter mixture."""

    radii = np.asarray(interface_cell_radii, dtype=np.float64)
    weights = np.asarray(target_executed_action_weights, dtype=np.float64)
    if radii.ndim != 3 or radii.shape[2] != radii.shape[0] + 1:
        raise ValueError("radii must have state/action/(states-plus-terminal) shape")
    if weights.shape != radii.shape[:2]:
        raise ValueError("target action weights must align with interface radii")
    if not np.all(np.isfinite(radii)) or np.any(radii < 0.0):
        raise ValueError("interface radii must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("target action weights must be finite and nonnegative")
    if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-12):
        raise ValueError("target action weights must sum to one in every state")

    target_cells = np.sum(weights[:, :, None] * radii, axis=1)
    num_states = radii.shape[0]
    matrix_radius = float(np.max(np.sum(target_cells[:, :num_states], axis=1)))
    terminal_radius = float(np.max(target_cells[:, num_states]))
    return TargetOperatorRadius(
        transient_operator_radius_inf=matrix_radius,
        terminal_vector_radius_inf=terminal_radius,
        target_cell_radii=tuple(
            tuple(float(value) for value in row) for row in target_cells
        ),
    )


def resolvent_plugin_certificate(
    *,
    learned_transient_operator: np.ndarray,
    learned_terminal_vector: np.ndarray,
    transient_operator_radius_inf: float,
    terminal_vector_radius_inf: float,
) -> ResolventPluginCertificate:
    """A data-computable robust resolvent perturbation certificate."""

    matrix = np.asarray(learned_transient_operator, dtype=np.float64)
    terminal = np.asarray(learned_terminal_vector, dtype=np.float64)
    matrix_radius = float(transient_operator_radius_inf)
    terminal_radius = float(terminal_vector_radius_inf)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("learned transient operator must be a nonempty square matrix")
    if terminal.shape != (matrix.shape[0],):
        raise ValueError("learned terminal vector must align with the operator")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(terminal)):
        raise ValueError("learned operator and terminal vector must be finite")
    if matrix_radius < 0.0 or terminal_radius < 0.0:
        raise ValueError("operator radii must be nonnegative")
    if not np.isfinite(matrix_radius) or not np.isfinite(terminal_radius):
        raise ValueError("operator radii must be finite")

    try:
        learned_resolvent = np.linalg.inv(np.eye(matrix.shape[0]) - matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("learned resolvent must be invertible") from error
    resolvent_norm = float(np.linalg.norm(learned_resolvent, ord=np.inf))
    stability_product = resolvent_norm * matrix_radius
    if stability_product >= 1.0:
        raise ValueError("robust resolvent condition must be strictly below one")
    learned_value = learned_resolvent @ terminal
    learned_value_norm = float(np.linalg.norm(learned_value, ord=np.inf))
    error_bound = (
        resolvent_norm
        / (1.0 - stability_product)
        * (terminal_radius + matrix_radius * learned_value_norm)
    )
    return ResolventPluginCertificate(
        learned_resolvent_norm_inf=resolvent_norm,
        stability_product=float(stability_product),
        learned_value_norm_inf=learned_value_norm,
        value_error_bound_inf=float(error_bound),
        learned_value=tuple(float(value) for value in learned_value),
    )


def interface_transfer_minimax_lower_bound(
    *,
    risk_parameter: float,
    high_resource: float,
    source_interface_probability: float,
    number_of_source_transitions: int,
) -> InterfaceTransferLowerBound:
    """Two-model lower bound with source coverage and exponential risk.

    The source executes the target-only informative interface with probability
    ``source_interface_probability``. Conditional on that interface, the two
    models use rare-event probabilities q and 2q, q=exp(-lambda*L). The target
    always executes the informative interface.
    """

    lam = float(risk_parameter)
    resource = float(high_resource)
    coverage = float(source_interface_probability)
    transitions = int(number_of_source_transitions)
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("risk parameter must be finite and positive")
    if not np.isfinite(resource) or resource <= 0.0:
        raise ValueError("high resource must be finite and positive")
    if not np.isfinite(coverage) or not 0.0 < coverage <= 1.0:
        raise ValueError("source interface probability must lie in (0, 1]")
    if transitions <= 0:
        raise ValueError("number of source transitions must be positive")
    rare = float(np.exp(-lam * resource))
    if rare > 0.25:
        raise ValueError("construction requires exp(-lambda*L) <= 1/4")

    tv_upper = float(min(1.0, transitions * coverage * rare))
    separation = float(1.0 - rare)
    minimax_lower = float(0.5 * separation * (1.0 - tv_upper))
    decision_lower = float(0.5 * (1.0 - tv_upper))
    threshold = float(1.0 / (4.0 * coverage * rare))
    constant_error = bool(
        transitions * coverage * rare
        <= 0.25 + 16.0 * np.finfo(np.float64).eps
    )
    return InterfaceTransferLowerBound(
        rare_event_probability=rare,
        source_interface_probability=coverage,
        number_of_source_transitions=transitions,
        source_law_total_variation_upper=tv_upper,
        target_mgf_separation=separation,
        minimax_absolute_error_lower=minimax_lower,
        minimax_decision_error_lower=decision_lower,
        constant_error_regime=constant_error,
        maximum_transitions_for_constant_error=threshold,
    )


__all__ = [
    "InterfaceTransferLowerBound",
    "ResolventPluginCertificate",
    "TargetOperatorRadius",
    "bernstein_interface_radii",
    "compose_target_operator_radii",
    "interface_transfer_minimax_lower_bound",
    "resolvent_plugin_certificate",
]
