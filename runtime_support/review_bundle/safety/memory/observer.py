from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .networks import ContractiveResidualMemory


def _vec3(value: np.ndarray | float, name: str, *, positive: bool = False) -> np.ndarray:
    array = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


@dataclass(frozen=True)
class IntervalObserverConfig:
    dt: float = 0.1
    position_gain: float = 0.85
    velocity_gain: float = 0.25
    acceleration_gain: float = 0.05
    sensor_error_bound: np.ndarray | float = 0.10
    jerk_residual_bound: np.ndarray | float = 12.0
    reset_velocity_bound: np.ndarray | float = 12.0
    reset_acceleration_bound: np.ndarray | float = 8.0
    innovation_tolerance: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        for name in ("position_gain", "velocity_gain", "acceleration_gain"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not np.isfinite(self.innovation_tolerance):
            raise ValueError("innovation_tolerance must be finite")
        if self.innovation_tolerance != 0.0:
            raise ValueError(
                "a certified interval observer cannot ignore a positive innovation gap"
            )
        for name in (
            "sensor_error_bound",
            "jerk_residual_bound",
            "reset_velocity_bound",
            "reset_acceleration_bound",
        ):
            object.__setattr__(self, name, _vec3(getattr(self, name), name, positive=True))


@dataclass(frozen=True)
class TrustedBaseIntervalCertificate:
    """Caller-supplied evidence for a fresh same-track base interval.

    The object records, rather than proves, the physical premises.  The observer
    only accepts it when every attested bound is no larger than the configured
    interval used at runtime.
    """

    track_id: str
    sensor_error_bound: np.ndarray | float
    velocity_bound: np.ndarray | float
    acceleration_bound: np.ndarray | float
    provenance: str
    measurement_epoch: int

    def __post_init__(self) -> None:
        if not self.track_id:
            raise ValueError("track_id must be nonempty")
        if not self.provenance:
            raise ValueError("base-certificate provenance must be nonempty")
        if self.measurement_epoch < 0:
            raise ValueError("measurement_epoch must be nonnegative")
        for name in (
            "sensor_error_bound",
            "velocity_bound",
            "acceleration_bound",
        ):
            object.__setattr__(
                self,
                name,
                _vec3(getattr(self, name), name, positive=True),
            )

    @classmethod
    def from_config(
        cls,
        config: IntervalObserverConfig,
        *,
        track_id: str,
        provenance: str,
        measurement_epoch: int,
    ) -> "TrustedBaseIntervalCertificate":
        return cls(
            track_id=track_id,
            sensor_error_bound=config.sensor_error_bound,
            velocity_bound=config.reset_velocity_bound,
            acceleration_bound=config.reset_acceleration_bound,
            provenance=provenance,
            measurement_epoch=measurement_epoch,
        )


@dataclass(frozen=True)
class IntervalObserverState:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    position_radius: np.ndarray
    velocity_radius: np.ndarray
    acceleration_radius: np.ndarray
    track_id: str | None
    certification_valid: bool
    invalid_reason: str | None
    certificate_epoch: int | None = None
    missed_frames: int = 0

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            object.__setattr__(self, name, _vec3(getattr(self, name), name))
        for name in ("position_radius", "velocity_radius", "acceleration_radius"):
            object.__setattr__(
                self,
                name,
                _vec3(getattr(self, name), name, positive=True),
            )
        if self.missed_frames < 0:
            raise ValueError("missed_frames must be nonnegative")
        if self.certification_valid and not self.track_id:
            raise ValueError("a valid certification state requires a track_id")
        if self.certification_valid and self.certificate_epoch is None:
            raise ValueError("a valid certification state requires a certificate_epoch")
        if not self.certification_valid and not self.invalid_reason:
            raise ValueError("an invalid certification state requires a reason")

    @property
    def vector(self) -> np.ndarray:
        return np.concatenate((self.position, self.velocity, self.acceleration))

    @property
    def radius(self) -> np.ndarray:
        return np.concatenate(
            (self.position_radius, self.velocity_radius, self.acceleration_radius)
        )


@dataclass(frozen=True)
class ObserverStep:
    state: IntervalObserverState
    innovation: np.ndarray
    normalized_innovation: float
    reset: bool
    dropout: bool
    association_uncertain: bool
    nominal_jerk: np.ndarray


class PhysicsMemoryObserver:
    """Explicit kinematic observer with optional recurrent nominal jerk.

    Certification is carried by the interval recursion. The recurrent model is
    allowed to improve the nominal center only; soundness additionally requires
    the declared jerk residual bound to contain the true-minus-nominal jerk.
    """

    def __init__(
        self,
        config: IntervalObserverConfig,
        residual_memory: ContractiveResidualMemory | None = None,
    ) -> None:
        self.config = config
        self.residual_memory = residual_memory
        self.state: IntervalObserverState | None = None
        self.hidden: torch.Tensor | None = None
        self._latest_certificate_epoch_by_track: dict[str, int] = {}

    def _certificate_is_compatible(
        self,
        certificate: TrustedBaseIntervalCertificate,
    ) -> bool:
        return bool(
            np.all(certificate.sensor_error_bound <= self.config.sensor_error_bound)
            and np.all(certificate.velocity_bound <= self.config.reset_velocity_bound)
            and np.all(
                certificate.acceleration_bound
                <= self.config.reset_acceleration_bound
            )
        )

    def _reset_state(
        self,
        measurement: np.ndarray,
        certificate: TrustedBaseIntervalCertificate | None,
        *,
        invalid_reason: str,
        retained_track_id: str | None = None,
        retained_certificate_epoch: int | None = None,
    ) -> IntervalObserverState:
        position = _vec3(measurement, "measurement")
        valid = bool(
            certificate is not None
            and self._certificate_is_compatible(certificate)
            and certificate.measurement_epoch
            > self._latest_certificate_epoch_by_track.get(certificate.track_id, -1)
        )
        if valid:
            self._latest_certificate_epoch_by_track[certificate.track_id] = (
                certificate.measurement_epoch
            )
        self.state = IntervalObserverState(
            position=position,
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            position_radius=self.config.sensor_error_bound,
            velocity_radius=self.config.reset_velocity_bound,
            acceleration_radius=self.config.reset_acceleration_bound,
            track_id=retained_track_id if not valid else certificate.track_id,
            certification_valid=valid,
            invalid_reason=None if valid else invalid_reason,
            certificate_epoch=(
                retained_certificate_epoch
                if not valid
                else certificate.measurement_epoch
            ),
        )
        if self.residual_memory is not None:
            self.hidden = self.residual_memory.initial_hidden(1)
        return self.state

    def initialize(
        self,
        measurement: np.ndarray,
        *,
        trusted_base_certificate: TrustedBaseIntervalCertificate | None = None,
    ) -> IntervalObserverState:
        return self._reset_state(
            measurement,
            trusted_base_certificate,
            invalid_reason="missing_trusted_base_certificate",
        )

    def _nominal_jerk(self, features: np.ndarray | None) -> np.ndarray:
        if self.residual_memory is None:
            return np.zeros(3)
        if features is None:
            raise ValueError("features are required by the residual memory")
        feature_array = np.asarray(features)
        if feature_array.shape != (self.residual_memory.input_layer.in_features,):
            raise ValueError("features have the wrong shape")
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("features must be finite")
        if self.hidden is None:
            self.hidden = self.residual_memory.initial_hidden(1)
        with torch.no_grad():
            parameter = next(self.residual_memory.parameters())
            tensor = torch.as_tensor(
                feature_array,
                device=parameter.device,
                dtype=parameter.dtype,
            ).reshape(1, -1)
            output, self.hidden = self.residual_memory(tensor, self.hidden)
        return output[0].detach().cpu().numpy().astype(np.float64)

    def _predict(
        self,
        nominal_jerk: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.state is None:
            raise RuntimeError("observer is not initialized")
        dt = self.config.dt
        identity = np.eye(3)
        transition = np.block(
            [
                [identity, dt * identity, 0.5 * dt**2 * identity],
                [np.zeros((3, 3)), identity, dt * identity],
                [np.zeros((3, 3)), np.zeros((3, 3)), identity],
            ]
        )
        jerk_map = np.concatenate(
            (
                (dt**3 / 6.0) * identity,
                (dt**2 / 2.0) * identity,
                dt * identity,
            ),
            axis=0,
        )
        center = transition @ self.state.vector + jerk_map @ nominal_jerk
        radius = np.abs(transition) @ self.state.radius
        radius += np.abs(jerk_map) @ self.config.jerk_residual_bound
        return center, radius

    def step(
        self,
        measurement: np.ndarray | None,
        *,
        features: np.ndarray | None = None,
        track_id: str | None = None,
        trusted_base_certificate: TrustedBaseIntervalCertificate | None = None,
        association_uncertain: bool = False,
    ) -> ObserverStep:
        if self.state is None:
            if measurement is None:
                raise RuntimeError("the first observer step requires a measurement")
            certificate = (
                None
                if association_uncertain
                or trusted_base_certificate is None
                or track_id != trusted_base_certificate.track_id
                else trusted_base_certificate
            )
            state = self.initialize(
                measurement,
                trusted_base_certificate=certificate,
            )
            return ObserverStep(
                state=state,
                innovation=np.zeros(3),
                normalized_innovation=0.0,
                reset=True,
                dropout=False,
                association_uncertain=association_uncertain,
                nominal_jerk=np.zeros(3),
            )
        nominal_jerk = self._nominal_jerk(features)
        predicted_center, predicted_radius = self._predict(nominal_jerk)
        if measurement is None:
            certification_valid = bool(
                self.state.certification_valid and not association_uncertain
            )
            self.state = IntervalObserverState(
                position=predicted_center[:3],
                velocity=predicted_center[3:6],
                acceleration=predicted_center[6:9],
                position_radius=predicted_radius[:3],
                velocity_radius=predicted_radius[3:6],
                acceleration_radius=predicted_radius[6:9],
                track_id=self.state.track_id,
                certificate_epoch=self.state.certificate_epoch,
                missed_frames=self.state.missed_frames + 1,
                certification_valid=certification_valid,
                invalid_reason=(
                    None
                    if certification_valid
                    else (
                        "association_uncertain"
                        if association_uncertain
                        else self.state.invalid_reason
                    )
                ),
            )
            return ObserverStep(
                state=self.state,
                innovation=np.full(3, np.nan),
                normalized_innovation=float("nan"),
                reset=False,
                dropout=True,
                association_uncertain=association_uncertain,
                nominal_jerk=nominal_jerk,
            )

        observed = _vec3(measurement, "measurement")
        innovation = observed - predicted_center[:3]
        admissible_innovation = predicted_radius[:3] + self.config.sensor_error_bound
        normalized = float(
            np.max(np.abs(innovation) / np.maximum(admissible_innovation, 1e-15))
        )
        inconsistent = bool(np.any(np.abs(innovation) > admissible_innovation))
        if association_uncertain or track_id is None:
            self._reset_state(
                observed,
                None,
                invalid_reason=(
                    "association_uncertain"
                    if association_uncertain
                    else "missing_track_id"
                ),
            )
            return ObserverStep(
                state=self.state,
                innovation=innovation,
                normalized_innovation=normalized,
                reset=True,
                dropout=False,
                association_uncertain=association_uncertain,
                nominal_jerk=nominal_jerk,
            )

        incoming_certificate = trusted_base_certificate
        if incoming_certificate is not None and incoming_certificate.track_id != track_id:
            incoming_certificate = None
        track_matches = bool(
            self.state.track_id is not None and track_id == self.state.track_id
        )
        if not track_matches:
            self._reset_state(
                observed,
                incoming_certificate,
                invalid_reason="track_identity_mismatch",
                retained_track_id=track_id,
            )
            return ObserverStep(
                state=self.state,
                innovation=innovation,
                normalized_innovation=normalized,
                reset=True,
                dropout=False,
                association_uncertain=False,
                nominal_jerk=nominal_jerk,
            )

        if inconsistent or not self.state.certification_valid:
            self._reset_state(
                observed,
                incoming_certificate,
                invalid_reason="fresh_reset_certificate_required",
                retained_track_id=self.state.track_id,
                retained_certificate_epoch=self.state.certificate_epoch,
            )
            return ObserverStep(
                state=self.state,
                innovation=innovation,
                normalized_innovation=normalized,
                reset=True,
                dropout=False,
                association_uncertain=False,
                nominal_jerk=nominal_jerk,
            )

        dt = self.config.dt
        gain = np.concatenate(
            (
                self.config.position_gain * np.eye(3),
                (self.config.velocity_gain / dt) * np.eye(3),
                (self.config.acceleration_gain / dt**2) * np.eye(3),
            ),
            axis=0,
        )
        observation = np.concatenate((np.eye(3), np.zeros((3, 6))), axis=1)
        corrected_center = predicted_center + gain @ innovation
        corrected_radius = (
            np.abs(np.eye(9) - gain @ observation) @ predicted_radius
            + np.abs(gain) @ self.config.sensor_error_bound
        )
        self.state = IntervalObserverState(
            position=corrected_center[:3],
            velocity=corrected_center[3:6],
            acceleration=corrected_center[6:9],
            position_radius=corrected_radius[:3],
            velocity_radius=corrected_radius[3:6],
            acceleration_radius=corrected_radius[6:9],
            track_id=self.state.track_id,
            certification_valid=True,
            invalid_reason=None,
            certificate_epoch=self.state.certificate_epoch,
        )
        return ObserverStep(
            state=self.state,
            innovation=innovation,
            normalized_innovation=normalized,
            reset=False,
            dropout=False,
            association_uncertain=association_uncertain,
            nominal_jerk=nominal_jerk,
        )
