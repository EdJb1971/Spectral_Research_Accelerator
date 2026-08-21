"""Strict runtime binding for a frozen motivating-experiment protocol.

The protocol says what was intended.  This module checks what the dataset and checkpoint
actually declare, then creates one content-addressed identity that evaluation can carry.  A
partial match is an error: provenance is not useful if a run can quietly substitute another
grid, split, representation, optimiser, rollout, or checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

from src.forecasting.adapter import ForecastContractError
from src.forecasting.artifact import LaboratoryModelArtifact
from src.forecasting.protocol import MotivatingExperimentProtocol


BINDING_SCHEMA = "motivating-experiment-runtime-binding/v1"
DATASET_BINDING_SCHEMA = "motivating-experiment-dataset-binding/v1"


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError(
            "runtime provenance must be finite JSON-serialisable data: %s" % exc) from exc


def provenance_sha256(value: Mapping[str, Any]) -> str:
    """Return the full canonical SHA-256 of a provenance mapping."""
    if not isinstance(value, Mapping):
        raise ForecastContractError("provenance must be a JSON object")
    return hashlib.sha256(_canonical(dict(value))).hexdigest()


def _at(record: Mapping[str, Any], path: str, mismatches: list[str]) -> Any:
    value: Any = record
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            mismatches.append("%s is missing" % path)
            return None
        value = value[component]
    return value


def _same(actual: Any, expected: Any) -> bool:
    try:
        return _canonical(actual) == _canonical(expected)
    except ForecastContractError:
        return False


def _expect(record: Mapping[str, Any], path: str, expected: Any,
            mismatches: list[str]) -> None:
    actual = _at(record, path, mismatches)
    if actual is not None and not _same(actual, expected):
        mismatches.append("%s is %r, expected %r" % (path, actual, expected))


def _utc_naive(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ForecastContractError("invalid protocol-bound timestamp %r" % value) from None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed.isoformat(timespec="seconds")


def _regular_spacing(values: Sequence[Any]) -> Union[float, None]:
    numbers = tuple(float(value) for value in values)
    if len(numbers) < 2:
        return None
    intervals = tuple(abs(right - left) for left, right in zip(numbers, numbers[1:]))
    if not all(math.isclose(value, intervals[0], rel_tol=0.0, abs_tol=1e-10)
               for value in intervals[1:]):
        return None
    return intervals[0]


@dataclass(frozen=True)
class DatasetProtocolBinding:
    schema: str
    protocol_sha256: str
    dataset_provenance_sha256: str
    variables: Tuple[str, ...]
    lead_frames: Tuple[int, ...]
    matched_contract_paths: Tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if self.schema != DATASET_BINDING_SCHEMA:
            raise ForecastContractError("unsupported dataset protocol binding schema")
        if not _is_sha256(self.protocol_sha256) or not _is_sha256(
                self.dataset_provenance_sha256):
            raise ForecastContractError("dataset protocol binding requires full SHA-256 identities")

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["lead_frames"] = list(self.lead_frames)
        record["matched_contract_paths"] = list(self.matched_contract_paths)
        return record


@dataclass(frozen=True)
class ExperimentProtocolBinding:
    schema: str
    protocol_sha256: str
    dataset_provenance_sha256: str
    artifact_checkpoint_sha256: str
    variables: Tuple[str, ...]
    lead_frames: Tuple[int, ...]
    binding_sha256: str
    claim_boundary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "lead_frames", tuple(self.lead_frames))
        if self.schema != BINDING_SCHEMA:
            raise ForecastContractError("unsupported experiment runtime binding schema")
        hashes = (self.protocol_sha256, self.dataset_provenance_sha256,
                  self.artifact_checkpoint_sha256, self.binding_sha256)
        if not all(_is_sha256(value) for value in hashes):
            raise ForecastContractError("experiment runtime binding requires full SHA-256 identities")
        identity = {
            "schema": self.schema, "protocol_sha256": self.protocol_sha256,
            "dataset_provenance_sha256": self.dataset_provenance_sha256,
            "artifact_checkpoint_sha256": self.artifact_checkpoint_sha256,
            "variables": self.variables, "lead_frames": self.lead_frames,
        }
        if hashlib.sha256(_canonical(identity)).hexdigest() != self.binding_sha256:
            raise ForecastContractError("experiment runtime binding SHA-256 mismatch")

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["lead_frames"] = list(self.lead_frames)
        return record

    @classmethod
    def from_provenance(cls, value: Mapping[str, Any]) -> "ExperimentProtocolBinding":
        if not isinstance(value, Mapping):
            raise ForecastContractError("experiment runtime binding must be a JSON object")
        fields = set(cls.__dataclass_fields__)
        supplied = set(value)
        if supplied != fields:
            raise ForecastContractError(
                "experiment runtime binding fields differ: missing=%r unknown=%r"
                % (sorted(fields - supplied), sorted(supplied - fields)))
        record = dict(value)
        record["variables"] = tuple(record["variables"])
        record["lead_frames"] = tuple(record["lead_frames"])
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid experiment runtime binding: %s" % exc) from exc


def bind_dataset_to_protocol(
    protocol: MotivatingExperimentProtocol,
    dataset_provenance: Mapping[str, Any],
) -> DatasetProtocolBinding:
    """Validate a prepared regional dataset against every observable protocol field."""
    if not isinstance(protocol, MotivatingExperimentProtocol):
        raise ForecastContractError("protocol must be a validated MotivatingExperimentProtocol")
    if not isinstance(dataset_provenance, Mapping):
        raise ForecastContractError("dataset_provenance must be a JSON object")
    record = dict(dataset_provenance)
    mismatches: list[str] = []
    expected = {
        "schema": "regional_forecast_dataset/v2",
        "source.dataset_identity.name": protocol.data.source_dataset,
        "source.dataset_identity.version": protocol.data.source_version,
        "config.variables": list(protocol.data.variables),
        "config.level_hpa": protocol.data.pressure_levels_hpa[0],
        "config.history_frames": protocol.timeline.history_frames,
        "config.lead_frames": list(protocol.timeline.lead_frames),
        "config.embargo_frames": protocol.splits.embargo_frames,
        "level_hpa": protocol.data.pressure_levels_hpa[0],
        "cadence.is_regular": True,
        "cadence.observed_cadence_hours": protocol.data.cadence_hours,
        "grid.shape": list(protocol.domain.grid_shape),
        "grid.sha256": protocol.domain.grid_coordinates_sha256,
        "grid.longitude_convention": protocol.domain.longitude_convention,
        "normalisation_contract.method": protocol.normalisation.method,
        "normalisation_contract.scope": protocol.normalisation.scope,
        "normalisation_contract.fitted_split": protocol.normalisation.fitted_split,
        "normalisation_contract.ddof": protocol.normalisation.ddof,
        "normalisation_contract.statistics_artifact_sha256": (
            protocol.normalisation.statistics_artifact_sha256),
        "timestamps.timezone": "UTC",
    }
    if len(protocol.data.pressure_levels_hpa) != 1:
        mismatches.append(
            "data.pressure_levels_hpa declares %r but RegionalForecastDataset is single-level"
            % (protocol.data.pressure_levels_hpa,))
    for path, value in expected.items():
        _expect(record, path, value, mismatches)

    latitude = _at(record, "grid.latitude", mismatches)
    longitude = _at(record, "grid.longitude", mismatches)
    if (isinstance(latitude, Sequence) and not isinstance(latitude, (str, bytes))
            and isinstance(longitude, Sequence) and not isinstance(longitude, (str, bytes))):
        calculated_grid_sha = provenance_sha256({
            "latitude": list(latitude), "longitude": list(longitude)})
        recorded_grid_sha = _at(record, "grid.sha256", mismatches)
        if recorded_grid_sha != calculated_grid_sha:
            mismatches.append(
                "grid.sha256 does not match the recorded latitude/longitude coordinates")
    if isinstance(latitude, Sequence) and not isinstance(latitude, (str, bytes)) and latitude:
        lat_values = tuple(float(v) for v in latitude)
        observed_order = "ascending" if lat_values[-1] > lat_values[0] else "descending"
        checks = (
            ("domain.latitude_bounds", [min(lat_values), max(lat_values)],
             list(protocol.domain.latitude_bounds)),
            ("domain.latitude_spacing_degrees", _regular_spacing(lat_values),
             protocol.domain.latitude_spacing_degrees),
            ("domain.latitude_order", observed_order, protocol.domain.latitude_order),
        )
        for path, actual, wanted in checks:
            if actual is None or not _same(actual, wanted):
                mismatches.append("%s observed %r, expected %r" % (path, actual, wanted))
    if isinstance(longitude, Sequence) and not isinstance(longitude, (str, bytes)) and longitude:
        lon_values = tuple(float(v) for v in longitude)
        checks = (
            ("domain.longitude_bounds", [min(lon_values), max(lon_values)],
             list(protocol.domain.longitude_bounds)),
            ("domain.longitude_spacing_degrees", _regular_spacing(lon_values),
             protocol.domain.longitude_spacing_degrees),
        )
        for path, actual, wanted in checks:
            if actual is None or not _same(actual, wanted):
                mismatches.append("%s observed %r, expected %r" % (path, actual, wanted))

    normalisation = _at(record, "normalisation", mismatches)
    if isinstance(normalisation, Mapping):
        normalisation_body = dict(normalisation)
        recorded_statistics_sha = normalisation_body.pop("artifact_hash", None)
        calculated_statistics_sha = provenance_sha256(normalisation_body)
        if recorded_statistics_sha != calculated_statistics_sha:
            mismatches.append(
                "normalisation.artifact_hash does not match the recorded training statistics")
        if recorded_statistics_sha != protocol.normalisation.statistics_artifact_sha256:
            mismatches.append(
                "normalisation.artifact_hash differs from the frozen protocol")

    split_paths = {
        "temporal_split.train.first_timestamp": protocol.splits.train_start,
        "temporal_split.train.last_timestamp": protocol.splits.train_end,
        "temporal_split.val.first_timestamp": protocol.splits.validation_start,
        "temporal_split.val.last_timestamp": protocol.splits.validation_end,
        "temporal_split.test.first_timestamp": protocol.splits.test_start,
        "temporal_split.test.last_timestamp": protocol.splits.test_end,
    }
    for path, wanted in split_paths.items():
        actual = _at(record, path, mismatches)
        if actual is not None and _utc_naive(str(actual)) != _utc_naive(wanted):
            mismatches.append("%s is %r, expected instant %r" % (path, actual, wanted))
    if mismatches:
        raise ForecastContractError(
            "dataset does not conform to frozen experiment protocol: " + "; ".join(mismatches))
    return DatasetProtocolBinding(
        schema=DATASET_BINDING_SCHEMA, protocol_sha256=protocol.fingerprint(),
        dataset_provenance_sha256=provenance_sha256(record),
        variables=protocol.data.variables, lead_frames=protocol.timeline.lead_frames,
        matched_contract_paths=tuple(expected) + tuple(split_paths) + (
            "domain.latitude_bounds", "domain.longitude_bounds",
            "domain.latitude_spacing_degrees", "domain.longitude_spacing_degrees",
            "domain.latitude_order"),
        claim_boundary=("Exact declared dataset/protocol conformance only; this binding does not "
                        "establish data correctness, model training quality, or forecast skill."),
    )


def protocol_training_provenance(
    protocol: MotivatingExperimentProtocol,
    dataset_binding: DatasetProtocolBinding,
) -> Dict[str, Any]:
    """Create the mandatory protocol identity block for a subsequently saved checkpoint."""
    if dataset_binding.protocol_sha256 != protocol.fingerprint():
        raise ForecastContractError("dataset binding belongs to a different experiment protocol")
    protocol_record = protocol.to_mapping()
    return {
        "protocol_sha256": protocol.fingerprint(),
        "dataset_provenance_sha256": dataset_binding.dataset_provenance_sha256,
        "model_version": protocol.model.model_version,
        "optimizer_contract": protocol_record["optimizer"],
        "rollout_contract": protocol_record["rollout"],
    }


def bind_artifact_to_protocol(
    protocol: MotivatingExperimentProtocol,
    dataset_binding: DatasetProtocolBinding,
    artifact: Union[LaboratoryModelArtifact, Mapping[str, Any]],
) -> ExperimentProtocolBinding:
    """Bind one verified checkpoint to the exact dataset and frozen training protocol."""
    if dataset_binding.protocol_sha256 != protocol.fingerprint():
        raise ForecastContractError("dataset binding belongs to a different experiment protocol")
    artifact_record = (artifact.to_provenance() if isinstance(artifact, LaboratoryModelArtifact)
                       else dict(artifact))
    mismatches: list[str] = []
    protocol_record = protocol.to_mapping()
    transform = protocol_record["transform"]
    expected = {
        "model_class": protocol.model.class_path,
        "model_config": protocol_record["model"]["config"],
        "representation_config": transform,
        "parameter_count": protocol.model.parameter_count,
        "trainable_parameter_count": protocol.model.trainable_parameter_count,
        "training_provenance.protocol_sha256": protocol.fingerprint(),
        "training_provenance.dataset_provenance_sha256": (
            dataset_binding.dataset_provenance_sha256),
        "training_provenance.model_version": protocol.model.model_version,
        "training_provenance.optimizer_contract": protocol_record["optimizer"],
        "training_provenance.rollout_contract": protocol_record["rollout"],
    }
    for path, value in expected.items():
        _expect(artifact_record, path, value, mismatches)
    checkpoint = _at(artifact_record, "checkpoint_sha256", mismatches)
    if not isinstance(checkpoint, str) or len(checkpoint) != 64:
        mismatches.append("checkpoint_sha256 must be a full SHA-256")
    if mismatches:
        raise ForecastContractError(
            "model artefact does not conform to frozen experiment protocol: "
            + "; ".join(mismatches))
    identity = {
        "schema": BINDING_SCHEMA,
        "protocol_sha256": protocol.fingerprint(),
        "dataset_provenance_sha256": dataset_binding.dataset_provenance_sha256,
        "artifact_checkpoint_sha256": checkpoint,
        "variables": tuple(protocol.data.variables),
        "lead_frames": tuple(protocol.timeline.lead_frames),
    }
    return ExperimentProtocolBinding(
        **identity, binding_sha256=hashlib.sha256(_canonical(identity)).hexdigest(),
        claim_boundary=("Exact protocol/dataset/checkpoint identity only; this binding does not "
                        "establish correct implementation, uncertainty, or scientific skill."),
    )


def validate_evaluation_binding(
    binding: ExperimentProtocolBinding,
    dataset_provenance: Mapping[str, Any],
    forecaster_provenance: Mapping[str, Any],
    variables: Sequence[str],
    lead_frames: Sequence[int],
) -> None:
    """Refuse evaluation if any input belongs to another frozen experiment."""
    mismatches = []
    if provenance_sha256(dataset_provenance) != binding.dataset_provenance_sha256:
        mismatches.append("dataset provenance SHA-256 differs from the bound dataset")
    checkpoint = _at(forecaster_provenance, "model_artifact.checkpoint_sha256", mismatches)
    if checkpoint != binding.artifact_checkpoint_sha256:
        mismatches.append("forecaster checkpoint differs from the bound artefact")
    if tuple(variables) != binding.variables:
        mismatches.append("evaluation variables differ from the frozen protocol")
    if tuple(lead_frames) != binding.lead_frames:
        mismatches.append("evaluation leads differ from the frozen protocol")
    if mismatches:
        raise ForecastContractError("evaluation protocol binding failed: " + "; ".join(mismatches))
