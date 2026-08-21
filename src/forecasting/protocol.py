"""Strict, portable contract for the motivating forecast experiment (roadmap T5.0a).

This module records a *declared* experimental design.  It does not infer missing choices from
the poster, a checkpoint, tensor shapes or platform defaults.  A protocol becomes frozen only
when every field is present and the record names and hashes the laboratory evidence from which
it was transcribed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union

from src.forecasting.adapter import ForecastContractError


PROTOCOL_SCHEMA = "motivating-forecast-protocol/v1"
PROTOCOL_FILENAME = "experiment-protocol.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"^(?:unknown|tbd|todo|n/?a|not[ -]?specified|unspecified|\?+)$", re.I)


class _FrozenJSONMap(Mapping[str, Any]):
    """Small recursively immutable mapping used inside a hashable protocol."""

    def __init__(self, items: Sequence[Tuple[str, Any]]) -> None:
        self._items = tuple(items)
        self._values = dict(self._items)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_FrozenJSONMap":
        return self


def _freeze_json(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        items = []
        for key in sorted(value):
            if not isinstance(key, str) or not key.strip():
                raise ForecastContractError("%s has a non-string or empty key" % path)
            items.append((key, _freeze_json(value[key], "%s.%s" % (path, key))))
        return _FrozenJSONMap(items)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, "%s[%d]" % (path, index))
                     for index, item in enumerate(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ForecastContractError("%s must contain only finite JSON values" % path)


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(_thaw_json(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError(
            "experiment protocol must contain finite JSON-serialisable values: %s" % exc) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_exact_keys(value: Any, expected: Sequence[str], section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForecastContractError("protocol section %r must be a JSON object" % section)
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    extra = sorted(set(value) - expected_set)
    if missing or extra:
        details = []
        if missing:
            details.append("missing %s" % ", ".join(missing))
        if extra:
            details.append("unknown %s" % ", ".join(extra))
        raise ForecastContractError("protocol section %r has %s" % (section, "; ".join(details)))
    return value


def _reject_placeholders(value: Any, path: str = "protocol") -> None:
    if isinstance(value, str):
        if not value.strip() or _PLACEHOLDER.fullmatch(value.strip()):
            raise ForecastContractError("%s must be explicit; placeholders are not accepted" % path)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ForecastContractError("%s has a non-string or empty key" % path)
            _reject_placeholders(item, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_placeholders(item, "%s[%d]" % (path, index))


def _strings(instance: Any, names: Sequence[str], section: str) -> None:
    for name in names:
        value = getattr(instance, name)
        if not isinstance(value, str):
            raise ForecastContractError("%s.%s must be a non-empty string" % (section, name))


def _positive_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForecastContractError("%s must be a positive finite number" % path)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ForecastContractError("%s must be a positive finite number" % path)
    return number


def _positive_int(value: Any, path: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ForecastContractError("%s must be a %s integer" % (path, qualifier))
    return value


def _sha256(value: str, path: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ForecastContractError("%s must be a lowercase 64-character SHA-256" % path)


def _timestamp(value: str, path: str) -> datetime:
    if not isinstance(value, str):
        raise ForecastContractError("%s must be an ISO-8601 timestamp" % path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ForecastContractError("%s must be an ISO-8601 timestamp" % path) from None
    if parsed.tzinfo is None:
        raise ForecastContractError("%s must include an explicit UTC offset" % path)
    return parsed


def _string_tuple(value: Any, path: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ForecastContractError("%s must be a non-empty sequence" % path)
    result = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ForecastContractError("%s must contain non-empty strings" % path)
    if len(set(result)) != len(result):
        raise ForecastContractError("%s must not contain duplicates" % path)
    return result


@dataclass(frozen=True)
class DomainContract:
    name: str
    latitude_bounds: Tuple[float, float]
    longitude_bounds: Tuple[float, float]
    grid_shape: Tuple[int, int]
    latitude_spacing_degrees: float
    longitude_spacing_degrees: float
    longitude_convention: str
    latitude_order: str
    grid_coordinates_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "latitude_bounds", tuple(self.latitude_bounds))
        object.__setattr__(self, "longitude_bounds", tuple(self.longitude_bounds))
        object.__setattr__(self, "grid_shape", tuple(self.grid_shape))
        _strings(self, ("name", "longitude_convention", "latitude_order",
                        "grid_coordinates_sha256"), "domain")
        _reject_placeholders(asdict(self), "domain")
        if len(self.latitude_bounds) != 2 or not all(math.isfinite(float(v)) for v in self.latitude_bounds):
            raise ForecastContractError("domain.latitude_bounds must contain two finite coordinates")
        if not -90 <= float(self.latitude_bounds[0]) < float(self.latitude_bounds[1]) <= 90:
            raise ForecastContractError("domain.latitude_bounds must be ordered south-to-north")
        if len(self.longitude_bounds) != 2 or not all(math.isfinite(float(v)) for v in self.longitude_bounds):
            raise ForecastContractError("domain.longitude_bounds must contain two finite coordinates")
        if float(self.longitude_bounds[0]) >= float(self.longitude_bounds[1]):
            raise ForecastContractError("domain.longitude_bounds must be strictly increasing without wrapping")
        if len(self.grid_shape) != 2 or any(_positive_int(v, "domain.grid_shape") < 2 for v in self.grid_shape):
            raise ForecastContractError("domain.grid_shape must be (latitude, longitude), both >= 2")
        _positive_number(self.latitude_spacing_degrees, "domain.latitude_spacing_degrees")
        _positive_number(self.longitude_spacing_degrees, "domain.longitude_spacing_degrees")
        if self.longitude_convention not in ("-180..180", "0..360"):
            raise ForecastContractError("domain.longitude_convention must be '-180..180' or '0..360'")
        if self.latitude_order not in ("ascending", "descending"):
            raise ForecastContractError("domain.latitude_order must be 'ascending' or 'descending'")
        _sha256(self.grid_coordinates_sha256, "domain.grid_coordinates_sha256")


@dataclass(frozen=True)
class DataContract:
    source_dataset: str
    source_version: str
    variables: Tuple[str, ...]
    pressure_levels_hpa: Tuple[int, ...]
    cadence_hours: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "pressure_levels_hpa", tuple(self.pressure_levels_hpa))
        _strings(self, ("source_dataset", "source_version"), "data")
        _reject_placeholders(asdict(self), "data")
        _string_tuple(self.variables, "data.variables")
        if not self.pressure_levels_hpa or len(set(self.pressure_levels_hpa)) != len(self.pressure_levels_hpa):
            raise ForecastContractError("data.pressure_levels_hpa must contain unique levels")
        for level in self.pressure_levels_hpa:
            _positive_int(level, "data.pressure_levels_hpa")
        _positive_number(self.cadence_hours, "data.cadence_hours")


@dataclass(frozen=True)
class TimelineContract:
    history_frames: int
    lead_frames: Tuple[int, ...]
    lead_durations_hours: Tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lead_frames", tuple(self.lead_frames))
        object.__setattr__(self, "lead_durations_hours", tuple(self.lead_durations_hours))
        _positive_int(self.history_frames, "timeline.history_frames")
        if not self.lead_frames or tuple(sorted(set(self.lead_frames))) != self.lead_frames:
            raise ForecastContractError("timeline.lead_frames must be unique, positive and increasing")
        for lead in self.lead_frames:
            _positive_int(lead, "timeline.lead_frames")
        if len(self.lead_durations_hours) != len(self.lead_frames):
            raise ForecastContractError("timeline must give one physical duration per frame lead")
        for duration in self.lead_durations_hours:
            _positive_number(duration, "timeline.lead_durations_hours")


@dataclass(frozen=True)
class TransformContract:
    name: str
    package: str
    package_version: str
    filters: Tuple[str, ...]
    decomposition_levels: int
    boundary_mode: str
    coefficient_packing: str
    transform_normalisation: str
    shift_invariant: bool
    directional: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", tuple(self.filters))
        _strings(self, ("name", "package", "package_version", "boundary_mode",
                        "coefficient_packing", "transform_normalisation"), "transform")
        _reject_placeholders(asdict(self), "transform")
        _string_tuple(self.filters, "transform.filters")
        _positive_int(self.decomposition_levels, "transform.decomposition_levels", allow_zero=True)
        if not isinstance(self.shift_invariant, bool) or not isinstance(self.directional, bool):
            raise ForecastContractError("transform invariance/directionality flags must be booleans")


@dataclass(frozen=True)
class NormalisationContract:
    method: str
    scope: str
    fitted_split: str
    ddof: int
    statistics_artifact_sha256: str

    def __post_init__(self) -> None:
        _strings(self, ("method", "scope", "fitted_split", "statistics_artifact_sha256"),
                 "normalisation")
        _reject_placeholders(asdict(self), "normalisation")
        if self.fitted_split != "train":
            raise ForecastContractError("normalisation.fitted_split must be exactly 'train'")
        _positive_int(self.ddof, "normalisation.ddof", allow_zero=True)
        _sha256(self.statistics_artifact_sha256, "normalisation.statistics_artifact_sha256")


@dataclass(frozen=True)
class SplitContract:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    embargo_frames: int

    def __post_init__(self) -> None:
        _strings(self, ("train_start", "train_end", "validation_start", "validation_end",
                        "test_start", "test_end"), "splits")
        _reject_placeholders(asdict(self), "splits")
        values = [_timestamp(getattr(self, field), "splits.%s" % field) for field in (
            "train_start", "train_end", "validation_start", "validation_end",
            "test_start", "test_end")]
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ForecastContractError(
                "split timestamps must be strictly ordered train -> validation -> test")
        _positive_int(self.embargo_frames, "splits.embargo_frames")


@dataclass(frozen=True)
class OptimizerContract:
    class_path: str
    package_version: str
    kwargs: Mapping[str, Any]
    scheduler_class_path: Optional[str]
    scheduler_kwargs: Mapping[str, Any]
    training_steps: int
    batch_size: int
    seed: int
    gradient_clip_norm: Optional[float]

    def __post_init__(self) -> None:
        _strings(self, ("class_path", "package_version"), "optimizer")
        if self.scheduler_class_path is not None and not isinstance(self.scheduler_class_path, str):
            raise ForecastContractError("optimizer.scheduler_class_path must be a string or null")
        if not isinstance(self.kwargs, Mapping) or not isinstance(self.scheduler_kwargs, Mapping):
            raise ForecastContractError("optimizer kwargs and scheduler_kwargs must be JSON objects")
        object.__setattr__(self, "kwargs", _freeze_json(self.kwargs, "optimizer.kwargs"))
        object.__setattr__(self, "scheduler_kwargs",
                           _freeze_json(self.scheduler_kwargs, "optimizer.scheduler_kwargs"))
        _reject_placeholders(asdict(self), "optimizer")
        _canonical_json(self.kwargs)
        _canonical_json(self.scheduler_kwargs)
        if self.scheduler_class_path is None and self.scheduler_kwargs:
            raise ForecastContractError("optimizer.scheduler_kwargs must be empty when scheduler is null")
        _positive_int(self.training_steps, "optimizer.training_steps")
        _positive_int(self.batch_size, "optimizer.batch_size")
        _positive_int(self.seed, "optimizer.seed", allow_zero=True)
        if self.gradient_clip_norm is not None:
            _positive_number(self.gradient_clip_norm, "optimizer.gradient_clip_norm")


@dataclass(frozen=True)
class RolloutContract:
    mode: str
    history_usage: str
    feedback: str
    training_strategy: str
    loss_lead_frames: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "loss_lead_frames", tuple(self.loss_lead_frames))
        _strings(self, ("mode", "history_usage", "feedback", "training_strategy"), "rollout")
        _reject_placeholders(asdict(self), "rollout")
        if self.mode not in ("autoregressive", "direct_multi_horizon"):
            raise ForecastContractError("rollout.mode must be autoregressive or direct_multi_horizon")
        if self.history_usage not in ("final_frame", "full_history"):
            raise ForecastContractError("rollout.history_usage must be final_frame or full_history")
        if self.feedback not in ("predicted_state", "none"):
            raise ForecastContractError("rollout.feedback must be predicted_state or none")
        if (self.mode == "autoregressive") != (self.feedback == "predicted_state"):
            raise ForecastContractError("autoregressive rollout requires predicted_state feedback; direct rollout requires none")
        if not self.loss_lead_frames or tuple(sorted(set(self.loss_lead_frames))) != self.loss_lead_frames:
            raise ForecastContractError("rollout.loss_lead_frames must be unique, positive and increasing")
        for lead in self.loss_lead_frames:
            _positive_int(lead, "rollout.loss_lead_frames")


@dataclass(frozen=True)
class ModelContract:
    class_path: str
    model_version: str
    config: Mapping[str, Any]
    parameter_count: int
    trainable_parameter_count: int

    def __post_init__(self) -> None:
        _strings(self, ("class_path", "model_version"), "model")
        if not isinstance(self.config, Mapping):
            raise ForecastContractError("model.config must be a JSON object")
        object.__setattr__(self, "config", _freeze_json(self.config, "model.config"))
        _reject_placeholders(asdict(self), "model")
        _canonical_json(self.config)
        _positive_int(self.parameter_count, "model.parameter_count")
        _positive_int(self.trainable_parameter_count, "model.trainable_parameter_count")
        if self.trainable_parameter_count > self.parameter_count:
            raise ForecastContractError("trainable parameter count cannot exceed total parameter count")


_T = TypeVar("_T")


def _construct(section: str, cls: Type[_T], value: Any) -> _T:
    fields = tuple(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    record = dict(_require_exact_keys(value, fields, section))
    tuple_fields = {
        "latitude_bounds", "longitude_bounds", "grid_shape", "variables",
        "pressure_levels_hpa", "lead_frames", "lead_durations_hours", "filters",
        "loss_lead_frames",
    }
    for name in tuple_fields & set(record):
        if not isinstance(record[name], (list, tuple)):
            raise ForecastContractError("%s.%s must be a JSON array" % (section, name))
        record[name] = tuple(record[name])
    try:
        return cls(**record)
    except TypeError as exc:
        raise ForecastContractError("invalid protocol section %r: %s" % (section, exc)) from exc


@dataclass(frozen=True)
class MotivatingExperimentProtocol:
    """Complete declared design with canonical identity and strict schema parsing."""

    schema: str
    protocol_revision: int
    experiment_id: str
    evidence_reference: str
    evidence_sha256: str
    domain: DomainContract
    data: DataContract
    timeline: TimelineContract
    transform: TransformContract
    normalisation: NormalisationContract
    splits: SplitContract
    optimizer: OptimizerContract
    rollout: RolloutContract
    model: ModelContract

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_SCHEMA:
            raise ForecastContractError("unsupported motivating experiment protocol schema %r" % self.schema)
        _positive_int(self.protocol_revision, "protocol_revision")
        _strings(self, ("experiment_id", "evidence_reference", "evidence_sha256"), "protocol")
        _reject_placeholders({
            "experiment_id": self.experiment_id,
            "evidence_reference": self.evidence_reference,
        })
        _sha256(self.evidence_sha256, "evidence_sha256")
        expected_durations = tuple(float(self.data.cadence_hours) * lead
                                   for lead in self.timeline.lead_frames)
        if len(expected_durations) != len(self.timeline.lead_durations_hours) or any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(self.timeline.lead_durations_hours,
                                            expected_durations)):
            raise ForecastContractError(
                "timeline physical lead durations must equal cadence_hours * lead_frames")
        if self.splits.embargo_frames < max(self.timeline.lead_frames):
            raise ForecastContractError("split embargo must be at least the longest lead")
        if any(lead not in self.timeline.lead_frames for lead in self.rollout.loss_lead_frames):
            raise ForecastContractError("rollout loss leads must be declared timeline leads")

    def to_mapping(self) -> Dict[str, Any]:
        """Return the canonical JSON-compatible protocol body (without its derived hash)."""
        return _thaw_json(asdict(self))

    def fingerprint(self) -> str:
        return _fingerprint(self.to_mapping())

    def __hash__(self) -> int:
        # Stable across processes, unlike Python's salted string hash.
        return int(self.fingerprint()[:16], 16)

    def to_provenance(self) -> Dict[str, Any]:
        return {"protocol": self.to_mapping(), "protocol_sha256": self.fingerprint()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MotivatingExperimentProtocol":
        fields = tuple(cls.__dataclass_fields__)
        record = dict(_require_exact_keys(value, fields, "protocol"))
        section_types = {
            "domain": DomainContract,
            "data": DataContract,
            "timeline": TimelineContract,
            "transform": TransformContract,
            "normalisation": NormalisationContract,
            "splits": SplitContract,
            "optimizer": OptimizerContract,
            "rollout": RolloutContract,
            "model": ModelContract,
        }
        for name, section_type in section_types.items():
            record[name] = _construct(name, section_type, record[name])
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid motivating experiment protocol: %s" % exc) from exc


def save_experiment_protocol(path: Union[str, os.PathLike[str]],
                             protocol: MotivatingExperimentProtocol) -> str:
    """Write a new canonical protocol envelope, refusing to replace existing evidence."""
    if not isinstance(protocol, MotivatingExperimentProtocol):
        raise ForecastContractError("protocol must be a validated MotivatingExperimentProtocol")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation makes the no-overwrite guarantee race-safe when laptop and HPC
        # jobs share an artefact directory.
        with target.open("xb") as handle:
            handle.write(_canonical_json(protocol.to_provenance()) + b"\n")
    except FileExistsError:
        raise FileExistsError(
            "refusing to overwrite frozen experiment protocol at %s" % target) from None
    return protocol.fingerprint()


def load_experiment_protocol(path: Union[str, os.PathLike[str]]) -> MotivatingExperimentProtocol:
    """Load, validate and integrity-check one canonical protocol envelope."""
    target = Path(path)
    try:
        envelope = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read motivating experiment protocol: %s" % exc) from exc
    envelope = _require_exact_keys(envelope, ("protocol", "protocol_sha256"), "envelope")
    protocol = MotivatingExperimentProtocol.from_mapping(envelope["protocol"])
    if envelope["protocol_sha256"] != protocol.fingerprint():
        raise ForecastContractError("motivating experiment protocol SHA-256 mismatch")
    return protocol
