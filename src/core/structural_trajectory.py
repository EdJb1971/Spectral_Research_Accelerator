"""Canonical structural trajectory contract for cross-domain mining (TG17.2).

The record deliberately preserves the translation boundary.  It is a projection beside a
content-addressed native record, not a replacement for that record and not a claim that native
magnitudes or meanings are comparable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence, Tuple

import numpy as np


SCHEMA = "structural-trajectory/v1"


class StructuralConformanceError(ValueError):
    """A native-to-structural translation violated its declared scientific contract."""


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _readonly(value: Any, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuralChannelDefinition:
    name: str
    semantics: str
    units: str
    formula: str
    benchmark_id: str

    def __post_init__(self) -> None:
        if not all(str(value).strip() for value in
                   (self.name, self.semantics, self.units, self.formula, self.benchmark_id)):
            raise StructuralConformanceError("a structural channel requires a name, semantics, "
                                             "units, exact formula and benchmark identity")


@dataclass(frozen=True)
class StructuralAdapterDeclaration:
    adapter_id: str
    adapter_version: str
    domain: str
    accepted_semantics: str
    accepted_units: str
    required_axes: Tuple[str, ...]
    required_roles: Tuple[str, ...]
    invariances: Tuple[str, ...]
    consumed_information: Tuple[str, ...]
    output_clock: str
    output_support: str
    missing_data_behavior: str
    legitimate_null_family: str
    leakage_risks: Tuple[str, ...]
    refused_operations: Tuple[str, ...]
    channels: Mapping[str, StructuralChannelDefinition]
    definition_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        channels = dict(self.channels)
        if not channels or set(channels) != {item.name for item in channels.values()}:
            raise StructuralConformanceError("adapter channels must be non-empty and keyed by "
                                             "their declared names")
        payload = {
            "adapter_id": self.adapter_id, "adapter_version": self.adapter_version,
            "domain": self.domain, "accepted_semantics": self.accepted_semantics,
            "accepted_units": self.accepted_units, "required_axes": self.required_axes,
            "required_roles": self.required_roles, "invariances": self.invariances,
            "consumed_information": self.consumed_information,
            "output_clock": self.output_clock, "output_support": self.output_support,
            "missing_data_behavior": self.missing_data_behavior,
            "legitimate_null_family": self.legitimate_null_family,
            "leakage_risks": self.leakage_risks,
            "refused_operations": self.refused_operations,
            "channels": {name: vars(channel) for name, channel in sorted(channels.items())},
        }
        object.__setattr__(self, "channels", MappingProxyType(channels))
        object.__setattr__(self, "definition_sha256", _digest(payload))


@dataclass(frozen=True)
class NativeStructuralRecord:
    domain: str
    source_id: str
    variable: str
    semantics: str
    units: str
    sample_times_seconds: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray
    native_scale_seconds: float
    native_locator: str
    assumption_violations: Tuple[str, ...] = ()
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        times = _readonly(self.sample_times_seconds, np.float64)
        values = _readonly(self.values, np.float64)
        valid = _readonly(self.valid_mask, bool)
        if times.ndim != 1 or values.shape != times.shape or valid.shape != times.shape:
            raise StructuralConformanceError("native time, value and validity arrays must be "
                                             "one-dimensional and equal length")
        if len(times) < 2 or not np.all(np.diff(times) > 0):
            raise StructuralConformanceError("native clock must contain at least two strictly "
                                             "increasing observations")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise StructuralConformanceError("native clocks and values must be finite; absence "
                                             "belongs in the validity mask")
        if not np.isfinite(self.native_scale_seconds) or self.native_scale_seconds <= 0:
            raise StructuralConformanceError("native scale must be a positive duration")
        object.__setattr__(self, "sample_times_seconds", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "content_sha256", _digest({
            "domain": self.domain, "source_id": self.source_id, "variable": self.variable,
            "semantics": self.semantics, "units": self.units,
            "native_scale_seconds": float(self.native_scale_seconds),
            "assumption_violations": tuple(self.assumption_violations),
            "array_sha256": _array_digest(times, values, valid),
        }))


@dataclass(frozen=True)
class StructuralScale:
    coordinate: float
    native_value: float
    native_units: str
    mapping: str


@dataclass(frozen=True)
class ChannelLineage:
    channel: str
    source_variable: str
    source_indices: np.ndarray
    operation: str
    parameters: Mapping[str, float]
    output_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_indices", _readonly(self.source_indices, np.int64))
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True)
class StructuralTrajectory:
    schema_id: str
    trajectory_id: str
    domain: str
    source_id: str
    variable: str
    native_semantics: str
    native_units: str
    support_start_seconds: np.ndarray
    support_end_seconds: np.ndarray
    valid_mask: np.ndarray
    structural_scales: Tuple[StructuralScale, ...]
    channels: Mapping[str, np.ndarray]
    channel_definitions: Mapping[str, StructuralChannelDefinition]
    adapter_id: str
    adapter_version: str
    adapter_definition_sha256: str
    adapter_config_sha256: str
    native_record_sha256: str
    native_record_locator: str
    native_record_retained: bool
    assumption_violations: Tuple[str, ...]
    lineage: Mapping[str, ChannelLineage]

    def __post_init__(self) -> None:
        starts = _readonly(self.support_start_seconds, np.float64)
        ends = _readonly(self.support_end_seconds, np.float64)
        valid = _readonly(self.valid_mask, bool)
        channels = {name: _readonly(values, np.float64)
                    for name, values in self.channels.items()}
        object.__setattr__(self, "support_start_seconds", starts)
        object.__setattr__(self, "support_end_seconds", ends)
        object.__setattr__(self, "valid_mask", valid)
        object.__setattr__(self, "channels", MappingProxyType(channels))
        object.__setattr__(self, "channel_definitions",
                           MappingProxyType(dict(self.channel_definitions)))
        object.__setattr__(self, "lineage", MappingProxyType(dict(self.lineage)))


def translate_standardized_level(record: NativeStructuralRecord,
                                 declaration: StructuralAdapterDeclaration,
                                 *, config: Mapping[str, Any] | None = None) -> StructuralTrajectory:
    """Benchmark-defined identity-clock projection; it never bins, fills or interpolates."""
    config = dict(config or {"ddof": 0})
    if config != {"ddof": 0}:
        raise StructuralConformanceError(
            "standardized_level accepts only its benchmarked population-standardization config")
    if record.domain != declaration.domain:
        raise StructuralConformanceError("adapter domain does not match the native record")
    if (record.semantics != declaration.accepted_semantics or
            record.units != declaration.accepted_units):
        raise StructuralConformanceError(
            "semantic leakage: native semantics and units are not those benchmarked by adapter")
    if set(declaration.channels) != {"standardized_level"}:
        raise StructuralConformanceError("this translator may emit only its benchmarked "
                                         "standardized_level channel")
    admitted = record.values[record.valid_mask]
    if len(admitted) < 2:
        raise StructuralConformanceError("standardization requires two valid native values")
    mean, scale = float(np.mean(admitted)), float(np.std(admitted, ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        raise StructuralConformanceError("standardized_level refuses a constant valid record")
    values = (record.values - mean) / scale
    ends = record.sample_times_seconds + record.native_scale_seconds
    output_sha = _array_digest(values)
    config_sha = _digest(config)
    trajectory_id = "structural:%s" % _digest({
        "native": record.content_sha256, "adapter": declaration.definition_sha256,
        "config": config_sha, "output": output_sha})
    return StructuralTrajectory(
        schema_id=SCHEMA, trajectory_id=trajectory_id, domain=record.domain,
        source_id=record.source_id, variable=record.variable,
        native_semantics=record.semantics, native_units=record.units,
        support_start_seconds=record.sample_times_seconds, support_end_seconds=ends,
        valid_mask=record.valid_mask,
        structural_scales=(StructuralScale(1.0, record.native_scale_seconds, "seconds",
                                           "coordinate * native_scale_seconds"),),
        channels={"standardized_level": values},
        channel_definitions=declaration.channels, adapter_id=declaration.adapter_id,
        adapter_version=declaration.adapter_version,
        adapter_definition_sha256=declaration.definition_sha256,
        adapter_config_sha256=config_sha, native_record_sha256=record.content_sha256,
        native_record_locator=record.native_locator, native_record_retained=True,
        assumption_violations=tuple(record.assumption_violations),
        lineage={"standardized_level": ChannelLineage(
            channel="standardized_level", source_variable=record.variable,
            source_indices=np.arange(len(record.values)),
            operation="(native_value - valid_native_mean) / valid_native_population_std",
            parameters={"valid_native_mean": mean, "valid_native_population_std": scale,
                        "ddof": 0.0}, output_sha256=output_sha)})


def assert_structural_conformance(trajectory: StructuralTrajectory,
                                  native: NativeStructuralRecord,
                                  declaration: StructuralAdapterDeclaration) -> None:
    """Reconstruct every output and reject laundering of semantics, gaps or clock support."""
    if trajectory.schema_id != SCHEMA or not trajectory.native_record_retained:
        raise StructuralConformanceError("canonical projection must retain its native record")
    if trajectory.native_record_sha256 != native.content_sha256:
        raise StructuralConformanceError("native record digest cannot be reconstructed")
    if trajectory.adapter_definition_sha256 != declaration.definition_sha256:
        raise StructuralConformanceError("adapter definition digest does not match")
    if trajectory.adapter_config_sha256 != _digest({"ddof": 0}):
        raise StructuralConformanceError("adapter configuration digest does not match")
    if (trajectory.native_semantics != native.semantics or
            trajectory.native_units != native.units or trajectory.domain != native.domain):
        raise StructuralConformanceError("semantic leakage changed native identity or units")
    expected_ends = native.sample_times_seconds + native.native_scale_seconds
    if (not np.array_equal(trajectory.support_start_seconds, native.sample_times_seconds) or
            not np.array_equal(trajectory.support_end_seconds, expected_ends) or
            not np.array_equal(trajectory.valid_mask, native.valid_mask)):
        raise StructuralConformanceError("undeclared interpolation, compaction or support change")
    if set(trajectory.channels) != set(declaration.channels):
        raise StructuralConformanceError("trajectory contains an unbenchmarked structural channel")
    if dict(trajectory.channel_definitions) != dict(declaration.channels):
        raise StructuralConformanceError("trajectory changed a benchmarked channel definition")
    if tuple(trajectory.assumption_violations) != tuple(native.assumption_violations):
        raise StructuralConformanceError("trajectory dropped or invented an assumption violation")
    if (len(trajectory.structural_scales) != 1 or
            trajectory.structural_scales[0].coordinate != 1.0 or
            trajectory.structural_scales[0].native_value != native.native_scale_seconds or
            trajectory.structural_scales[0].native_units != "seconds"):
        raise StructuralConformanceError("structural-to-native scale mapping cannot be reconstructed")
    for name, values in trajectory.channels.items():
        lineage = trajectory.lineage.get(name)
        if lineage is None or not np.array_equal(lineage.source_indices,
                                                  np.arange(len(native.values))):
            raise StructuralConformanceError("every canonical value must retain its native index")
        mean = lineage.parameters.get("valid_native_mean")
        scale = lineage.parameters.get("valid_native_population_std")
        rebuilt = (native.values - mean) / scale
        if not np.allclose(values, rebuilt, rtol=0.0, atol=1e-12):
            raise StructuralConformanceError("canonical values do not reconstruct from lineage")
        if lineage.output_sha256 != _array_digest(values):
            raise StructuralConformanceError("canonical channel digest does not match its values")


def mine_structural_peak(trajectory: StructuralTrajectory,
                         channel: str = "standardized_level") -> Mapping[str, Any]:
    """The first domain-blind mining seam: consume only the canonical contract."""
    if channel not in trajectory.channels:
        raise StructuralConformanceError("requested structural channel is unavailable")
    values = trajectory.channels[channel]
    admitted = np.flatnonzero(trajectory.valid_mask)
    if not len(admitted):
        raise StructuralConformanceError("a structural peak needs at least one valid support")
    index = int(admitted[int(np.argmax(values[admitted]))])
    return {"trajectory_id": trajectory.trajectory_id, "channel": channel,
            "support_start_seconds": float(trajectory.support_start_seconds[index]),
            "support_end_seconds": float(trajectory.support_end_seconds[index]),
            "value": float(values[index]), "units": trajectory.channel_definitions[channel].units}


def with_support_for_conformance_test(trajectory: StructuralTrajectory,
                                      starts: Sequence[float]) -> StructuralTrajectory:
    """Build a changed immutable record for conformance tests without mutating production data."""
    return replace(trajectory, support_start_seconds=np.asarray(starts, dtype=np.float64))


__all__ = ["SCHEMA", "ChannelLineage", "NativeStructuralRecord",
           "StructuralAdapterDeclaration", "StructuralChannelDefinition",
           "StructuralConformanceError", "StructuralScale", "StructuralTrajectory",
           "assert_structural_conformance", "mine_structural_peak",
           "translate_standardized_level", "with_support_for_conformance_test"]
