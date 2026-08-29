"""Declared reductions from irregular profiles to channel series (TG12.2c).

The reduction is scientific content, not formatting.  It declares which acquisition axes it
consumes, derives the resulting domain declaration, and enters the content identity of the
channel series.  Two reductions over identical profile bytes therefore cannot share provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from src.core.channel_series import ChannelSeries, assert_presence_contract
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError
from src.core.registry import Registry
from src.data_layer.profiles import ProfileCollection, stable_sha256


PROFILE_REDUCTIONS: Registry["ProfileReduction"] = Registry("profile reduction")
REDUCTION_SCHEMA = "spectral.profile-reduction.v1"


def _finite_number(config: Mapping[str, Any], key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(key, config.get(key), "a finite number") from exc
    if not np.isfinite(value):
        raise InvalidParameterError(key, value, "a finite number")
    return value


def _variable(collection: ProfileCollection, config: Mapping[str, Any]) -> str:
    name = str(config.get("variable", collection.spec.variables[0])).strip().lower()
    if name not in collection.measures:
        raise InvalidParameterError("variable", name,
                                    "one of the acquired measures %s"
                                    % sorted(collection.measures))
    return name


def _clock_is_regular(times: np.ndarray) -> bool:
    if times.size < 3:
        return True
    intervals = np.diff(times)
    return bool(np.allclose(intervals, intervals[0], rtol=0.0,
                            atol=1e-9 * max(abs(float(intervals[0])), 1.0)))


@dataclass(frozen=True)
class ReducedProfiles:
    series: ChannelSeries
    declaration: DomainDeclaration
    reduction: Mapping[str, Any]

    def describe(self) -> Dict[str, Any]:
        return {
            "schema": REDUCTION_SCHEMA,
            "reduction": dict(self.reduction),
            "derived_declaration": self.declaration.describe(),
            "n_frames": self.series.n_times, "n_channels": self.series.n_channels,
            "channels": [str(value) for value in self.series.channels],
            "channel_records": list(self.series.channel_records),
            "clock": {
                "strictly_increasing": True,
                "regular": bool(self.series.provenance.get("clock_regular")),
                "cadence_seconds": self.series.provenance.get("cadence_seconds"),
            },
            "content_sha256": self.series.provenance["content_sha256"],
            "claim_boundary": (
                "A registered reduction records how acquisition axes became scalar channels. "
                "Its digest establishes identity and reproducibility, not that the reduction "
                "is appropriate for a particular scientific question."),
        }


class ProfileReduction:
    """One registered axis-consuming reduction."""

    name = ""
    consumes_axes: Tuple[str, ...] = ()
    output_axis = ""

    def normalise(self, collection: ProfileCollection,
                  configuration: Mapping[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def derive_declaration(self, parent: DomainDeclaration,
                           configuration: Mapping[str, Any]) -> DomainDeclaration:
        raise NotImplementedError

    def apply(self, collection: ProfileCollection, parent: DomainDeclaration,
              configuration: Mapping[str, Any]) -> ReducedProfiles:
        config = self.normalise(collection, configuration)
        declaration = self.derive_declaration(parent, config)
        series, details = self.reduce(collection, config)
        reduction_body = {
            "schema": REDUCTION_SCHEMA, "name": self.name,
            "configuration": config, "consumes_axes": list(self.consumes_axes),
            "output_axis": self.output_axis,
            "parent_collection_sha256": collection.collection_sha256(),
            "derived_declaration": declaration.describe(), **details,
        }
        reduction_sha256 = stable_sha256(reduction_body)
        provenance = {**dict(series.provenance),
                      "profile_collection_sha256": collection.collection_sha256(),
                      "profile_request_sha256": collection.spec.request_sha256(),
                      "reduction": reduction_body, "reduction_sha256": reduction_sha256,
                      "content_sha256": reduction_sha256}
        series = ChannelSeries(
            channels=series.channels, times_seconds=series.times_seconds,
            measures=series.measures, present=series.present, usable=series.usable,
            support_parent_px=series.support_parent_px,
            unusable_reason=series.unusable_reason, provenance=provenance)
        assert_presence_contract(series, declaration.violations, declaration.name)
        return ReducedProfiles(series=series, declaration=declaration,
                               reduction={**reduction_body,
                                          "reduction_sha256": reduction_sha256})

    def reduce(self, collection: ProfileCollection,
               configuration: Mapping[str, Any]) -> Tuple[ChannelSeries, Dict[str, Any]]:
        raise NotImplementedError


class PerFloatAtPressure(ProfileReduction):
    """One interpolated scalar per float and profile, without invented simultaneity."""

    name = "per_float_at_pressure"
    consumes_axes = ("pressure", "latitude", "longitude")
    output_axis = "float_id"

    def normalise(self, collection: ProfileCollection,
                  configuration: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = sorted(set(configuration) - {"variable", "pressure_dbar",
                                                "max_interpolation_gap_dbar"})
        if unknown:
            raise InvalidParameterError("reduction configuration", unknown,
                                        "only variable, pressure_dbar and "
                                        "max_interpolation_gap_dbar")
        pressure = _finite_number(configuration, "pressure_dbar", 100.0)
        gap = _finite_number(configuration, "max_interpolation_gap_dbar", 25.0)
        if not collection.spec.pressure_min_dbar <= pressure \
                <= collection.spec.pressure_max_dbar:
            raise InvalidParameterError("pressure_dbar", pressure,
                                        "inside the acquired pressure range")
        if gap <= 0:
            raise InvalidParameterError("max_interpolation_gap_dbar", gap,
                                        "a positive physical span")
        return {"variable": _variable(collection, configuration),
                "pressure_dbar": pressure, "max_interpolation_gap_dbar": gap,
                "interpolation": "linear_bracket_no_extrapolation"}

    def derive_declaration(self, parent: DomainDeclaration,
                           configuration: Mapping[str, Any]) -> DomainDeclaration:
        return DomainDeclaration(
            name=parent.name,
            description=("%s Reduced to one %s value per float profile at %.6g dbar; "
                         "latitude and longitude remain recorded per-sample attributes."
                         % (parent.description, configuration["variable"],
                            configuration["pressure_dbar"])),
            axes=(AxisSpec(name="cycle_time", role="time", units="s"),
                  AxisSpec(name="float_id", role="category", ordered=False)),
            licence=parent.licence, violations=tuple(parent.violations),
            lag_policy=parent.lag_policy,
            declared_floor_frames=parent.declared_floor_frames,
            declared_floor_basis=parent.declared_floor_basis,
            provenance={**dict(parent.provenance), "derived_from": parent.name,
                        "profile_reduction": self.name,
                        "axis_consumption": list(self.consumes_axes),
                        "per_sample_attributes": ["latitude", "longitude"]})

    @staticmethod
    def _interpolate(pressure: np.ndarray, values: np.ndarray, target: float,
                     maximum_gap: float) -> float:
        valid = np.isfinite(values)
        p, y = pressure[valid], values[valid]
        if p.size < 1:
            return float("nan")
        exact = np.flatnonzero(np.isclose(p, target, rtol=0.0, atol=1e-9))
        if exact.size:
            return float(y[int(exact[0])])
        right = int(np.searchsorted(p, target, side="right"))
        if right == 0 or right == p.size:
            return float("nan")
        left = right - 1
        span = float(p[right] - p[left])
        if span <= 0 or span > maximum_gap:
            return float("nan")
        weight = (target - float(p[left])) / span
        return float(y[left] + weight * (y[right] - y[left]))

    def reduce(self, collection: ProfileCollection,
               configuration: Mapping[str, Any]) -> Tuple[ChannelSeries, Dict[str, Any]]:
        variable = str(configuration["variable"])
        values = [self._interpolate(p, v, float(configuration["pressure_dbar"]),
                                    float(configuration["max_interpolation_gap_dbar"]))
                  for p, v in zip(collection.pressure_dbar, collection.measures[variable])]
        by_platform: Dict[str, list[int]] = {}
        for index, (platform, value) in enumerate(zip(collection.platform_ids, values)):
            if np.isfinite(value):
                by_platform.setdefault(platform, []).append(index)
        admitted = sorted(platform for platform, indices in by_platform.items()
                          if len(indices) >= 2)
        excluded = {platform: len(indices) for platform, indices in sorted(by_platform.items())
                    if len(indices) < 2}
        if len(admitted) < 2:
            raise InvalidParameterError(
                "per-float reduction", {key: len(value) for key, value in by_platform.items()},
                "at least two floats with at least two valid reduced profiles each. Increase "
                "the time/region range, choose a pressure covered by more profiles, or widen "
                "the declared interpolation gap; a one-float result has no cross-channel pair")
        selected_indices = sorted(index for platform in admitted for index in by_platform[platform])
        clock = np.unique(collection.times_seconds[selected_indices])
        matrix = np.full((clock.size, len(admitted)), np.nan, dtype=np.float64)
        present = np.zeros(matrix.shape, dtype=bool)
        locations: Dict[str, list[Dict[str, float]]] = {platform: [] for platform in admitted}
        row_for = {float(value): index for index, value in enumerate(clock)}
        col_for = {platform: index for index, platform in enumerate(admitted)}
        for index in selected_indices:
            platform = collection.platform_ids[index]
            row, column = row_for[float(collection.times_seconds[index])], col_for[platform]
            if present[row, column]:
                raise InvalidParameterError(
                    "profile clock", collection.profile_ids[index],
                    "at most one profile per float at an exact timestamp. Combining two "
                    "profiles would be an undeclared reduction")
            matrix[row, column] = values[index]
            present[row, column] = True
            locations[platform].append({"time_seconds": float(collection.times_seconds[index]),
                                        "latitude": float(collection.latitude[index]),
                                        "longitude": float(collection.longitude[index])})
        series = ChannelSeries(
            channels=admitted, times_seconds=clock, measures={variable: matrix},
            present=present,
            provenance={"adapter": "profile_reduction", "clock_regular": _clock_is_regular(clock),
                        "cadence_seconds": None, "per_sample_locations": locations})
        return series, {"excluded_platforms_with_fewer_than_two_values": excluded,
                        "intermittent_support_preserved": True,
                        "aggregation_window": None}


class DepthBinMean(ProfileReduction):
    """Mean within declared pressure bins, pooling float identity and position."""

    name = "depth_bin_mean"
    consumes_axes = ("float_id", "latitude", "longitude", "pressure")
    output_axis = "pressure_bin"

    def normalise(self, collection: ProfileCollection,
                  configuration: Mapping[str, Any]) -> Dict[str, Any]:
        unknown = sorted(set(configuration) - {"variable", "bin_edges_dbar"})
        if unknown:
            raise InvalidParameterError("reduction configuration", unknown,
                                        "only variable and bin_edges_dbar")
        raw = configuration.get("bin_edges_dbar", [0.0, 50.0, 100.0, 200.0])
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            raise InvalidParameterError("bin_edges_dbar", raw,
                                        "at least three strictly increasing pressure edges")
        try:
            edges = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError("bin_edges_dbar", raw,
                                        "finite pressure values") from exc
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise InvalidParameterError("bin_edges_dbar", raw,
                                        "finite strictly increasing pressure values")
        if edges[0] < collection.spec.pressure_min_dbar \
                or edges[-1] > collection.spec.pressure_max_dbar:
            raise InvalidParameterError("bin_edges_dbar", edges,
                                        "inside the acquired pressure range")
        return {"variable": _variable(collection, configuration),
                "bin_edges_dbar": edges, "statistic": "arithmetic_mean_of_valid_levels"}

    def derive_declaration(self, parent: DomainDeclaration,
                           configuration: Mapping[str, Any]) -> DomainDeclaration:
        violations = tuple(value for value in parent.violations
                           if value != "non_stationary_support")
        if "aggregated_values" not in violations:
            violations += ("aggregated_values",)
        return DomainDeclaration(
            name=parent.name,
            description=("%s Reduced to pressure-bin means pooled over float identity and "
                         "position; the bin edges are part of the reduction identity."
                         % parent.description),
            axes=(AxisSpec(name="cycle_time", role="time", units="s"),
                  AxisSpec(name="pressure_bin", role="category", units="dbar", ordered=True)),
            licence=parent.licence, violations=violations,
            lag_policy=parent.lag_policy,
            declared_floor_frames=parent.declared_floor_frames,
            declared_floor_basis=parent.declared_floor_basis,
            provenance={**dict(parent.provenance), "derived_from": parent.name,
                        "profile_reduction": self.name,
                        "axis_consumption": list(self.consumes_axes),
                        "aggregation_window": {"axis": "pressure", "units": "dbar",
                                               "edges": list(configuration["bin_edges_dbar"])}})

    def reduce(self, collection: ProfileCollection,
               configuration: Mapping[str, Any]) -> Tuple[ChannelSeries, Dict[str, Any]]:
        edges = np.asarray(configuration["bin_edges_dbar"], dtype=np.float64)
        labels = ["%.6g-%.6g dbar" % (edges[i], edges[i + 1])
                  for i in range(edges.size - 1)]
        order = np.argsort(collection.times_seconds, kind="stable")
        clock = np.asarray(collection.times_seconds, dtype=np.float64)[order]
        if np.any(np.diff(clock) <= 0):
            raise InvalidParameterError(
                "profile times", "duplicate exact timestamps",
                "strictly increasing profile times for depth-bin reduction. Averaging profiles "
                "that happen to share a timestamp would be a second, undeclared reduction")
        matrix = np.full((clock.size, len(labels)), np.nan, dtype=np.float64)
        variable = str(configuration["variable"])
        for row, index in enumerate(order):
            pressure = collection.pressure_dbar[int(index)]
            values = collection.measures[variable][int(index)]
            for column in range(len(labels)):
                upper_closed = column == len(labels) - 1
                inside = ((pressure >= edges[column]) &
                          (pressure <= edges[column + 1] if upper_closed
                           else pressure < edges[column + 1]) & np.isfinite(values))
                if np.any(inside):
                    matrix[row, column] = float(np.mean(values[inside]))
        series = ChannelSeries(
            channels=labels, times_seconds=clock, measures={variable: matrix}, present=None,
            provenance={"adapter": "profile_reduction", "clock_regular": _clock_is_regular(clock),
                        "cadence_seconds": None,
                        "aggregation_window": {"axis": "pressure", "units": "dbar",
                                               "edges": edges.tolist()}})
        return series, {"intermittent_support_preserved": False,
                        "aggregation_window": {"axis": "pressure", "units": "dbar",
                                               "edges": edges.tolist()}}


PROFILE_REDUCTIONS.add(
    PerFloatAtPressure.name, PerFloatAtPressure(),
    description="One QC-admitted, bracket-interpolated scalar per float profile.",
    params={"variable": {"type": "string"}, "pressure_dbar": {"type": "number"},
            "max_interpolation_gap_dbar": {"type": "number", "exclusiveMinimum": 0}},
    capabilities={"preserves_non_stationary_support": True,
                  "introduces_aggregated_values": False})
PROFILE_REDUCTIONS.add(
    DepthBinMean.name, DepthBinMean(),
    description="Arithmetic means in explicit pressure bins, pooling float identity and position.",
    params={"variable": {"type": "string"},
            "bin_edges_dbar": {"type": "array", "minItems": 3}},
    capabilities={"preserves_non_stationary_support": False,
                  "introduces_aggregated_values": True})


def reduce_profiles(collection: ProfileCollection, parent: DomainDeclaration, name: str,
                    configuration: Mapping[str, Any]) -> ReducedProfiles:
    return PROFILE_REDUCTIONS.get(name).apply(collection, parent, configuration)


__all__ = ["PROFILE_REDUCTIONS", "ProfileReduction", "ReducedProfiles", "reduce_profiles"]
