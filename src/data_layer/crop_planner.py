"""Transform-derived, metadata-only crop planning (R13).

A store probe answers what a source is.  This module answers a different, request-specific
question: whether one exact native-cell selection supports one exact registered transform and
analysis depth, and what the smallest defensible expansion would cost.  Keeping those records
separate prevents a transform preference from being laundered into a fact about the store.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.core.errors import InvalidParameterError
from src.transform_engine.registry import TRANSFORMS

PLAN_SCHEMA = "transform-acquisition-plan/v1"
ANALYSIS_SCHEMA = "transform-support-request/v1"

# R13's existing practical floor, now a named *analysis* policy rather than a copied filter
# constant. The transform supplies the contaminated margin; this policy supplies the minimum
# uncontaminated parent-grid span on which cross-scale spatial statistics are considered useful.
RECOMMENDED_VALID_PARENT_SIDE = 128
RECOMMENDATION_POLICY = "r13-cross-scale-valid-interior/v1"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _coordinate_observation(dataset: Any, name: str) -> Dict[str, Any]:
    """Bind a plan to the exact coordinate values used to derive native bounds."""
    values = np.ascontiguousarray(np.asarray(dataset[name].values))
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("ascii"))
    digest.update(_canonical(list(values.shape)))
    digest.update(values.tobytes(order="C"))
    return {"name": name, "dtype": str(values.dtype),
            "shape": [int(v) for v in values.shape],
            "values_sha256": digest.hexdigest()}


def _ceil_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(int(value) / int(multiple)) * int(multiple))


def _next_power_two(value: int) -> int:
    out = 1
    while out < int(value):
        out *= 2
    return out


@dataclass(frozen=True)
class TransformSupportRequest:
    """The transform identity that makes a crop-size statement meaningful."""

    transform_family: str = "dtcwt"
    levels: int = 4
    wavelet: str = "db2"
    boundary_mode: str = "periodic"
    dtcwt_level1: str = "near_sym_b"
    dtcwt_qshift: str = "qshift_b"
    recommendation_policy: str = RECOMMENDATION_POLICY

    def __post_init__(self) -> None:
        if isinstance(self.levels, bool) or int(self.levels) != self.levels or self.levels < 1:
            raise InvalidParameterError("analysis.levels", self.levels, "an integer >= 1")
        if self.recommendation_policy != RECOMMENDATION_POLICY:
            raise InvalidParameterError(
                "analysis.recommendation_policy", self.recommendation_policy,
                RECOMMENDATION_POLICY)
        try:
            entry = TRANSFORMS.get(str(self.transform_family))
        except Exception as exc:
            raise InvalidParameterError(
                "analysis.transform_family", self.transform_family,
                "a registered transform with an R13 support contract") from exc
        if entry.support is None:
            raise InvalidParameterError(
                "analysis.transform_family", self.transform_family,
                "a registered transform with an R13 support contract; this transform does "
                "not declare boundary support and cannot size an acquisition")
        # Invoke the implementation-owned support callback now, so bad filter names fail at
        # request construction rather than after a remote store has been opened.
        try:
            entry.support(int(self.levels), self.config())
        except Exception as exc:
            raise InvalidParameterError(
                "analysis.transform_config", self.config(),
                "filter settings accepted by the registered %s implementation (%s: %s)"
                % (self.transform_family, type(exc).__name__, exc)) from exc

    def config(self) -> Dict[str, Any]:
        if self.transform_family == "swt":
            return {"wavelet": self.wavelet, "mode": self.boundary_mode}
        if self.transform_family == "dtcwt":
            return {"level1": self.dtcwt_level1, "qshift": self.dtcwt_qshift}
        return {}

    def support(self) -> Dict[str, Any]:
        callback = TRANSFORMS.get(self.transform_family).support
        assert callback is not None  # enforced in __post_init__
        return callback(int(self.levels), self.config())

    def canonical(self) -> Dict[str, Any]:
        return {"schema": ANALYSIS_SCHEMA, "transform_family": self.transform_family,
                "levels": int(self.levels), "config": self.config(),
                "recommendation_policy": self.recommendation_policy}

    def digest(self) -> str:
        return _sha256(self.canonical())


def assess_shape(height: int, width: int,
                 analysis: TransformSupportRequest) -> Dict[str, Any]:
    """Assess one native-cell shape without opening a source or reading values."""
    if min(int(height), int(width)) < 1:
        raise InvalidParameterError("crop_shape", [height, width],
                                    "positive native-cell dimensions")
    support = analysis.support()
    coarsest = support["levels"][-1]
    parent_margin = int(coarsest["margin_parent_px"])
    native_margin = int(coarsest["margin_native_px"])
    factor = int(coarsest["sampling_factor"])
    alignment = int(support["alignment_cells"])

    # Absolute means one uncontaminated coefficient on each coarsest native axis. Both the
    # parent support and the decimated-native support must fit; taking the maximum prevents a
    # decimated transform from calling padded coefficients valid.
    absolute_raw = max(2 * parent_margin + 1,
                       factor * (2 * native_margin + 1))
    absolute = _ceil_multiple(absolute_raw, alignment)

    # The practical policy is stated in parent-grid span. Translate it onto the coarsest native
    # grid conservatively, then preserve the programme's dyadic operational crop convention.
    recommended_native_valid = int(math.ceil(RECOMMENDED_VALID_PARENT_SIDE / factor))
    recommended_raw = max(2 * parent_margin + RECOMMENDED_VALID_PARENT_SIDE,
                          factor * (2 * native_margin + recommended_native_valid))
    recommended = _next_power_two(_ceil_multiple(recommended_raw, alignment))

    level_rows = []
    for row in support["levels"]:
        sampling = int(row["sampling_factor"])
        p_margin = int(row["margin_parent_px"])
        n_margin = int(row["margin_native_px"])
        native_shape = [int(height) // sampling, int(width) // sampling]
        level_rows.append({**dict(row),
            "valid_parent_shape": [max(0, int(height) - 2 * p_margin),
                                   max(0, int(width) - 2 * p_margin)],
            "native_shape": native_shape,
            "valid_native_shape": [max(0, native_shape[0] - 2 * n_margin),
                                   max(0, native_shape[1] - 2 * n_margin)]})

    current = min(int(height), int(width))
    if current >= recommended:
        verdict = "recommended"
    elif current >= absolute:
        verdict = "technical_only"
    else:
        verdict = "insufficient"
    return {
        "analysis": analysis.canonical(),
        "analysis_sha256": analysis.digest(),
        "current_shape": [int(height), int(width)],
        "verdict": verdict,
        "meets_absolute_minimum": current >= absolute,
        "meets_recommended_minimum": current >= recommended,
        "absolute_minimum": {
            "shape": [absolute, absolute],
            "unrounded_required_side": absolute_raw,
            "basis": ("at least one uncontaminated coefficient on each coarsest native axis; "
                      "technical computability only, not meaningful statistical support"),
        },
        "recommended_minimum": {
            "shape": [recommended, recommended],
            "unrounded_required_side": recommended_raw,
            "valid_parent_side_policy": RECOMMENDED_VALID_PARENT_SIDE,
            "basis": ("R13 practical cross-scale policy: at least %d uncontaminated "
                      "parent-grid cells of span at the coarsest level, translated through "
                      "the transform's native sampling and rounded to a dyadic operational "
                      "crop" % RECOMMENDED_VALID_PARENT_SIDE),
        },
        "alignment_cells": alignment,
        "support_source": ("registered transform implementation; no filter length is copied "
                           "into the acquisition layer"),
        "levels": level_rows,
    }


def _expanded_positions(selected: Sequence[int], total: int,
                        target: int) -> Optional[Tuple[int, int]]:
    if int(total) < int(target) or not selected:
        return None
    start, stop = min(int(v) for v in selected), max(int(v) for v in selected)
    current = stop - start + 1
    if current >= target:
        return start, stop
    missing = target - current
    start -= missing // 2
    stop += missing - missing // 2
    if start < 0:
        stop += -start
        start = 0
    if stop >= total:
        start -= stop - total + 1
        stop = total - 1
    return max(0, start), min(total - 1, stop)


def _suggestion(dataset: Any, crop: Any, positions: Mapping[str, Sequence[int]],
                lat_name: str, lon_name: str, target: int) -> Dict[str, Any]:
    from src.data_layer import zarr_source

    lat_span = _expanded_positions(positions.get(lat_name, ()),
                                   int(dataset.sizes[lat_name]), target)
    lon_span = _expanded_positions(positions.get(lon_name, ()),
                                   int(dataset.sizes[lon_name]), target)
    if lat_span is None or lon_span is None:
        return {
            "feasible": False,
            "target_shape": [target, target],
            "reason": ("the store itself has only %dx%d native cells on these axes, so it "
                       "cannot supply the requested minimum without changing source or "
                       "analysis depth" % (int(dataset.sizes[lat_name]),
                                            int(dataset.sizes[lon_name]))),
        }
    lat = np.asarray(dataset[lat_name].values, dtype=np.float64)
    lon = np.asarray(dataset[lon_name].values, dtype=np.float64)
    lat_values = lat[list(range(lat_span[0], lat_span[1] + 1))]
    lon_values = lon[list(range(lon_span[0], lon_span[1] + 1))]
    bounds = {
        "lat_min": float(np.min(lat_values)), "lat_max": float(np.max(lat_values)),
        "lon_min": float(np.min(lon_values)), "lon_max": float(np.max(lon_values)),
    }
    expanded = replace(crop, **bounds)
    assessment = zarr_source.assess_access_pattern(dataset, expanded)
    selection = assessment["selection"]
    actual = [int(selection[lat_name]), int(selection[lon_name])]
    return {"feasible": True, "target_shape": [target, target],
            "actual_shape": actual, "bounds": bounds,
            "cost": {key: assessment[key] for key in (
                "bytes_wanted", "bytes_fetched_estimate", "megabytes_fetched_estimate",
                "amplification", "chunk_hostile", "byte_basis")}}


def plan_acquisition(dataset: Any, crop: Any,
                     analysis: TransformSupportRequest,
                     *, structure: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build a complete metadata-only acquisition plan for one open dataset."""
    from src.data_layer import zarr_source

    lat_name = "latitude" if "latitude" in dataset.sizes else "lat"
    lon_name = "longitude" if "longitude" in dataset.sizes else "lon"
    selection, positions = zarr_source._selection_plan(dataset, crop, lat_name, lon_name)
    geometry = assess_shape(int(selection[lat_name]), int(selection[lon_name]), analysis)
    observed = structure or zarr_source.describe_store(dataset, crop.variables)
    # Bounds are derived from coordinate values, not merely axis lengths. Include their exact
    # byte observations so a mutable store cannot silently reuse the identity of an older plan.
    source_observation = {
        "uri": crop.uri,
        "structure": observed,
        "coordinates": {
            lat_name: _coordinate_observation(dataset, lat_name),
            lon_name: _coordinate_observation(dataset, lon_name),
        },
    }
    absolute_target = int(geometry["absolute_minimum"]["shape"][0])
    recommended_target = int(geometry["recommended_minimum"]["shape"][0])
    suggestions = {
        "absolute": _suggestion(dataset, crop, positions, lat_name, lon_name,
                                absolute_target),
        "recommended": _suggestion(dataset, crop, positions, lat_name, lon_name,
                                   recommended_target),
    }
    core = {
        "schema": PLAN_SCHEMA,
        "crop": crop.to_provenance(),
        "analysis": analysis.canonical(),
        "source_observation_sha256": _sha256(source_observation),
        "geometry": geometry,
        "suggestions": suggestions,
    }
    return {**core, "plan_sha256": _sha256(core),
            "claim_boundary": ("metadata and coordinate arithmetic only; no field value was "
                               "read, and a recommended crop is a support/cost plan rather "
                               "than evidence that the eventual analysis will find anything")}
