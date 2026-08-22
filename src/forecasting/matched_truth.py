"""Lazy ERA5/CDS truth matching for external regional forecasts (roadmap T5.6d).

This module turns the canonical regional pressure-level cube into the two physical datasets
consumed by :mod:`src.forecasting.ensemble_evaluation`.  Coordinate lookup is exact and lazy:
initialization analyses and verifying valid times are selected by integer index without
interpolation, nearest-neighbour matching, unit conversion or loading field values.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import xarray as xr

from src.data_layer.regional_forecast import VARIABLE_ALIASES
from src.forecasting.adapter import ForecastContractError
from src.forecasting.ensemble_evaluation import INITIAL_STATE_SCHEMA, MATCHED_TRUTH_SCHEMA
from src.forecasting.external_cube import (
    RegionalForecastCube,
    fcn3_variable_unit,
    grid_coordinates_sha256,
)


MATCHED_TRUTH_BUILDER_SCHEMA = "matched-truth-builder/fcn3-era5-v1"
FRESH_HOLDOUT_ROLE = "fresh_post_2019_holdout"
PUBLISHED_OVERLAP_ROLE = "published_partition_diagnostic"
FCN3_PUBLISHED_TRAIN_START = np.datetime64("1980-01-01T00:00:00", "ns")
FCN3_PUBLISHED_TEST_START = np.datetime64("2016-01-01T00:00:00", "ns")
FCN3_PUBLISHED_EVALUATION_START = np.datetime64("2018-01-01T00:00:00", "ns")
FCN3_POST_PUBLISHED_START = np.datetime64("2020-01-01T00:00:00", "ns")
_HEX = re.compile(r"^[0-9a-fA-F]+$")


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                             allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError("source provenance must be finite JSON: %s" % exc) from exc
    return hashlib.sha256(encoded).hexdigest()


def _datetime_ns(values: Any, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.datetime64):
        raise ForecastContractError("%s must be a one-dimensional datetime64 coordinate" % label)
    result = array.astype("datetime64[ns]")
    if np.any(np.isnat(result)):
        raise ForecastContractError("%s contains NaT" % label)
    return result


def _timestamp(value: str, label: str) -> np.datetime64:
    if not isinstance(value, str) or not value.strip():
        raise ForecastContractError("%s must be an explicit ISO timestamp" % label)
    try:
        result = np.datetime64(value, "ns")
    except (TypeError, ValueError):
        raise ForecastContractError("%s must be an explicit ISO timestamp" % label) from None
    if np.isnat(result):
        raise ForecastContractError("%s must not be NaT" % label)
    return result


def _duration_ns(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.timedelta64):
        raise ForecastContractError("forecast lead_time must be a one-dimensional timedelta64 axis")
    result = array.astype("timedelta64[ns]").astype("int64")
    if np.any(result <= 0) or np.any(np.diff(result) <= 0):
        raise ForecastContractError("forecast lead_time must be positive and strictly increasing")
    return result


def _period(value: np.datetime64) -> str:
    if value < FCN3_PUBLISHED_TRAIN_START:
        return "before_published_record"
    if value < FCN3_PUBLISHED_TEST_START:
        return "published_train_1980_2015"
    if value < FCN3_PUBLISHED_EVALUATION_START:
        return "published_test_2016_2017"
    if value < FCN3_POST_PUBLISHED_START:
        return "published_evaluation_2018_2019"
    return "post_published_2020_onward"


def _exact_indices(source_times: np.ndarray, wanted: np.ndarray, label: str) -> np.ndarray:
    source_ns = source_times.astype("datetime64[ns]").astype("int64")
    wanted_ns = wanted.astype("datetime64[ns]").astype("int64")
    positions = np.searchsorted(source_ns, wanted_ns)
    valid = positions < len(source_ns)
    matched = np.zeros(valid.shape, dtype=bool)
    matched[valid] = source_ns[positions[valid]] == wanted_ns[valid]
    if not np.all(matched):
        missing = wanted.reshape(-1)[np.flatnonzero(~matched.reshape(-1))[:5]]
        rendered = [np.datetime_as_string(value, unit="s") for value in missing]
        raise ForecastContractError(
            "%s are absent from the ERA5 source time axis: %r; nearest-time matching is refused"
            % (label, rendered))
    return positions


@dataclass(frozen=True)
class MatchedTruthBuild:
    """Lazy truth pair plus a content-addressed selection receipt."""

    truth: xr.Dataset
    initial_state: xr.Dataset
    provenance: Mapping[str, Any]

    def fingerprint(self) -> str:
        return _canonical_hash(self.provenance)


def build_matched_truth(
    source: xr.Dataset,
    source_manifest: Mapping[str, Any],
    forecast: RegionalForecastCube,
    *,
    split: str,
    split_start: str,
    split_end: str,
    evaluation_role: str,
    level_hpa: int = 850,
) -> MatchedTruthBuild:
    """Select exact initialization analyses and valid-time truth from canonical ERA5/CDS.

    ``fresh_post_2019_holdout`` accepts only samples wholly on or after 2020-01-01.
    ``published_partition_diagnostic`` permits FCN3's published 1980--2019 partitions but marks
    the result as diagnostic. Dates before the published 1980 record are refused in v1 because
    their relationship to the pinned model has not been declared.
    """
    if not isinstance(source, xr.Dataset):
        raise ForecastContractError("source must be an xarray Dataset")
    if not isinstance(forecast, RegionalForecastCube):
        raise ForecastContractError("forecast must be a validated RegionalForecastCube")
    if not isinstance(source_manifest, Mapping):
        raise ForecastContractError("source_manifest must be a provenance mapping")
    content_hash = source_manifest.get("content_hash")
    if (not isinstance(content_hash, str) or len(content_hash) not in (32, 64)
            or not _HEX.fullmatch(content_hash)):
        raise ForecastContractError("source_manifest.content_hash must be a 32- or 64-character hex digest")
    if not isinstance(split, str) or not split.strip() or split.lower() == "train":
        raise ForecastContractError("split must be explicit and held out from training")
    if evaluation_role not in (FRESH_HOLDOUT_ROLE, PUBLISHED_OVERLAP_ROLE):
        raise ForecastContractError(
            "evaluation_role must be fresh_post_2019_holdout or published_partition_diagnostic")
    if isinstance(level_hpa, bool) or int(level_hpa) != level_hpa or int(level_hpa) != 850:
        raise ForecastContractError("T5.6d requires the declared 850-hPa bridge")

    forecast_ds = forecast.dataset
    required_forecast_provenance = {
        "schema", "request_sha256", "result_sha256", "artifact_sha256", "validation_sha256",
        "grid_shape", "grid_coordinates_sha256",
    }
    missing_forecast = sorted(required_forecast_provenance - set(forecast.provenance))
    if missing_forecast:
        raise ForecastContractError("regional forecast provenance is missing %s" % missing_forecast)
    if forecast.provenance["schema"] != "regional-external-forecast-cube/fcn3-v1":
        raise ForecastContractError("regional forecast provenance schema is not accepted")
    for name in ("request_sha256", "result_sha256", "artifact_sha256", "validation_sha256"):
        value = forecast.provenance[name]
        if not isinstance(value, str) or len(value) != 64 or not _HEX.fullmatch(value):
            raise ForecastContractError("regional forecast %s must be a SHA-256" % name)
    required_dims = {"time", "ensemble", "lead_time", "lat", "lon"}
    if set(forecast_ds.dims) != required_dims:
        raise ForecastContractError("regional forecast does not have the canonical dimension set")
    variables = tuple(forecast_ds.data_vars)
    if not variables:
        raise ForecastContractError("regional forecast contains no variables")
    source_times = _datetime_ns(source.coords.get("time", []), "ERA5 source time")
    if len(source_times) < 2 or np.any(np.diff(source_times.astype("int64")) <= 0):
        raise ForecastContractError("ERA5 source time must be unique and strictly increasing")
    initial_times = _datetime_ns(forecast_ds.time.values, "forecast initialization time")
    if len(initial_times) == 0 or np.any(np.diff(initial_times.astype("int64")) <= 0):
        raise ForecastContractError("forecast initialization time must be unique and strictly increasing")
    lead_ns = _duration_ns(forecast_ds.lead_time.values)
    valid_times = initial_times[:, None] + lead_ns[None, :].astype("timedelta64[ns]")

    lower, upper = _timestamp(split_start, "split_start"), _timestamp(split_end, "split_end")
    if lower > upper:
        raise ForecastContractError("split_start must not be after split_end")
    all_required = np.concatenate((initial_times, valid_times.reshape(-1)))
    if np.any(all_required < lower) or np.any(all_required > upper):
        raise ForecastContractError(
            "every initialization and valid time must lie inside the declared held-out split")
    periods = tuple(sorted({_period(value) for value in all_required}))
    if "before_published_record" in periods:
        raise ForecastContractError("dates before FCN3's published 1980 record are unsupported in v1")
    has_published_overlap = any(value != "post_published_2020_onward" for value in periods)
    if evaluation_role == FRESH_HOLDOUT_ROLE and has_published_overlap:
        raise ForecastContractError(
            "fresh_post_2019_holdout cannot include FCN3 published train/test/evaluation dates")
    if evaluation_role == PUBLISHED_OVERLAP_ROLE and not has_published_overlap:
        raise ForecastContractError(
            "published_partition_diagnostic requires at least one 1980-2019 sample; use the fresh role")

    lat_name = "latitude" if "latitude" in source.coords else "lat" if "lat" in source.coords else None
    lon_name = "longitude" if "longitude" in source.coords else "lon" if "lon" in source.coords else None
    if lat_name is None or lon_name is None:
        raise ForecastContractError("ERA5 source requires latitude/longitude coordinates")
    source_lat = np.asarray(source[lat_name].values)
    source_lon = np.asarray(source[lon_name].values)
    if (not np.array_equal(source_lat, forecast_ds.lat.values)
            or not np.array_equal(source_lon, forecast_ds.lon.values)):
        raise ForecastContractError(
            "ERA5 latitude/longitude must exactly match the regional forecast; regridding is refused")
    grid_hash = grid_coordinates_sha256(source_lat, source_lon)
    if list(forecast.provenance["grid_shape"]) != [len(source_lat), len(source_lon)]:
        raise ForecastContractError("regional forecast grid shape provenance does not match ERA5")
    if forecast.provenance.get("grid_coordinates_sha256") != grid_hash:
        raise ForecastContractError("regional forecast grid provenance does not match ERA5 coordinates")

    initial_indices = _exact_indices(source_times, initial_times, "initialization times")
    valid_indices = _exact_indices(source_times, valid_times, "forecast valid times")
    initial_indexer = xr.DataArray(initial_indices, dims=("time",))
    valid_indexer = xr.DataArray(valid_indices, dims=("time", "lead_time"))
    truth_arrays: Dict[str, xr.DataArray] = {}
    initial_arrays: Dict[str, xr.DataArray] = {}
    resolved_names: Dict[str, str] = {}
    for forecast_name in variables:
        suffix = str(int(level_hpa))
        if not forecast_name.endswith(suffix):
            raise ForecastContractError("forecast variable %r is not an 850-hPa channel" % forecast_name)
        canonical = forecast_name[:-len(suffix)]
        if canonical not in VARIABLE_ALIASES:
            raise ForecastContractError("forecast variable %r has no canonical ERA5 bridge" % forecast_name)
        matches = [name for name in VARIABLE_ALIASES[canonical] if name in source.data_vars]
        if len(matches) != 1:
            raise ForecastContractError(
                "ERA5 source must contain exactly one alias for %s; found %r" % (canonical, matches))
        name = matches[0]
        array = source[name]
        if "level" not in array.dims or "level" not in source.coords:
            raise ForecastContractError("ERA5 source variable %s must retain an explicit level axis" % name)
        levels = np.asarray(source.level.values)
        level_matches = np.flatnonzero(levels.astype(float) == float(level_hpa))
        if len(level_matches) != 1:
            raise ForecastContractError("ERA5 source must contain exactly one 850-hPa level")
        array = array.isel(level=int(level_matches[0]), drop=True)
        if set(array.dims) != {"time", lat_name, lon_name}:
            raise ForecastContractError(
                "ERA5 variable %s must contain only time, level, latitude and longitude" % name)
        if not np.issubdtype(array.dtype, np.floating):
            raise ForecastContractError("ERA5 variable %s must have a floating dtype" % name)
        unit = fcn3_variable_unit(forecast_name)
        if array.attrs.get("units") != unit:
            raise ForecastContractError("ERA5 %s units must be %r" % (name, unit))
        array = array.transpose("time", lat_name, lon_name).rename({lat_name: "lat", lon_name: "lon"})
        initial = array.isel(time=initial_indexer).assign_coords(time=forecast_ds.time.values)
        truth = array.isel(time=valid_indexer).assign_coords(
            time=forecast_ds.time.values, lead_time=forecast_ds.lead_time.values)
        initial.name = truth.name = forecast_name
        initial.attrs = {"units": unit}
        truth.attrs = {"units": unit}
        initial_arrays[forecast_name] = initial.transpose("time", "lat", "lon")
        truth_arrays[forecast_name] = truth.transpose("time", "lead_time", "lat", "lon")
        resolved_names[forecast_name] = name

    source_sha256 = _canonical_hash(source_manifest)
    selection = {
        "schema": MATCHED_TRUTH_BUILDER_SCHEMA,
        "source_sha256": source_sha256,
        "source_content_hash": content_hash.lower(),
        "forecast_request_sha256": forecast.provenance.get("request_sha256"),
        "forecast_result_sha256": forecast.provenance.get("result_sha256"),
        "forecast_artifact_sha256": forecast.provenance.get("artifact_sha256"),
        "forecast_validation_sha256": forecast.provenance.get("validation_sha256"),
        "forecast_provenance_sha256": _canonical_hash(dict(forecast.provenance)),
        "forecast_grid_coordinates_sha256": grid_hash,
        "variables": list(variables),
        "resolved_source_variables": resolved_names,
        "level_hpa": int(level_hpa),
        "split": split,
        "split_start": np.datetime_as_string(lower, unit="s"),
        "split_end": np.datetime_as_string(upper, unit="s"),
        "evaluation_role": evaluation_role,
        "fcn3_periods": list(periods),
        "initialization_times": [np.datetime_as_string(value, unit="s") for value in initial_times],
        "lead_durations_ns": [int(value) for value in lead_ns],
        "valid_times_sha256": _canonical_hash([
            np.datetime_as_string(value, unit="s") for value in valid_times.reshape(-1)]),
        "selection_semantics": "exact integer time/grid/level selection; no interpolation or conversion",
        "claim_boundary": (
            "This receipt establishes exact lazy ERA5 matching and declared FCN3-period overlap, "
            "not source independence, forecast skill, calibration or statistical significance."),
    }
    builder_sha256 = _canonical_hash(selection)
    common_attrs = {
        "source_sha256": source_sha256,
        "source_content_hash": content_hash.lower(),
        "builder_schema": MATCHED_TRUTH_BUILDER_SCHEMA,
        "builder_sha256": builder_sha256,
        "split": split,
        "split_start": selection["split_start"],
        "split_end": selection["split_end"],
        "evaluation_role": evaluation_role,
        "fcn3_periods": ",".join(periods),
        "write_complete": "true",
    }
    truth_ds = xr.Dataset(truth_arrays, coords={
        "time": forecast_ds.time.values, "lead_time": forecast_ds.lead_time.values,
        "lat": forecast_ds.lat.values, "lon": forecast_ds.lon.values,
    }, attrs={"schema": MATCHED_TRUTH_SCHEMA, **common_attrs})
    initial_ds = xr.Dataset(initial_arrays, coords={
        "time": forecast_ds.time.values, "lat": forecast_ds.lat.values,
        "lon": forecast_ds.lon.values,
    }, attrs={"schema": INITIAL_STATE_SCHEMA, **common_attrs})
    provenance = {**selection, "builder_sha256": builder_sha256,
                  "truth_schema": MATCHED_TRUTH_SCHEMA,
                  "initial_state_schema": INITIAL_STATE_SCHEMA,
                  "lazy": all(getattr(value.data, "chunks", None) is not None
                              for value in tuple(truth_arrays.values()) + tuple(initial_arrays.values()))}
    return MatchedTruthBuild(truth_ds, initial_ds, provenance)
