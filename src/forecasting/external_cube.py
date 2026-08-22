"""Lazy, bounded-memory acceptance of canonical external forecast cubes (T5.6b).

The optional FCN3 worker writes ordinary NetCDF4 or Zarr.  This module is deliberately
worker-independent: it uses only the portable scientific stack, binds the bytes to the
T5.6a request/result manifests, validates one storage chunk at a time, and returns an xarray
view of an explicitly declared regional crop.  It never imports Earth2Studio or CUDA code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from src.forecasting.adapter import ForecastContractError
from src.forecasting.external_fcn3 import (
    FCN3_GRID_DEGREES,
    FCN3_GRID_SHAPE,
    FCN3_VARIABLES,
    ExternalForecastResult,
    ExternalForecastRun,
    verify_external_forecast_result,
)


CANONICAL_FORECAST_CUBE_SCHEMA = "canonical-forecast-cube/fcn3-v1"
FORECAST_CUBE_VALIDATION_SCHEMA = "forecast-cube-validation/fcn3-v1"
CANONICAL_DIMS = ("time", "ensemble", "lead_time", "lat", "lon")
DEFAULT_MAX_CHUNK_BYTES = 64 * 1024 * 1024


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grid_coordinates_sha256(latitude: Sequence[float], longitude: Sequence[float]) -> str:
    """Return the canonical identity used by FCN3 input and output grid contracts."""
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    if lat.ndim != 1 or lon.ndim != 1 or not np.issubdtype(lat.dtype, np.number) or not np.issubdtype(
            lon.dtype, np.number):
        raise ForecastContractError("latitude and longitude must be one-dimensional numeric arrays")
    lat64, lon64 = lat.astype("float64"), lon.astype("float64")
    if not np.isfinite(lat64).all() or not np.isfinite(lon64).all():
        raise ForecastContractError("latitude and longitude coordinates must be finite")
    return _canonical_hash({"latitude": lat64.tolist(), "longitude": lon64.tolist()})


def fcn3_variable_unit(variable: str) -> str:
    """Canonical SI unit for an Earth2Studio FCN3 variable identifier."""
    if variable not in FCN3_VARIABLES:
        raise ForecastContractError("no canonical FCN3 unit is registered for %r" % variable)
    if variable in ("u10m", "v10m", "u100m", "v100m") or variable.startswith(("u", "v")):
        return "m s-1"
    if variable == "tcwv":
        return "kg m-2"
    if variable == "t2m" or variable.startswith("t"):
        return "K"
    if variable == "msl":
        return "Pa"
    if variable.startswith("q"):
        return "kg kg-1"
    if variable.startswith("z"):
        return "m2 s-2"
    raise ForecastContractError("no canonical FCN3 unit is registered for %r" % variable)


@dataclass(frozen=True)
class GeographicBounds:
    """An inclusive, grid-aligned, non-wrapping regional selection."""

    south: float
    north: float
    west: float
    east: float
    longitude_convention: str
    label: str = "New Zealand"

    def __post_init__(self) -> None:
        values = (self.south, self.north, self.west, self.east)
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(float(value)) for value in values):
            raise ForecastContractError("regional bounds must be finite numbers")
        if not self.south < self.north:
            raise ForecastContractError("regional south must be strictly less than north")
        if self.south < -90.0 or self.north > 90.0:
            raise ForecastContractError("regional latitude bounds must lie within -90..90")
        if not self.west < self.east:
            raise ForecastContractError(
                "regional west must be less than east; antimeridian-wrapping crops require "
                "an explicit future contract")
        if self.longitude_convention not in ("0..360", "-180..180"):
            raise ForecastContractError("regional longitude convention must be 0..360 or -180..180")
        lon_min = 0.0 if self.longitude_convention == "0..360" else -180.0
        lon_max = 360.0 if self.longitude_convention == "0..360" else 180.0
        if self.west < lon_min or self.east >= lon_max:
            raise ForecastContractError(
                "regional longitude bounds fall outside the declared %s convention"
                % self.longitude_convention)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ForecastContractError("regional bounds require a non-empty label")

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ForecastCubeValidation:
    schema: str
    request_sha256: str
    result_sha256: str
    artifact_sha256: str
    artifact_format: str
    dimensions: Tuple[Tuple[str, int], ...]
    variables: Tuple[str, ...]
    units: Tuple[Tuple[str, str], ...]
    grid_coordinates_sha256: str
    finite_values: bool
    checked_values: int
    chunks_checked: int
    maximum_observed_chunk_bytes: int
    maximum_allowed_chunk_bytes: int
    claim_boundary: str

    def to_mapping(self) -> Dict[str, Any]:
        record = asdict(self)
        record["dimensions"] = {name: size for name, size in self.dimensions}
        record["units"] = {name: unit for name, unit in self.units}
        return record

    def fingerprint(self) -> str:
        return _canonical_hash(self.to_mapping())


def _expected_grid(request: ExternalForecastRun) -> Tuple[np.ndarray, np.ndarray]:
    if request.input.latitude_order == "descending":
        lat = np.linspace(90.0, -90.0, FCN3_GRID_SHAPE[0], dtype="float64")
    else:
        lat = np.linspace(-90.0, 90.0, FCN3_GRID_SHAPE[0], dtype="float64")
    start = 0.0 if request.input.longitude_convention == "0..360" else -180.0
    lon = start + np.arange(FCN3_GRID_SHAPE[1], dtype="float64") * FCN3_GRID_DEGREES
    return lat, lon


def _expected_times(values: Sequence[str]) -> np.ndarray:
    result = []
    for value in values:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        utc = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        result.append(np.datetime64(utc, "ns"))
    return np.asarray(result, dtype="datetime64[ns]")


def _require_exact_coordinate(dataset: xr.Dataset, name: str, expected: np.ndarray) -> None:
    if name not in dataset.coords:
        raise ForecastContractError("canonical forecast cube is missing coordinate %r" % name)
    actual = np.asarray(dataset.coords[name].values)
    if actual.ndim != 1 or actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ForecastContractError(
            "canonical forecast cube coordinate %r differs from the request" % name)


def _validate_structure(dataset: xr.Dataset, request: ExternalForecastRun) -> Tuple[str, Tuple[Tuple[str, str], ...]]:
    if dataset.attrs.get("forecast_cube_schema") != CANONICAL_FORECAST_CUBE_SCHEMA:
        raise ForecastContractError(
            "forecast_cube_schema must equal %s" % CANONICAL_FORECAST_CUBE_SCHEMA)
    if dataset.attrs.get("request_sha256") != request.fingerprint():
        raise ForecastContractError("forecast cube request_sha256 differs from the originating request")
    if dataset.attrs.get("write_complete") != "true":
        raise ForecastContractError(
            "forecast cube write_complete must be the string 'true'; partial worker output is refused")

    expected_sizes = {
        "time": len(request.input.initialization_times),
        "ensemble": request.rollout.ensemble_size,
        "lead_time": request.rollout.steps,
        "lat": FCN3_GRID_SHAPE[0],
        "lon": FCN3_GRID_SHAPE[1],
    }
    actual_sizes = {name: int(size) for name, size in dataset.sizes.items()}
    if actual_sizes != expected_sizes:
        raise ForecastContractError(
            "canonical forecast cube dimensions differ: expected=%r actual=%r"
            % (expected_sizes, actual_sizes))

    if set(dataset.coords) != set(CANONICAL_DIMS):
        raise ForecastContractError(
            "canonical forecast cube coordinates must be exactly %r; got %r"
            % (CANONICAL_DIMS, tuple(dataset.coords)))
    if set(dataset.data_vars) != set(request.output.variables):
        raise ForecastContractError(
            "forecast variables differ: expected=%r actual=%r"
            % (request.output.variables, tuple(dataset.data_vars)))

    lat, lon = _expected_grid(request)
    _require_exact_coordinate(dataset, "lat", lat)
    _require_exact_coordinate(dataset, "lon", lon)
    _require_exact_coordinate(dataset, "time", _expected_times(request.input.initialization_times))
    _require_exact_coordinate(dataset, "ensemble", np.asarray(request.rollout.member_seeds))
    leads = np.asarray([np.timedelta64(int(round(hours * 3600)), "s")
                        for hours in request.rollout.lead_hours], dtype="timedelta64[ns]")
    actual_leads = np.asarray(dataset.coords["lead_time"].values)
    if np.issubdtype(actual_leads.dtype, np.timedelta64):
        normalised_leads = actual_leads.astype("timedelta64[ns]")
    elif np.issubdtype(actual_leads.dtype, np.number):
        unit = dataset.coords["lead_time"].attrs.get("units")
        factors = {"nanoseconds": 1, "microseconds": 1_000, "milliseconds": 1_000_000,
                   "seconds": 1_000_000_000, "minutes": 60_000_000_000,
                   "hours": 3_600_000_000_000, "days": 86_400_000_000_000}
        if unit not in factors:
            raise ForecastContractError(
                "numeric lead_time requires an explicit CF duration unit; got %r" % unit)
        as_ns = actual_leads.astype("float64") * factors[unit]
        if not np.isfinite(as_ns).all() or not np.equal(as_ns, np.rint(as_ns)).all():
            raise ForecastContractError("lead_time cannot be represented exactly in nanoseconds")
        normalised_leads = np.rint(as_ns).astype("int64").astype("timedelta64[ns]")
    else:
        raise ForecastContractError("lead_time must be timedelta64 or a numeric CF duration")
    if normalised_leads.ndim != 1 or not np.array_equal(normalised_leads, leads):
        raise ForecastContractError("canonical forecast cube coordinate 'lead_time' differs from the request")

    actual_grid_hash = grid_coordinates_sha256(dataset.lat.values, dataset.lon.values)
    if actual_grid_hash != request.input.grid_coordinates_sha256:
        raise ForecastContractError(
            "forecast grid coordinate SHA-256 differs from input.grid_coordinates_sha256")

    units = []
    for variable in request.output.variables:
        array = dataset[variable]
        if tuple(array.dims) != CANONICAL_DIMS:
            raise ForecastContractError(
                "%s dimensions must be exactly %r; got %r"
                % (variable, CANONICAL_DIMS, tuple(array.dims)))
        if not np.issubdtype(array.dtype, np.floating):
            raise ForecastContractError("%s values must use a floating-point dtype" % variable)
        expected_unit = fcn3_variable_unit(variable)
        if array.attrs.get("units") != expected_unit:
            raise ForecastContractError(
                "%s units must be %r; got %r" %
                (variable, expected_unit, array.attrs.get("units")))
        units.append((variable, expected_unit))
    return actual_grid_hash, tuple(units)


def _scan_finite(dataset: xr.Dataset, variables: Sequence[str],
                 max_chunk_bytes: int) -> Tuple[int, int, int]:
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int) or max_chunk_bytes <= 0:
        raise ForecastContractError("max_chunk_bytes must be a positive integer")
    checked = chunks_checked = observed_max = 0
    for variable in variables:
        array = dataset[variable]
        chunks = getattr(array.data, "chunks", None)
        if chunks is None:
            prospective = int(array.size * array.dtype.itemsize)
            if prospective > max_chunk_bytes:
                raise ForecastContractError(
                    "%s is not lazily chunked and would require %d bytes; rechunk the worker "
                    "artifact to at most %d bytes" % (variable, prospective, max_chunk_bytes))
            blocks = (array.data,)
        else:
            largest = int(math.prod(max(axis) for axis in chunks) * array.dtype.itemsize)
            if largest > max_chunk_bytes:
                raise ForecastContractError(
                    "%s has a decompressed storage chunk of up to %d bytes; rechunk the worker "
                    "artifact to at most %d bytes" % (variable, largest, max_chunk_bytes))
            blocks = array.data.to_delayed().ravel()
        for ordinal, block in enumerate(blocks):
            values = np.asarray(block.compute() if hasattr(block, "compute") else block)
            nbytes = int(values.nbytes)
            if nbytes > max_chunk_bytes:
                raise ForecastContractError(
                    "%s chunk %d exceeded the declared memory bound" % (variable, ordinal))
            if not np.isfinite(values).all():
                bad = int(values.size - np.count_nonzero(np.isfinite(values)))
                raise ForecastContractError(
                    "%s chunk %d contains %d non-finite forecast values" %
                    (variable, ordinal, bad))
            checked += int(values.size)
            chunks_checked += 1
            observed_max = max(observed_max, nbytes)
    return checked, chunks_checked, observed_max


class CanonicalForecastCube:
    """Validated global cube with a live lazy xarray handle."""

    def __init__(self, dataset: xr.Dataset, validation: ForecastCubeValidation,
                 request: ExternalForecastRun) -> None:
        self._dataset = dataset
        self.validation = validation
        self.request = request

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    def crop(self, bounds: GeographicBounds) -> "RegionalForecastCube":
        if not isinstance(bounds, GeographicBounds):
            raise ForecastContractError("bounds must be validated GeographicBounds")
        if bounds.longitude_convention != self.request.input.longitude_convention:
            raise ForecastContractError(
                "regional and global longitude conventions differ; implicit conversion is refused")
        lat = np.asarray(self._dataset.lat.values, dtype="float64")
        lon = np.asarray(self._dataset.lon.values, dtype="float64")

        def exact_index(values: np.ndarray, target: float, name: str) -> int:
            indices = np.flatnonzero(values == float(target))
            if len(indices) != 1:
                raise ForecastContractError(
                    "%s bound %r is not exactly present on the 0.25-degree grid; interpolation "
                    "and implicit rounding are refused" % (name, target))
            return int(indices[0])

        south = exact_index(lat, bounds.south, "south")
        north = exact_index(lat, bounds.north, "north")
        west = exact_index(lon, bounds.west, "west")
        east = exact_index(lon, bounds.east, "east")
        lat_slice = slice(min(south, north), max(south, north) + 1)
        lon_slice = slice(west, east + 1)
        regional = self._dataset.isel(lat=lat_slice, lon=lon_slice)
        provenance = {
            "schema": "regional-external-forecast-cube/fcn3-v1",
            "request_sha256": self.validation.request_sha256,
            "result_sha256": self.validation.result_sha256,
            "artifact_sha256": self.validation.artifact_sha256,
            "validation_sha256": self.validation.fingerprint(),
            "bounds": bounds.to_mapping(),
            "selection": "inclusive exact-coordinate subset; no interpolation",
            "grid_shape": [int(regional.sizes["lat"]), int(regional.sizes["lon"])],
            "grid_coordinates_sha256": grid_coordinates_sha256(
                regional.lat.values, regional.lon.values),
            "lazy": all(getattr(regional[name].data, "chunks", None) is not None
                        for name in self.request.output.variables),
            "claim_boundary": (
                "The crop is a lazy exact subset of a schema-, identity-, unit- and finite-value-"
                "validated global artifact. It does not establish meteorological accuracy, "
                "ensemble calibration, spectral fidelity or forecast skill."),
        }
        return RegionalForecastCube(regional, provenance)

    def close(self) -> None:
        self._dataset.close()

    def __enter__(self) -> "CanonicalForecastCube":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


@dataclass
class RegionalForecastCube:
    """Lazy regional view; its global parent owns the underlying file handle."""

    dataset: xr.Dataset
    provenance: Mapping[str, Any]


def open_canonical_forecast_cube(
    artifact_path: Union[str, Path],
    request: ExternalForecastRun,
    result: ExternalForecastResult,
    *,
    max_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES,
) -> CanonicalForecastCube:
    """Open, authenticate and completely validate a canonical FCN3 cube lazily.

    "Completely" refers to identity, schema, coordinates, units and finiteness.  Scientific
    skill and calibration are deliberately outside this acceptance boundary.  Every forecast
    value is inspected, but only one declared storage chunk is materialised at a time.
    """
    if not isinstance(request, ExternalForecastRun) or not isinstance(result, ExternalForecastResult):
        raise ForecastContractError("request and result must be validated FCN3 contracts")
    path = Path(artifact_path)
    verify_external_forecast_result(request, result, path)
    try:
        if result.artifact_format == "netcdf4":
            dataset = xr.open_dataset(
                path, engine="h5netcdf", chunks={}, decode_timedelta=False)
        else:
            dataset = xr.open_zarr(path, chunks={}, consolidated=None)
    except Exception as exc:
        raise ForecastContractError(
            "cannot lazily open %s forecast cube: %s" % (result.artifact_format, exc)) from exc
    try:
        grid_hash, units = _validate_structure(dataset, request)
        # Per-variable arrays are label-addressed. Storage backends may enumerate them in a
        # different order, so restore the request order after exact set validation.
        dataset = dataset[list(request.output.variables)]
        checked, chunks_checked, observed = _scan_finite(
            dataset, request.output.variables, max_chunk_bytes)
        dimensions = tuple((name, int(dataset.sizes[name])) for name in CANONICAL_DIMS)
        validation = ForecastCubeValidation(
            schema=FORECAST_CUBE_VALIDATION_SCHEMA,
            request_sha256=request.fingerprint(), result_sha256=result.result_sha256,
            artifact_sha256=result.artifact_sha256, artifact_format=result.artifact_format,
            dimensions=dimensions, variables=request.output.variables, units=units,
            grid_coordinates_sha256=grid_hash, finite_values=True,
            checked_values=checked, chunks_checked=chunks_checked,
            maximum_observed_chunk_bytes=observed,
            maximum_allowed_chunk_bytes=max_chunk_bytes,
            claim_boundary=(
                "Identity, canonical schema, exact axes, SI units and finite values are verified. "
                "Meteorological accuracy, ensemble calibration, spectral fidelity and forecast "
                "skill are NOT established."),
        )
        return CanonicalForecastCube(dataset, validation, request)
    except Exception:
        dataset.close()
        raise
