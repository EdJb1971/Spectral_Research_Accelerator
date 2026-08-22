"""Resumable regional ERA5 acquisition through the Copernicus CDS (T5.2c / D43a).

CDS is a queued materialisation service, not an interactive fallback source.  This module
therefore downloads explicit monthly shards and converts them into the same content-addressed,
time-chunked local Zarr cache used by :mod:`src.data_layer.regional_forecast`.  Training never
needs credentials or network access and a laptop/HPC cache differs only by its directory.

The live client is optional.  Importing this module does not import ``cdsapi`` and no request is
submitted unless both ``allow_network=True`` (or ``SPECTRALEARTH_ALLOW_NETWORK=1``) and a CDS
client are available.  Credentials remain in the standard CDS client configuration and are
never read into provenance.
"""

from __future__ import annotations

import calendar
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.regional_forecast import CANONICAL_VARIABLES, VARIABLE_ALIASES
from src.data_layer.zarr_source import (
    NETWORK_ENV_VAR,
    CropSpec,
    cache_path,
    check_crop_size,
    is_cached,
    manifest_path,
    streaming_content_hash,
)


CDS_DATASET = "reanalysis-era5-pressure-levels"
CDS_VARIABLES: Mapping[str, str] = {
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "z": "geopotential",
}
PRESSURE_LEVELS = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875,
    900, 925, 950, 975, 1000,
)
ACQUISITION_SCHEMA = "cds-regional-acquisition/v1"
STORAGE_SAFETY_FACTOR = 2.0
MINIMUM_FREE_RESERVE_BYTES = 5 * 1024 ** 3
PER_SHARD_OVERHEAD_BYTES = 16 * 1024 ** 2


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            "CDS request", value, "finite JSON-serialisable values: %s" % exc
        ) from exc


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_date(value: str, name: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(name, value, "an ISO calendar date YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise InvalidParameterError(name, value, "an exact ISO calendar date YYYY-MM-DD")
    return parsed


@dataclass(frozen=True)
class CDSRegionalRequest:
    """Complete, hashable request for a regular regional ERA5 pressure-level record."""

    variables: Tuple[str, ...]
    date_start: str
    date_end: str
    hours_utc: Tuple[int, ...]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    pressure_levels: Tuple[int, ...] = (850,)
    grid_degrees: float = 0.25
    dataset: str = CDS_DATASET
    data_format: str = "netcdf"
    n_levels_analysis: int = 4

    def __post_init__(self) -> None:
        start = _parse_date(self.date_start, "date_start")
        end = _parse_date(self.date_end, "date_end")
        if end < start:
            raise InvalidParameterError(
                "date_start/date_end", (self.date_start, self.date_end),
                "an inclusive range with date_start <= date_end")
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise InvalidParameterError("variables", self.variables,
                                        "a non-empty sequence of unique canonical variables")
        unknown = sorted(set(self.variables) - set(CANONICAL_VARIABLES))
        if unknown:
            raise InvalidParameterError(
                "variables", unknown, "canonical ERA5 pressure variables t/q/u/v/z")
        if not self.hours_utc or len(set(self.hours_utc)) != len(self.hours_utc):
            raise InvalidParameterError("hours_utc", self.hours_utc,
                                        "one or more unique UTC hours")
        if tuple(sorted(self.hours_utc)) != self.hours_utc or any(
                isinstance(hour, bool) or int(hour) != hour or not 0 <= hour <= 23
                for hour in self.hours_utc):
            raise InvalidParameterError("hours_utc", self.hours_utc,
                                        "strictly increasing integer hours from 0 to 23")
        if not self.pressure_levels or len(set(self.pressure_levels)) != len(self.pressure_levels):
            raise InvalidParameterError("pressure_levels", self.pressure_levels,
                                        "one or more unique official ERA5 pressure levels")
        invalid_levels = sorted(set(self.pressure_levels) - set(PRESSURE_LEVELS))
        if invalid_levels:
            raise InvalidParameterError("pressure_levels", invalid_levels,
                                        "official ERA5 pressure levels")
        if not (-90.0 <= self.lat_min < self.lat_max <= 90.0):
            raise InvalidParameterError("latitude bounds", (self.lat_min, self.lat_max),
                                        "-90 <= lat_min < lat_max <= 90")
        if not (-180.0 <= self.lon_min < self.lon_max <= 180.0):
            raise InvalidParameterError(
                "longitude bounds", (self.lon_min, self.lon_max),
                "-180 <= lon_min < lon_max <= 180; split a dateline-crossing request")
        if not math.isfinite(self.grid_degrees) or self.grid_degrees <= 0:
            raise InvalidParameterError("grid_degrees", self.grid_degrees,
                                        "a finite positive angular spacing")
        for name, span in (
                ("latitude", self.lat_max - self.lat_min),
                ("longitude", self.lon_max - self.lon_min)):
            intervals = span / self.grid_degrees
            if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9):
                raise InvalidParameterError(
                    "%s bounds/grid" % name,
                    (self.lat_min, self.lat_max, self.lon_min, self.lon_max,
                     self.grid_degrees),
                    "bounds separated by an integer number of grid intervals; server-side "
                    "snapping would change the frozen crop")
        if self.dataset != CDS_DATASET:
            raise InvalidParameterError("dataset", self.dataset,
                                        "the reviewed ERA5 pressure-level product %s" % CDS_DATASET)
        if self.data_format != "netcdf":
            raise InvalidParameterError(
                "data_format", self.data_format,
                "netcdf for the current dependency-light ingestion path; GRIB is not silently relabelled")
        if isinstance(self.n_levels_analysis, bool) or self.n_levels_analysis < 1:
            raise InvalidParameterError("n_levels_analysis", self.n_levels_analysis,
                                        "a positive transform depth")

    def canonical(self) -> Dict[str, Any]:
        return {
            "schema": ACQUISITION_SCHEMA,
            "dataset": self.dataset,
            "product_type": "reanalysis",
            "variables": list(self.variables),
            "cds_variables": [CDS_VARIABLES[name] for name in self.variables],
            "date_start": self.date_start,
            "date_end": self.date_end,
            "hours_utc": list(self.hours_utc),
            "pressure_levels": list(self.pressure_levels),
            "bbox_north_west_south_east": [
                self.lat_max, self.lon_min, self.lat_min, self.lon_max,
            ],
            "grid_degrees": self.grid_degrees,
            "data_format": self.data_format,
        }

    def request_sha256(self) -> str:
        return _stable_hash(self.canonical())

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["hours_utc"] = list(self.hours_utc)
        record["pressure_levels"] = list(self.pressure_levels)
        record["request_sha256"] = self.request_sha256()
        record["area_order"] = "north, west, south, east"
        record["credential_policy"] = "standard CDS client configuration; credentials never recorded"
        return record

    @classmethod
    def from_provenance(cls, record: Mapping[str, Any]) -> "CDSRegionalRequest":
        fields = set(cls.__dataclass_fields__)
        values = {key: value for key, value in record.items() if key in fields}
        required = fields - {"pressure_levels", "grid_degrees", "dataset", "data_format",
                             "n_levels_analysis"}
        missing = required - set(values)
        if missing:
            raise InvalidParameterError(
                "CDS provenance", sorted(record), "a request containing %s" % sorted(missing))
        for key in ("variables", "hours_utc", "pressure_levels"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_crop_spec(self) -> CropSpec:
        return CropSpec(
            store="cds:%s" % self.dataset,
            variables=tuple(self.variables),
            time_start="%sT%02d:00:00" % (self.date_start, self.hours_utc[0]),
            time_end="%sT%02d:00:00" % (self.date_end, self.hours_utc[-1]),
            lat_min=self.lat_min, lat_max=self.lat_max,
            lon_min=self.lon_min, lon_max=self.lon_max,
            levels=tuple(self.pressure_levels), n_levels_analysis=self.n_levels_analysis)


@dataclass(frozen=True)
class CDSShard:
    year: int
    month: int
    days: Tuple[int, ...]
    request: Mapping[str, Any]
    request_sha256: str
    filename: str

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["days"] = list(self.days)
        return record


def plan_monthly_shards(spec: CDSRegionalRequest) -> Tuple[CDSShard, ...]:
    """Split an inclusive request into queue-friendly calendar-month requests."""
    start = _parse_date(spec.date_start, "date_start")
    end = _parse_date(spec.date_end, "date_end")
    cursor = date(start.year, start.month, 1)
    shards: List[CDSShard] = []
    while cursor <= end:
        final_day = calendar.monthrange(cursor.year, cursor.month)[1]
        month_start = max(start, cursor)
        month_end = min(end, date(cursor.year, cursor.month, final_day))
        days = tuple(range(month_start.day, month_end.day + 1))
        request: Dict[str, Any] = {
            "product_type": ["reanalysis"],
            "variable": [CDS_VARIABLES[name] for name in spec.variables],
            "year": ["%04d" % cursor.year],
            "month": ["%02d" % cursor.month],
            "day": ["%02d" % day for day in days],
            "time": ["%02d:00" % hour for hour in spec.hours_utc],
            "pressure_level": [str(level) for level in spec.pressure_levels],
            "area": [spec.lat_max, spec.lon_min, spec.lat_min, spec.lon_max],
            "grid": [spec.grid_degrees, spec.grid_degrees],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        request_hash = _stable_hash({"dataset": spec.dataset, "request": request})
        filename = "%04d-%02d-%s.nc" % (cursor.year, cursor.month, request_hash[:12])
        shards.append(CDSShard(
            year=cursor.year, month=cursor.month, days=days, request=request,
            request_sha256=request_hash, filename=filename))
        cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
    return tuple(shards)


def estimate_cds_storage(
    spec: CDSRegionalRequest,
    *,
    shards: Optional[Sequence[CDSShard]] = None,
) -> Dict[str, Any]:
    """Conservative pre-download storage estimate from the frozen request geometry.

    CDS does not publish the eventual NetCDF compression ratio before materialisation.  The
    refusal boundary therefore starts from float32 payload bytes, doubles them independently
    for the NetCDF and Zarr artifacts, and adds fixed per-shard container overhead.  It never
    treats hoped-for compression as available capacity.
    """
    selected = tuple(plan_monthly_shards(spec) if shards is None else shards)
    frame_count = sum(len(shard.days) * len(spec.hours_utc) for shard in selected)
    latitude_points = int(math.ceil(
        (spec.lat_max - spec.lat_min) / spec.grid_degrees - 1e-12)) + 1
    longitude_points = int(math.ceil(
        (spec.lon_max - spec.lon_min) / spec.grid_degrees - 1e-12)) + 1
    value_count = (frame_count * len(spec.pressure_levels) * latitude_points
                   * longitude_points * len(spec.variables))
    raw_value_bytes = int(value_count * 4)
    artifact_bytes = int(math.ceil(raw_value_bytes * STORAGE_SAFETY_FACTOR)
                         + len(selected) * PER_SHARD_OVERHEAD_BYTES)
    return {
        "basis": "float32 values x 2 safety factor + 16 MiB container overhead per shard",
        "compression_credit_assumed": False,
        "frames": int(frame_count),
        "latitude_points_upper_bound": latitude_points,
        "longitude_points_upper_bound": longitude_points,
        "levels": len(spec.pressure_levels),
        "variables": len(spec.variables),
        "raw_value_bytes": raw_value_bytes,
        "artifact_bytes_upper_bound": artifact_bytes,
        "shards": len(selected),
    }


def _storage_location(path: Path) -> Tuple[Path, str]:
    candidate = path.expanduser().absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise DataSourceError("cannot resolve a filesystem for storage preflight", path=str(path))
    resolved = candidate.resolve()
    drive = os.path.splitdrive(str(resolved))[0].upper()
    identity = drive or "device:%s" % os.stat(resolved).st_dev
    return resolved, identity


def preflight_cds_storage(
    spec: CDSRegionalRequest,
    *,
    download_dir: Union[str, os.PathLike[str]],
    cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
    minimum_free_reserve_bytes: int = MINIMUM_FREE_RESERVE_BYTES,
) -> Dict[str, Any]:
    """Refuse an acquisition whose peak working set plus reserve does not fit.

    Completed, tracked shards already consume filesystem space, so only missing shards count
    toward additional download capacity.  Cache conversion still budgets the complete Zarr
    artifact.  Requirements are combined when download and cache directories share a volume.
    """
    if isinstance(minimum_free_reserve_bytes, bool) \
            or int(minimum_free_reserve_bytes) != minimum_free_reserve_bytes \
            or minimum_free_reserve_bytes < 0:
        raise InvalidParameterError(
            "minimum_free_reserve_bytes", minimum_free_reserve_bytes,
            "a non-negative integer byte reserve")
    download_path = Path(download_dir)
    shards = plan_monthly_shards(spec)
    completed_names = set()
    state_path = download_path / "acquisition.json"
    if state_path.exists():
        state = _read_state(state_path, spec, shards)
        completed_names = set(state.get("completed_shards", {}))
    remaining = tuple(shard for shard in shards if shard.filename not in completed_names)
    download_estimate = estimate_cds_storage(spec, shards=remaining)
    cache_estimate = estimate_cds_storage(spec)

    roles = [("download", download_path, download_estimate["artifact_bytes_upper_bound"])]
    if cache_dir is not None:
        roles.append(("cache", Path(cache_dir), cache_estimate["artifact_bytes_upper_bound"]))
    grouped: Dict[str, Dict[str, Any]] = {}
    for role, path, required in roles:
        existing, identity = _storage_location(path)
        entry = grouped.setdefault(identity, {
            "volume": identity, "roles": [], "existing_probe": existing,
            "working_bytes_required": 0,
        })
        entry["roles"].append(role)
        entry["working_bytes_required"] += int(required)

    reports = []
    for entry in grouped.values():
        usage = shutil.disk_usage(entry.pop("existing_probe"))
        working = int(entry["working_bytes_required"])
        reserve = max(int(minimum_free_reserve_bytes), int(math.ceil(working * 0.10)))
        total_required = working + reserve
        report = {
            **entry,
            "free_bytes": int(usage.free),
            "reserve_bytes": reserve,
            "total_free_required": total_required,
            "passes": int(usage.free) >= total_required,
        }
        reports.append(report)
    failures = [report for report in reports if not report["passes"]]
    result = {
        "status": "READY" if not failures else "INSUFFICIENT_SPACE",
        "download_estimate": download_estimate,
        "cache_estimate": cache_estimate if cache_dir is not None else None,
        "completed_shards": len(completed_names),
        "remaining_shards": len(remaining),
        "volumes": sorted(reports, key=lambda record: record["volume"]),
    }
    if failures:
        raise DataSourceError(
            "CDS storage preflight failed before network use: peak working bytes plus the "
            "free-space reserve do not fit",
            storage_preflight=result)
    return result


def _network_enabled(allow_network: Optional[bool]) -> bool:
    if allow_network is not None:
        return bool(allow_network)
    return os.environ.get(NETWORK_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def _read_state(path: Path, spec: CDSRegionalRequest,
                shards: Sequence[CDSShard]) -> Dict[str, Any]:
    planned = [shard.to_provenance() for shard in shards]
    if not path.exists():
        return {
            "schema": ACQUISITION_SCHEMA,
            "request": spec.to_provenance(),
            "planned_shards": planned,
            "completed_shards": {},
            "claim_boundary": "request/download provenance only; ERA5 agreement is a separate live check",
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("CDS acquisition state is unreadable: %s" % exc, path=str(path)) from exc
    if state.get("schema") != ACQUISITION_SCHEMA:
        raise DataSourceError("unsupported CDS acquisition state schema", path=str(path))
    if state.get("request", {}).get("request_sha256") != spec.request_sha256():
        raise DataSourceError("download directory belongs to a different CDS request", path=str(path))
    if state.get("planned_shards") != planned:
        raise DataSourceError("stored CDS shard plan does not match the current request", path=str(path))
    return state


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(state) + b"\n")
    os.replace(temporary, path)


def _validate_netcdf_shard(path: Path) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise DataSourceError("CDS shard is missing or empty", path=str(path))
    import xarray as xr
    try:
        dataset = xr.open_dataset(path)
        try:
            sizes = {str(key): int(value) for key, value in dataset.sizes.items()}
            variables = sorted(str(name) for name in dataset.data_vars)
        finally:
            dataset.close()
    except Exception as exc:
        raise DataSourceError("CDS shard is not readable NetCDF: %s" % exc, path=str(path)) from exc
    return {"bytes": path.stat().st_size, "sha256": _file_hash(path),
            "sizes": sizes, "variables": variables}


def acquire_cds_shards(
    spec: CDSRegionalRequest,
    download_dir: Union[str, os.PathLike[str]],
    *,
    client: Optional[Any] = None,
    allow_network: Optional[bool] = None,
) -> Dict[str, Any]:
    """Download missing monthly shards atomically and return replayable acquisition state."""
    if not _network_enabled(allow_network):
        raise DataSourceError(
            "CDS acquisition is network-disabled; set %s=1 or pass allow_network=True explicitly"
            % NETWORK_ENV_VAR)
    storage_preflight = preflight_cds_storage(spec, download_dir=download_dir)
    if client is None:
        try:
            import cdsapi
        except ImportError as exc:
            raise DataSourceError(
                "CDS acquisition requires the optional cdsapi>=0.7.7 dependency") from exc
        try:
            client = cdsapi.Client()
        except Exception as exc:
            raise DataSourceError(
                "CDS client setup failed (%s); configure the standard CDS API token and accept the dataset licence"
                % type(exc).__name__) from exc

    root = Path(download_dir)
    root.mkdir(parents=True, exist_ok=True)
    shards = plan_monthly_shards(spec)
    state_path = root / "acquisition.json"
    state = _read_state(state_path, spec, shards)
    completed = dict(state.get("completed_shards", {}))
    downloaded = resumed = 0
    for shard in shards:
        target = root / shard.filename
        existing = completed.get(shard.filename)
        if existing is not None:
            observed = _validate_netcdf_shard(target)
            if (existing.get("sha256") != observed["sha256"] or
                    existing.get("request_sha256") != shard.request_sha256):
                raise DataSourceError(
                    "completed CDS shard failed integrity verification; refuse silent replacement",
                    path=str(target))
            resumed += 1
            continue
        if target.exists():
            raise DataSourceError(
                "untracked CDS shard already exists; remove or quarantine it before resuming",
                path=str(target))
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            partial.unlink()
        try:
            client.retrieve(spec.dataset, dict(shard.request), str(partial))
            validation = _validate_netcdf_shard(partial)
            os.replace(partial, target)
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            if isinstance(exc, DataSourceError):
                raise
            raise DataSourceError(
                "CDS shard retrieval failed (%s); incomplete file removed"
                % type(exc).__name__, path=str(target)) from exc
        completed[shard.filename] = {
            **validation,
            "request_sha256": shard.request_sha256,
            "dataset": spec.dataset,
        }
        state["completed_shards"] = completed
        _write_state(state_path, state)
        downloaded += 1
    state["run"] = {
        "downloaded_shards": downloaded,
        "resumed_shards": resumed,
        "complete": len(completed) == len(shards),
        "total_bytes": int(sum(int(value["bytes"]) for value in completed.values())),
        "storage_preflight": storage_preflight,
    }
    _write_state(state_path, state)
    return state


def _expected_times(spec: CDSRegionalRequest) -> List[datetime]:
    cursor = _parse_date(spec.date_start, "date_start")
    end = _parse_date(spec.date_end, "date_end")
    values: List[datetime] = []
    while cursor <= end:
        values.extend(datetime(cursor.year, cursor.month, cursor.day, hour)
                      for hour in spec.hours_utc)
        cursor += timedelta(days=1)
    return values


def _normalise_downloaded_dataset(
    dataset: Any,
    spec: CDSRegionalRequest,
    *,
    expected_times: Optional[Sequence[datetime]] = None,
) -> Any:
    import numpy as np

    coordinate_renames = {
        old: new for old, new in (("valid_time", "time"), ("pressure_level", "level"),
                                  ("lat", "latitude"), ("lon", "longitude"))
        if old in dataset.coords and new not in dataset.coords
    }
    if coordinate_renames:
        dataset = dataset.rename(coordinate_renames)
    required_coordinates = {"time", "level", "latitude", "longitude"}
    if not required_coordinates.issubset(dataset.coords):
        raise DataSourceError(
            "CDS output lacks required coordinates %s; found %s"
            % (sorted(required_coordinates), sorted(dataset.coords)))

    renames: Dict[str, str] = {}
    for canonical in spec.variables:
        aliases = VARIABLE_ALIASES[canonical]
        matches = [name for name in aliases if name in dataset.data_vars]
        if len(matches) != 1:
            raise DataSourceError(
                "CDS output must contain exactly one alias for %s; found %s"
                % (canonical, matches))
        if matches[0] != canonical:
            renames[matches[0]] = canonical
    dataset = dataset.rename(renames)[list(spec.variables)]
    extra_dims = set(dataset.dims) - required_coordinates
    if extra_dims:
        raise DataSourceError(
            "CDS output has unresolved dimensions %s; no implicit member/expver selection is allowed"
            % sorted(extra_dims))
    dataset = dataset.sel(level=list(spec.pressure_levels))
    dataset = dataset.sortby("time")
    observed_times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    required_times = np.asarray(
        list(expected_times) if expected_times is not None else _expected_times(spec),
        dtype="datetime64[ns]",
    )
    if not np.array_equal(observed_times, required_times):
        raise DataSourceError(
            "CDS output timestamps do not exactly match the requested calendar dates/hours")
    lat = np.asarray(dataset.latitude.values, dtype=float)
    lon = np.asarray(dataset.longitude.values, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or len(lat) < 2 or len(lon) < 2:
        raise DataSourceError("CDS output must have one-dimensional non-trivial latitude/longitude")
    if not np.allclose(np.abs(np.diff(lat)), spec.grid_degrees, rtol=0.0, atol=1e-8) or not np.allclose(
            np.abs(np.diff(lon)), spec.grid_degrees, rtol=0.0, atol=1e-8):
        raise DataSourceError("CDS output grid spacing does not match the declared request")
    if not (math.isclose(float(lat.min()), spec.lat_min, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(float(lat.max()), spec.lat_max, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(float(lon.min()), spec.lon_min, rel_tol=0.0, abs_tol=1e-8)
            and math.isclose(float(lon.max()), spec.lon_max, rel_tol=0.0, abs_tol=1e-8)):
        raise DataSourceError(
            "CDS output grid bounds do not exactly match the frozen request; server-side "
            "coordinate snapping is not accepted")
    if any(not np.issubdtype(dataset[name].dtype, np.number) for name in spec.variables):
        raise DataSourceError("CDS output variables must be numeric")
    # Canonical dimension order makes append semantics and the logical content hash
    # independent of how an individual NetCDF writer happened to order its axes.
    return dataset.transpose("time", "level", "latitude", "longitude")


def _shard_expected_times(shard: CDSShard, spec: CDSRegionalRequest) -> Tuple[datetime, ...]:
    return tuple(datetime(shard.year, shard.month, day, hour)
                 for day in shard.days for hour in spec.hours_utc)


def materialise_cds(
    spec: CDSRegionalRequest,
    *,
    download_dir: Union[str, os.PathLike[str]],
    cache_dir: Optional[str] = None,
    time_chunk: int = 32,
    check_size: bool = True,
    client: Optional[Any] = None,
    allow_network: Optional[bool] = None,
) -> Dict[str, Any]:
    """Acquire and convert a complete CDS request into the canonical local Zarr cache."""
    if isinstance(time_chunk, bool) or int(time_chunk) != time_chunk or time_chunk < 1:
        raise InvalidParameterError("time_chunk", time_chunk, "a positive integer frame count")
    crop = spec.to_crop_spec()
    if is_cached(crop, cache_dir):
        with open(manifest_path(crop, cache_dir), "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest["cache_hit"] = True
        manifest["bytes_transferred"] = 0
        return manifest
    storage_preflight = preflight_cds_storage(
        spec, download_dir=download_dir,
        cache_dir=Path(cache_path(crop, cache_dir)).parent)
    state = acquire_cds_shards(
        spec, download_dir, client=client, allow_network=allow_network)
    if not state.get("run", {}).get("complete"):
        raise DataSourceError("CDS acquisition is incomplete; cache materialisation refused")

    import numpy as np
    import xarray as xr

    started = time.monotonic()
    shards = plan_monthly_shards(spec)
    expected_frame_count = len(_expected_times(spec))
    path = Path(cache_path(crop, cache_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DataSourceError("unmanifested CDS cache path already exists", path=str(path))
    temporary_root = Path(tempfile.mkdtemp(
        dir=str(path.parent), prefix=".%s." % crop.content_key()))
    temporary_store = temporary_root / "cache.zarr"

    reference_lat = reference_lon = reference_level = None
    variables_seen: Optional[Tuple[str, ...]] = None
    frames_written = 0
    max_frames_in_memory = 0
    first_write = True
    geometry: Optional[Dict[str, Any]] = None
    chunking: Optional[Dict[str, int]] = None
    try:
        for shard in shards:
            shard_path = Path(download_dir) / shard.filename
            with xr.open_dataset(shard_path) as opened:
                normalised = _normalise_downloaded_dataset(
                    opened, spec, expected_times=_shard_expected_times(shard, spec))
                lat = np.asarray(normalised.latitude.values)
                lon = np.asarray(normalised.longitude.values)
                level = np.asarray(normalised.level.values)
                names = tuple(sorted(str(name) for name in normalised.data_vars))
                if reference_lat is None:
                    reference_lat, reference_lon, reference_level = lat, lon, level
                    variables_seen = names
                    height = int(normalised.sizes["latitude"])
                    width = int(normalised.sizes["longitude"])
                    geometry = (check_crop_size(height, width, spec.n_levels_analysis)
                                if check_size else {"ok": None, "skipped": "check_size=False"})
                    chunking = {
                        "time": min(int(time_chunk), expected_frame_count),
                        "level": int(normalised.sizes["level"]),
                        "latitude": height,
                        "longitude": width,
                    }
                elif (not np.array_equal(lat, reference_lat)
                      or not np.array_equal(lon, reference_lon)
                      or not np.array_equal(level, reference_level)
                      or names != variables_seen):
                    raise DataSourceError(
                        "CDS monthly shards do not share exact coordinates and variables; "
                        "concatenation would mix different grids or schemas")

                assert chunking is not None
                for start in range(0, int(normalised.sizes["time"]), int(time_chunk)):
                    stop = min(start + int(time_chunk), int(normalised.sizes["time"]))
                    block = normalised.isel(time=slice(start, stop)).load()
                    max_frames_in_memory = max(max_frames_in_memory, stop - start)
                    if not all(bool(np.isfinite(block[name].values).all())
                               for name in spec.variables):
                        raise DataSourceError(
                            "CDS output contains non-finite values; no implicit imputation is allowed")
                    block = block.chunk(chunking)
                    encoding = {}
                    for name in block.data_vars:
                        for key in ("chunks", "preferred_chunks"):
                            block[name].encoding.pop(key, None)
                        encoding[name] = {"chunks": tuple(
                            int(chunking.get(str(dim), block.sizes[dim]))
                            for dim in block[name].dims)}
                    if first_write:
                        block.to_zarr(
                            temporary_store, mode="w", consolidated=False, encoding=encoding)
                        first_write = False
                    else:
                        block.to_zarr(
                            temporary_store, mode="a", append_dim="time", consolidated=False)
                    frames_written += stop - start

        if first_write or frames_written != expected_frame_count:
            raise DataSourceError(
                "CDS cache contains %d frames after streaming, expected %d"
                % (frames_written, expected_frame_count))

        import zarr
        zarr.consolidate_metadata(str(temporary_store))
        with xr.open_zarr(temporary_store, consolidated=True) as completed:
            observed_times = np.asarray(completed.time.values).astype("datetime64[ns]")
            expected_times = np.asarray(_expected_times(spec), dtype="datetime64[ns]")
            if not np.array_equal(observed_times, expected_times):
                raise DataSourceError(
                    "streamed CDS cache timestamps do not exactly match the complete request")
            content_hash = streaming_content_hash(completed, time_block=int(time_chunk))
            final_shape = {key: int(value) for key, value in completed.sizes.items()}

        # Publish only a complete, validated store. The manifest follows, so an interrupted
        # conversion is never discoverable as a cache hit.
        os.replace(temporary_store, path)
        manifest = {
            "content_key": crop.content_key(),
            "cache_path": str(path),
            "cache_hit": False,
            "spec": crop.to_provenance(),
            "shape": final_shape,
            "variables": list(variables_seen or ()),
            "content_hash": content_hash,
            "bytes_transferred": int(state["run"]["total_bytes"]),
            "megabytes_transferred": round(int(state["run"]["total_bytes"]) / 1e6, 3),
            "elapsed_s": round(time.monotonic() - started, 3),
            "remote_chunk_structure": "CDS monthly regional NetCDF shards",
            "access_assessment": {"route": "regional_server_side_subset", "chunk_amplification": None},
            "geometry": geometry,
            "cache_chunking": chunking,
            "rechunk_rationale": "monthly CDS shards converted to bounded time chunks for local/HPC workers",
            "materialisation": {
                "strategy": "monthly shards appended in bounded time blocks",
                "maximum_source_frames_in_memory": int(max_frames_in_memory),
                "requested_time_block_frames": int(time_chunk),
                "full_record_loaded": False,
                "publication": "temporary sibling Zarr renamed only after complete validation",
                "content_hash": "logical variable-major stream; independent of Zarr chunking",
            },
            "storage_preflight": storage_preflight,
            "acquisition": state,
            "source_route": "Copernicus Climate Data Store API",
            "independent_overlap_check": "NOT RUN",
        }
        manifest_file = Path(manifest_path(crop, cache_dir))
        temporary = manifest_file.with_suffix(manifest_file.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, manifest_file)
        return manifest
    except Exception:
        if path.is_dir() and not Path(manifest_path(crop, cache_dir)).exists():
            shutil.rmtree(path)
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def rematerialise_cds_from_provenance(
    record: Mapping[str, Any],
    *,
    download_dir: Union[str, os.PathLike[str]],
    cache_dir: Optional[str] = None,
    time_chunk: int = 32,
    check_size: bool = True,
    client: Optional[Any] = None,
    allow_network: Optional[bool] = None,
) -> Dict[str, Any]:
    """Replay a CDS cache from its acquisition or cache manifest without local-path guessing."""
    request_record: Any = record
    if "acquisition" in request_record:
        request_record = request_record["acquisition"]
    if "request" in request_record:
        request_record = request_record["request"]
    if not isinstance(request_record, Mapping):
        raise InvalidParameterError("CDS provenance", type(request_record).__name__,
                                    "a manifest or request mapping")
    spec = CDSRegionalRequest.from_provenance(request_record)
    return materialise_cds(
        spec, download_dir=download_dir, cache_dir=cache_dir, time_chunk=time_chunk,
        check_size=check_size, client=client, allow_network=allow_network)


def _comma_tuple(value: str, cast: Any) -> Tuple[Any, ...]:
    try:
        parsed = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid comma-separated value %r" % value) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or materialise an explicit regional ERA5 CDS request")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "materialise"):
        child = subparsers.add_parser(command)
        child.add_argument("--variables", default="t,q,u,v,z",
                           help="comma-separated canonical variables (default: t,q,u,v,z)")
        child.add_argument("--date-start", required=True, help="inclusive YYYY-MM-DD")
        child.add_argument("--date-end", required=True, help="inclusive YYYY-MM-DD")
        child.add_argument("--hours", required=True,
                           help="explicit comma-separated UTC hours, e.g. 0,6,12,18")
        child.add_argument("--lat", nargs=2, type=float, required=True,
                           metavar=("SOUTH", "NORTH"))
        child.add_argument("--lon", nargs=2, type=float, required=True,
                           metavar=("WEST", "EAST"))
        child.add_argument("--pressure-levels", default="850",
                           help="comma-separated hPa levels (default: 850)")
        child.add_argument("--grid", type=float, default=0.25, help="degree spacing")
        child.add_argument("--analysis-levels", type=int, default=4,
                           help="transform depth used for the valid-interior floor")
        if command == "materialise":
            child.add_argument("--download-dir", required=True,
                               help="resumable monthly NetCDF shard directory")
            child.add_argument("--cache-dir", required=True,
                               help="portable local/shared Zarr cache directory")
            child.add_argument("--time-chunk", type=int, default=32,
                               help="maximum frames per local Zarr time chunk")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _cli_parser().parse_args(argv)
    spec = CDSRegionalRequest(
        variables=_comma_tuple(args.variables, str),
        date_start=args.date_start, date_end=args.date_end,
        hours_utc=_comma_tuple(args.hours, int),
        lat_min=args.lat[0], lat_max=args.lat[1],
        lon_min=args.lon[0], lon_max=args.lon[1],
        pressure_levels=_comma_tuple(args.pressure_levels, int),
        grid_degrees=args.grid, n_levels_analysis=args.analysis_levels)
    if args.command == "plan":
        result = {
            "request": spec.to_provenance(),
            "monthly_shards": [shard.to_provenance() for shard in plan_monthly_shards(spec)],
            "storage_estimate": estimate_cds_storage(spec),
            "network_used": False,
        }
    else:
        result = materialise_cds(
            spec, download_dir=args.download_dir, cache_dir=args.cache_dir,
            time_chunk=args.time_chunk)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
