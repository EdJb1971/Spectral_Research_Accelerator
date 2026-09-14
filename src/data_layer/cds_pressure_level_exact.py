"""Exact-timestamp ERA5 pressure-level acquisition for guarded holdout studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_single_level import _parse_timestamp
from src.data_layer.cds_source import PRESSURE_LEVELS, _validate_netcdf_shard
from src.data_layer.zarr_source import NETWORK_ENV_VAR


PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"
# CDS names this relative-vorticity parameter ``vorticity``.  The phrase
# ``relative_vorticity`` is its semantic description, but CDS resolves that string to parameter
# 157 (relative humidity), so the request identifier must remain the tested catalogue name here.
PRESSURE_LEVEL_VARIABLES: Mapping[str, str] = {"vo": "vorticity"}
PRESSURE_LEVEL_EXACT_SCHEMA = "cds-pressure-level-exact-acquisition/v1"
PER_SHARD_OVERHEAD_BYTES = 16 * 1024 ** 2
MINIMUM_FREE_RESERVE_BYTES = 5 * 1024 ** 3


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True)
class CDSPressureLevelExactRequest:
    variables: Tuple[str, ...]
    pressure_levels: Tuple[int, ...]
    timestamps: Tuple[str, ...]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grid_degrees: float = 0.25
    dataset: str = PRESSURE_LEVEL_DATASET
    data_format: str = "netcdf"

    def __post_init__(self) -> None:
        if self.dataset != PRESSURE_LEVEL_DATASET:
            raise InvalidParameterError(
                "dataset", self.dataset, "the reviewed ERA5 pressure-level product")
        if self.data_format != "netcdf":
            raise InvalidParameterError("data_format", self.data_format, "netcdf")
        if (not self.variables or len(set(self.variables)) != len(self.variables)
                or set(self.variables) - set(PRESSURE_LEVEL_VARIABLES)):
            raise InvalidParameterError(
                "variables", self.variables, "the reviewed canonical variable: vo")
        if (not self.pressure_levels
                or len(set(self.pressure_levels)) != len(self.pressure_levels)
                or set(self.pressure_levels) - set(PRESSURE_LEVELS)):
            raise InvalidParameterError(
                "pressure_levels", self.pressure_levels,
                "one or more unique reviewed ERA5 pressure levels")
        if not self.timestamps or len(set(self.timestamps)) != len(self.timestamps):
            raise InvalidParameterError(
                "timestamps", self.timestamps, "one or more unique exact UTC timestamps")
        parsed = tuple(_parse_timestamp(value) for value in self.timestamps)
        if tuple(sorted(parsed)) != parsed:
            raise InvalidParameterError("timestamps", self.timestamps,
                                        "strictly increasing timestamps")
        if not (-90.0 <= self.lat_min < self.lat_max <= 90.0):
            raise InvalidParameterError("latitude bounds", (self.lat_min, self.lat_max),
                                        "-90 <= south < north <= 90")
        if not (-180.0 <= self.lon_min < self.lon_max <= 180.0):
            raise InvalidParameterError(
                "longitude bounds", (self.lon_min, self.lon_max),
                "-180 <= west < east <= 180; split a dateline crossing")
        if not math.isfinite(self.grid_degrees) or self.grid_degrees <= 0:
            raise InvalidParameterError("grid_degrees", self.grid_degrees,
                                        "a finite positive angular spacing")
        for name, span in (("latitude", self.lat_max - self.lat_min),
                           ("longitude", self.lon_max - self.lon_min)):
            intervals = span / self.grid_degrees
            if not math.isclose(intervals, round(intervals), rel_tol=0.0, abs_tol=1e-9):
                raise InvalidParameterError(
                    "%s bounds/grid" % name, span,
                    "bounds separated by an integer number of grid intervals")

    def canonical(self) -> Dict[str, Any]:
        return {
            "schema": PRESSURE_LEVEL_EXACT_SCHEMA,
            "dataset": self.dataset,
            "product_type": "reanalysis",
            "variables": list(self.variables),
            "cds_variables": [PRESSURE_LEVEL_VARIABLES[name] for name in self.variables],
            "pressure_levels": list(self.pressure_levels),
            "timestamps": list(self.timestamps),
            "bbox_north_west_south_east": [
                self.lat_max, self.lon_min, self.lat_min, self.lon_max],
            "grid_degrees": self.grid_degrees,
            "data_format": self.data_format,
        }

    def request_sha256(self) -> str:
        return _stable_hash(self.canonical())

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["pressure_levels"] = list(self.pressure_levels)
        record["timestamps"] = list(self.timestamps)
        record["request_sha256"] = self.request_sha256()
        record["area_order"] = "north, west, south, east"
        record["credential_policy"] = (
            "standard CDS client configuration; credentials never recorded")
        return record


@dataclass(frozen=True)
class CDSPressureLevelExactShard:
    timestamp: str
    request: Mapping[str, Any]
    request_sha256: str
    filename: str

    def to_provenance(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "request": dict(self.request),
                "request_sha256": self.request_sha256, "filename": self.filename}


def plan_pressure_level_exact_shards(
    spec: CDSPressureLevelExactRequest,
) -> Tuple[CDSPressureLevelExactShard, ...]:
    shards: List[CDSPressureLevelExactShard] = []
    for stamp in spec.timestamps:
        value = _parse_timestamp(stamp)
        request = {
            "product_type": ["reanalysis"],
            "variable": [PRESSURE_LEVEL_VARIABLES[name] for name in spec.variables],
            "pressure_level": [str(level) for level in spec.pressure_levels],
            "year": ["%04d" % value.year],
            "month": ["%02d" % value.month],
            "day": ["%02d" % value.day],
            "time": ["%02d:00" % value.hour],
            "area": [spec.lat_max, spec.lon_min, spec.lat_min, spec.lon_max],
            "grid": [spec.grid_degrees, spec.grid_degrees],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        digest = _stable_hash({"dataset": spec.dataset, "request": request})
        filename = "%s-%s.nc" % (value.strftime("%Y%m%dT%H%M"), digest[:12])
        shards.append(CDSPressureLevelExactShard(
            timestamp=stamp, request=request, request_sha256=digest, filename=filename))
    return tuple(shards)


def estimate_pressure_level_exact_storage(
    spec: CDSPressureLevelExactRequest,
) -> Dict[str, Any]:
    latitude_points = int(round((spec.lat_max - spec.lat_min) / spec.grid_degrees)) + 1
    longitude_points = int(round((spec.lon_max - spec.lon_min) / spec.grid_degrees)) + 1
    frames = len(spec.timestamps)
    raw = (frames * latitude_points * longitude_points
           * len(spec.variables) * len(spec.pressure_levels) * 4)
    shards = len(plan_pressure_level_exact_shards(spec))
    return {
        "basis": "float32 values x 2 safety factor + 16 MiB container overhead per shard",
        "compression_credit_assumed": False,
        "frames": frames,
        "latitude_points_upper_bound": latitude_points,
        "longitude_points_upper_bound": longitude_points,
        "variables": len(spec.variables),
        "levels": len(spec.pressure_levels),
        "raw_value_bytes": raw,
        "artifact_bytes_upper_bound": raw * 2 + shards * PER_SHARD_OVERHEAD_BYTES,
        "shards": shards,
    }


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(state) + b"\n")
    os.replace(temporary, path)


def _read_state(path: Path, spec: CDSPressureLevelExactRequest,
                shards: Sequence[CDSPressureLevelExactShard]) -> Dict[str, Any]:
    planned = [shard.to_provenance() for shard in shards]
    if not path.exists():
        return {
            "schema": PRESSURE_LEVEL_EXACT_SCHEMA,
            "request": spec.to_provenance(),
            "planned_shards": planned,
            "completed_shards": {},
            "claim_boundary": "request/download provenance only; holdout is not evaluated",
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "pressure-level exact acquisition state is unreadable: %s" % exc,
            path=str(path)) from exc
    if (state.get("schema") != PRESSURE_LEVEL_EXACT_SCHEMA
            or state.get("request", {}).get("request_sha256") != spec.request_sha256()
            or state.get("planned_shards") != planned):
        raise DataSourceError(
            "pressure-level exact acquisition state does not match the exact request")
    return state


def _validate_pressure_level_shard(path: Path) -> Dict[str, Any]:
    """Validate container integrity and prove the returned parameter is vorticity."""
    validation = _validate_netcdf_shard(path)
    try:
        import xarray as xr
        with xr.open_dataset(path) as dataset:
            if len(dataset.data_vars) != 1:
                raise DataSourceError(
                    "pressure-level exact shard must contain one field", path=str(path))
            name = str(next(iter(dataset.data_vars)))
            attrs = dataset[name].attrs
            semantic_names = {
                name.lower(), str(attrs.get("GRIB_shortName", "")).lower(),
                str(attrs.get("GRIB_cfVarName", "")).lower(),
                str(attrs.get("GRIB_name", "")).lower(),
                str(attrs.get("standard_name", "")).lower(),
            }
            if not semantic_names.intersection({"vo", "vorticity", "relative_vorticity"}):
                raise DataSourceError(
                    "pressure-level exact shard is not relative vorticity; returned %s" % name,
                    path=str(path))
            levels = (dataset.coords.get("pressure_level")
                      if "pressure_level" in dataset.coords else dataset.coords.get("level"))
            if levels is None or list(levels.values.reshape(-1).astype(float)) != [850.0]:
                raise DataSourceError(
                    "pressure-level exact shard is not the declared 850 hPa field",
                    path=str(path))
    except DataSourceError:
        raise
    except Exception as exc:
        raise DataSourceError(
            "pressure-level exact shard semantics are unreadable: %s" % exc,
            path=str(path)) from exc
    return validation


def acquire_pressure_level_exact_shards(
    spec: CDSPressureLevelExactRequest,
    download_dir: Union[str, os.PathLike[str]],
    *,
    explicit_network_authorisation: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Atomically acquire exact pressure-level shards after both network gates."""
    if explicit_network_authorisation is not True:
        raise DataSourceError(
            "pressure-level exact acquisition is refused: explicit experiment network "
            "authorisation is required")
    if os.environ.get(NETWORK_ENV_VAR, "").strip().lower() not in ("1", "true", "yes"):
        raise DataSourceError(
            "pressure-level exact acquisition is network-disabled; set %s=1" % NETWORK_ENV_VAR)
    estimate = estimate_pressure_level_exact_storage(spec)
    root = Path(download_dir)
    probe = root.absolute()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    reserve = max(MINIMUM_FREE_RESERVE_BYTES,
                  int(math.ceil(estimate["artifact_bytes_upper_bound"] * 0.10)))
    if usage.free < estimate["artifact_bytes_upper_bound"] + reserve:
        raise DataSourceError(
            "pressure-level exact acquisition storage preflight failed before network")
    if client is None:
        try:
            import cdsapi
            client = cdsapi.Client()
        except Exception as exc:
            raise DataSourceError(
                "pressure-level exact acquisition requires a configured CDS client") from exc

    root.mkdir(parents=True, exist_ok=True)
    shards = plan_pressure_level_exact_shards(spec)
    state_path = root / "acquisition.json"
    state = _read_state(state_path, spec, shards)
    completed = dict(state.get("completed_shards", {}))
    downloaded = resumed = 0
    for shard in shards:
        target = root / shard.filename
        existing = completed.get(shard.filename)
        if existing is not None:
            observed = _validate_pressure_level_shard(target)
            if (existing.get("sha256") != observed["sha256"]
                    or existing.get("request_sha256") != shard.request_sha256):
                raise DataSourceError(
                    "completed pressure-level shard failed integrity verification")
            resumed += 1
            continue
        if target.exists():
            raise DataSourceError("untracked pressure-level shard already exists", path=str(target))
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            partial.unlink()
        try:
            client.retrieve(spec.dataset, dict(shard.request), str(partial))
            validation = _validate_pressure_level_shard(partial)
            os.replace(partial, target)
        except Exception as exc:
            if partial.exists():
                partial.unlink()
            if isinstance(exc, DataSourceError):
                raise
            raise DataSourceError(
                "pressure-level CDS shard retrieval failed (%s)" % type(exc).__name__,
                path=str(target)) from exc
        completed[shard.filename] = {
            **validation, "request_sha256": shard.request_sha256,
            "dataset": spec.dataset, "timestamp": shard.timestamp,
        }
        state["completed_shards"] = completed
        _write_state(state_path, state)
        downloaded += 1
    state["run"] = {
        "downloaded_shards": downloaded,
        "resumed_shards": resumed,
        "complete": len(completed) == len(shards),
        "total_bytes": sum(int(item["bytes"]) for item in completed.values()),
        "storage_estimate": estimate,
    }
    _write_state(state_path, state)
    return state
