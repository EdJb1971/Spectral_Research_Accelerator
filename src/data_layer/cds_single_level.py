"""Exact-timestamp ERA5 single-level acquisition for T4E.38.

This is deliberately separate from :mod:`cds_source`: pressure-level requests require a level
axis and are a different CDS product. Sharing a permissive request object would allow MSLP to be
submitted to the wrong product or a pressure field to lose its declared level.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_source import _validate_netcdf_shard
from src.data_layer.zarr_source import NETWORK_ENV_VAR


SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"
SINGLE_LEVEL_VARIABLES: Mapping[str, str] = {"msl": "mean_sea_level_pressure"}
SINGLE_LEVEL_ACQUISITION_SCHEMA = "cds-single-level-exact-acquisition/v1"
PER_SHARD_OVERHEAD_BYTES = 16 * 1024 ** 2
MINIMUM_FREE_RESERVE_BYTES = 5 * 1024 ** 3


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace(" ", "T"))
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            "timestamps", value, "an exact ISO timestamp YYYY-MM-DD HH:MM:SS") from exc
    if parsed.minute or parsed.second or parsed.microsecond or parsed.tzinfo is not None:
        raise InvalidParameterError(
            "timestamps", value, "a timezone-naive exact whole UTC hour")
    return parsed


@dataclass(frozen=True)
class CDSSingleLevelExactRequest:
    variables: Tuple[str, ...]
    timestamps: Tuple[str, ...]
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    grid_degrees: float = 0.25
    dataset: str = SINGLE_LEVEL_DATASET
    data_format: str = "netcdf"

    def __post_init__(self) -> None:
        if self.dataset != SINGLE_LEVEL_DATASET:
            raise InvalidParameterError(
                "dataset", self.dataset, "the reviewed ERA5 single-level product")
        if self.data_format != "netcdf":
            raise InvalidParameterError("data_format", self.data_format, "netcdf")
        if not self.variables or len(set(self.variables)) != len(self.variables):
            raise InvalidParameterError("variables", self.variables,
                                        "one or more unique single-level variables")
        unknown = sorted(set(self.variables) - set(SINGLE_LEVEL_VARIABLES))
        if unknown:
            raise InvalidParameterError(
                "variables", unknown, "a reviewed canonical variable: msl")
        if not self.timestamps or len(set(self.timestamps)) != len(self.timestamps):
            raise InvalidParameterError("timestamps", self.timestamps,
                                        "one or more unique exact UTC timestamps")
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
            "schema": SINGLE_LEVEL_ACQUISITION_SCHEMA,
            "dataset": self.dataset,
            "product_type": "reanalysis",
            "variables": list(self.variables),
            "cds_variables": [SINGLE_LEVEL_VARIABLES[name] for name in self.variables],
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
        record["timestamps"] = list(self.timestamps)
        record["request_sha256"] = self.request_sha256()
        record["area_order"] = "north, west, south, east"
        record["credential_policy"] = (
            "standard CDS client configuration; credentials never recorded")
        return record


@dataclass(frozen=True)
class CDSSingleLevelShard:
    timestamp: str
    request: Mapping[str, Any]
    request_sha256: str
    filename: str

    def to_provenance(self) -> Dict[str, Any]:
        return {"timestamp": self.timestamp, "request": dict(self.request),
                "request_sha256": self.request_sha256, "filename": self.filename}


def plan_exact_timestamp_shards(
    spec: CDSSingleLevelExactRequest,
) -> Tuple[CDSSingleLevelShard, ...]:
    shards: List[CDSSingleLevelShard] = []
    for stamp in spec.timestamps:
        value = _parse_timestamp(stamp)
        request = {
            "product_type": ["reanalysis"],
            "variable": [SINGLE_LEVEL_VARIABLES[name] for name in spec.variables],
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
        shards.append(CDSSingleLevelShard(
            timestamp=stamp, request=request, request_sha256=digest, filename=filename))
    return tuple(shards)


def estimate_single_level_storage(spec: CDSSingleLevelExactRequest) -> Dict[str, Any]:
    latitude_points = int(round((spec.lat_max - spec.lat_min) / spec.grid_degrees)) + 1
    longitude_points = int(round((spec.lon_max - spec.lon_min) / spec.grid_degrees)) + 1
    frames = len(spec.timestamps)
    raw = frames * latitude_points * longitude_points * len(spec.variables) * 4
    shards = len(plan_exact_timestamp_shards(spec))
    return {
        "basis": "float32 values x 2 safety factor + 16 MiB container overhead per shard",
        "compression_credit_assumed": False,
        "frames": frames,
        "latitude_points_upper_bound": latitude_points,
        "longitude_points_upper_bound": longitude_points,
        "variables": len(spec.variables),
        "raw_value_bytes": raw,
        "artifact_bytes_upper_bound": raw * 2 + shards * PER_SHARD_OVERHEAD_BYTES,
        "shards": shards,
    }


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(state) + b"\n")
    os.replace(temporary, path)


def _read_state(path: Path, spec: CDSSingleLevelExactRequest,
                shards: Sequence[CDSSingleLevelShard]) -> Dict[str, Any]:
    planned = [shard.to_provenance() for shard in shards]
    if not path.exists():
        return {
            "schema": SINGLE_LEVEL_ACQUISITION_SCHEMA,
            "request": spec.to_provenance(),
            "planned_shards": planned,
            "completed_shards": {},
            "claim_boundary": "request/download provenance only; alignment is not measured",
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "single-level acquisition state is unreadable: %s" % exc, path=str(path)) from exc
    if (state.get("schema") != SINGLE_LEVEL_ACQUISITION_SCHEMA
            or state.get("request", {}).get("request_sha256") != spec.request_sha256()
            or state.get("planned_shards") != planned):
        raise DataSourceError("single-level acquisition state does not match the exact request")
    return state


def _network_enabled() -> bool:
    return os.environ.get(NETWORK_ENV_VAR, "").strip().lower() in ("1", "true", "yes")


def acquire_single_level_shards(
    spec: CDSSingleLevelExactRequest,
    download_dir: Union[str, os.PathLike[str]],
    *,
    explicit_network_authorisation: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Atomically acquire exact timestamp shards after two independent network gates."""
    if explicit_network_authorisation is not True:
        raise DataSourceError(
            "single-level acquisition is refused: explicit experiment network authorisation "
            "is required")
    if not _network_enabled():
        raise DataSourceError(
            "single-level acquisition is network-disabled; set %s=1" % NETWORK_ENV_VAR)
    estimate = estimate_single_level_storage(spec)
    root = Path(download_dir)
    probe = root.absolute()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    reserve = max(MINIMUM_FREE_RESERVE_BYTES,
                  int(math.ceil(estimate["artifact_bytes_upper_bound"] * 0.10)))
    if usage.free < estimate["artifact_bytes_upper_bound"] + reserve:
        raise DataSourceError("single-level acquisition storage preflight failed before network")
    if client is None:
        try:
            import cdsapi
            client = cdsapi.Client()
        except Exception as exc:
            raise DataSourceError(
                "single-level acquisition requires a configured CDS client") from exc

    root.mkdir(parents=True, exist_ok=True)
    shards = plan_exact_timestamp_shards(spec)
    state_path = root / "acquisition.json"
    state = _read_state(state_path, spec, shards)
    completed = dict(state.get("completed_shards", {}))
    downloaded = resumed = 0
    for shard in shards:
        target = root / shard.filename
        existing = completed.get(shard.filename)
        if existing is not None:
            observed = _validate_netcdf_shard(target)
            if (existing.get("sha256") != observed["sha256"]
                    or existing.get("request_sha256") != shard.request_sha256):
                raise DataSourceError("completed single-level shard failed integrity verification")
            resumed += 1
            continue
        if target.exists():
            raise DataSourceError("untracked single-level shard already exists", path=str(target))
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
                "single-level CDS shard retrieval failed (%s)" % type(exc).__name__,
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
