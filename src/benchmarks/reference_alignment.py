"""Guarded T4E.38 MSLP reference-alignment acquisition plan."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from src.benchmarks.catalogue_join import great_circle_km
from src.core.adoption import adoption_state
from src.core.errors import DataSourceError
from src.core.publication import publish_new_bytes
from src.data_layer.cds_source import _validate_netcdf_shard
from src.data_layer.cds_single_level import (
    CDSSingleLevelExactRequest,
    acquire_single_level_shards,
    estimate_single_level_storage,
    plan_exact_timestamp_shards,
)
from src.data_layer.era5_overlap import encoding_step
from src.data_layer.zarr_source import streaming_content_sha256


REFERENCE_DECLARATION_SCHEMA = "t4e38-reference-alignment-declaration/v1"
DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e38-reference-alignment-declaration.json")
DEFAULT_DOWNLOAD_ROOT = Path("data/cds_downloads/t4e38_mslp")
DEFAULT_RECORD_ROOT = Path("data/wrapped_records/t4e38")
DEFAULT_OUTPUT = Path("measurements/t4e38_reference_alignment.json")
REFERENCE_RECORD_SCHEMA = "t4e38-mslp-record/v1"
REFERENCE_MEASUREMENT_SCHEMA = "t4e38-reference-alignment/v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_sha256(body: Mapping[str, Any]) -> str:
    unsigned = dict(body)
    supplied = unsigned.pop("receipt_sha256", None)
    actual = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if supplied != actual:
        raise DataSourceError("source measurement receipt digest is invalid")
    return actual


def _bound_json(path: Path, expected_file_sha256: str, label: str) -> Dict[str, Any]:
    try:
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
            raise DataSourceError("%s bytes do not match the T4E.38 declaration" % label)
        return json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataSourceError("%s is unreadable: %s" % (label, exc), path=str(path)) from exc


def load_reference_declaration(
    path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("T4E.38 declaration is unreadable: %s" % exc) from exc
    request = body.get("single_level_request", {})
    decision = body.get("decision_declared_before_acquisition", {})
    if (body.get("schema") != REFERENCE_DECLARATION_SCHEMA
            or body.get("task") != "T4E.38"):
        raise DataSourceError("unsupported T4E.38 reference-alignment declaration")
    if (request.get("dataset") != "reanalysis-era5-single-levels"
            or request.get("canonical_variable") != "msl"
            or request.get("expected_total_shards") != 36
            or request.get("expected_frames_after_join") != 18
            or len(request.get("segments", [])) != 2):
        raise DataSourceError("T4E.38 declaration lacks the exact MSLP request")
    if ("At least 10" not in str(decision.get(
            "VERTICAL_QUANTITY_SEPARATION_DOMINANT", ""))
            or "At least 10" not in str(decision.get(
                "CATALOGUE_REANALYSIS_ALIGNMENT_DOMINANT", ""))
            or "NOT_AN_ACCEPTANCE" not in str(decision.get("no_acceptance", ""))):
        raise DataSourceError("T4E.38 declaration lacks the frozen three-way decision")
    return body


def require_current_adoption(
    declaration_path: Union[str, os.PathLike[str]],
) -> Dict[str, Any]:
    source = Path(declaration_path)
    state = adoption_state(source.parent, source.name)
    if not state.get("adopted"):
        raise DataSourceError(
            "T4E.38 acquisition is refused: the declaration has not been adopted by a maintainer")
    if not state.get("signature_still_reaches_the_declaration"):
        raise DataSourceError(
            "T4E.38 acquisition is refused: the adoption does not bind the current declaration")
    return state


def _verified_timestamps(declaration: Mapping[str, Any]) -> Tuple[str, ...]:
    evidence = declaration["source_evidence"]
    t4e36 = _bound_json(
        Path(evidence["t4e36_measurement"]),
        evidence["t4e36_measurement_file_sha256"], "T4E.36 measurement")
    if (_receipt_sha256(t4e36) != evidence["t4e36_measurement_receipt_sha256"]
            or t4e36.get("VERDICT") != "FAIL"):
        raise DataSourceError("T4E.36 source is not the declared failed measurement")
    t4e37 = _bound_json(
        Path(evidence["t4e37_measurement"]),
        evidence["t4e37_measurement_file_sha256"], "T4E.37 measurement")
    if (_receipt_sha256(t4e37) != evidence["t4e37_measurement_receipt_sha256"]
            or t4e37.get("OUTCOME") != "FIELD_REFERENCE_SEPARATION_DOMINANT"):
        raise DataSourceError("T4E.37 source is not the declared attribution measurement")
    rows = t4e36.get("paths", {}).get("raw_field", {}).get("rows", [])
    if len(rows) != declaration["frozen_population"]["expected_rows"]:
        raise DataSourceError("T4E.38 requires exactly the 18 T4E.36 rows")
    identities = [(str(row["sid"]), str(row["time"])) for row in rows]
    if len(set(identities)) != 18:
        raise DataSourceError("T4E.38 fixed rows must have 18 unique sid/time identities")
    timestamps = tuple(row["time"] for row in rows)
    if tuple(sorted(timestamps)) != timestamps or len(set(timestamps)) != 18:
        raise DataSourceError("T4E.38 exact timestamps must be unique and chronological")
    return timestamps


def plan_reference_alignment(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    declaration_path = Path(declaration_path)
    declaration = load_reference_declaration(declaration_path)
    timestamps = _verified_timestamps(declaration)
    request = declaration["single_level_request"]
    segments: List[Dict[str, Any]] = []
    for segment in request["segments"]:
        spec = CDSSingleLevelExactRequest(
            variables=(request["canonical_variable"],), timestamps=timestamps,
            lat_min=float(segment["lat_min"]), lat_max=float(segment["lat_max"]),
            lon_min=float(segment["lon_min"]), lon_max=float(segment["lon_max"]),
            grid_degrees=float(request["grid_degrees"]),
            dataset=request["dataset"], data_format=request["data_format"])
        shards = plan_exact_timestamp_shards(spec)
        estimate = estimate_single_level_storage(spec)
        if (len(shards) != request["expected_timestamp_shards_per_segment"]
                or [estimate["latitude_points_upper_bound"],
                    estimate["longitude_points_upper_bound"]]
                != request["expected_segment_shape"]):
            raise DataSourceError("T4E.38 derived shard count or segment shape drifted")
        segments.append({
            "name": segment["name"], "request": spec.to_provenance(),
            "shards": [shard.to_provenance() for shard in shards],
            "storage_estimate": estimate,
        })
    if sum(len(item["shards"]) for item in segments) != request["expected_total_shards"]:
        raise DataSourceError("T4E.38 total shard count drifted")
    return {
        "task": "T4E.38",
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
        "status": declaration["status"],
        "timestamps": list(timestamps),
        "segments": segments,
        "total_shards": sum(len(item["shards"]) for item in segments),
        "total_raw_value_bytes": sum(
            item["storage_estimate"]["raw_value_bytes"] for item in segments),
        "record_opened": False,
        "network_used": False,
    }


def acquire_reference_alignment(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    *,
    download_root: Union[str, os.PathLike[str]] = DEFAULT_DOWNLOAD_ROOT,
    explicit_network_authorisation: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    declaration_path = Path(declaration_path)
    adoption = require_current_adoption(declaration_path)
    if explicit_network_authorisation is not True:
        raise DataSourceError(
            "T4E.38 acquisition is refused: pass the experiment-specific network authorisation")
    plan = plan_reference_alignment(declaration_path)
    states = {}
    for segment in plan["segments"]:
        request = segment["request"]
        spec = CDSSingleLevelExactRequest(
            variables=tuple(request["variables"]), timestamps=tuple(request["timestamps"]),
            lat_min=request["lat_min"], lat_max=request["lat_max"],
            lon_min=request["lon_min"], lon_max=request["lon_max"],
            grid_degrees=request["grid_degrees"], dataset=request["dataset"],
            data_format=request["data_format"])
        states[segment["name"]] = acquire_single_level_shards(
            spec, Path(download_root) / segment["name"],
            explicit_network_authorisation=True, client=client)
    return {
        "schema": "t4e38-reference-alignment-acquisition/v1",
        "declaration_sha256": plan["declaration_sha256"],
        "adoption": adoption,
        "segments": states,
    }


def _segment_spec(segment: Mapping[str, Any]) -> CDSSingleLevelExactRequest:
    request = segment["request"]
    return CDSSingleLevelExactRequest(
        variables=tuple(request["variables"]), timestamps=tuple(request["timestamps"]),
        lat_min=float(request["lat_min"]), lat_max=float(request["lat_max"]),
        lon_min=float(request["lon_min"]), lon_max=float(request["lon_max"]),
        grid_degrees=float(request["grid_degrees"]), dataset=str(request["dataset"]),
        data_format=str(request["data_format"]))


def _load_complete_acquisition(directory: Path, segment: Mapping[str, Any]) -> Dict[str, Any]:
    state_path = directory / "acquisition.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "T4E.38 materialisation requires a readable acquisition state: %s" % exc,
            path=str(state_path)) from exc
    spec = _segment_spec(segment)
    planned = plan_exact_timestamp_shards(spec)
    if (state.get("schema") != "cds-single-level-exact-acquisition/v1"
            or state.get("request", {}).get("request_sha256") != spec.request_sha256()
            or state.get("planned_shards") != [item.to_provenance() for item in planned]):
        raise DataSourceError("T4E.38 acquisition state does not match the frozen segment")
    completed = state.get("completed_shards", {})
    if len(completed) != len(planned) or state.get("run", {}).get("complete") is not True:
        raise DataSourceError(
            "T4E.38 acquisition is incomplete: %d of %d shards are recorded"
            % (len(completed), len(planned)), path=str(state_path))
    for shard in planned:
        recorded = completed.get(shard.filename)
        if not isinstance(recorded, Mapping):
            raise DataSourceError("T4E.38 acquisition is missing %s" % shard.filename)
        observed = _validate_netcdf_shard(directory / shard.filename)
        if (recorded.get("sha256") != observed["sha256"]
                or int(recorded.get("bytes", -1)) != observed["bytes"]
                or recorded.get("request_sha256") != shard.request_sha256
                or recorded.get("timestamp") != shard.timestamp):
            raise DataSourceError(
                "T4E.38 acquisition shard failed content/request verification",
                path=str(directory / shard.filename))
    return state


def _normalise_single_frame(dataset: Any, spec: CDSSingleLevelExactRequest,
                            timestamp: str) -> Any:
    import numpy as np
    import xarray as xr

    if "msl" not in dataset.data_vars or set(dataset.data_vars) != {"msl"}:
        raise DataSourceError("T4E.38 shard must contain only the canonical msl field")
    time_name = "valid_time" if "valid_time" in dataset.dims else "time"
    if time_name not in dataset.dims or int(dataset.sizes[time_name]) != 1:
        raise DataSourceError("T4E.38 shard must contain exactly one declared timestamp")
    try:
        frame = dataset["msl"].isel({time_name: 0}).transpose("latitude", "longitude")
        latitudes = np.asarray(frame.latitude.values, dtype=np.float64)
        longitudes = np.asarray(frame.longitude.values, dtype=np.float64)
        values = np.asarray(frame.values)
        observed_time = np.asarray(dataset[time_name].values).reshape(-1)[0]
    except (KeyError, ValueError, TypeError) as exc:
        raise DataSourceError("T4E.38 shard axes are unreadable: %s" % exc) from exc
    expected_time = np.datetime64(timestamp.replace(" ", "T"))
    if np.datetime64(observed_time, "ns") != np.datetime64(expected_time, "ns"):
        raise DataSourceError("T4E.38 shard timestamp does not match its exact request")
    expected_latitudes = np.linspace(
        spec.lat_max, spec.lat_min,
        int(round((spec.lat_max - spec.lat_min) / spec.grid_degrees)) + 1)
    expected_longitudes = np.linspace(
        spec.lon_min, spec.lon_max,
        int(round((spec.lon_max - spec.lon_min) / spec.grid_degrees)) + 1)
    if (values.shape != (len(expected_latitudes), len(expected_longitudes))
            or not np.allclose(latitudes, expected_latitudes, rtol=0.0, atol=1e-8)
            or not np.allclose(longitudes, expected_longitudes, rtol=0.0, atol=1e-8)
            or not bool(np.isfinite(values).all())):
        raise DataSourceError("T4E.38 shard does not match the declared finite regular grid")
    return xr.Dataset(
        {"msl": (("time", "latitude", "longitude"), values[None, :, :])},
        coords={"time": [expected_time], "latitude": latitudes,
                "longitude": longitudes})


def join_mslp_segments(parent: Any, complement: Any,
                       parent_spec: CDSSingleLevelExactRequest,
                       complement_spec: CDSSingleLevelExactRequest,
                       timestamp: str) -> Tuple[Any, Dict[str, Any]]:
    """Join one exact-time pair after measuring its source-encoding seam tolerance."""
    import numpy as np
    import xarray as xr

    left = _normalise_single_frame(parent, parent_spec, timestamp)
    right = _normalise_single_frame(complement, complement_spec, timestamp)
    if not np.array_equal(left.latitude.values, right.latitude.values):
        raise DataSourceError("T4E.38 segments disagree on latitude coordinates")
    wrapped = np.asarray(right.longitude.values, dtype=np.float64) + 360.0
    if not math.isclose(float(left.longitude.values[-1]), float(wrapped[0]),
                        rel_tol=0.0, abs_tol=1e-8):
        raise DataSourceError("T4E.38 segments do not meet at the 180-degree seam")
    left_values = np.asarray(left.msl.values[0])
    right_values = np.asarray(right.msl.values[0])
    left_step = encoding_step(left_values)
    right_step = encoding_step(right_values)
    if left_step is None or right_step is None:
        raise DataSourceError(
            "T4E.38 source encoding tolerance is not measurable for %s" % timestamp)
    tolerance = max(float(left_step), float(right_step))
    difference = float(np.max(np.abs(left_values[:, -1] - right_values[:, 0])))
    if not math.isfinite(difference) or difference > tolerance:
        raise DataSourceError(
            "T4E.38 seam mismatch at %s: %.12g exceeds source encoding tolerance %.12g"
            % (timestamp, difference, tolerance))
    right = right.assign_coords(longitude=wrapped).isel(longitude=slice(1, None))
    joined = xr.concat(
        [left, right], dim="longitude", data_vars="minimal", coords="minimal",
        compat="equals", join="exact").transpose("time", "latitude", "longitude")
    longitude = np.asarray(joined.longitude.values, dtype=np.float64)
    if (len(longitude) != 321
            or not np.allclose(np.diff(longitude), parent_spec.grid_degrees,
                               rtol=0.0, atol=1e-8)):
        raise DataSourceError("T4E.38 joined longitude geometry is not the declared 321 points")
    return joined, {
        "timestamp": timestamp,
        "coordinate": 180.0,
        "retained": "parent 180E column",
        "complement_duplicate_removed": True,
        "maximum_absolute_difference": difference,
        "source_encoding_tolerance": tolerance,
        "passed": True,
    }


def materialise_reference_record(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    *,
    download_root: Union[str, os.PathLike[str]] = DEFAULT_DOWNLOAD_ROOT,
    record_root: Union[str, os.PathLike[str]] = DEFAULT_RECORD_ROOT,
) -> Dict[str, Any]:
    """Publish the 18 seam-checked MSLP frames as one immutable wrapped record."""
    import numpy as np
    import xarray as xr
    import zarr

    declaration_path = Path(declaration_path)
    declaration = load_reference_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    plan = plan_reference_alignment(declaration_path)
    segments = {item["name"]: item for item in plan["segments"]}
    if set(segments) != {"parent", "complement"}:
        raise DataSourceError("T4E.38 requires exactly parent and complement segments")
    root = Path(download_root)
    states = {
        name: _load_complete_acquisition(root / name, segments[name])
        for name in ("parent", "complement")}
    specs = {name: _segment_spec(segments[name]) for name in states}
    shards = {name: plan_exact_timestamp_shards(specs[name]) for name in states}
    source_hashes = {
        name: {shard.filename: _file_sha256(root / name / shard.filename)
               for shard in shards[name]}
        for name in states}

    frames = []
    seams = []
    for index, timestamp in enumerate(plan["timestamps"]):
        left_path = root / "parent" / shards["parent"][index].filename
        right_path = root / "complement" / shards["complement"][index].filename
        with xr.open_dataset(left_path) as left, xr.open_dataset(right_path) as right:
            frame, seam = join_mslp_segments(
                left, right, specs["parent"], specs["complement"], timestamp)
            frames.append(frame.load())
            seams.append(seam)
    record = xr.concat(frames, dim="time", data_vars="minimal", coords="minimal",
                       compat="equals", join="exact")
    if (dict(record.sizes) != {"time": 18, "latitude": 161, "longitude": 321}
            or not bool(np.isfinite(record.msl.values).all())):
        raise DataSourceError("T4E.38 materialised record has unexpected shape or values")
    if any(_file_sha256(root / name / filename) != digest
           for name, files in source_hashes.items() for filename, digest in files.items()):
        raise DataSourceError("a T4E.38 acquisition shard changed during materialisation")

    destination = Path(record_root)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".t4e38.", dir=str(destination)))
    temporary_store = temporary_root / "record.zarr"
    try:
        chunked = record.chunk({"time": 1, "latitude": 161, "longitude": 321})
        chunked.to_zarr(
            temporary_store, mode="w", consolidated=False,
            encoding={"msl": {"chunks": (1, 161, 321)}})
        zarr.consolidate_metadata(str(temporary_store))
        with xr.open_zarr(temporary_store, consolidated=True) as opened:
            record_sha256 = streaming_content_sha256(opened, time_block=1)
            shape = {str(name): int(value) for name, value in opened.sizes.items()}
        record_path = destination / (record_sha256 + ".zarr")
        if record_path.exists():
            with xr.open_zarr(record_path, consolidated=True) as existing:
                if (streaming_content_sha256(existing, time_block=1) != record_sha256
                        or {str(k): int(v) for k, v in existing.sizes.items()} != shape):
                    raise DataSourceError(
                        "content-addressed T4E.38 record contains different content",
                        path=str(record_path))
            shutil.rmtree(temporary_store)
        else:
            os.rename(temporary_store, record_path)
        coordinates_sha256 = _sha256({
            "time": [str(value) for value in record.time.values.astype("datetime64[s]")],
            "latitude": [float(value) for value in record.latitude.values],
            "longitude": [float(value) for value in record.longitude.values],
        })
        receipt = {
            "schema": REFERENCE_RECORD_SCHEMA,
            "record_sha256": record_sha256,
            "record_path": str(record_path).replace("\\", "/"),
            "coordinates_sha256": coordinates_sha256,
            "shape": shape,
            "declaration": str(declaration_path).replace("\\", "/"),
            "declaration_sha256": _file_sha256(declaration_path),
            "adoption": adoption,
            "sources": {
                name: {
                    "directory": str(root / name).replace("\\", "/"),
                    "acquisition_sha256": _file_sha256(root / name / "acquisition.json"),
                    "request_sha256": specs[name].request_sha256(),
                    "shards": source_hashes[name],
                    "preserved_byte_for_byte": True,
                } for name in ("parent", "complement")
            },
            "seam": {"checks": seams, "all_passed": True},
            "network_used": False,
            "claim_boundary": declaration["claim_boundary"],
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        receipt_path = destination / (receipt["receipt_sha256"] + ".json")
        payload = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if receipt_path.exists():
            if receipt_path.read_bytes() != payload:
                raise DataSourceError(
                    "content-addressed T4E.38 receipt contains different bytes",
                    path=str(receipt_path))
        else:
            publish_new_bytes(receipt_path, payload, "T4E.38 MSLP record receipt")
        return receipt
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def _load_reference_record_receipt(path: Path) -> Dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("T4E.38 record receipt is unreadable: %s" % exc) from exc
    if receipt.get("schema") != REFERENCE_RECORD_SCHEMA:
        raise DataSourceError("unsupported T4E.38 record receipt schema")
    if _receipt_sha256(receipt) != receipt.get("receipt_sha256"):
        raise DataSourceError("T4E.38 record receipt digest is invalid")
    return receipt


def _coordinate_sha256(dataset: Any) -> str:
    return _sha256({
        "time": [str(value) for value in dataset.time.values.astype("datetime64[s]")],
        "latitude": [float(value) for value in dataset.latitude.values],
        "longitude": [float(value) for value in dataset.longitude.values],
    })


def basin_walk(values: Any, latitudes: Any, longitudes: Any,
               catalogue_position: Tuple[float, float], *,
               direction: str) -> Dict[str, Any]:
    """Apply the frozen catalogue-seeded 3x3 path with deterministic ties."""
    import numpy as np

    field = np.asarray(values, dtype=np.float64)
    latitude = np.asarray(latitudes, dtype=np.float64)
    longitude = np.asarray(longitudes, dtype=np.float64)
    if (field.ndim != 2 or field.shape != (len(latitude), len(longitude))
            or not bool(np.isfinite(field).all())):
        raise DataSourceError("basin walk requires one finite field on its declared axes")
    if direction not in ("minimum", "maximum"):
        raise DataSourceError("basin walk direction must be minimum or maximum")
    cat_lat, cat_lon = (float(catalogue_position[0]), float(catalogue_position[1]))
    row = int(np.argmin(np.abs(latitude - cat_lat)))
    column = int(np.argmin(np.abs(longitude - cat_lon)))
    seed = [row, column]
    visited = set()
    path = []
    while True:
        if row in (0, len(latitude) - 1) or column in (0, len(longitude) - 1):
            return {
                "status": "REFUSED_BOUNDARY", "seed_grid_index": seed,
                "path": path, "steps": max(0, len(path) - 1), "centre": None,
            }
        current = (row, column)
        if current in visited:
            return {
                "status": "REFUSED_REPEAT", "seed_grid_index": seed,
                "path": path, "steps": max(0, len(path) - 1), "centre": None,
            }
        visited.add(current)
        path.append({
            "grid_index": [row, column], "lat": float(latitude[row]),
            "lon": float(longitude[column]), "field_value": float(field[row, column]),
        })
        candidates = [
            (r, c) for r in range(row - 1, row + 2)
            for c in range(column - 1, column + 2)]
        target = (min(field[r, c] for r, c in candidates)
                  if direction == "minimum"
                  else max(field[r, c] for r, c in candidates))
        tied = [(r, c) for r, c in candidates if field[r, c] == target]
        next_row, next_column = min(
            tied, key=lambda item: (
                great_circle_km(
                    (cat_lat, cat_lon),
                    (float(latitude[item[0]]), float(longitude[item[1]]))),
                item[0], item[1]))
        if (next_row, next_column) == current:
            centre = dict(path[-1])
            return {
                "status": "CENTRE_FOUND", "seed_grid_index": seed,
                "path": path, "steps": len(path) - 1, "centre": centre,
            }
        if (next_row, next_column) in visited:
            return {
                "status": "REFUSED_REPEAT", "seed_grid_index": seed,
                "path": path, "steps": max(0, len(path) - 1), "centre": None,
            }
        row, column = next_row, next_column


def reference_alignment_decision(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = (
        "VERTICAL_QUANTITY_SEPARATION_CANDIDATE",
        "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE",
        "EXACT_TIE", "REFUSED")
    counts = {label: 0 for label in labels}
    if len(rows) != 18:
        raise DataSourceError("T4E.38 decision requires exactly 18 fixed rows")
    for row in rows:
        label = str(row.get("classification"))
        if label not in counts:
            raise DataSourceError("T4E.38 row has no declared comparison classification")
        counts[label] += 1
    if counts[labels[0]] >= 10:
        outcome = "VERTICAL_QUANTITY_SEPARATION_DOMINANT"
    elif counts[labels[1]] >= 10:
        outcome = "CATALOGUE_REANALYSIS_ALIGNMENT_DOMINANT"
    else:
        outcome = "MIXED"
    return {"outcome": outcome, "counts": counts, "majority_needed": 10,
            "population_denominator": 18, "is_acceptance_verdict": False}


def _fixed_rows(declaration: Mapping[str, Any]) -> List[Dict[str, Any]]:
    timestamps = _verified_timestamps(declaration)
    evidence = declaration["source_evidence"]
    measurement = _bound_json(
        Path(evidence["t4e36_measurement"]),
        evidence["t4e36_measurement_file_sha256"], "T4E.36 measurement")
    rows = measurement.get("paths", {}).get("raw_field", {}).get("rows", [])
    if [str(row["time"]) for row in rows] != list(timestamps):
        raise DataSourceError("T4E.38 fixed row order drifted from the declared timestamps")
    return [dict(row) for row in rows]


def evaluate_reference_alignment(
    record_receipt_path: Union[str, os.PathLike[str]],
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    *,
    output_path: Union[str, os.PathLike[str]] = DEFAULT_OUTPUT,
    hash_block_frames: int = 1,
) -> Dict[str, Any]:
    """Compare frozen MSLP and vorticity basin centres on exactly the 18 declared rows."""
    import numpy as np
    import xarray as xr
    from src.benchmarks.wrapped_join import _load_record_receipt

    declaration_path = Path(declaration_path)
    declaration = load_reference_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    rows = _fixed_rows(declaration)
    mslp_receipt_path = Path(record_receipt_path)
    mslp_receipt = _load_reference_record_receipt(mslp_receipt_path)
    declaration_sha256 = _file_sha256(declaration_path)
    if mslp_receipt.get("declaration_sha256") != declaration_sha256:
        raise DataSourceError("T4E.38 MSLP record belongs to a different declaration")

    source = declaration["source_evidence"]
    t4e36 = _bound_json(
        Path(source["t4e36_measurement"]), source["t4e36_measurement_file_sha256"],
        "T4E.36 measurement")
    vorticity_receipt_path = Path(t4e36["wrapped_record_receipt"])
    vorticity_receipt = _load_record_receipt(vorticity_receipt_path)
    if vorticity_receipt.get("receipt_sha256") != t4e36["wrapped_record_receipt_sha256"]:
        raise DataSourceError("T4E.36 vorticity record receipt identity drifted")

    measured = []
    mslp_path = Path(mslp_receipt["record_path"])
    vorticity_path = Path(vorticity_receipt["record_path"])
    with xr.open_zarr(mslp_path, consolidated=True) as mslp, \
            xr.open_zarr(vorticity_path, consolidated=True) as vorticity:
        if (streaming_content_sha256(mslp, time_block=hash_block_frames)
                != mslp_receipt["record_sha256"]
                or _coordinate_sha256(mslp) != mslp_receipt["coordinates_sha256"]):
            raise DataSourceError("T4E.38 MSLP record content or coordinates drifted")
        if (streaming_content_sha256(vorticity, time_block=32)
                != vorticity_receipt["record_sha256"]):
            raise DataSourceError("T4E.36 vorticity record content drifted")
        latitudes = np.asarray(mslp.latitude.values, dtype=np.float64)
        longitudes = np.asarray(mslp.longitude.values, dtype=np.float64)
        if (not np.array_equal(latitudes, np.asarray(vorticity.latitude.values))
                or not np.array_equal(longitudes, np.asarray(vorticity.longitude.values))):
            raise DataSourceError("MSLP and vorticity records do not share the declared grid")
        for source_row in rows:
            stamp = np.datetime64(str(source_row["time"]).replace(" ", "T"))
            try:
                pressure = np.asarray(
                    mslp.msl.sel(time=stamp).squeeze(drop=True).values, dtype=np.float64)
                signed_vorticity = -np.asarray(
                    vorticity.vo.sel(time=stamp, level=850.0).squeeze(drop=True).values,
                    dtype=np.float64)
            except (KeyError, ValueError) as exc:
                raise DataSourceError(
                    "a frozen T4E.38 timestamp is absent from an input record") from exc
            catalogue = (float(source_row["lat"]), float(source_row["lon"]))
            pressure_walk = basin_walk(
                pressure, latitudes, longitudes, catalogue, direction="minimum")
            vorticity_walk = basin_walk(
                signed_vorticity, latitudes, longitudes, catalogue, direction="maximum")
            if (pressure_walk["status"] != "CENTRE_FOUND"
                    or vorticity_walk["status"] != "CENTRE_FOUND"):
                classification = "REFUSED"
                pressure_distance = vorticity_distance = centres_distance = None
            else:
                pressure_centre = pressure_walk["centre"]
                vorticity_centre = vorticity_walk["centre"]
                pressure_position = (pressure_centre["lat"], pressure_centre["lon"])
                vorticity_position = (vorticity_centre["lat"], vorticity_centre["lon"])
                pressure_distance = great_circle_km(catalogue, pressure_position)
                vorticity_distance = great_circle_km(catalogue, vorticity_position)
                centres_distance = great_circle_km(pressure_position, vorticity_position)
                classification = (
                    "VERTICAL_QUANTITY_SEPARATION_CANDIDATE"
                    if pressure_distance < vorticity_distance else
                    "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE"
                    if pressure_distance > vorticity_distance else "EXACT_TIE")
            measured.append({
                "storm": str(source_row["storm"]), "sid": str(source_row["sid"]),
                "time": str(source_row["time"]),
                "catalogue_position": {"lat": catalogue[0], "lon": catalogue[1]},
                "mslp_basin": pressure_walk,
                "negated_vorticity_basin": vorticity_walk,
                "catalogue_to_mslp_km": pressure_distance,
                "catalogue_to_vorticity_km": vorticity_distance,
                "mslp_to_vorticity_km": centres_distance,
                "classification": classification,
                "t4e36_inside_radius": int(source_row["inside_radius"]),
            })

    decision = reference_alignment_decision(measured)
    result = {
        "schema": REFERENCE_MEASUREMENT_SCHEMA,
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": declaration_sha256,
        "adoption": adoption,
        "sources": {
            "mslp_record_receipt": str(mslp_receipt_path).replace("\\", "/"),
            "mslp_record_receipt_sha256": mslp_receipt["receipt_sha256"],
            "vorticity_record_receipt": str(vorticity_receipt_path).replace("\\", "/"),
            "vorticity_record_receipt_sha256": vorticity_receipt["receipt_sha256"],
            "t4e36_measurement": source["t4e36_measurement"],
            "t4e36_measurement_file_sha256": source["t4e36_measurement_file_sha256"],
        },
        "population": {
            "rows": len(measured), "catalogue_census_rerun": False,
            "failures_only_selected": False, "forecast_test_opened": False,
        },
        "method": {
            "catalogue_seeded": True, "neighbourhood": "3x3",
            "tie_break": "great_circle_to_catalogue_then_lower_row_then_lower_column",
            "mslp_direction": "minimum", "vorticity_negated": True,
            "vorticity_direction_after_negation": "maximum",
            "search_radius_used": False,
            "distance_function": "catalogue_join.great_circle_km",
        },
        "rows": measured,
        "decision": decision,
        "OUTCOME": decision["outcome"],
        "VERDICT": "NOT_AN_ACCEPTANCE",
        "network_used": False,
        "claim_boundary": declaration["claim_boundary"],
    }
    result["receipt_sha256"] = _sha256(result)
    try:
        publish_new_bytes(
            output_path,
            json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "T4E.38 reference-alignment measurement")
    except FileExistsError as exc:
        raise DataSourceError(str(exc), path=str(output_path)) from exc
    return result
