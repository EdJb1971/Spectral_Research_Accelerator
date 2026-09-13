"""Exact-field acquisition, materialisation and evaluation for T4E.39."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.benchmarks.catalogue_join import great_circle_km
from src.benchmarks.reference_alignment import basin_walk
from src.benchmarks.reference_holdout import (
    DEFAULT_CENSUS,
    DEFAULT_DECLARATION,
    HOLDOUT_CENSUS_SCHEMA,
    load_holdout_declaration,
    require_current_adoption,
)
from src.core.errors import DataSourceError
from src.core.publication import publish_new_bytes
from src.data_layer.cds_pressure_level_exact import (
    CDSPressureLevelExactRequest,
    PRESSURE_LEVEL_EXACT_SCHEMA,
    acquire_pressure_level_exact_shards,
    estimate_pressure_level_exact_storage,
    plan_pressure_level_exact_shards,
)
from src.data_layer.cds_single_level import (
    CDSSingleLevelExactRequest,
    SINGLE_LEVEL_ACQUISITION_SCHEMA,
    acquire_single_level_shards,
    estimate_single_level_storage,
    plan_exact_timestamp_shards,
)
from src.data_layer.cds_source import _validate_netcdf_shard
from src.data_layer.era5_overlap import encoding_step
from src.data_layer.zarr_source import streaming_content_sha256


DEFAULT_DOWNLOAD_ROOT = Path("data/cds_downloads/t4e39_holdout")
DEFAULT_RECORD_ROOT = Path("data/wrapped_records/t4e39")
DEFAULT_OUTPUT = Path("measurements/t4e39_reference_holdout_evaluation.json")
HOLDOUT_FIELD_RECORD_SCHEMA = "t4e39-reference-holdout-field-record/v1"
HOLDOUT_EVALUATION_SCHEMA = "t4e39-reference-holdout-evaluation/v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_receipt_identity(body: Mapping[str, Any], label: str) -> str:
    """Recompute an artifact receipt from its own content and refuse a supplied digest.

    Public because the review surface re-verifies the same receipts it displays; a reader
    that trusts a digest written beside the content it describes is checking nothing.
    """
    unsigned = dict(body)
    supplied = unsigned.pop("receipt_sha256", None)
    actual = _sha256(unsigned)
    if supplied != actual:
        raise DataSourceError("%s receipt identity is invalid" % label)
    return actual


_verified_receipt = verify_receipt_identity


def load_holdout_census(
    census_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
) -> Dict[str, Any]:
    path = Path(census_path)
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("T4E.39 census is unreadable: %s" % exc, path=str(path)) from exc
    declaration_path = Path(declaration_path)
    if (body.get("schema") != HOLDOUT_CENSUS_SCHEMA
            or body.get("declaration_sha256") != _file_sha256(declaration_path)
            or _verified_receipt(body, "T4E.39 census") != body.get("receipt_sha256")):
        raise DataSourceError("T4E.39 census does not bind the current declaration")
    rows = body.get("selected_rows", [])
    if (body.get("status") != "READY_FOR_EXACT_ACQUISITION"
            or len(rows) != body.get("selected_storms")
            or len(rows) < body.get("minimum_selected_storms", 10)
            or len({(row.get("sid"), row.get("time")) for row in rows}) != len(rows)):
        raise DataSourceError("T4E.39 census is not ready for exact acquisition")
    return body


def _spec(field: str, segment: Mapping[str, Any], timestamps: Tuple[str, ...],
          acquisition: Mapping[str, Any]) -> Any:
    common = dict(
        timestamps=timestamps,
        lat_min=float(segment["lat_min"]), lat_max=float(segment["lat_max"]),
        lon_min=float(segment["lon_min"]), lon_max=float(segment["lon_max"]),
        grid_degrees=float(acquisition["grid_degrees"]))
    if field == "msl":
        return CDSSingleLevelExactRequest(
            variables=("msl",), dataset=acquisition["mslp"]["dataset"], **common)
    return CDSPressureLevelExactRequest(
        variables=("vo",), pressure_levels=(int(
            acquisition["vorticity"]["pressure_level_hpa"]),),
        dataset=acquisition["vorticity"]["dataset"], **common)


def _plan_shards(spec: Any) -> Tuple[Any, ...]:
    if isinstance(spec, CDSSingleLevelExactRequest):
        return plan_exact_timestamp_shards(spec)
    return plan_pressure_level_exact_shards(spec)


def _estimate(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, CDSSingleLevelExactRequest):
        return estimate_single_level_storage(spec)
    return estimate_pressure_level_exact_storage(spec)


def plan_holdout_fields(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    census_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
) -> Dict[str, Any]:
    declaration_path = Path(declaration_path)
    declaration = load_holdout_declaration(declaration_path)
    census = load_holdout_census(census_path, declaration_path)
    timestamps = tuple(str(row["time"]) for row in census["selected_rows"])
    if tuple(sorted(timestamps)) != timestamps or len(set(timestamps)) != len(timestamps):
        raise DataSourceError("T4E.39 census timestamps are not unique and chronological")
    acquisition = declaration["future_exact_acquisition"]
    requests = []
    for field in ("msl", "vo"):
        for segment in acquisition["field_segments"]:
            spec = _spec(field, segment, timestamps, acquisition)
            shards = _plan_shards(spec)
            estimate = _estimate(spec)
            if ([estimate["latitude_points_upper_bound"],
                 estimate["longitude_points_upper_bound"]]
                    != acquisition["expected_segment_shape"]
                    or len(shards) != len(timestamps)):
                raise DataSourceError("T4E.39 exact request shape or shard count drifted")
            requests.append({
                "field": field, "segment": segment["name"],
                "request": spec.to_provenance(),
                "shards": [item.to_provenance() for item in shards],
                "storage_estimate": estimate,
            })
    return {
        "task": "T4E.39", "phase": "EXACT_FIELD_ACQUISITION",
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": _file_sha256(declaration_path),
        "census": str(Path(census_path)).replace("\\", "/"),
        "census_receipt_sha256": census["receipt_sha256"],
        "selected_storms": len(timestamps), "timestamps": list(timestamps),
        "requests": requests,
        "total_shards": sum(len(item["shards"]) for item in requests),
        "total_raw_value_bytes": sum(
            item["storage_estimate"]["raw_value_bytes"] for item in requests),
        "record_opened": False, "network_used": False,
    }


def _spec_from_plan(item: Mapping[str, Any]) -> Any:
    request = item["request"]
    common = dict(
        variables=tuple(request["variables"]), timestamps=tuple(request["timestamps"]),
        lat_min=float(request["lat_min"]), lat_max=float(request["lat_max"]),
        lon_min=float(request["lon_min"]), lon_max=float(request["lon_max"]),
        grid_degrees=float(request["grid_degrees"]), dataset=str(request["dataset"]),
        data_format=str(request["data_format"]))
    if item["field"] == "msl":
        return CDSSingleLevelExactRequest(**common)
    return CDSPressureLevelExactRequest(
        pressure_levels=tuple(int(v) for v in request["pressure_levels"]), **common)


def acquire_holdout_fields(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    census_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
    *, download_root: Union[str, os.PathLike[str]] = DEFAULT_DOWNLOAD_ROOT,
    explicit_network_authorisation: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    declaration_path = Path(declaration_path)
    adoption = require_current_adoption(declaration_path)
    if explicit_network_authorisation is not True:
        raise DataSourceError(
            "T4E.39 acquisition is refused: pass the experiment-specific network authorisation")
    plan = plan_holdout_fields(declaration_path, census_path)
    states = {}
    root = Path(download_root)
    for item in plan["requests"]:
        spec = _spec_from_plan(item)
        key = "%s_%s" % (item["field"], item["segment"])
        target = root / item["field"] / item["segment"]
        if item["field"] == "msl":
            state = acquire_single_level_shards(
                spec, target, explicit_network_authorisation=True, client=client)
        else:
            state = acquire_pressure_level_exact_shards(
                spec, target, explicit_network_authorisation=True, client=client)
        states[key] = state
    return {
        "schema": "t4e39-reference-holdout-acquisition/v1",
        "declaration_sha256": plan["declaration_sha256"],
        "census_receipt_sha256": plan["census_receipt_sha256"],
        "adoption": adoption, "requests": states,
    }


def _load_complete(directory: Path, item: Mapping[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    spec = _spec_from_plan(item)
    shards = _plan_shards(spec)
    state_path = directory / "acquisition.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "T4E.39 materialisation requires a readable acquisition state: %s" % exc,
            path=str(state_path)) from exc
    expected_schema = (SINGLE_LEVEL_ACQUISITION_SCHEMA if item["field"] == "msl"
                       else PRESSURE_LEVEL_EXACT_SCHEMA)
    if (state.get("schema") != expected_schema
            or state.get("request", {}).get("request_sha256") != spec.request_sha256()
            or state.get("planned_shards") != [shard.to_provenance() for shard in shards]
            or state.get("run", {}).get("complete") is not True):
        raise DataSourceError("T4E.39 acquisition state is incomplete or does not match its request")
    completed = state.get("completed_shards", {})
    if len(completed) != len(shards):
        raise DataSourceError("T4E.39 acquisition state omits a planned shard")
    for shard in shards:
        observed = _validate_netcdf_shard(directory / shard.filename)
        recorded = completed.get(shard.filename, {})
        if (recorded.get("sha256") != observed["sha256"]
                or int(recorded.get("bytes", -1)) != observed["bytes"]
                or recorded.get("request_sha256") != shard.request_sha256):
            raise DataSourceError("T4E.39 shard failed content/request verification")
    return spec, state


def _normalise_frame(dataset: Any, field: str, spec: Any, timestamp: str) -> Any:
    import numpy as np
    import xarray as xr

    if field not in dataset.data_vars or set(dataset.data_vars) != {field}:
        raise DataSourceError("T4E.39 shard must contain only its canonical field")
    time_name = "valid_time" if "valid_time" in dataset.dims else "time"
    if time_name not in dataset.dims or int(dataset.sizes[time_name]) != 1:
        raise DataSourceError("T4E.39 shard must contain exactly one timestamp")
    array = dataset[field].isel({time_name: 0})
    level_name = "pressure_level" if "pressure_level" in array.dims else "level"
    if field == "vo":
        if level_name in array.dims:
            if int(array.sizes[level_name]) != 1:
                raise DataSourceError("T4E.39 vorticity shard must contain exactly one level")
            observed_level = float(array[level_name].values.reshape(-1)[0])
            if not math.isclose(observed_level, 850.0, rel_tol=0.0, abs_tol=1e-8):
                raise DataSourceError("T4E.39 vorticity shard is not 850 hPa")
            array = array.isel({level_name: 0})
    try:
        array = array.transpose("latitude", "longitude")
        latitudes = np.asarray(array.latitude.values, dtype=np.float64)
        longitudes = np.asarray(array.longitude.values, dtype=np.float64)
        values = np.asarray(array.values)
        observed_time = np.asarray(dataset[time_name].values).reshape(-1)[0]
    except (KeyError, ValueError, TypeError) as exc:
        raise DataSourceError("T4E.39 shard axes are unreadable: %s" % exc) from exc
    expected_time = np.datetime64(timestamp.replace(" ", "T"))
    expected_latitudes = np.linspace(
        spec.lat_max, spec.lat_min,
        int(round((spec.lat_max - spec.lat_min) / spec.grid_degrees)) + 1)
    expected_longitudes = np.linspace(
        spec.lon_min, spec.lon_max,
        int(round((spec.lon_max - spec.lon_min) / spec.grid_degrees)) + 1)
    if (np.datetime64(observed_time, "ns") != np.datetime64(expected_time, "ns")
            or values.shape != (len(expected_latitudes), len(expected_longitudes))
            or not np.allclose(latitudes, expected_latitudes, rtol=0.0, atol=1e-8)
            or not np.allclose(longitudes, expected_longitudes, rtol=0.0, atol=1e-8)
            or not bool(np.isfinite(values).all())):
        raise DataSourceError("T4E.39 shard does not match the declared finite regular grid")
    return xr.DataArray(
        values[None, :, :], dims=("time", "latitude", "longitude"), name=field,
        coords={"time": [expected_time], "latitude": latitudes, "longitude": longitudes})


def join_field_segments(parent: Any, complement: Any, field: str,
                        parent_spec: Any, complement_spec: Any,
                        timestamp: str) -> Tuple[Any, Dict[str, Any]]:
    import numpy as np
    import xarray as xr

    left = _normalise_frame(parent, field, parent_spec, timestamp)
    right = _normalise_frame(complement, field, complement_spec, timestamp)
    if not np.array_equal(left.latitude.values, right.latitude.values):
        raise DataSourceError("T4E.39 segments disagree on latitude coordinates")
    wrapped = np.asarray(right.longitude.values, dtype=np.float64) + 360.0
    if not math.isclose(float(left.longitude.values[-1]), float(wrapped[0]),
                        rel_tol=0.0, abs_tol=1e-8):
        raise DataSourceError("T4E.39 segments do not meet at the 180-degree seam")
    left_values = np.asarray(left.values[0])
    right_values = np.asarray(right.values[0])
    left_step, right_step = encoding_step(left_values), encoding_step(right_values)
    if left_step is None or right_step is None:
        raise DataSourceError("T4E.39 source encoding tolerance is not measurable")
    tolerance = max(float(left_step), float(right_step))
    difference = float(np.max(np.abs(left_values[:, -1] - right_values[:, 0])))
    if not math.isfinite(difference) or difference > tolerance:
        raise DataSourceError(
            "T4E.39 %s seam mismatch at %s: %.12g exceeds %.12g"
            % (field, timestamp, difference, tolerance))
    right = right.assign_coords(longitude=wrapped).isel(longitude=slice(1, None))
    joined = xr.concat([left, right], dim="longitude").transpose(
        "time", "latitude", "longitude")
    if (joined.shape != (1, 161, 321)
            or not np.allclose(np.diff(joined.longitude.values), parent_spec.grid_degrees,
                               rtol=0.0, atol=1e-8)):
        raise DataSourceError("T4E.39 joined field geometry is not 1x161x321")
    return joined, {
        "field": field, "timestamp": timestamp, "coordinate": 180.0,
        "retained": "parent 180E column", "complement_duplicate_removed": True,
        "maximum_absolute_difference": difference,
        "source_encoding_tolerance": tolerance, "passed": True,
    }


def materialise_holdout_record(
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    census_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
    *, download_root: Union[str, os.PathLike[str]] = DEFAULT_DOWNLOAD_ROOT,
    record_root: Union[str, os.PathLike[str]] = DEFAULT_RECORD_ROOT,
) -> Dict[str, Any]:
    import numpy as np
    import xarray as xr
    import zarr

    declaration_path = Path(declaration_path)
    declaration = load_holdout_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    plan = plan_holdout_fields(declaration_path, census_path)
    items = {(item["field"], item["segment"]): item for item in plan["requests"]}
    expected = {(field, segment) for field in ("msl", "vo")
                for segment in ("parent", "complement")}
    if set(items) != expected:
        raise DataSourceError("T4E.39 requires exactly four field/segment requests")
    root = Path(download_root)
    specs, shards, source_hashes = {}, {}, {}
    for key, item in items.items():
        directory = root / key[0] / key[1]
        specs[key], _ = _load_complete(directory, item)
        shards[key] = _plan_shards(specs[key])
        source_hashes[key] = {
            shard.filename: _file_sha256(directory / shard.filename)
            for shard in shards[key]}
    frames, seams = [], []
    for index, timestamp in enumerate(plan["timestamps"]):
        fields = {}
        for field in ("msl", "vo"):
            left_path = root / field / "parent" / shards[(field, "parent")][index].filename
            right_path = root / field / "complement" / shards[(field, "complement")][index].filename
            with xr.open_dataset(left_path) as left, xr.open_dataset(right_path) as right:
                joined, seam = join_field_segments(
                    left, right, field, specs[(field, "parent")],
                    specs[(field, "complement")], timestamp)
                fields[field] = joined.load()
                seams.append(seam)
        frames.append(xr.Dataset(fields))
    record = xr.concat(frames, dim="time", data_vars="minimal", coords="minimal",
                       compat="equals", join="exact")
    expected_shape = {"time": len(plan["timestamps"]), "latitude": 161, "longitude": 321}
    if ({str(k): int(v) for k, v in record.sizes.items()} != expected_shape
            or set(record.data_vars) != {"msl", "vo"}
            or not bool(np.isfinite(record.msl.values).all())
            or not bool(np.isfinite(record.vo.values).all())):
        raise DataSourceError("T4E.39 materialised record has unexpected shape or values")
    for key, hashes in source_hashes.items():
        directory = root / key[0] / key[1]
        if any(_file_sha256(directory / name) != digest for name, digest in hashes.items()):
            raise DataSourceError("a T4E.39 shard changed during materialisation")

    destination = Path(record_root)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".t4e39.", dir=str(destination)))
    temporary_store = temporary_root / "record.zarr"
    try:
        record.chunk({"time": 1, "latitude": 161, "longitude": 321}).to_zarr(
            temporary_store, mode="w", consolidated=False,
            encoding={field: {"chunks": (1, 161, 321)} for field in ("msl", "vo")})
        zarr.consolidate_metadata(str(temporary_store))
        with xr.open_zarr(temporary_store, consolidated=True) as opened:
            record_sha256 = streaming_content_sha256(opened, time_block=1)
            shape = {str(k): int(v) for k, v in opened.sizes.items()}
        record_path = destination / (record_sha256 + ".zarr")
        if record_path.exists():
            with xr.open_zarr(record_path, consolidated=True) as existing:
                if streaming_content_sha256(existing, time_block=1) != record_sha256:
                    raise DataSourceError("existing T4E.39 content-addressed record drifted")
            shutil.rmtree(temporary_store)
        else:
            os.rename(temporary_store, record_path)
        receipt = {
            "schema": HOLDOUT_FIELD_RECORD_SCHEMA,
            "record_sha256": record_sha256,
            "record_path": str(record_path).replace("\\", "/"), "shape": shape,
            "declaration": str(declaration_path).replace("\\", "/"),
            "declaration_sha256": plan["declaration_sha256"],
            "census": str(Path(census_path)).replace("\\", "/"),
            "census_receipt_sha256": plan["census_receipt_sha256"],
            "adoption": adoption,
            "sources": {
                "%s_%s" % key: {
                    "directory": str(root / key[0] / key[1]).replace("\\", "/"),
                    "acquisition_sha256": _file_sha256(
                        root / key[0] / key[1] / "acquisition.json"),
                    "request_sha256": specs[key].request_sha256(),
                    "shards": source_hashes[key], "preserved_byte_for_byte": True,
                } for key in sorted(expected)
            },
            "seam": {"checks": seams, "all_passed": True},
            "network_used": False, "claim_boundary": declaration["claim_boundary"],
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        receipt_path = destination / (receipt["receipt_sha256"] + ".json")
        payload = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if receipt_path.exists():
            if receipt_path.read_bytes() != payload:
                raise DataSourceError("existing T4E.39 receipt bytes differ")
        else:
            publish_new_bytes(receipt_path, payload, "T4E.39 exact-field record receipt")
        return receipt
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def holdout_decision(rows: Sequence[Mapping[str, Any]], minimum: int = 10) -> Dict[str, Any]:
    labels = (
        "VERTICAL_QUANTITY_SEPARATION_CANDIDATE",
        "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE", "EXACT_TIE", "REFUSED")
    counts = {label: 0 for label in labels}
    for row in rows:
        label = str(row.get("classification"))
        if label not in counts:
            raise DataSourceError("T4E.39 row carries an undeclared classification")
        counts[label] += 1
    total = len(rows)
    needed = math.floor(total / 2) + 1
    if total < minimum:
        outcome = "INSUFFICIENT_HOLDOUT_POPULATION"
    elif counts[labels[0]] >= needed:
        outcome = "HOLDOUT_SUPPORTS_VERTICAL_QUANTITY_CANDIDATE"
    elif counts[labels[1]] >= needed:
        outcome = "HOLDOUT_FAVOURS_CATALOGUE_REANALYSIS_ALIGNMENT"
    else:
        outcome = "HOLDOUT_MIXED"
    return {
        "counts": counts, "population_denominator": total,
        "strict_majority_needed": needed, "minimum_selected_storms": minimum,
        "outcome": outcome, "is_acceptance_verdict": False,
    }


def evaluate_holdout(
    record_receipt_path: Union[str, os.PathLike[str]],
    declaration_path: Union[str, os.PathLike[str]] = DEFAULT_DECLARATION,
    census_path: Union[str, os.PathLike[str]] = DEFAULT_CENSUS,
    *, output_path: Union[str, os.PathLike[str]] = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    import numpy as np
    import xarray as xr

    declaration_path = Path(declaration_path)
    declaration = load_holdout_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    census = load_holdout_census(census_path, declaration_path)
    receipt_path = Path(record_receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("T4E.39 field receipt is unreadable: %s" % exc) from exc
    if (receipt.get("schema") != HOLDOUT_FIELD_RECORD_SCHEMA
            or _verified_receipt(receipt, "T4E.39 field record") != receipt.get("receipt_sha256")
            or receipt.get("declaration_sha256") != _file_sha256(declaration_path)
            or receipt.get("census_receipt_sha256") != census["receipt_sha256"]):
        raise DataSourceError("T4E.39 field record does not bind the declaration and census")
    record_path = Path(receipt["record_path"])
    measured = []
    with xr.open_zarr(record_path, consolidated=True) as record:
        if streaming_content_sha256(record, time_block=1) != receipt["record_sha256"]:
            raise DataSourceError("T4E.39 field record content drifted")
        latitudes = np.asarray(record.latitude.values, dtype=np.float64)
        longitudes = np.asarray(record.longitude.values, dtype=np.float64)
        for source in census["selected_rows"]:
            stamp = np.datetime64(str(source["time"]).replace(" ", "T"))
            try:
                pressure = np.asarray(record.msl.sel(time=stamp).values, dtype=np.float64)
                vorticity = -np.asarray(record.vo.sel(time=stamp).values, dtype=np.float64)
            except (KeyError, ValueError) as exc:
                raise DataSourceError("a census timestamp is absent from the field record") from exc
            catalogue = (float(source["lat"]), float(source["lon"]))
            pressure_walk = basin_walk(
                pressure, latitudes, longitudes, catalogue, direction="minimum")
            vorticity_walk = basin_walk(
                vorticity, latitudes, longitudes, catalogue, direction="maximum")
            if (pressure_walk["status"] != "CENTRE_FOUND"
                    or vorticity_walk["status"] != "CENTRE_FOUND"):
                classification = "REFUSED"
                pressure_distance = vorticity_distance = centres_distance = None
            else:
                p = pressure_walk["centre"]
                v = vorticity_walk["centre"]
                p_position, v_position = (p["lat"], p["lon"]), (v["lat"], v["lon"])
                pressure_distance = great_circle_km(catalogue, p_position)
                vorticity_distance = great_circle_km(catalogue, v_position)
                centres_distance = great_circle_km(p_position, v_position)
                classification = (
                    "VERTICAL_QUANTITY_SEPARATION_CANDIDATE"
                    if pressure_distance < vorticity_distance else
                    "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE"
                    if pressure_distance > vorticity_distance else "EXACT_TIE")
            measured.append({
                "storm": source["storm"], "sid": source["sid"], "time": source["time"],
                "catalogue_position": {"lat": catalogue[0], "lon": catalogue[1]},
                "catalogue_radius_km": source["catalogue_radius_km"],
                "mslp_basin": pressure_walk, "negated_vorticity_basin": vorticity_walk,
                "catalogue_to_mslp_km": pressure_distance,
                "catalogue_to_vorticity_km": vorticity_distance,
                "mslp_to_vorticity_km": centres_distance,
                "classification": classification,
            })
    decision = holdout_decision(measured, int(census["minimum_selected_storms"]))
    result = {
        "schema": HOLDOUT_EVALUATION_SCHEMA,
        "declaration": str(declaration_path).replace("\\", "/"),
        "declaration_sha256": _file_sha256(declaration_path), "adoption": adoption,
        "sources": {
            "census": str(Path(census_path)).replace("\\", "/"),
            "census_receipt_sha256": census["receipt_sha256"],
            "field_record_receipt": str(receipt_path).replace("\\", "/"),
            "field_record_receipt_sha256": receipt["receipt_sha256"],
        },
        "population": {"rows": len(measured), "catalogue_census_rerun": False},
        "method": {
            "catalogue_seeded": True, "neighbourhood": "3x3",
            "tie_break": "great_circle_to_catalogue_then_lower_row_then_lower_column",
            "mslp_direction": "minimum", "vorticity_negated": True,
            "vorticity_direction_after_negation": "maximum", "search_radius_used": False,
        },
        "rows": measured, "decision": decision, "OUTCOME": decision["outcome"],
        "VERDICT": "NOT_AN_ACCEPTANCE", "network_used": False,
        "claim_boundary": declaration["claim_boundary"],
    }
    result["receipt_sha256"] = _sha256(result)
    try:
        publish_new_bytes(
            output_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            "T4E.39 holdout evaluation")
    except FileExistsError as exc:
        raise DataSourceError(str(exc), path=str(output_path)) from exc
    return result
