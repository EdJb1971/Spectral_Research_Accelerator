"""Guarded, seam-safe materialisation for the T4E.36 wrapped ERA5 record.

The complementary CDS request is a second acquisition.  It never edits the T4E.18
parent shards.  This module admits it only under a current maintainer adoption, requires
an additional call-site network authorisation before delegating to CDS, and joins the two
segments through a checked 180-degree seam.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.core.adoption import adoption_state, digest_of
from src.core.errors import DataSourceError, InvalidParameterError
from src.core.publication import publish_new_bytes
from src.data_layer.cds_source import (
    CDSRegionalRequest,
    _expected_times,
    _file_hash,
    _normalise_downloaded_dataset,
    _shard_expected_times,
    _validate_netcdf_shard,
    acquire_cds_shards,
    plan_monthly_shards,
)
from src.data_layer.era5_overlap import encoding_step
from src.data_layer.zarr_source import streaming_content_sha256


WRAPPED_DECLARATION_SCHEMA = "t4e36-wrapped-acquisition-declaration/v1"
WRAPPED_RECORD_SCHEMA = "t4e36-wrapped-record/v1"
DEFAULT_DECLARATION = Path(
    "data/identity_calibration/t4e36-wrapped-acquisition-declaration.json")
DEFAULT_PARENT_DIR = Path("data/cds_downloads/t4e18_vorticity")
DEFAULT_COMPLEMENT_DIR = Path("data/cds_downloads/t4e36_eastward_vorticity")
DEFAULT_RECORD_DIR = Path("data/wrapped_records/t4e36")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def load_wrapped_declaration(path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    source = Path(path)
    try:
        body = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "T4E.36 declaration is unreadable: %s" % exc, path=str(source)) from exc
    if body.get("schema") != WRAPPED_DECLARATION_SCHEMA:
        raise DataSourceError(
            "unsupported wrapped-acquisition declaration schema", path=str(source))
    if body.get("task") != "T4E.36":
        raise DataSourceError("wrapped-acquisition declaration is not T4E.36", path=str(source))
    request = body.get("complementary_request")
    geometry = body.get("wrapped_geometry")
    if not isinstance(request, Mapping) or not isinstance(geometry, Mapping):
        raise DataSourceError("T4E.36 declaration lacks request or wrapped geometry")
    frozen = body.get("frozen_extraction", {})
    gate = body.get("gate_declared_before_acquisition", {})
    if (frozen.get("n_surrogates") != 99 or frozen.get("seed") != 1234
            or frozen.get("primary_path") != "raw_field: representation 'identity'"
            or "wavelet 'db2', level 3" not in str(frozen.get("diagnostic_path", ""))):
        raise DataSourceError("T4E.36 declaration does not carry the frozen extraction contract")
    if ("raw_field" not in str(gate.get("primary_scientific_path", ""))
            or "9 of the fixed 18" not in str(gate.get("condition_1", ""))
            or "9 of the fixed 18" not in str(gate.get("condition_2", ""))
            or "SWT-plane counts cannot rescue" not in str(gate.get("decision", ""))):
        raise DataSourceError("T4E.36 declaration does not carry the frozen primary gate")
    return body


def request_from_declaration(declaration: Mapping[str, Any]) -> CDSRegionalRequest:
    declared = declaration["complementary_request"]
    request = dict(declared)
    for derived in ("expected_monthly_shards", "expected_frames", "expected_segment_shape"):
        request.pop(derived, None)
    for name in ("variables", "hours_utc", "pressure_levels"):
        request[name] = tuple(request[name])
    spec = CDSRegionalRequest(**request)
    shards = plan_monthly_shards(spec)
    frames = len(_expected_times(spec))
    shape = [
        int(round((spec.lat_max - spec.lat_min) / spec.grid_degrees)) + 1,
        int(round((spec.lon_max - spec.lon_min) / spec.grid_degrees)) + 1,
    ]
    if int(declared["expected_monthly_shards"]) != len(shards):
        raise DataSourceError("declared monthly-shard count disagrees with the frozen request")
    if int(declared["expected_frames"]) != frames:
        raise DataSourceError("declared frame count disagrees with the frozen request")
    if list(declared["expected_segment_shape"]) != shape:
        raise DataSourceError("declared segment shape disagrees with the frozen request")
    geometry = declaration["wrapped_geometry"]
    parent_bounds = [float(value) for value in geometry["parent_longitude"]]
    wrapped_bounds = [spec.lon_min + 360.0, spec.lon_max + 360.0]
    if ([float(value) for value in geometry["complement_longitude_as_requested"]]
            != [spec.lon_min, spec.lon_max]
            or [float(value) for value in geometry["complement_longitude_after_wrapping"]]
            != wrapped_bounds
            or not math.isclose(parent_bounds[1], wrapped_bounds[0],
                                rel_tol=0.0, abs_tol=1e-9)
            or [float(value) for value in geometry["canonical_joined_longitude"]]
            != [parent_bounds[0], wrapped_bounds[1]]):
        raise DataSourceError("declared wrapped geometry disagrees with the two source segments")
    joined_points = int(round(
        (float(geometry["canonical_joined_longitude"][1])
         - float(geometry["canonical_joined_longitude"][0])) / spec.grid_degrees)) + 1
    if int(geometry["expected_joined_longitude_points"]) != joined_points:
        raise DataSourceError("declared joined shape disagrees with the wrapped geometry")
    return spec


def require_current_adoption(
    declaration_path: Union[str, os.PathLike[str]],
) -> Dict[str, Any]:
    """Return the adoption state only when its digest reaches the current declaration."""
    source = Path(declaration_path)
    state = adoption_state(source.parent, source.name)
    if not state.get("adopted"):
        raise DataSourceError(
            "T4E.36 acquisition is refused: the declaration has not been adopted by a "
            "maintainer", declaration=source.name, adoption_file=state["adoption_file"])
    if not state.get("signature_still_reaches_the_declaration"):
        raise DataSourceError(
            "T4E.36 acquisition is refused: the adoption does not bind the current declaration",
            declaration=source.name, adoption_file=state["adoption_file"])
    return state


def acquire_wrapped_complement(
    declaration_path: Union[str, os.PathLike[str]],
    download_dir: Union[str, os.PathLike[str]],
    *,
    explicit_network_authorisation: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Acquire the complement after two distinct human-controlled gates.

    Adoption and network consent are intentionally independent.  Even when the general
    network environment variable is enabled, this experiment does not submit a request unless
    its caller also supplies ``explicit_network_authorisation=True``.
    """
    declaration = load_wrapped_declaration(declaration_path)
    adopted = require_current_adoption(declaration_path)
    if explicit_network_authorisation is not True:
        raise DataSourceError(
            "T4E.36 acquisition is network-refused: pass the experiment-specific explicit "
            "network authorisation after reviewing the adopted declaration")
    spec = request_from_declaration(declaration)
    state = acquire_cds_shards(
        spec, download_dir, client=client, allow_network=None)
    return {
        "schema": "t4e36-complement-acquisition/v1",
        "declaration": str(Path(declaration_path)).replace("\\", "/"),
        "declaration_sha256": digest_of(Path(declaration_path)),
        "adoption": adopted,
        "acquisition": state,
    }


def _load_acquisition(directory: Path, expected: CDSRegionalRequest) -> Dict[str, Any]:
    state_path = directory / "acquisition.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            "wrapped record requires a readable acquisition state: %s" % exc,
            path=str(state_path)) from exc
    if state.get("request", {}).get("request_sha256") != expected.request_sha256():
        raise DataSourceError(
            "acquisition state belongs to a different request", path=str(state_path))
    shards = plan_monthly_shards(expected)
    completed = state.get("completed_shards", {})
    if len(completed) != len(shards):
        raise DataSourceError(
            "acquisition is incomplete: %d of %d monthly shards are recorded"
            % (len(completed), len(shards)), path=str(state_path))
    for shard in shards:
        recorded = completed.get(shard.filename)
        if not isinstance(recorded, Mapping):
            raise DataSourceError("acquisition is missing planned shard %s" % shard.filename)
        observed = _validate_netcdf_shard(directory / shard.filename)
        if (recorded.get("sha256") != observed["sha256"]
                or recorded.get("request_sha256") != shard.request_sha256):
            raise DataSourceError(
                "acquisition shard failed content/request integrity verification",
                path=str(directory / shard.filename))
    return state


def _parent_request(complement: CDSRegionalRequest,
                    declaration: Mapping[str, Any]) -> CDSRegionalRequest:
    west, east = declaration["wrapped_geometry"]["parent_longitude"]
    return CDSRegionalRequest(
        variables=complement.variables, date_start=complement.date_start,
        date_end=complement.date_end, hours_utc=complement.hours_utc,
        lat_min=complement.lat_min, lat_max=complement.lat_max,
        lon_min=float(west), lon_max=float(east),
        pressure_levels=complement.pressure_levels,
        grid_degrees=complement.grid_degrees,
        dataset=complement.dataset, data_format=complement.data_format,
        # The parent was acquired with four analysis levels. This field participates in its
        # provenance identity even though it does not alter CDS response bytes.
        n_levels_analysis=4)


def join_wrapped_segments(
    parent: Any,
    complement: Any,
    parent_spec: CDSRegionalRequest,
    complement_spec: CDSRegionalRequest,
    *,
    expected_times: Optional[Sequence[Any]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Validate, wrap, seam-check and concatenate two already-open xarray datasets."""
    import numpy as np
    import xarray as xr

    left = _normalise_downloaded_dataset(
        parent, parent_spec, expected_times=expected_times)
    right = _normalise_downloaded_dataset(
        complement, complement_spec, expected_times=expected_times)
    for coordinate in ("time", "level", "latitude"):
        if not np.array_equal(left[coordinate].values, right[coordinate].values):
            raise DataSourceError(
                "wrapped segments disagree on %s coordinates" % coordinate)

    wrapped = np.asarray(right.longitude.values, dtype=np.float64) + 360.0
    if not math.isclose(float(left.longitude.values[-1]), float(wrapped[0]),
                        rel_tol=0.0, abs_tol=1e-8):
        raise DataSourceError("wrapped segments do not meet at the declared 180-degree seam")

    tolerances: Dict[str, float] = {}
    seam_max: Dict[str, float] = {}
    for name in parent_spec.variables:
        left_values = np.asarray(left[name].values)
        right_values = np.asarray(right[name].values)
        maximum_tolerance = 0.0
        maximum_difference = 0.0
        # GRIB packing chooses a scale and reference per message. Measure the lattice on each
        # time/level field; pooling a month can erase the very lattice being measured.
        for time_index in range(left_values.shape[0]):
            for level_index in range(left_values.shape[1]):
                left_step = encoding_step(left_values[time_index, level_index])
                right_step = encoding_step(right_values[time_index, level_index])
                if left_step is None or right_step is None:
                    raise DataSourceError(
                        "source encoding tolerance is not measurable for %s at time index %d, "
                        "level index %d; seam agreement cannot be judged without inventing a "
                        "tolerance" % (name, time_index, level_index))
                tolerance = max(float(left_step), float(right_step))
                difference = float(np.max(np.abs(
                    left_values[time_index, level_index, :, -1]
                    - right_values[time_index, level_index, :, 0])))
                if not math.isfinite(difference) or difference > tolerance:
                    raise DataSourceError(
                        "wrapped 180-degree seam mismatch for %s at time index %d, level index "
                        "%d: maximum difference %.12g exceeds source encoding tolerance %.12g"
                        % (name, time_index, level_index, difference, tolerance))
                maximum_tolerance = max(maximum_tolerance, tolerance)
                maximum_difference = max(maximum_difference, difference)
        tolerances[name] = maximum_tolerance
        seam_max[name] = maximum_difference

    right = right.assign_coords(longitude=wrapped).isel(longitude=slice(1, None))
    joined = xr.concat(
        [left, right], dim="longitude", data_vars="minimal", coords="minimal",
        compat="equals", join="exact")
    longitude = np.asarray(joined.longitude.values, dtype=np.float64)
    expected_points = int(round(
        (complement_spec.lon_max + 360.0 - parent_spec.lon_min)
        / complement_spec.grid_degrees)) + 1
    if (len(longitude) != expected_points
            or not np.allclose(np.diff(longitude), complement_spec.grid_degrees,
                               rtol=0.0, atol=1e-8)):
        raise DataSourceError(
            "seam-deduplicated longitude geometry does not match the declared regular union")
    return joined.transpose("time", "level", "latitude", "longitude"), {
        "coordinate": 180.0,
        "retained": "parent 180E column",
        "complement_duplicate_removed": True,
        "maximum_absolute_difference": seam_max,
        "source_encoding_tolerance": tolerances,
        "passed": True,
    }


def materialise_wrapped_record(
    declaration_path: Union[str, os.PathLike[str]],
    *,
    parent_dir: Union[str, os.PathLike[str]] = DEFAULT_PARENT_DIR,
    complement_dir: Union[str, os.PathLike[str]] = DEFAULT_COMPLEMENT_DIR,
    record_dir: Union[str, os.PathLike[str]] = DEFAULT_RECORD_DIR,
    time_chunk: int = 32,
) -> Dict[str, Any]:
    """Stream all checked month pairs into one immutable content-addressed Zarr record."""
    if isinstance(time_chunk, bool) or int(time_chunk) != time_chunk or time_chunk < 1:
        raise InvalidParameterError("time_chunk", time_chunk, "a positive integer")
    declaration_path = Path(declaration_path)
    declaration = load_wrapped_declaration(declaration_path)
    adoption = require_current_adoption(declaration_path)
    complement_spec = request_from_declaration(declaration)
    parent_spec = _parent_request(complement_spec, declaration)
    parent_root, complement_root = Path(parent_dir), Path(complement_dir)
    _load_acquisition(parent_root, parent_spec)
    complement_state = _load_acquisition(complement_root, complement_spec)
    parent_shards = plan_monthly_shards(parent_spec)
    complement_shards = plan_monthly_shards(complement_spec)

    # Bind and later recheck every parent byte. The parent is input only and is never opened for
    # writing; the second check makes that invariant executable rather than documentary.
    parent_hashes = {
        shard.filename: _file_hash(parent_root / shard.filename) for shard in parent_shards}

    import numpy as np
    import xarray as xr
    import zarr

    destination = Path(record_dir)
    destination.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".t4e36.", dir=str(destination)))
    temporary_store = temporary_root / "record.zarr"
    first = True
    frames = 0
    seam_reports = []
    try:
        for left_shard, right_shard in zip(parent_shards, complement_shards):
            expected = _shard_expected_times(left_shard, parent_spec)
            with xr.open_dataset(parent_root / left_shard.filename) as parent, \
                    xr.open_dataset(complement_root / right_shard.filename) as complement:
                joined, seam = join_wrapped_segments(
                    parent, complement, parent_spec, complement_spec,
                    expected_times=expected)
                seam_reports.append({
                    "year": left_shard.year, "month": left_shard.month, **seam})
                for start in range(0, int(joined.sizes["time"]), int(time_chunk)):
                    block = joined.isel(time=slice(start, start + time_chunk)).load()
                    if not all(bool(np.isfinite(block[name].values).all())
                               for name in complement_spec.variables):
                        raise DataSourceError("wrapped record contains non-finite values")
                    chunks = {
                        "time": min(time_chunk, int(block.sizes["time"])),
                        "level": int(block.sizes["level"]),
                        "latitude": int(block.sizes["latitude"]),
                        "longitude": int(block.sizes["longitude"]),
                    }
                    block = block.chunk(chunks)
                    encoding = None
                    if first:
                        encoding = {name: {"chunks": tuple(
                            chunks[dim] for dim in block[name].dims)}
                                    for name in block.data_vars}
                    block.to_zarr(
                        temporary_store, mode="w" if first else "a",
                        append_dim=None if first else "time", consolidated=False,
                        encoding=encoding)
                    first = False
                    frames += int(block.sizes["time"])
        expected_frames = len(_expected_times(complement_spec))
        if first or frames != expected_frames:
            raise DataSourceError(
                "wrapped record contains %d frames, expected %d" % (frames, expected_frames))
        zarr.consolidate_metadata(str(temporary_store))
        with xr.open_zarr(temporary_store, consolidated=True) as opened:
            content_sha256 = streaming_content_sha256(opened, time_block=time_chunk)
            shape = {str(k): int(v) for k, v in opened.sizes.items()}
        if any(_file_hash(parent_root / name) != digest
               for name, digest in parent_hashes.items()):
            raise DataSourceError("a T4E.18 parent shard changed during materialisation")
        record_path = destination / (content_sha256 + ".zarr")
        if record_path.exists():
            with xr.open_zarr(record_path, consolidated=True) as existing:
                if streaming_content_sha256(existing, time_block=time_chunk) != content_sha256:
                    raise DataSourceError(
                        "content-addressed wrapped-record path contains different content",
                        path=str(record_path))
            shutil.rmtree(temporary_store)
        else:
            os.rename(temporary_store, record_path)

        receipt = {
            "schema": WRAPPED_RECORD_SCHEMA,
            "record_sha256": content_sha256,
            "record_path": str(record_path).replace("\\", "/"),
            "shape": shape,
            "declaration": str(declaration_path).replace("\\", "/"),
            "declaration_sha256": digest_of(declaration_path),
            "adoption": adoption,
            "parent": {
                "directory": str(parent_root).replace("\\", "/"),
                "request_sha256": parent_spec.request_sha256(),
                "shards": parent_hashes,
                "preserved_byte_for_byte": True,
            },
            "complement": {
                "directory": str(complement_root).replace("\\", "/"),
                "request_sha256": complement_spec.request_sha256(),
                "shards": {name: value["sha256"] for name, value in
                           complement_state["completed_shards"].items()},
            },
            "seam": {"monthly_checks": seam_reports, "all_passed": True},
            "network_used": False,
            "claim_boundary": declaration["claim_boundary"],
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        receipt_path = destination / (receipt["receipt_sha256"] + ".json")
        payload = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if receipt_path.exists():
            if receipt_path.read_bytes() != payload:
                raise DataSourceError(
                    "content-addressed wrapped-record receipt contains different bytes",
                    path=str(receipt_path))
        else:
            publish_new_bytes(
                receipt_path, payload, "T4E.36 wrapped-record receipt")
        return receipt
    except Exception:
        if temporary_store.exists():
            shutil.rmtree(temporary_store)
        raise
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
