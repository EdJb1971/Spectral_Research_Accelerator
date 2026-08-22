"""Content-bound, bounded verification of CDS values against WeatherBench ERA5.

The direct CDS route solves the regional-access problem, but a successful download is not
evidence that coordinate normalisation and variable decoding preserved ERA5 values.  This
module compares a deliberately small cached overlap obtained through WeatherBench 2, writes an
immutable receipt, and binds that receipt to the CDS cache manifest.  Neither input may fall
back to the network during verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.regional_forecast import CANONICAL_VARIABLES, VARIABLE_ALIASES
from src.data_layer.zarr_source import (
    CATALOGUE,
    CropSpec,
    manifest_path,
    open_cached_lazy,
)

OVERLAP_SCHEMA = "era5-independent-overlap/v1"
DEFAULT_ATOL: Mapping[str, float] = {
    "t": 1e-4,
    "q": 1e-9,
    "u": 1e-5,
    "v": 1e-5,
    "z": 1e-3,
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def overlap_receipt_path(spec: CropSpec, cache_dir: Optional[str] = None) -> Path:
    return Path(manifest_path(spec, cache_dir)).with_suffix(".overlap.json")


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError("refusing to overwrite ERA5 overlap receipt at %s" % path) from None
        except OSError as exc:
            raise DataSourceError(
                "cannot atomically publish ERA5 overlap receipt at %s: %s" % (path, exc)) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_overlap_receipt(path: Union[str, os.PathLike[str]]) -> Dict[str, Any]:
    try:
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("ERA5 overlap receipt is unreadable: %s" % exc) from exc
    if not isinstance(envelope, Mapping) or set(envelope) != {"receipt", "receipt_sha256"}:
        raise DataSourceError("ERA5 overlap receipt has unknown or missing envelope fields")
    receipt = envelope["receipt"]
    _validate_receipt_structure(receipt)
    if envelope["receipt_sha256"] != _sha256(receipt):
        raise DataSourceError("ERA5 overlap receipt hash does not authenticate its contents")
    return {**dict(receipt), "receipt_sha256": envelope["receipt_sha256"]}


def _validate_receipt_structure(receipt: Any) -> None:
    required = {
        "schema", "primary", "independent", "level_hpa", "coordinates_exact",
        "coordinate_sha256", "shape", "variables", "passed", "bounded_execution",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != required \
            or receipt.get("schema") != OVERLAP_SCHEMA:
        raise DataSourceError("ERA5 overlap receipt has an unsupported or incomplete schema")
    if not isinstance(receipt.get("primary"), Mapping) \
            or not isinstance(receipt.get("independent"), Mapping) \
            or not isinstance(receipt.get("variables"), Mapping) \
            or not isinstance(receipt.get("bounded_execution"), Mapping):
        raise DataSourceError("ERA5 overlap receipt contains malformed evidence blocks")
    if not isinstance(receipt.get("shape"), list) or len(receipt["shape"]) != 3 \
            or any(isinstance(value, bool) or int(value) != value or value < 2
                   for value in receipt["shape"]):
        raise DataSourceError("ERA5 overlap receipt has an invalid overlap shape")
    if not isinstance(receipt.get("coordinate_sha256"), str) \
            or len(receipt["coordinate_sha256"]) != 64:
        raise DataSourceError("ERA5 overlap receipt has an invalid coordinate fingerprint")


def _resolved_array(dataset: Any, variable: str, level_hpa: float) -> Tuple[Any, str]:
    matches = [name for name in VARIABLE_ALIASES[variable] if name in dataset.data_vars]
    if len(matches) != 1:
        raise DataSourceError(
            "ERA5 overlap requires exactly one alias for %s; found %s" % (variable, matches))
    array = dataset[matches[0]]
    if "level" in array.dims:
        levels = np.asarray(array.level.values, dtype=np.float64)
        indices = np.flatnonzero(levels == float(level_hpa))
        if len(indices) != 1:
            raise DataSourceError(
                "ERA5 overlap level %.6g is not present exactly once" % float(level_hpa))
        array = array.isel(level=int(indices[0]))
    elif level_hpa is not None:
        raise DataSourceError("ERA5 overlap variable %s has no pressure-level axis" % variable)
    lat_name = "latitude" if "latitude" in array.dims else "lat"
    lon_name = "longitude" if "longitude" in array.dims else "lon"
    if set(array.dims) != {"time", lat_name, lon_name}:
        raise DataSourceError(
            "ERA5 overlap variable %s has unsupported dimensions %s" % (variable, list(array.dims)))
    return array.transpose("time", lat_name, lon_name), matches[0]


def _unit_token(value: Any) -> str:
    return "".join(character for character in str(value or "").lower()
                   if character.isalnum())


def _units_compatible(variable: str, primary: Any, independent: Any) -> bool:
    left, right = _unit_token(primary), _unit_token(independent)
    if not left or not right:
        return False
    accepted = {
        "t": {"k", "kelvin"},
        "q": {"kgkg1", "kgkg01", "1", "dimensionless"},
        "u": {"ms1", "ms01", "mss1"},
        "v": {"ms1", "ms01", "mss1"},
        "z": {"m2s2", "m2s02"},
    }
    return left == right or (left in accepted[variable] and right in accepted[variable])


def _coordinate_hash(times: np.ndarray, latitude: np.ndarray, longitude: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(times.astype("datetime64[ns]").astype("int64")).tobytes())
    digest.update(np.ascontiguousarray(latitude, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(longitude, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _attach_receipt(primary: CropSpec, cache_dir: Optional[str], receipt: Mapping[str, Any],
                    receipt_sha256: str) -> None:
    path = Path(manifest_path(primary, cache_dir))
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataSourceError("CDS cache manifest is unreadable: %s" % exc) from exc
    existing = manifest.get("independent_overlap_receipt_sha256")
    if existing is not None and existing != receipt_sha256:
        raise DataSourceError("CDS cache is already bound to a different overlap receipt")
    if manifest.get("content_key") != receipt["primary"]["content_key"] \
            or manifest.get("content_hash") != receipt["primary"]["content_hash"]:
        raise DataSourceError("CDS cache identity changed before overlap receipt publication")
    manifest["independent_overlap_check"] = "PASS" if receipt["passed"] else "FAIL"
    manifest["independent_overlap_receipt_sha256"] = receipt_sha256
    manifest["independent_overlap_receipt"] = dict(receipt)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(manifest) + b"\n")
    os.replace(temporary, path)


def validate_overlap_evidence(
    manifest: Mapping[str, Any], *, variable: Optional[str] = None,
    level_hpa: Optional[float] = None,
) -> Mapping[str, Any]:
    """Validate that a manifest's PASS is backed by content-bound independent evidence."""
    receipt = manifest.get("independent_overlap_receipt")
    fingerprint = manifest.get("independent_overlap_receipt_sha256")
    if manifest.get("independent_overlap_check") != "PASS" or not isinstance(receipt, Mapping):
        raise DataSourceError("independent WeatherBench overlap evidence is not a recorded PASS")
    _validate_receipt_structure(receipt)
    if fingerprint != _sha256(receipt):
        raise DataSourceError("independent WeatherBench overlap evidence is missing or tampered")
    if not receipt.get("passed") or not receipt.get("coordinates_exact"):
        raise DataSourceError("independent WeatherBench overlap receipt did not pass")
    primary = receipt.get("primary") or {}
    if primary.get("content_key") != manifest.get("content_key") \
            or primary.get("content_hash") != manifest.get("content_hash"):
        raise DataSourceError("independent overlap receipt belongs to different CDS content")
    request_hash = (manifest.get("acquisition_request_sha256")
                    or (((manifest.get("acquisition") or {}).get("request") or {})
                        .get("request_sha256")))
    if primary.get("request_sha256") != request_hash:
        raise DataSourceError("independent overlap receipt belongs to a different CDS request")
    if (receipt.get("independent") or {}).get("store") not in CATALOGUE:
        raise DataSourceError("independent overlap receipt is not from a catalogued WeatherBench route")
    if variable is not None and variable not in receipt.get("variables", {}):
        raise DataSourceError("independent overlap receipt does not cover variable %s" % variable)
    if level_hpa is not None and float(receipt.get("level_hpa")) != float(level_hpa):
        raise DataSourceError("independent overlap receipt does not cover level %.6g" % level_hpa)
    return receipt


def verify_cached_era5_overlap(
    primary: CropSpec,
    independent: CropSpec,
    *,
    primary_cache_dir: Optional[str] = None,
    independent_cache_dir: Optional[str] = None,
    variables: Sequence[str],
    level_hpa: float,
    rtol: float = 1e-6,
    atol: Optional[Mapping[str, float]] = None,
    block_frames: int = 8,
) -> Dict[str, Any]:
    """Compare a local CDS cache with a local WeatherBench overlap and publish evidence."""
    selected = tuple(variables)
    if not selected or len(set(selected)) != len(selected) \
            or set(selected) - set(CANONICAL_VARIABLES):
        raise InvalidParameterError("variables", selected, "unique canonical variables t/q/u/v/z")
    if independent.store not in CATALOGUE:
        raise InvalidParameterError(
            "independent.store", independent.store, "a catalogued WeatherBench 2 ERA5 store")
    if not primary.store.startswith("cds:"):
        raise InvalidParameterError("primary.store", primary.store, "the direct CDS route")
    if isinstance(block_frames, bool) or int(block_frames) != block_frames or block_frames < 1:
        raise InvalidParameterError("block_frames", block_frames, "a positive integer")
    if not math.isfinite(rtol) or rtol < 0:
        raise InvalidParameterError("rtol", rtol, "a finite non-negative tolerance")
    tolerances = ({name: DEFAULT_ATOL[name] for name in selected}
                  if atol is None else dict(atol))
    if set(tolerances) != set(selected) or any(
            not math.isfinite(float(value)) or float(value) < 0 for value in tolerances.values()):
        raise InvalidParameterError("atol", tolerances,
                                    "one finite non-negative tolerance per selected variable")

    receipt_path = overlap_receipt_path(primary, primary_cache_dir)
    if receipt_path.exists():
        existing = load_overlap_receipt(receipt_path)
        sha = existing.pop("receipt_sha256")
        same_design = (
            (existing.get("primary") or {}).get("content_key") == primary.content_key()
            and (existing.get("independent") or {}).get("content_key") == independent.content_key()
            and float(existing.get("level_hpa")) == float(level_hpa)
            and set(existing.get("variables", {})) == set(selected)
            and int((existing.get("bounded_execution") or {}).get("block_frames", 0))
                == int(block_frames)
            and all(
                float(existing["variables"][name].get("rtol")) == float(rtol)
                and float(existing["variables"][name].get("atol"))
                    == float(tolerances[name])
                for name in selected))
        if not same_design:
            raise DataSourceError(
                "existing immutable overlap receipt belongs to a different comparison design")
        _attach_receipt(primary, primary_cache_dir, existing, sha)
        return {**existing, "receipt_sha256": sha}

    primary_ds, primary_manifest = open_cached_lazy(primary, primary_cache_dir)
    independent_ds, independent_manifest = open_cached_lazy(independent, independent_cache_dir)
    try:
        if primary_manifest.get("source_route") != "Copernicus Climate Data Store API":
            raise DataSourceError("primary overlap cache is not the direct CDS route")
        for label, spec, manifest in (("primary", primary, primary_manifest),
                                      ("independent", independent, independent_manifest)):
            if manifest.get("content_key") != spec.content_key() or not manifest.get("content_hash"):
                raise DataSourceError("%s overlap cache identity is incomplete" % label)

        reference_array, _ = _resolved_array(independent_ds, selected[0], level_hpa)
        ilat_name = "latitude" if "latitude" in reference_array.dims else "lat"
        ilon_name = "longitude" if "longitude" in reference_array.dims else "lon"
        times = np.asarray(reference_array.time.values).astype("datetime64[ns]")
        latitude = np.asarray(reference_array[ilat_name].values, dtype=np.float64)
        longitude = np.asarray(reference_array[ilon_name].values, dtype=np.float64)
        if times.size < 2 or latitude.size < 2 or longitude.size < 2:
            raise DataSourceError("independent overlap must contain at least two points per axis")

        details: Dict[str, Any] = {}
        for variable in selected:
            left, left_name = _resolved_array(primary_ds, variable, level_hpa)
            right, right_name = _resolved_array(independent_ds, variable, level_hpa)
            llat = "latitude" if "latitude" in left.dims else "lat"
            llon = "longitude" if "longitude" in left.dims else "lon"
            rlat = "latitude" if "latitude" in right.dims else "lat"
            rlon = "longitude" if "longitude" in right.dims else "lon"
            right = right.sel(time=times, **{rlat: latitude, rlon: longitude})
            try:
                left = left.sel(time=times, **{llat: latitude, llon: longitude})
            except KeyError as exc:
                raise DataSourceError(
                    "CDS cache does not contain the independent overlap coordinates exactly") from exc
            if (not np.array_equal(np.asarray(left.time.values).astype("datetime64[ns]"), times)
                    or not np.array_equal(np.asarray(left[llat].values, dtype=np.float64), latitude)
                    or not np.array_equal(np.asarray(left[llon].values, dtype=np.float64), longitude)):
                raise DataSourceError("CDS and WeatherBench overlap coordinates are not exact")
            left_unit, right_unit = left.attrs.get("units"), right.attrs.get("units")
            if not _units_compatible(variable, left_unit, right_unit):
                raise DataSourceError(
                    "CDS and WeatherBench units are missing or incompatible for %s: %r vs %r"
                    % (variable, left_unit, right_unit))

            count = mismatches = 0
            sum_abs = sum_squared = max_abs = 0.0
            for start in range(0, len(times), int(block_frames)):
                stop = min(start + int(block_frames), len(times))
                a = np.asarray(left.isel(time=slice(start, stop)).values, dtype=np.float64)
                b = np.asarray(right.isel(time=slice(start, stop)).values, dtype=np.float64)
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise DataSourceError("ERA5 overlap contains non-finite values")
                delta = np.abs(a - b)
                count += int(delta.size)
                sum_abs += float(delta.sum(dtype=np.float64))
                sum_squared += float(np.square(delta).sum(dtype=np.float64))
                max_abs = max(max_abs, float(delta.max(initial=0.0)))
                mismatches += int(np.count_nonzero(
                    ~np.isclose(a, b, rtol=float(rtol), atol=float(tolerances[variable]))))
            details[variable] = {
                "primary_name": left_name, "independent_name": right_name,
                "primary_units": str(left_unit), "independent_units": str(right_unit),
                "units_compatible": True, "atol": float(tolerances[variable]),
                "rtol": float(rtol), "value_count": count,
                "mismatch_count": mismatches, "allclose": mismatches == 0,
                "max_abs_error": max_abs, "mean_abs_error": sum_abs / count,
                "rmse": math.sqrt(sum_squared / count),
            }

        receipt: Dict[str, Any] = {
            "schema": OVERLAP_SCHEMA,
            "primary": {
                "route": primary_manifest.get("source_route"),
                "store": primary.store, "content_key": primary.content_key(),
                "content_hash": primary_manifest["content_hash"],
                "request_sha256": (((primary_manifest.get("acquisition") or {}).get("request") or {})
                                    .get("request_sha256")),
            },
            "independent": {
                "route": "WeatherBench 2 public ERA5 Zarr",
                "store": independent.store, "content_key": independent.content_key(),
                "content_hash": independent_manifest["content_hash"],
            },
            "level_hpa": float(level_hpa), "coordinates_exact": True,
            "coordinate_sha256": _coordinate_hash(times, latitude, longitude),
            "shape": [int(len(times)), int(len(latitude)), int(len(longitude))],
            "variables": details,
            "passed": all(record["allclose"] for record in details.values()),
            "bounded_execution": {
                "network_used": False, "block_frames": int(block_frames),
                "maximum_frames_per_source_in_memory": int(block_frames),
                "full_overlap_loaded": False,
            },
        }
        receipt_sha256 = _sha256(receipt)
        _atomic_write_new(receipt_path, _canonical_json({
            "receipt": receipt, "receipt_sha256": receipt_sha256}) + b"\n")
        _attach_receipt(primary, primary_cache_dir, receipt, receipt_sha256)
        return {**receipt, "receipt_sha256": receipt_sha256}
    finally:
        primary_ds.close()
        independent_ds.close()


def _load_manifest_crop(path: Union[str, os.PathLike[str]]) -> Tuple[CropSpec, str]:
    manifest_path_value = Path(path)
    try:
        manifest = json.loads(manifest_path_value.read_text(encoding="utf-8"))
        spec = CropSpec.from_provenance(manifest["spec"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, InvalidParameterError) as exc:
        raise DataSourceError("cache manifest does not contain a valid CropSpec: %s" % exc) from exc
    if manifest.get("content_key") != spec.content_key():
        raise DataSourceError("cache manifest content key does not match its CropSpec")
    return spec, str(manifest_path_value.parent)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a cached CDS crop against a cached WeatherBench 2 overlap")
    parser.add_argument("--primary-manifest", required=True,
                        help="CDS cache manifest JSON")
    parser.add_argument("--independent-manifest", required=True,
                        help="WeatherBench cache manifest JSON")
    parser.add_argument("--variables", required=True,
                        help="comma-separated canonical variables")
    parser.add_argument("--level-hpa", required=True, type=float)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--block-frames", type=int, default=8)
    args = parser.parse_args(argv)
    primary, primary_cache = _load_manifest_crop(args.primary_manifest)
    independent, independent_cache = _load_manifest_crop(args.independent_manifest)
    variables = tuple(part.strip() for part in args.variables.split(",") if part.strip())
    result = verify_cached_era5_overlap(
        primary, independent, primary_cache_dir=primary_cache,
        independent_cache_dir=independent_cache, variables=variables,
        level_hpa=args.level_hpa, rtol=args.rtol, block_frames=args.block_frames)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
