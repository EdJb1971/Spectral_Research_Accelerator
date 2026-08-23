"""Read a user-supplied field back into the platform (roadmap T3.5.24).

The counterpart to `exporters.py`, and the half that makes the platform usable on data it did
not produce. Until this existed, real data could arrive only two ways: a file placed in `data/`
by hand under one of three fixed names, or a Zarr crop streamed from WeatherBench 2. A
researcher with a NetCDF file from anywhere else - their own model, a colleague, a CDS
download, a previous export of this platform - had no way in.

**Accepts what the platform emits, and what a researcher already has:**

*   ``.nc`` / ``.nc4`` / ``.netcdf`` - NetCDF3 or NetCDF4/HDF5, via `xarray`.
*   ``.zarr.zip`` - a zipped Zarr store, which is exactly what `export_field` writes.
*   ``.csv`` - a plain numeric grid; `#`-commented provenance headers written by our own
    exporter are parsed back rather than skipped.
*   ``.json`` - our own export format, round-tripped exactly.

**Provenance is reconstructed, not assumed.** An imported file is *not* platform output by
default: `origin` is recorded as ``user_upload`` with the filename and a SHA-256 of the bytes,
so a result derived from it can be traced to the exact file. Where the file carries a
provenance record of its own - because this platform wrote it - that record is preserved
underneath, so a round trip does not launder away the fact that the data was simulated.

**The hard part is the one that looks trivial: is it 2D?** A NetCDF file from ERA5 is
``(time, level, lat, lon)``. Silently taking ``[0, 0]`` would import *a* field and never say
which - and every subsequent statistic would describe an arbitrary timestep the researcher did
not choose. So extra dimensions are **reported and must be selected**, by index, and the
selection is recorded in the provenance.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.axes import resolve_axis_roles
from src.core.errors import DataSourceError, InvalidParameterError

#: Extensions this module can read, longest first so `.zarr.zip` wins over `.zip`.
SUFFIXES = (".zarr.zip", ".nc", ".nc4", ".netcdf", ".cdf", ".csv", ".json")

#: Refuse anything larger. A browser upload of a global 0.25 degree field is ~4 MB per level
#: per timestep; 256 MB is generous for a regional crop and small enough that a mistake is a
#: rejection rather than an out-of-memory kill.
MAX_UPLOAD_BYTES = 256 * 1024 * 1024

#: The names this module used to consult for the two spatial axes now live in
#: `src.core.axes.AXIS_NAME_HINTS`, where they are one registered convention among several
#: rather than a fact about axes (TG1.1). A caller who knows the roles passes `axis_roles`
#: and is believed; a caller who does not gets the same answer as before, with a record of
#: how it was reached.


def detect_format(filename: str) -> str:
    """Format from the filename, since a browser upload has no reliable media type."""
    lowered = (filename or "").lower()
    for suffix in SUFFIXES:
        if lowered.endswith(suffix):
            if suffix == ".zarr.zip":
                return "zarr"
            if suffix in (".nc", ".nc4", ".netcdf", ".cdf"):
                return "netcdf"
            return suffix.lstrip(".")
    raise InvalidParameterError(
        "filename", filename,
        "a file ending in one of %s. The format is taken from the extension because a "
        "browser upload's media type is unreliable - Windows reports NetCDF as "
        "application/octet-stream." % ", ".join(SUFFIXES))


def _check_size(payload: bytes) -> None:
    if len(payload) > MAX_UPLOAD_BYTES:
        raise InvalidParameterError(
            "file", "%.1f MB" % (len(payload) / 1e6),
            "a file under %d MB. A global 0.25 degree field is about 4 MB per level per "
            "timestep, so this is far larger than a regional crop should be - which usually "
            "means the wrong file, or one that has not been subset yet."
            % (MAX_UPLOAD_BYTES // (1024 * 1024)))
    if not payload:
        raise InvalidParameterError("file", "0 bytes", "a non-empty file")


def content_hash(payload: bytes) -> str:
    """SHA-256 of the uploaded bytes, so a finding can name the exact file it came from."""
    return hashlib.sha256(payload).hexdigest()[:32]


# --------------------------------------------------------------------------- inspection

def _per_variable_roles(axis_roles: Optional[Dict[str, str]], all_dims: Sequence[str],
                        var_dims: Sequence[Any]) -> Optional[Dict[str, str]]:
    """Narrow a file-level role declaration to one variable's axes.

    Filtering is the right behaviour - variables in one file need not share axes - but only
    after checking the declaration against the file as a whole. A role declared for an axis
    that appears nowhere is a typo, and silently filtering it away would leave the axis it was
    meant for to be inferred, which is the failure this parameter exists to prevent.
    """
    if not axis_roles:
        return None
    unknown = [name for name in axis_roles if name not in set(all_dims)]
    if unknown:
        raise InvalidParameterError(
            "axis_roles", unknown,
            "roles only for axes this file has, which are %s" % (sorted(set(all_dims)),))
    present = {str(d) for d in var_dims}
    return {name: role for name, role in axis_roles.items() if name in present}


def inspect(payload: bytes, filename: str,
            axis_roles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Describe an upload **without** committing to a 2D slice of it.

    Two calls, deliberately: `inspect` then `read_field`. A file with a time axis has no single
    field in it, and the platform must not choose which one on the researcher's behalf. This
    reports the variables, the dimensions and which of them still need an index.
    """
    _check_size(payload)
    fmt = detect_format(filename)
    record: Dict[str, Any] = {
        "filename": os.path.basename(filename),
        "format": fmt,
        "bytes": len(payload),
        "content_hash": content_hash(payload),
    }

    if fmt in ("csv", "json"):
        field, coords, embedded = _read_flat(payload, fmt)
        record.update({
            "variables": {"field": {"dims": ["y", "x"],
                                    "shape": [len(field), len(field[0])],
                                    "extra_dims": {}}},
            "default_variable": "field",
            "embedded_provenance": embedded,
            "needs_selection": False,
            "coords": {k: list(v) for k, v in coords.items()},
        })
        return record

    dataset = _open_dataset(payload, fmt)
    try:
        variables: Dict[str, Any] = {}
        all_dims = [str(d) for var in dataset.data_vars.values() for d in var.dims]
        for name, var in dataset.data_vars.items():
            declared = _per_variable_roles(axis_roles, all_dims, var.dims)
            resolution = _resolve_axes(var.dims, declared)
            lat, lon = _spatial_dims(var.dims, declared)
            extra = {str(d): int(dataset.sizes[d]) for d in var.dims
                     if d not in (lat, lon)}
            variables[str(name)] = {
                "dims": [str(d) for d in var.dims],
                "shape": [int(s) for s in var.shape],
                "dtype": str(var.dtype),
                "units": var.attrs.get("units"),
                "spatial_dims": [lat, lon] if lat and lon else None,
                # How each role was reached: declared, by registered name, or by position.
                "axes": resolution.describe(),
                # These are the axes that have no meaning for a 2D field and must be pinned.
                "extra_dims": extra,
            }
        if not variables:
            raise DataSourceError(
                "%r contains no data variables, only coordinates. It may be a grid "
                "definition rather than a field." % os.path.basename(filename),
                filename=filename)
        record.update({
            "variables": variables,
            "default_variable": sorted(variables)[0],
            "embedded_provenance": {k: v for k, v in dataset.attrs.items()},
            "needs_selection": any(v["extra_dims"] for v in variables.values()),
            "coords": {str(k): [float(x) for x in dataset[k].values.ravel()[:4096]]
                       for k in dataset.coords
                       if dataset[k].ndim == 1 and dataset[k].dtype.kind in "fiu"},
        })
        return record
    finally:
        dataset.close()


def _resolve_axes(dims: Sequence[Any],
                  axis_roles: Optional[Dict[str, str]] = None):
    """Roles for this variable's axes: declared if the caller said so, otherwise inferred.

    The inference is unchanged - registered names first, trailing axes second - but it is now
    performed by `src.core.axes`, which records *which* of those two produced each answer.
    That record is what a reader needs: "the spatial axes were chosen by position" is a
    caveat on every anisotropy number in the result, and it used to be invisible.
    """
    return resolve_axis_roles(dims, axis_roles)


def _spatial_dims(dims: Sequence[Any],
                  axis_roles: Optional[Dict[str, str]] = None
                  ) -> Tuple[Optional[str], Optional[str]]:
    """The two spatial axes in ``(row, column)`` order, or ``(None, None)``.

    A single resolved spatial axis is reported as none: half a grid is not a grid, and the
    caller's next act would be to pair it with whatever sat beside it.
    """
    row, column = _resolve_axes(dims, axis_roles).spatial_pair()
    return (row, column) if row and column else (None, None)


def _open_dataset(payload: bytes, fmt: str):
    """Open NetCDF or zipped-Zarr bytes as an xarray Dataset."""
    import xarray as xr

    tmpdir = tempfile.mkdtemp(prefix="spectralearth-import-")
    try:
        if fmt == "netcdf":
            path = os.path.join(tmpdir, "upload.nc")
            with open(path, "wb") as handle:
                handle.write(payload)
            try:
                dataset = xr.open_dataset(path)
            except Exception as exc:
                raise DataSourceError(
                    "this file could not be read as NetCDF (%s: %s). If it is NetCDF4/HDF5, "
                    "the `h5netcdf` engine must be installed - see requirements.txt."
                    % (type(exc).__name__, exc))
            # Loaded eagerly so the temporary directory can be removed: a lazily-opened
            # dataset holds a file handle, and deleting the file underneath it produces an
            # error at first access, far from this line.
            dataset.load()
            return dataset

        archive = os.path.join(tmpdir, "upload.zip")
        with open(archive, "wb") as handle:
            handle.write(payload)
        try:
            with zipfile.ZipFile(archive) as zf:
                _reject_unsafe_paths(zf)
                zf.extractall(os.path.join(tmpdir, "store"))
        except zipfile.BadZipFile:
            raise DataSourceError(
                "this file is named .zarr.zip but is not a valid zip archive.")
        root = _find_zarr_root(os.path.join(tmpdir, "store"))
        dataset = xr.open_zarr(root)
        dataset.load()
        return dataset
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _reject_unsafe_paths(archive: zipfile.ZipFile) -> None:
    """Refuse absolute paths and `..` traversal in an uploaded archive.

    `extractall` follows them, so an uploaded zip could otherwise write outside the temporary
    directory. This is the one place the platform accepts arbitrary bytes from outside, and
    the check is cheap.
    """
    for name in archive.namelist():
        if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
            raise DataSourceError(
                "the archive contains an unsafe path (%r). Entries must be relative and "
                "must not traverse upwards." % name)


def _find_zarr_root(directory: str) -> str:
    """A `.zarr` store may sit at the root of the zip or one level down."""
    if os.path.exists(os.path.join(directory, ".zgroup")):
        return directory
    for entry in sorted(os.listdir(directory)):
        candidate = os.path.join(directory, entry)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, ".zgroup")):
            return candidate
    raise DataSourceError(
        "no Zarr store was found inside the archive - no `.zgroup` at the root or one level "
        "down. A zipped Zarr store is a *directory* of chunk files, not a single file.")


# --------------------------------------------------------------------------- reading

def read_field(
    payload: bytes,
    filename: str,
    variable: Optional[str] = None,
    selection: Optional[Dict[str, int]] = None,
    axis_roles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Read one 2D field, with its coordinates and a reconstructed provenance record.

    ``selection`` pins every non-spatial dimension by integer index. It is **required** when
    such dimensions exist: importing `[0, 0]` silently would produce a field the researcher
    did not choose, and every number computed from it would describe an arbitrary timestep.
    """
    _check_size(payload)
    fmt = detect_format(filename)
    digest = content_hash(payload)

    if fmt in ("csv", "json"):
        field, coords, embedded = _read_flat(payload, fmt)
        # CSV and JSON have no place for a variable attribute, so our own exporter records
        # the units in the provenance block. Reading them back from there is what makes the
        # round trip lossless rather than merely numerically correct.
        units = (embedded or {}).get("units") or None
        variable = (embedded or {}).get("variable") or "field"
        # A flat grid's axes are not inferred: this reader built them, so their roles are
        # declared rather than guessed, and the record says `declared` for both.
        return _result(field, coords, filename, fmt, digest, embedded, {}, units, variable,
                       axes=_resolve_axes(("y", "x"), {"y": "space", "x": "space"}))

    dataset = _open_dataset(payload, fmt)
    try:
        names = sorted(str(n) for n in dataset.data_vars)
        chosen = variable or (names[0] if len(names) == 1 else None)
        if chosen is None:
            raise InvalidParameterError(
                "variable", None,
                "one of %s. The file holds several variables and choosing one for you would "
                "silently decide what is being analysed." % ", ".join(names))
        if chosen not in names:
            raise InvalidParameterError("variable", chosen, "one of %s" % ", ".join(names))

        var = dataset[chosen]
        declared = _per_variable_roles(
            axis_roles, [str(d) for v in dataset.data_vars.values() for d in v.dims],
            var.dims)
        resolution = _resolve_axes(var.dims, declared)
        lat, lon = _spatial_dims(var.dims, declared)
        extra = [str(d) for d in var.dims if d not in (lat, lon)]
        selection = {str(k): int(v) for k, v in (selection or {}).items()}

        missing = [d for d in extra if d not in selection]
        if missing:
            raise InvalidParameterError(
                "selection", selection,
                "an index for every non-spatial dimension (%s). %r is %dD; taking index 0 "
                "without being asked would import a field you did not choose, and every "
                "statistic computed from it would describe that arbitrary slice."
                % (", ".join("%s: 0..%d" % (d, int(dataset.sizes[d]) - 1) for d in missing),
                   chosen, var.ndim))

        for dim, index in selection.items():
            if dim not in extra:
                raise InvalidParameterError(
                    "selection", {dim: index},
                    "indices only for the non-spatial dimensions %s" % (extra or "(none)"))
            size = int(dataset.sizes[dim])
            if not 0 <= index < size:
                raise InvalidParameterError(
                    "selection[%s]" % dim, index, "an index in 0..%d" % (size - 1))
            var = var.isel({dim: index})

        values = var.values
        if values.ndim != 2:
            raise DataSourceError(
                "after selection the field is still %dD (dims %s). This usually means the "
                "spatial axes were not recognised. Name them latitude/longitude or x/y, or "
                "pass `axis_roles` to say which axes are spatial - a declared role is "
                "believed and never second-guessed by a name."
                % (values.ndim, list(var.dims)))

        coords: Dict[str, List[float]] = {}
        for dim, key in ((lat, "lat"), (lon, "lon")):
            if dim and dim in var.coords:
                coords[key] = [float(v) for v in var[dim].values.ravel()]

        return _result(
            [[float(v) for v in row] for row in values], coords, filename, fmt, digest,
            {k: v for k, v in dataset.attrs.items()}, selection,
            var.attrs.get("units"), chosen, axes=resolution)
    finally:
        dataset.close()


def _read_flat(payload: bytes, fmt: str):
    """CSV or JSON written by `exporters`, or a plain numeric grid."""
    import numpy as np

    if fmt == "json":
        try:
            body = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise DataSourceError("this file is not valid JSON (%s)." % exc)
        data = body.get("data", body if isinstance(body, list) else None)
        if data is None:
            raise DataSourceError(
                "this JSON has no `data` key and is not a bare 2D array. An export from this "
                "platform has `data`, `coords` and `provenance`.")
        array = np.asarray(data, dtype=np.float64)
        coords = {k: [float(x) for x in v] for k, v in (body.get("coords") or {}).items()}
        return _as_2d(array), coords, body.get("provenance", {})

    text = payload.decode("utf-8", errors="replace")
    embedded: Dict[str, Any] = {}
    coords: Dict[str, List[float]] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        match = re.match(r"#\s*([^:]+):\s*(.*)$", line)
        if not match:
            continue
        key, value = match.group(1).strip(), match.group(2).strip()
        if key.startswith("coord."):
            try:
                coords[key[len("coord."):]] = [float(v) for v in value.split(",")]
            except ValueError:
                pass
        else:
            embedded[key] = value
    try:
        array = np.loadtxt(io.StringIO(text), delimiter=",", comments="#", ndmin=2)
    except ValueError as exc:
        raise DataSourceError(
            "this CSV could not be parsed as a numeric grid (%s). Rows must be "
            "comma-separated numbers of equal length; comment lines start with '#'." % exc)
    return _as_2d(array), coords, embedded


def _as_2d(array):
    if array.ndim != 2:
        raise DataSourceError(
            "the file contains a %dD array; a field must be 2D." % array.ndim)
    if array.size == 0:
        raise DataSourceError("the file contains no values.")
    return [[float(v) for v in row] for row in array]


def _result(field, coords, filename, fmt, digest, embedded, selection, units, variable,
            axes=None):
    """Assemble the field plus the provenance that must travel with it.

    ``is_simulated`` is inherited from the file when the file says so, and is otherwise
    ``None`` - **not** False. An uploaded file of unknown origin is not evidence that the data
    is observational, and recording a confident `False` would be the platform asserting
    something nobody told it.
    """
    inherited_simulated = None
    for key in ("is_simulated", "simulated"):
        if key in (embedded or {}):
            raw = embedded[key]
            inherited_simulated = (raw if isinstance(raw, bool)
                                   else str(raw).strip().lower() in ("true", "1", "yes"))
            break

    return {
        "field_data": field,
        "coords": coords,
        "units": units,
        "variable": variable,
        "provenance": {
            "origin": "user_upload",
            "filename": os.path.basename(filename),
            "format": fmt,
            "content_hash": digest,
            "variable": variable,
            "selection": selection or {},
            "units": units,
            "shape": [len(field), len(field[0])],
            # Which axes were taken to be spatial, and on whose authority. A pair chosen by
            # position is a caveat on every orientation statistic downstream, and before
            # TG1.1 that choice left no trace in the record at all.
            "axes": axes.describe() if axes is not None else None,
            # Preserved rather than flattened away: a round trip through this platform must
            # not launder a simulated field into an apparently observational one.
            "embedded_provenance": embedded or {},
            "is_simulated": inherited_simulated,
            "origin_note": (
                "Uploaded by the user. The platform did not produce this data and cannot "
                "vouch for it; `is_simulated` is null unless the file itself declared it."),
        },
    }
