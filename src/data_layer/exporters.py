"""Export fields and tables in the formats a researcher actually opens (T3.5.23).

**The gap this closes.** Before this module the platform could not emit a single file. Every
field, spectrum, metric, hypothesis and benchmark result lived and died inside a browser tab.
A tool whose output cannot leave it is not a research tool, whatever the quality of its
mathematics.

**Provenance travels *inside* the file, never beside it.** A CSV in a downloads folder with no
record of the seed, the units, the grid or whether the data was simulated is indistinguishable
from any other CSV six months later - and that is precisely when it matters. So:

*   **CSV** carries a commented header block (``# key: value``) before the numbers. Comment
    lines are skipped by `numpy.loadtxt`, `pandas.read_csv(comment="#")` and every spreadsheet
    importer worth using, so the record costs nothing to the reader who ignores it.
*   **JSON** carries it as a top-level ``provenance`` object beside ``data``.
*   **NetCDF and Zarr** carry it as dataset *attributes*, which is where `xarray` puts them
    back when the file is reopened - so the round trip is lossless, and asserted to be.

**Simulated data is labelled in a way that survives the file.** Every export whose provenance
says the field was fabricated gets a ``WARNING`` attribute stating so in words. A researcher
who reopens the file in six months, or a colleague who receives it, sees it without having to
know to look.

Formats and why each is here:

*   ``csv``    - opens anywhere, no dependencies, the universal fallback.
*   ``json``   - structure preserved, and the only format that round-trips nested provenance
                 without flattening it.
*   ``netcdf`` - what an atmospheric scientist actually loads: coordinates, units and
                 attributes in one file, straight into `xarray`.
*   ``zarr``   - chunked and cloud-native, returned as a zip of the store, for fields too large
                 to want as one blob.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.core.errors import InvalidParameterError

#: Formats `export_field` accepts. `png`/`svg` are deliberately absent: those are rendered by
#: the client from the live plot, because a server-side re-render would be a *different*
#: picture from the one on screen, and a figure that does not match what the researcher saw is
#: worse than no figure.
FIELD_FORMATS = ("csv", "json", "netcdf", "zarr")

#: Formats `export_table` accepts. A table has no coordinates to preserve, so the array
#: formats would add nothing but a dependency.
TABLE_FORMATS = ("csv", "json")

MEDIA_TYPES = {
    "csv": "text/csv",
    "json": "application/json",
    "netcdf": "application/x-netcdf",
    "zarr": "application/zip",
}

EXTENSIONS = {"csv": "csv", "json": "json", "netcdf": "nc", "zarr": "zarr.zip"}

#: Stated in the header of every export so the reader knows what produced it.
GENERATOR = "SpectralEarth"

SIMULATED_WARNING = (
    "THIS DATA IS SIMULATED. It was fabricated by the platform's synthetic generator or by a "
    "fallback source, and is NOT an observation. Do not present it as measurement.")

UNSEEDED_WARNING = (
    "THIS DATA IS NOT REPRODUCIBLE. At least one stochastic step ran without a seed, so "
    "re-running the same request will not reproduce these values.")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _validate_format(fmt: str, allowed: Sequence[str]) -> str:
    fmt = (fmt or "").strip().lower()
    if fmt not in allowed:
        raise InvalidParameterError("format", fmt, "one of %s" % ", ".join(allowed))
    return fmt


def build_provenance(metadata: Optional[Dict[str, Any]] = None,
                     **extra: Any) -> Dict[str, Any]:
    """Assemble the record that goes into the file.

    Warnings are *derived* rather than passed in, so a caller cannot forget to add one. The
    two that matter - simulated data and an unseeded draw - are the two facts that change what
    a number means, and both are inferable from the metadata the platform already carries.
    """
    record: Dict[str, Any] = {
        "generator": GENERATOR,
        "exported_at": _now(),
    }
    record.update(metadata or {})
    record.update(extra)

    warnings: List[str] = list(record.get("warnings") or [])
    if record.get("is_simulated") or record.get("simulated"):
        warnings.append(SIMULATED_WARNING)
    if record.get("reproducible") is False or record.get("seeded") is False:
        warnings.append(UNSEEDED_WARNING)
    if warnings:
        record["warnings"] = warnings
    return record


def _flatten(record: Dict[str, Any], prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten nested provenance to `key: value` lines for the CSV header and NetCDF attrs.

    NetCDF attributes must be scalars or arrays - a nested dict raises on write - so nesting
    is flattened with dotted keys rather than dropped. Dropping it would silently lose exactly
    the parts of the record that describe the crop, the correction and the seed.
    """
    out: List[Tuple[str, str]] = []
    for key, value in record.items():
        name = "%s.%s" % (prefix, key) if prefix else str(key)
        if isinstance(value, dict):
            out.extend(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            if value and all(isinstance(v, dict) for v in value):
                for i, item in enumerate(value):
                    out.extend(_flatten(item, "%s.%d" % (name, i)))
            else:
                out.append((name, ", ".join(str(v) for v in value)))
        elif value is None:
            out.append((name, ""))
        else:
            out.append((name, str(value)))
    return out


# --------------------------------------------------------------------------- fields

def export_field(
    field: Sequence[Sequence[float]],
    fmt: str = "csv",
    coords: Optional[Dict[str, Sequence[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    variable: str = "field",
    units: Optional[str] = None,
) -> bytes:
    """Serialise a 2D field with its coordinates and provenance.

    ``units`` is a separate argument rather than one more metadata key because it is the one
    attribute that changes what the numbers *are*. It is written to the place each format
    expects it - the CSV header, the JSON body, and the NetCDF/Zarr variable attribute that
    `xarray` shows on reopen.
    """
    import numpy as np

    fmt = _validate_format(fmt, FIELD_FORMATS)
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2:
        raise InvalidParameterError("field", "%dD array" % array.ndim, "a 2D field")
    if array.size == 0:
        raise InvalidParameterError("field", "empty", "a field with at least one value")

    record = build_provenance(metadata, variable=variable, units=units,
                              shape=list(array.shape))

    if fmt == "csv":
        return _field_to_csv(array, coords, record)
    if fmt == "json":
        return _field_to_json(array, coords, record)
    return _field_to_xarray(array, coords, record, fmt, variable, units)


def _field_to_csv(array, coords, record) -> bytes:
    import numpy as np

    buffer = io.StringIO()
    for key, value in _flatten(record):
        buffer.write("# %s: %s\n" % (key, value))
    if coords:
        for name, values in coords.items():
            buffer.write("# coord.%s: %s\n"
                         % (name, ", ".join("%.6g" % v for v in values)))
    buffer.write("# ---\n")
    # 17 significant digits, not 10. IEEE-754 double needs 17 to round-trip exactly; at
    # 10 the CSV silently loses seven digits and a re-imported field differs from the one
    # exported - measured on a live export/import loop, where CSV was the only format that
    # came back changed. That is the precision D36 was fixed to stop discarding at the HTTP
    # boundary, so throwing it away again in the file format would be the same defect one
    # layer out. The cost is roughly 50% more bytes.
    np.savetxt(buffer, array, delimiter=",", fmt="%.17g")
    return buffer.getvalue().encode("utf-8")


def _field_to_json(array, coords, record) -> bytes:
    payload = {
        "provenance": record,
        "coords": {k: [float(v) for v in vals] for k, vals in (coords or {}).items()},
        "data": [[float(v) for v in row] for row in array],
    }
    return json.dumps(payload, indent=2, sort_keys=False,
                      default=str).encode("utf-8")


def _coord_arrays(array, coords):
    """Pick the y/x coordinate vectors, falling back to indices, and check their lengths."""
    import numpy as np

    height, width = array.shape
    coords = coords or {}
    y = coords.get("lat", coords.get("y"))
    x = coords.get("lon", coords.get("x"))
    y_name = "lat" if "lat" in coords else "y"
    x_name = "lon" if "lon" in coords else "x"
    y = np.asarray(y, dtype=np.float64) if y is not None else np.arange(height, dtype=float)
    x = np.asarray(x, dtype=np.float64) if x is not None else np.arange(width, dtype=float)
    if y.size != height or x.size != width:
        raise InvalidParameterError(
            "coords", {"y": int(y.size), "x": int(x.size)},
            "coordinate vectors matching the field shape %dx%d. Writing a file whose axes "
            "disagree with its data would produce a plot that is wrong in a way nothing "
            "downstream can detect." % (height, width))
    return (y_name, y), (x_name, x)


def _field_to_xarray(array, coords, record, fmt, variable, units) -> bytes:
    import xarray as xr

    (y_name, y), (x_name, x) = _coord_arrays(array, coords)
    attrs = {k: v for k, v in _flatten(record)}
    dataset = xr.Dataset(
        {variable: ((y_name, x_name), array, {"units": units or "unknown"})},
        coords={y_name: y, x_name: x},
        attrs=attrs,
    )
    if fmt == "netcdf":
        # Written to a temporary file and read back rather than serialised in memory.
        # `to_netcdf()` with no path only supports the scipy engine, which emits **NetCDF3
        # classic** - no groups, no compression, 32-bit offsets. That is not what an
        # atmospheric scientist means by "a NetCDF file", so the h5netcdf engine (the one
        # declared in requirements.txt after D33) writes real NetCDF4/HDF5 to disk and the
        # bytes are read from there.
        tmpdir = tempfile.mkdtemp(prefix="spectralearth-nc-")
        try:
            path = os.path.join(tmpdir, "field.nc")
            dataset.to_netcdf(path, engine="h5netcdf")
            with open(path, "rb") as handle:
                return handle.read()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # Zarr is a directory, so it is written to a temporary one and zipped. The zip is what
    # crosses HTTP; `unzip` then `xr.open_zarr` restores it exactly.
    tmpdir = tempfile.mkdtemp(prefix="spectralearth-zarr-")
    try:
        store = os.path.join(tmpdir, "field.zarr")
        dataset.to_zarr(store, mode="w", consolidated=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for root, _dirs, files in os.walk(store):
                for name in files:
                    full = os.path.join(root, name)
                    archive.write(full, os.path.relpath(full, tmpdir))
        return buffer.getvalue()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------- tables

def export_table(
    rows: Sequence[Dict[str, Any]],
    fmt: str = "csv",
    columns: Optional[Sequence[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Serialise a list of records - hypotheses, benchmark results, a spectrum, metrics.

    An empty table is exported rather than refused, with the row count in the provenance. "No
    rows" is a real result - a scan that found nothing after correction is the *correct*
    outcome on null data - and refusing to export it would make the honest answer the one you
    cannot save.
    """
    import numpy as np

    fmt = _validate_format(fmt, TABLE_FORMATS)
    rows = list(rows)
    record = build_provenance(metadata, n_rows=len(rows))

    if columns is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen

    if fmt == "json":
        return json.dumps({"provenance": record, "columns": list(columns), "rows": rows},
                          indent=2, default=str).encode("utf-8")

    buffer = io.StringIO()
    for key, value in _flatten(record):
        buffer.write("# %s: %s\n" % (key, value))
    buffer.write("# ---\n")
    import csv as _csv

    writer = _csv.DictWriter(buffer, fieldnames=list(columns), extrasaction="ignore",
                             lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _scalar(row.get(k)) for k in columns})
    return buffer.getvalue().encode("utf-8")


def _scalar(value: Any) -> Any:
    """Flatten a nested value into something a CSV cell can hold, without losing it."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str)
    return value


def filename(stem: str, fmt: str) -> str:
    """A safe, dated filename. Dated because two exports of the same thing are common."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (stem or "export"))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "%s_%s.%s" % (safe or "export", stamp, EXTENSIONS[fmt])
