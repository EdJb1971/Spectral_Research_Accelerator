"""Export tests (roadmap T3.5.23).

Every assertion here is a **round trip**: the file is written, reopened with the library a
researcher would actually use, and compared to what went in. A serialiser that produces
plausible-looking bytes nobody can read back is the failure mode worth testing for, and it is
invisible to any test that only checks the response was 200.

The other half is provenance. A CSV in a downloads folder six months later, with no record of
the seed, the units or whether the data was simulated, is indistinguishable from any other CSV
- so the tests assert the record is *inside* each file and survives the round trip.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest

from src.core.errors import InvalidParameterError
from src.data_layer import exporters

xr = pytest.importorskip("xarray")


FIELD = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
COORDS = {"lat": [10.0, 5.0], "lon": [0.0, 1.0, 2.0]}


def _write(tmp_path, payload: bytes, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(payload)
    return str(path)


# ======================================================== formats round-trip

def test_csv_round_trips_through_numpy(tmp_path):
    """The header must be readable by a human and skippable by a machine."""
    payload = exporters.export_field(FIELD, "csv", coords=COORDS,
                                     metadata={"seed": 42}).decode("utf-8")
    values = np.loadtxt(io.StringIO(payload), delimiter=",", comments="#")
    assert values.tolist() == FIELD
    assert "# seed: 42" in payload
    assert "# generator: SpectralEarth" in payload


def test_csv_header_records_the_coordinates(tmp_path):
    """Without them the CSV is a bare grid of numbers with no position on Earth."""
    payload = exporters.export_field(FIELD, "csv", coords=COORDS).decode("utf-8")
    assert "# coord.lat: 10, 5" in payload
    assert "# coord.lon: 0, 1, 2" in payload


def test_json_round_trips_with_nested_provenance():
    """JSON is the only format that keeps nesting; the test asserts it is not flattened."""
    metadata = {"source": {"name": "simulated", "kind": "simulated"}, "seed": 7}
    payload = json.loads(exporters.export_field(FIELD, "json", coords=COORDS,
                                                metadata=metadata))
    assert payload["data"] == FIELD
    assert payload["coords"]["lat"] == [10.0, 5.0]
    assert payload["provenance"]["source"]["name"] == "simulated"


def test_netcdf_is_netcdf4_not_netcdf3(tmp_path):
    """`to_netcdf()` in memory only supports scipy, which writes NetCDF3 classic.

    NetCDF3 has no groups, no compression and 32-bit offsets - not what an atmospheric
    scientist means by "a NetCDF file". The HDF5 magic number is the check that the right
    thing was written.
    """
    payload = exporters.export_field(FIELD, "netcdf", coords=COORDS, units="K")
    assert payload[:4] == b"\x89HDF", "not an HDF5/NetCDF4 file"


def test_netcdf_round_trips_values_coords_units_and_attrs(tmp_path):
    payload = exporters.export_field(FIELD, "netcdf", coords=COORDS, variable="t2m",
                                     units="K", metadata={"seed": 42, "is_simulated": True})
    dataset = xr.open_dataset(_write(tmp_path, payload, "f.nc"))
    try:
        assert dataset["t2m"].values.tolist() == FIELD
        assert dataset["lat"].values.tolist() == [10.0, 5.0]
        assert dataset["t2m"].attrs["units"] == "K"
        assert dataset.attrs["seed"] == "42"
        assert any("SIMULATED" in v for v in dataset.attrs.values()
                   if isinstance(v, str))
    finally:
        dataset.close()


def test_zarr_zip_round_trips(tmp_path):
    payload = exporters.export_field(FIELD, "zarr", coords=COORDS, variable="t2m",
                                     units="K", metadata={"seed": 42})
    archive = _write(tmp_path, payload, "f.zip")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(str(tmp_path / "out"))
    dataset = xr.open_zarr(str(tmp_path / "out" / "field.zarr"))
    try:
        assert dataset["t2m"].values.tolist() == FIELD
        assert dataset["lon"].values.tolist() == [0.0, 1.0, 2.0]
        assert dataset.attrs["seed"] == "42"
    finally:
        dataset.close()


def test_every_declared_field_format_produces_bytes():
    for fmt in exporters.FIELD_FORMATS:
        payload = exporters.export_field(FIELD, fmt, coords=COORDS)
        assert isinstance(payload, bytes) and payload, fmt


# ======================================================== provenance

def test_simulated_data_is_warned_about_in_words():
    """A flag a reader has to know to look for is not a warning."""
    for fmt in ("csv", "json"):
        payload = exporters.export_field(FIELD, fmt, metadata={"is_simulated": True})
        assert b"THIS DATA IS SIMULATED" in payload, fmt


def test_unseeded_data_is_warned_about_in_words():
    payload = exporters.export_field(FIELD, "csv", metadata={"reproducible": False})
    assert b"NOT REPRODUCIBLE" in payload


def test_warnings_are_derived_not_supplied():
    """A caller must not be able to omit them by forgetting.

    `build_provenance` computes the warnings from the metadata the platform already carries,
    so an export path that never thinks about warnings still emits the right ones.
    """
    record = exporters.build_provenance({"is_simulated": True, "reproducible": False})
    assert len(record["warnings"]) == 2


def test_real_observational_data_gets_no_false_warning():
    record = exporters.build_provenance({"is_simulated": False, "reproducible": True})
    assert "warnings" not in record


def test_nested_provenance_is_flattened_not_dropped():
    """NetCDF attributes cannot hold a dict; dropping the nesting would lose the seed."""
    payload = exporters.export_field(
        FIELD, "csv", metadata={"crop": {"store": "era5", "levels": [850, 500]}}).decode()
    assert "# crop.store: era5" in payload
    assert "# crop.levels: 850, 500" in payload


def test_a_list_of_records_is_flattened_by_index():
    payload = exporters.export_field(
        FIELD, "csv",
        metadata={"perturbations": [{"type": "noise", "seed": 1}]}).decode()
    assert "# perturbations.0.type: noise" in payload
    assert "# perturbations.0.seed: 1" in payload


# ======================================================== refusals

def test_mismatched_coordinates_are_refused():
    """A file whose axes disagree with its data plots wrongly and nothing downstream knows."""
    with pytest.raises(InvalidParameterError) as excinfo:
        exporters.export_field(FIELD, "netcdf", coords={"lat": [1.0], "lon": [0.0, 1.0, 2.0]})
    assert "matching the field shape" in str(excinfo.value)


def test_unknown_format_names_the_supported_ones():
    with pytest.raises(InvalidParameterError) as excinfo:
        exporters.export_field(FIELD, "parquet")
    assert "csv" in str(excinfo.value) and "netcdf" in str(excinfo.value)


def test_png_is_not_a_server_side_field_format():
    """Figures are rendered client-side on purpose; a server re-render is a different picture."""
    assert "png" not in exporters.FIELD_FORMATS
    with pytest.raises(InvalidParameterError):
        exporters.export_field(FIELD, "png")


def test_non_2d_input_is_refused():
    with pytest.raises(InvalidParameterError):
        exporters.export_field([1.0, 2.0, 3.0], "csv")


# ======================================================== tables

def test_table_csv_round_trips_through_csv_reader():
    import csv as _csv

    payload = exporters.export_table(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}], "csv").decode("utf-8")
    body = "\n".join(l for l in payload.splitlines() if not l.startswith("#"))
    rows = list(_csv.DictReader(io.StringIO(body)))
    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_table_columns_are_the_union_across_rows():
    """A row missing a key must not silently truncate the header for every other row."""
    payload = exporters.export_table([{"a": 1}, {"b": 2}], "csv").decode()
    header = [l for l in payload.splitlines() if not l.startswith("#")][0]
    assert header == "a,b"


def test_table_nested_values_are_kept_as_json_not_stringified_dicts():
    payload = exporters.export_table([{"stats": {"q": 0.01}}], "csv").decode()
    assert '{""q"": 0.01}' in payload or '{"q": 0.01}' in payload


def test_an_empty_table_still_exports():
    """"Nothing survived correction" is a real result and must be saveable.

    Refusing to export it would make the honest outcome the one you cannot record, which is
    precisely backwards for a platform whose null benchmarks exist to produce that answer.
    """
    payload = exporters.export_table([], "csv", metadata={"scan": "all pairs"}).decode()
    assert "# n_rows: 0" in payload


def test_table_format_list_excludes_array_formats():
    assert set(exporters.TABLE_FORMATS) == {"csv", "json"}


# ======================================================== filenames

def test_filename_is_dated_and_safe():
    name = exporters.filename("my field/../etc", "netcdf")
    assert name.endswith(".nc")
    assert "/" not in name and ".." not in name
    assert "T" in name and name.count("_") >= 1


def test_zarr_filename_says_it_is_a_zip():
    assert exporters.filename("f", "zarr").endswith(".zarr.zip")


# ======================================================== HTTP surface

@pytest.mark.parametrize("fmt", ["csv", "json", "netcdf", "zarr"])
def test_export_field_endpoint_returns_a_named_attachment(client, fmt):
    response = client.post("/api/v1/export/field", json={
        "field_data": FIELD, "format": fmt, "coords": COORDS,
        "metadata": {"is_simulated": True}, "variable": "t2m", "units": "K",
        "name": "demo",
    })
    assert response.status_code == 200
    assert response.content
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment; filename=")
    assert exporters.EXTENSIONS[fmt] in disposition
    # Without this header a cross-origin browser cannot read the filename it was given.
    assert "Content-Disposition" in response.headers["access-control-expose-headers"]
    assert response.headers["content-type"].startswith(
        exporters.MEDIA_TYPES[fmt].split(";")[0])


def test_export_field_endpoint_embeds_provenance(client):
    response = client.post("/api/v1/export/field", json={
        "field_data": FIELD, "format": "csv", "coords": COORDS,
        "metadata": {"is_simulated": True, "seed": 99}, "variable": "t2m",
        "units": "K", "name": "demo",
    })
    body = response.content.decode("utf-8")
    assert "# seed: 99" in body
    assert "THIS DATA IS SIMULATED" in body


def test_export_table_endpoint_returns_csv(client):
    response = client.post("/api/v1/export/table", json={
        "rows": [{"q_value": 0.01}], "format": "csv", "columns": None,
        "metadata": {"what": "hypotheses"}, "name": "hyp",
    })
    assert response.status_code == 200
    assert b"q_value" in response.content


def test_export_endpoint_rejects_an_unsupported_format(client):
    response = client.post("/api/v1/export/field",
                           json={"field_data": FIELD, "format": "png"})
    assert response.status_code == 400
    assert "csv" in response.json()["detail"]


def test_export_endpoint_rejects_mismatched_coordinates(client):
    response = client.post("/api/v1/export/field", json={
        "field_data": FIELD, "format": "netcdf",
        "coords": {"lat": [1.0], "lon": [0.0, 1.0, 2.0]},
    })
    assert response.status_code == 400


# ======================================================== reproducibility (D34)

def test_perturbation_endpoint_is_reproducible_when_seeded(client):
    """Defect D34: the engine took a seed; the endpoint never passed one."""
    payload = {"field_data": [[float(i + j) for j in range(8)] for i in range(8)],
               "perturbations": [{"type": "noise", "noise_type": "gaussian",
                                  "level": 0.2, "seed": 42}]}
    first = client.post("/api/v1/synthetic/perturb", json=payload).json()
    second = client.post("/api/v1/synthetic/perturb", json=payload).json()
    assert first["perturbed_field"] == second["perturbed_field"]
    assert first["reproducible"] is True
    assert first["provenance"][0]["seed"] == 42


def test_perturbation_endpoint_reports_when_it_was_not_seeded(client):
    """An unseeded draw is a property of the result, and is reported rather than hidden."""
    payload = {"field_data": [[float(i + j) for j in range(8)] for i in range(8)],
               "perturbations": [{"type": "noise", "noise_type": "gaussian", "level": 0.2}]}
    body = client.post("/api/v1/synthetic/perturb", json=payload).json()
    assert body["reproducible"] is False
    assert body["provenance"][0]["seeded"] is False


def test_a_deterministic_perturbation_is_reproducible_without_a_seed(client):
    """Rotation has no randomness, so demanding a seed for it would be noise in the record."""
    payload = {"field_data": [[float(i + j) for j in range(8)] for i in range(8)],
               "perturbations": [{"type": "rotation", "angle": 90.0}]}
    body = client.post("/api/v1/synthetic/perturb", json=payload).json()
    assert body["reproducible"] is True
