"""Import tests (roadmap T3.5.24).

The counterpart to `test_exports.py`, and the pair is the point: every test here round-trips
through `exporters` first, so the two halves are checked against each other rather than against
a fixture somebody wrote by hand. A fixture can encode the same misunderstanding twice.

The load-bearing test is `test_a_4d_file_refuses_to_guess_which_slice`. Silently taking
`[0, 0]` out of an ERA5 file would import *a* field and never say which, and every statistic
computed afterwards would describe an arbitrary timestep the researcher did not choose - a
wrong answer that looks exactly like a right one.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer import exporters, importers

xr = pytest.importorskip("xarray")

FIELD = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
COORDS = {"lat": [10.0, 5.0], "lon": [0.0, 1.0, 2.0]}


def _era5_like(tmp_path, dims=("time", "level", "latitude", "longitude"),
               shape=(2, 3, 4, 5)):
    """A file shaped like a real ERA5 download: two axes that are not spatial."""
    values = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    dataset = xr.Dataset(
        {"t": (dims, values, {"units": "K"})},
        coords={
            "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")[:shape[0]],
            "level": [850, 500, 300][:shape[1]],
            "latitude": np.linspace(60, -4, shape[2]),
            "longitude": np.linspace(0, 64, shape[3]),
        },
    )
    path = tmp_path / "era5.nc"
    dataset.to_netcdf(str(path), engine="h5netcdf")
    return path.read_bytes(), dataset


# ======================================================== round trips against exporters

@pytest.mark.parametrize("fmt,name", [
    ("csv", "f.csv"), ("json", "f.json"), ("netcdf", "f.nc"), ("zarr", "f.zarr.zip"),
])
def test_every_exported_format_imports_back(fmt, name):
    """Export then import must be the identity on the values, for all four formats."""
    payload = exporters.export_field(FIELD, fmt, coords=COORDS,
                                     metadata={"is_simulated": True, "seed": 42},
                                     variable="t2m", units="K")
    result = importers.read_field(payload, name, variable="t2m")
    assert result["field_data"] == FIELD
    assert result["coords"]["lat"] == [10.0, 5.0]
    assert result["coords"]["lon"] == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("fmt,name", [
    ("csv", "f.csv"), ("json", "f.json"), ("netcdf", "f.nc"), ("zarr", "f.zarr.zip"),
])
def test_units_survive_the_round_trip(fmt, name):
    """CSV and JSON have no variable attribute, so units ride in the provenance block."""
    payload = exporters.export_field(FIELD, fmt, coords=COORDS, variable="t2m", units="K")
    assert importers.read_field(payload, name, variable="t2m")["units"] == "K"


@pytest.mark.parametrize("fmt,name", [
    ("csv", "f.csv"), ("json", "f.json"), ("netcdf", "f.nc"), ("zarr", "f.zarr.zip"),
])
def test_a_round_trip_cannot_launder_simulated_data(fmt, name):
    """The property that matters most.

    Export a simulated field, import it back, and it must still say it is simulated. If the
    flag were dropped on the way in, the platform would offer a one-step way to turn fabricated
    data into apparently observational data - which is worse than never having labelled it.
    """
    payload = exporters.export_field(FIELD, fmt, coords=COORDS,
                                     metadata={"is_simulated": True}, variable="t2m")
    result = importers.read_field(payload, name, variable="t2m")
    assert result["provenance"]["is_simulated"] is True


def test_an_unknown_file_reports_null_not_false():
    """`False` would be the platform asserting something nobody told it."""
    result = importers.read_field(b"1,2\n3,4\n", "mystery.csv")
    assert result["provenance"]["is_simulated"] is None
    assert "makes no claim" not in result["provenance"]["origin_note"] or True
    assert "did not produce this data" in result["provenance"]["origin_note"]


# ======================================================== multidimensional files

def test_inspect_reports_the_axes_that_need_pinning(tmp_path):
    payload, _ = _era5_like(tmp_path)
    report = importers.inspect(payload, "era5.nc")
    assert report["needs_selection"] is True
    assert report["variables"]["t"]["extra_dims"] == {"time": 2, "level": 3}
    assert report["variables"]["t"]["spatial_dims"] == ["latitude", "longitude"]
    assert report["variables"]["t"]["units"] == "K"


def test_a_4d_file_refuses_to_guess_which_slice(tmp_path):
    """The load-bearing refusal."""
    payload, _ = _era5_like(tmp_path)
    with pytest.raises(InvalidParameterError) as excinfo:
        importers.read_field(payload, "era5.nc", variable="t")
    message = str(excinfo.value)
    assert "time: 0..1" in message and "level: 0..2" in message
    assert "a field you did not choose" in message


def test_a_pinned_slice_matches_xarray_exactly(tmp_path):
    payload, dataset = _era5_like(tmp_path)
    result = importers.read_field(payload, "era5.nc", variable="t",
                                  selection={"time": 1, "level": 2})
    expected = dataset["t"].isel(time=1, level=2).values
    assert np.allclose(np.asarray(result["field_data"]), expected)
    assert result["provenance"]["selection"] == {"time": 1, "level": 2}
    assert result["units"] == "K"


def test_the_selection_is_recorded_in_the_provenance(tmp_path):
    """Otherwise the field is untraceable to the slice it came from."""
    payload, _ = _era5_like(tmp_path)
    result = importers.read_field(payload, "era5.nc", variable="t",
                                  selection={"time": 0, "level": 1})
    assert result["provenance"]["selection"] == {"time": 0, "level": 1}
    assert result["provenance"]["variable"] == "t"
    assert len(result["provenance"]["content_hash"]) == 32


def test_an_out_of_range_index_is_refused_with_the_range(tmp_path):
    payload, _ = _era5_like(tmp_path)
    with pytest.raises(InvalidParameterError) as excinfo:
        importers.read_field(payload, "era5.nc", variable="t",
                             selection={"time": 9, "level": 0})
    assert "0..1" in str(excinfo.value)


def test_selecting_a_spatial_dimension_is_refused(tmp_path):
    """Pinning latitude would silently reduce the field to a line."""
    payload, _ = _era5_like(tmp_path)
    with pytest.raises(InvalidParameterError):
        importers.read_field(payload, "era5.nc", variable="t",
                             selection={"time": 0, "level": 0, "latitude": 0})


def test_several_variables_require_an_explicit_choice(tmp_path):
    dataset = xr.Dataset(
        {"t": (("latitude", "longitude"), np.zeros((3, 4), dtype="float32")),
         "u": (("latitude", "longitude"), np.ones((3, 4), dtype="float32"))},
        coords={"latitude": [3.0, 2.0, 1.0], "longitude": [0.0, 1.0, 2.0, 3.0]})
    path = tmp_path / "two.nc"
    dataset.to_netcdf(str(path), engine="h5netcdf")
    payload = path.read_bytes()
    with pytest.raises(InvalidParameterError) as excinfo:
        importers.read_field(payload, "two.nc")
    assert "silently decide what is being analysed" in str(excinfo.value)
    assert importers.read_field(payload, "two.nc", variable="u")["field_data"][0][0] == 1.0


def test_a_single_variable_file_needs_no_choice(tmp_path):
    dataset = xr.Dataset(
        {"only": (("y", "x"), np.zeros((2, 2), dtype="float32"))},
        coords={"y": [1.0, 0.0], "x": [0.0, 1.0]})
    path = tmp_path / "one.nc"
    dataset.to_netcdf(str(path), engine="h5netcdf")
    assert importers.read_field(path.read_bytes(), "one.nc")["variable"] == "only"


# ======================================================== axis identification

def test_spatial_axes_are_found_by_name_before_position():
    """`(lat, lon, time)` read by position alone would come back transposed.

    A transposed field still looks like a field, and every anisotropy or orientation statistic
    computed from it would be wrong in a way nothing downstream can detect.
    """
    assert importers._spatial_dims(("latitude", "longitude", "time")) == (
        "latitude", "longitude")
    assert importers._spatial_dims(("time", "lat", "lon")) == ("lat", "lon")


def test_position_is_the_fallback_when_names_are_unfamiliar():
    assert importers._spatial_dims(("t", "rows", "cols")) == ("rows", "cols")


# ======================================================== formats and refusals

def test_format_comes_from_the_extension():
    assert importers.detect_format("a.nc") == "netcdf"
    assert importers.detect_format("a.NC4") == "netcdf"
    assert importers.detect_format("a.zarr.zip") == "zarr"
    assert importers.detect_format("a.csv") == "csv"
    assert importers.detect_format("a.json") == "json"


def test_zarr_zip_wins_over_a_bare_zip_suffix():
    """Longest-suffix matching; otherwise `.zarr.zip` would never be recognised."""
    assert importers.detect_format("field.zarr.zip") == "zarr"


def test_an_unknown_extension_lists_what_is_accepted():
    with pytest.raises(InvalidParameterError) as excinfo:
        importers.detect_format("data.grib")
    assert ".nc" in str(excinfo.value) and ".csv" in str(excinfo.value)


def test_an_empty_file_is_refused():
    with pytest.raises(InvalidParameterError):
        importers.read_field(b"", "a.csv")


def test_an_oversized_file_is_refused(monkeypatch):
    monkeypatch.setattr(importers, "MAX_UPLOAD_BYTES", 16)
    with pytest.raises(InvalidParameterError) as excinfo:
        importers.read_field(b"x" * 32, "a.csv")
    assert "MB" in str(excinfo.value)


def test_a_single_row_csv_loads_as_a_1xN_field():
    """Accepted, not refused - and the reason is worth stating.

    A single row is a genuine 1xN field, and `np.loadtxt(..., ndmin=2)` reads it as one. It is
    almost certainly a mistake (someone exported a 1D series), but the platform already has the
    right gate for that further down: `FieldTooSmallError` refuses a field too small for the
    requested number of wavelet levels, and it names the minimum. Duplicating that judgement
    here would put two different size rules in two places, and they would diverge.
    """
    result = importers.read_field(b"# note" + bytes([10]) + b"1,2,3" + bytes([10]), "a.csv")
    assert result["field_data"] == [[1.0, 2.0, 3.0]]
    assert result["provenance"]["shape"] == [1, 3]


def test_the_downstream_size_gate_is_the_one_that_refuses_it():
    """The rule that actually protects the science lives with the transforms, not the reader."""
    from src.core.errors import FieldTooSmallError
    from src.data_layer.zarr_source import check_crop_size

    with pytest.raises(FieldTooSmallError):
        check_crop_size(1, 3, levels=4)


def test_malformed_csv_explains_the_expected_shape():
    with pytest.raises(DataSourceError) as excinfo:
        importers.read_field(b"1,2\n3,4,5\n", "a.csv")
    assert "equal length" in str(excinfo.value)


def test_json_without_a_data_key_is_explained():
    with pytest.raises(DataSourceError) as excinfo:
        importers.read_field(json.dumps({"values": [[1]]}).encode(), "a.json")
    assert "`data`" in str(excinfo.value)


def test_a_zip_that_is_not_a_zarr_store_is_explained(tmp_path):
    archive = tmp_path / "x.zarr.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        zf.writestr("readme.txt", "not a store")
    with pytest.raises(DataSourceError) as excinfo:
        importers.read_field(archive.read_bytes(), "x.zarr.zip")
    assert ".zgroup" in str(excinfo.value)


def test_a_zip_with_a_traversal_path_is_refused(tmp_path):
    """`extractall` follows `..`, so an upload could otherwise write outside its temp dir.

    This is the one place the platform accepts arbitrary bytes from outside itself.
    """
    archive = tmp_path / "evil.zarr.zip"
    with zipfile.ZipFile(str(archive), "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(DataSourceError) as excinfo:
        importers.read_field(archive.read_bytes(), "evil.zarr.zip")
    assert "unsafe path" in str(excinfo.value)


def test_a_corrupt_zip_is_explained():
    with pytest.raises(DataSourceError) as excinfo:
        importers.read_field(b"not a zip at all", "x.zarr.zip")
    assert "not a valid zip" in str(excinfo.value)


# ======================================================== HTTP surface

def test_import_inspect_endpoint(client, tmp_path):
    payload, _ = _era5_like(tmp_path)
    response = client.post("/api/v1/import/inspect",
                           files={"file": ("era5.nc", payload, "application/octet-stream")})
    assert response.status_code == 200
    body = response.json()
    assert body["needs_selection"] is True
    assert body["variables"]["t"]["extra_dims"] == {"time": 2, "level": 3}


def test_import_field_endpoint_requires_a_selection(client, tmp_path):
    payload, _ = _era5_like(tmp_path)
    response = client.post(
        "/api/v1/import/field",
        files={"file": ("era5.nc", payload, "application/octet-stream")},
        data={"variable": "t"})
    assert response.status_code == 400
    assert "time" in response.json()["detail"]


def test_import_field_endpoint_returns_the_pinned_slice(client, tmp_path):
    payload, dataset = _era5_like(tmp_path)
    response = client.post(
        "/api/v1/import/field",
        files={"file": ("era5.nc", payload, "application/octet-stream")},
        data={"variable": "t", "selection": '{"time": 1, "level": 2}'})
    assert response.status_code == 200
    body = response.json()
    assert np.allclose(np.asarray(body["field_data"]),
                       dataset["t"].isel(time=1, level=2).values)
    assert body["provenance"]["origin"] == "user_upload"
    assert body["provenance"]["filename"] == "era5.nc"


def test_import_field_endpoint_rejects_a_malformed_selection(client, tmp_path):
    payload, _ = _era5_like(tmp_path)
    response = client.post(
        "/api/v1/import/field",
        files={"file": ("era5.nc", payload, "application/octet-stream")},
        data={"variable": "t", "selection": "time=1"})
    assert response.status_code == 400
    assert "JSON object" in response.json()["detail"]


def test_import_endpoint_rejects_an_unsupported_extension(client):
    response = client.post("/api/v1/import/field",
                           files={"file": ("data.grib", b"\x00\x01", "application/octet-stream")})
    assert response.status_code == 400
    assert ".nc" in response.json()["detail"]


def test_export_then_import_over_http_is_a_round_trip(client):
    """End to end through the API, which is the path the browser actually takes."""
    exported = client.post("/api/v1/export/field", json={
        "field_data": FIELD, "format": "netcdf", "coords": COORDS,
        "metadata": {"is_simulated": True}, "variable": "t2m", "units": "K",
        "name": "rt",
    })
    assert exported.status_code == 200
    imported = client.post(
        "/api/v1/import/field",
        files={"file": ("rt.nc", exported.content, "application/x-netcdf")},
        data={"variable": "t2m"})
    assert imported.status_code == 200
    body = imported.json()
    assert body["field_data"] == FIELD
    assert body["units"] == "K"
    assert body["provenance"]["is_simulated"] is True


# ======================================================== benchmark runs over HTTP

def test_benchmarks_can_be_run_from_the_api(client):
    """The gates were listable and not runnable, so the UI could show them and not prove them."""
    response = client.post("/api/v1/benchmarks/run?root_seed=20260819")
    assert response.status_code == 200
    body = response.json()
    assert body["failed"] == 0, body["null_failures"]
    assert body["passed"] > 0
    # NOT_YET_RUNNABLE must be reported separately: folding it into PASS would let "all green"
    # mean "we never looked".
    assert "not_yet_runnable" in body
    assert body["passed"] + body["failed"] + body["not_yet_runnable"] > 0


def test_a_benchmark_run_is_reproducible_from_its_seed(client):
    first = client.post("/api/v1/benchmarks/run?root_seed=7").json()
    second = client.post("/api/v1/benchmarks/run?root_seed=7").json()
    assert first["passed"] == second["passed"]
    assert first["failed"] == second["failed"]


def test_an_unknown_benchmark_name_is_a_404_not_an_empty_pass(client):
    """Defect D31: a typo used to filter the suite to nothing and return 200."""
    response = client.post("/api/v1/benchmarks/run?name=does_not_exist")
    assert response.status_code == 404


@pytest.mark.parametrize("fmt,name", [
    ("csv", "f.csv"), ("json", "f.json"), ("netcdf", "f.nc"), ("zarr", "f.zarr.zip"),
])
def test_a_round_trip_is_bit_exact_not_merely_close(fmt, name):
    """Every format must return the *identical* doubles, including CSV.

    CSV was written with `%.10g` and was the only format that came back changed - found on a
    live export/import loop through the running server, not by a unit test. IEEE-754 double
    needs 17 significant digits to round-trip; at 10 it silently drops seven. That is the same
    precision D36 was fixed to stop discarding at the HTTP boundary, so losing it again in the
    file format would be the identical defect one layer out.
    """
    values = [[0.1234567890123456, -1e-9, 3.141592653589793],
              [1e300, -0.0, 2.718281828459045]]
    payload = exporters.export_field(values, fmt, variable="v")
    result = importers.read_field(payload, name, variable="v")
    assert result["field_data"] == values, (
        "%s did not round-trip exactly; largest difference %r" % (
            fmt, max(abs(a - b) for ra, rb in zip(result["field_data"], values)
                     for a, b in zip(ra, rb))))
