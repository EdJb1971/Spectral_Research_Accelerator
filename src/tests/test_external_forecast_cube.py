"""Acceptance tests for T5.6b canonical forecast-cube import and lazy NZ cropping."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr

from src.forecasting import (
    CANONICAL_FORECAST_CUBE_SCHEMA,
    ExternalForecastRun,
    ForecastContractError,
    GeographicBounds,
    fcn3_variable_unit,
    grid_coordinates_sha256,
    open_canonical_forecast_cube,
    seal_external_forecast_result,
)
from src.tests.test_external_fcn3 import SHA_A, complete_request


def _coordinates(request):
    lat = np.linspace(90.0, -90.0, 721, dtype="float64")
    lon = np.arange(1440, dtype="float64") * 0.25
    time = np.asarray([np.datetime64(value.replace("+00:00", ""), "ns")
                       for value in request.input.initialization_times])
    lead = np.asarray([np.timedelta64(int(hours), "h")
                       for hours in request.rollout.lead_hours], dtype="timedelta64[ns]")
    ensemble = np.asarray(request.rollout.member_seeds, dtype="int64")
    return {"time": time, "ensemble": ensemble, "lead_time": lead, "lat": lat, "lon": lon}


def _request(*, artifact_format="zarr", variables=("t850",), steps=2, members=2):
    record = complete_request(worker=True)
    record["rollout"].update(
        steps=steps, ensemble_size=members,
        member_seeds=list(range(7103, 7103 + members)))
    record["output"].update(
        variables=list(variables), artifact_format=artifact_format,
        artifact_reference=("artifacts/test/global.zarr" if artifact_format == "zarr"
                            else "artifacts/test/global.nc"))
    lat = np.linspace(90.0, -90.0, 721, dtype="float64")
    lon = np.arange(1440, dtype="float64") * 0.25
    record["input"]["grid_coordinates_sha256"] = grid_coordinates_sha256(lat, lon)
    return ExternalForecastRun.from_mapping(record)


def _write_cube(path, request, *, chunk_lat=181, chunk_lon=360, mutation=None):
    coords = _coordinates(request)
    shape = tuple(len(coords[name]) for name in ("time", "ensemble", "lead_time", "lat", "lon"))
    chunks = (1, 1, 1, chunk_lat, chunk_lon)
    arrays = {}
    for index, variable in enumerate(request.output.variables):
        values = da.full(shape, float(index + 1), chunks=chunks, dtype="float32")
        arrays[variable] = xr.DataArray(
            values, dims=("time", "ensemble", "lead_time", "lat", "lon"),
            attrs={"units": fcn3_variable_unit(variable)})
    dataset = xr.Dataset(arrays, coords=coords, attrs={
        "forecast_cube_schema": CANONICAL_FORECAST_CUBE_SCHEMA,
        "request_sha256": request.fingerprint(),
        "write_complete": "true",
    })
    if mutation is not None:
        mutation(dataset)
    if request.output.artifact_format == "zarr":
        dataset.to_zarr(path, mode="w", consolidated=True)
    else:
        encoding = {name: {"chunksizes": chunks, "compression": "gzip", "compression_opts": 1}
                    for name in request.output.variables}
        dataset.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def _sealed(request, path):
    return seal_external_forecast_result(
        request, path, peak_vram_bytes=1, peak_ram_bytes=1,
        wall_time_seconds=1.0, worker_log_sha256=SHA_A)


@pytest.mark.parametrize("artifact_format", ["zarr", "netcdf4"])
def test_opens_authenticates_scans_and_keeps_global_cube_lazy(tmp_path, artifact_format):
    request = _request(artifact_format=artifact_format, steps=1, members=1)
    path = tmp_path / ("cube.zarr" if artifact_format == "zarr" else "cube.nc")
    _write_cube(path, request)
    result = _sealed(request, path)

    with open_canonical_forecast_cube(path, request, result, max_chunk_bytes=2_000_000) as cube:
        assert cube.validation.finite_values is True
        assert cube.validation.checked_values == 721 * 1440
        assert cube.validation.chunks_checked == 16
        assert cube.validation.units == (("t850", "K"),)
        assert cube.validation.maximum_observed_chunk_bytes <= 2_000_000
        assert cube.dataset.t850.chunks is not None
        assert cube.validation.claim_boundary.endswith("NOT established.")


def test_all_fcn3_unit_families_have_explicit_si_semantics():
    assert fcn3_variable_unit("u10m") == "m s-1"
    assert fcn3_variable_unit("t2m") == "K"
    assert fcn3_variable_unit("msl") == "Pa"
    assert fcn3_variable_unit("tcwv") == "kg m-2"
    assert fcn3_variable_unit("q850") == "kg kg-1"
    assert fcn3_variable_unit("z850") == "m2 s-2"
    with pytest.raises(ForecastContractError, match="no canonical FCN3 unit"):
        fcn3_variable_unit("precipitation")


def test_crops_exact_declared_nz_box_lazily_and_preserves_lineage(tmp_path):
    request = _request(variables=("t850", "q850", "u850", "v850", "z850"))
    path = tmp_path / "cube.zarr"
    _write_cube(path, request)
    result = _sealed(request, path)
    cube = open_canonical_forecast_cube(path, request, result, max_chunk_bytes=2_000_000)
    try:
        bounds = GeographicBounds(
            south=-47.0, north=-34.0, west=166.0, east=179.75,
            longitude_convention="0..360", label="declared NZ fixture")
        regional = cube.crop(bounds)
        assert regional.dataset.sizes["lat"] == 53
        assert regional.dataset.sizes["lon"] == 56
        assert all(regional.dataset[name].chunks is not None for name in request.output.variables)
        assert regional.provenance["lazy"] is True
        assert regional.provenance["artifact_sha256"] == result.artifact_sha256
        assert regional.provenance["selection"].startswith("inclusive exact-coordinate")
        assert np.asarray(regional.dataset.t850.isel(time=0, ensemble=0, lead_time=0)).shape == (53, 56)
    finally:
        cube.close()


@pytest.mark.parametrize("change,match", [
    (lambda ds: ds.attrs.update(write_complete="false"), "write_complete"),
    (lambda ds: ds.attrs.update(request_sha256="a" * 64), "request_sha256"),
    (lambda ds: ds.t850.attrs.update(units="degC"), "units must be 'K'"),
    (lambda ds: ds.assign_coords(ensemble=[999, 1000]), "coordinate 'ensemble'"),
    (lambda ds: ds.assign_coords(
        lead_time=np.asarray([np.timedelta64(0, "h"), np.timedelta64(6, "h")])),
     "coordinate 'lead_time'"),
])
def test_refuses_incomplete_or_semantically_different_cube(tmp_path, change, match):
    request = _request()
    path = tmp_path / "bad.zarr"

    def mutation(ds):
        changed = change(ds)
        if isinstance(changed, xr.Dataset):
            # assign_coords returns a new Dataset; copy its coordinates into the fixture object.
            for name in changed.coords:
                ds.coords[name] = changed.coords[name]

    _write_cube(path, request, mutation=mutation)
    result = _sealed(request, path)
    with pytest.raises(ForecastContractError, match=match):
        open_canonical_forecast_cube(path, request, result)


def test_refuses_nonfinite_values_and_oversized_decompressed_chunks(tmp_path):
    request = _request(steps=1, members=1)
    path = tmp_path / "bad-values.zarr"
    _write_cube(path, request, chunk_lat=721, chunk_lon=1440)
    zarr.open_group(path, mode="r+")["t850"][0, 0, 0, 0, 0] = np.nan
    result = _sealed(request, path)
    with pytest.raises(ForecastContractError, match="rechunk the worker artifact"):
        open_canonical_forecast_cube(path, request, result, max_chunk_bytes=1_000_000)
    with pytest.raises(ForecastContractError, match="non-finite forecast values"):
        open_canonical_forecast_cube(path, request, result, max_chunk_bytes=5_000_000)


@pytest.mark.parametrize("mutation,match", [
    (lambda ds: ds.__setitem__("t850", ds.t850.transpose(
        "time", "lead_time", "ensemble", "lat", "lon")), "dimensions must be exactly"),
    (lambda ds: ds.__setitem__("unexpected", ds.t850), "forecast variables differ"),
    (lambda ds: ds.coords.update({"lat": (
        "lat", np.concatenate((np.asarray([89.75]), ds.lat.values[1:])))}), "coordinate 'lat'"),
])
def test_refuses_dimension_variable_or_global_grid_drift(tmp_path, mutation, match):
    request = _request(steps=1, members=1)
    path = tmp_path / "structural-drift.zarr"
    _write_cube(path, request, mutation=mutation)
    result = _sealed(request, path)
    with pytest.raises(ForecastContractError, match=match):
        open_canonical_forecast_cube(path, request, result)


def test_refuses_off_grid_or_implicitly_converted_regional_bounds(tmp_path):
    request = _request(steps=1, members=1)
    path = tmp_path / "cube.zarr"
    _write_cube(path, request)
    result = _sealed(request, path)
    cube = open_canonical_forecast_cube(path, request, result)
    try:
        with pytest.raises(ForecastContractError, match="implicit conversion"):
            cube.crop(GeographicBounds(-47, -34, 166, 179.75, "-180..180"))
        with pytest.raises(ForecastContractError, match="not exactly present"):
            cube.crop(GeographicBounds(-47.1, -34, 166, 179.75, "0..360"))
    finally:
        cube.close()


def test_artifact_tampering_is_detected_before_xarray_open(monkeypatch, tmp_path):
    request = _request(steps=1, members=1)
    path = tmp_path / "cube.zarr"
    _write_cube(path, request)
    result = _sealed(request, path)
    zarr.open_group(path, mode="r+")["t850"][0, 0, 0, 0, 0] = 99.0
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("xarray must not open unauthenticated bytes")

    monkeypatch.setattr(xr, "open_zarr", forbidden)
    with pytest.raises(ForecastContractError, match="artifact SHA-256 mismatch"):
        open_canonical_forecast_cube(path, request, result)
    assert called is False
