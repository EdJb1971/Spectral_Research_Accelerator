"""Acceptance tests for T5.6d lazy ERA5/CDS matched-truth construction."""

from __future__ import annotations

import copy

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from src.forecasting import (
    FRESH_HOLDOUT_ROLE,
    INITIAL_STATE_SCHEMA,
    MATCHED_TRUTH_BUILDER_SCHEMA,
    MATCHED_TRUTH_SCHEMA,
    PUBLISHED_OVERLAP_ROLE,
    ForecastContractError,
    RegionalForecastCube,
    build_matched_truth,
    evaluate_regional_ensemble,
    grid_coordinates_sha256,
)


SHA = "a" * 64


def _fixture(*, start="2020-01-01T00:00:00", variables=("t850", "q850")):
    times = np.datetime64(start, "ns") + np.arange(7) * np.timedelta64(6, "h")
    init = times[[0, 2]]
    leads = np.asarray([6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
    lat = np.asarray([-45.0, -44.75])
    lon = np.asarray([170.0, 170.25, 170.5])
    ensemble = np.asarray([101, 202], dtype="int64")
    units = {"t": "K", "q": "kg kg-1"}
    arrays = {}
    for offset, name in enumerate(("t", "q")):
        values = (np.arange(len(times) * len(lat) * len(lon), dtype="float32")
                  .reshape(len(times), 1, len(lat), len(lon)) + offset * 1000)
        arrays[name] = xr.DataArray(
            da.from_array(values, chunks=(2, 1, 1, 3)),
            dims=("time", "level", "latitude", "longitude"),
            attrs={"units": units[name]},
        )
    source = xr.Dataset(arrays, coords={
        "time": times, "level": [850], "latitude": lat, "longitude": lon,
    })
    forecast_arrays = {}
    for name in variables:
        unit = "K" if name == "t850" else "kg kg-1"
        forecast_arrays[name] = xr.DataArray(
            da.zeros((2, 2, 2, 2, 3), chunks=(1, 2, 1, 1, 3), dtype="float32"),
            dims=("time", "ensemble", "lead_time", "lat", "lon"), attrs={"units": unit})
    forecast_ds = xr.Dataset(forecast_arrays, coords={
        "time": init, "ensemble": ensemble, "lead_time": leads, "lat": lat, "lon": lon,
    })
    forecast = RegionalForecastCube(forecast_ds, {
        "schema": "regional-external-forecast-cube/fcn3-v1",
        "request_sha256": SHA, "result_sha256": SHA, "artifact_sha256": SHA,
        "validation_sha256": SHA, "grid_shape": [2, 3],
        "grid_coordinates_sha256": grid_coordinates_sha256(lat, lon),
    })
    manifest = {
        "content_hash": "b" * 32,
        "source_route": "Copernicus Climate Data Store API",
        "spec": {"dataset": "reanalysis-era5-pressure-levels", "level": [850]},
    }
    end = np.datetime_as_string(times[-1], unit="s")
    return source, manifest, forecast, end


def _build(source, manifest, forecast, end, **kwargs):
    return build_matched_truth(
        source, manifest, forecast, split="external_test",
        split_start=np.datetime_as_string(forecast.dataset.time.values[0], unit="s"), split_end=end,
        evaluation_role=FRESH_HOLDOUT_ROLE, **kwargs)


def test_builds_exact_lazy_initialization_and_valid_time_cubes():
    source, manifest, forecast, end = _fixture()
    built = _build(source, manifest, forecast, end)
    assert built.truth.t850.dims == ("time", "lead_time", "lat", "lon")
    assert built.initial_state.t850.dims == ("time", "lat", "lon")
    assert built.truth.t850.chunks is not None and built.initial_state.t850.chunks is not None
    expected_initial = source.t.isel(level=0, time=[0, 2]).values
    expected_truth = source.t.isel(
        level=0, time=xr.DataArray([[1, 2], [3, 4]], dims=("time", "lead_time"))).values
    assert np.array_equal(built.initial_state.t850.values, expected_initial)
    assert np.array_equal(built.truth.t850.values, expected_truth)
    assert built.truth.attrs["schema"] == MATCHED_TRUTH_SCHEMA
    assert built.initial_state.attrs["schema"] == INITIAL_STATE_SCHEMA
    assert built.provenance["schema"] == MATCHED_TRUTH_BUILDER_SCHEMA
    assert built.provenance["lazy"] is True


def test_content_addressed_receipt_is_deterministic_and_changes_with_manifest_or_selection():
    source, manifest, forecast, end = _fixture()
    first = _build(source, manifest, forecast, end)
    second = _build(source, copy.deepcopy(manifest), forecast, end)
    changed_manifest = copy.deepcopy(manifest)
    changed_manifest["content_hash"] = "c" * 32
    changed = _build(source, changed_manifest, forecast, end)
    assert first.fingerprint() == second.fingerprint()
    assert first.truth.attrs["source_sha256"] == second.truth.attrs["source_sha256"]
    assert first.fingerprint() != changed.fingerprint()
    assert first.truth.attrs["builder_sha256"] == first.provenance["builder_sha256"]
    assert len(first.provenance["valid_times_sha256"]) == 64


def test_outputs_enter_matched_evaluator_without_adapter_or_eager_copy():
    source, manifest, forecast, end = _fixture(variables=("t850",))
    built = _build(source, manifest, forecast, end)
    result = evaluate_regional_ensemble(forecast, built.truth, built.initial_state,
                                        max_tile_bytes=256)
    assert result.initialization_count == 2
    assert result.truth_provenance["source_sha256"] == built.truth.attrs["source_sha256"]
    assert result.truth_provenance["builder_sha256"] == built.provenance["builder_sha256"]
    assert result.metrics["6.0"]["t850"]["persistence"]["rmse"] > 0


def test_published_partitions_require_an_explicit_diagnostic_role():
    source, manifest, forecast, end = _fixture(start="2018-01-01T00:00:00")
    with pytest.raises(ForecastContractError, match="fresh_post_2019_holdout"):
        _build(source, manifest, forecast, end)
    built = build_matched_truth(
        source, manifest, forecast, split="published_evaluation",
        split_start="2018-01-01T00:00:00", split_end=end,
        evaluation_role=PUBLISHED_OVERLAP_ROLE)
    assert built.provenance["fcn3_periods"] == ["published_evaluation_2018_2019"]
    assert built.truth.attrs["evaluation_role"] == PUBLISHED_OVERLAP_ROLE


def test_refuses_calling_post_2019_data_a_published_partition_diagnostic():
    source, manifest, forecast, end = _fixture()
    with pytest.raises(ForecastContractError, match="requires at least one"):
        build_matched_truth(
            source, manifest, forecast, split="test",
            split_start="2020-01-01", split_end=end,
            evaluation_role=PUBLISHED_OVERLAP_ROLE)


@pytest.mark.parametrize("mutation,match", [
    (lambda source, forecast: source.assign_coords(time=source.time.values + np.timedelta64(1, "h")),
     "absent from the ERA5 source"),
    (lambda source, forecast: source.assign_coords(longitude=[170.0, 170.25, 170.75]),
     "exactly match"),
    (lambda source, forecast: source.assign_coords(level=[700]), "exactly one 850"),
    (lambda source, forecast: source.assign_coords(time=np.asarray([
        source.time.values[0], source.time.values[1], source.time.values[1],
        *source.time.values[3:]])), "unique and strictly increasing"),
])
def test_refuses_missing_duplicate_or_semantically_different_coordinates(mutation, match):
    source, manifest, forecast, end = _fixture()
    source = mutation(source, forecast)
    with pytest.raises(ForecastContractError, match=match):
        _build(source, manifest, forecast, end)


@pytest.mark.parametrize("mutation,match", [
    (lambda source, manifest: source.t.attrs.update(units="degC"), "units must be 'K'"),
    (lambda source, manifest: manifest.update(content_hash="not-a-hash"), "content_hash"),
    (lambda source, manifest: source.__setitem__("temperature", source.t), "exactly one alias"),
])
def test_refuses_unit_conversion_anonymous_content_or_ambiguous_aliases(mutation, match):
    source, manifest, forecast, end = _fixture()
    mutation(source, manifest)
    with pytest.raises(ForecastContractError, match=match):
        _build(source, manifest, forecast, end)


def test_split_guard_covers_valid_times_not_only_initializations():
    source, manifest, forecast, _end = _fixture()
    with pytest.raises(ForecastContractError, match="every initialization and valid time"):
        build_matched_truth(
            source, manifest, forecast, split="test",
            split_start="2020-01-01T00:00:00", split_end="2020-01-01T12:00:00",
            evaluation_role=FRESH_HOLDOUT_ROLE)


def test_refuses_unsupported_pre_1980_period_even_when_held_out():
    source, manifest, forecast, end = _fixture(start="1979-01-01T00:00:00")
    with pytest.raises(ForecastContractError, match="before FCN3's published 1980"):
        build_matched_truth(
            source, manifest, forecast, split="historical",
            split_start="1979-01-01", split_end=end,
            evaluation_role=PUBLISHED_OVERLAP_ROLE)
