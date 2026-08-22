"""Acceptance tests for T5.6c matched-truth regional ensemble evaluation."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from src.forecasting import (
    INITIAL_STATE_SCHEMA,
    MATCHED_TRUTH_SCHEMA,
    ForecastContractError,
    RegionalForecastCube,
    evaluate_regional_ensemble,
    grid_coordinates_sha256,
)


SHA = "a" * 64


def _fixture(*, variables=("t850",), forecast_values=None, truth_values=None,
             initial_values=None):
    coords = {
        "time": np.asarray(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        "ensemble": np.asarray([11, 22, 33], dtype="int64"),
        "lead_time": np.asarray([6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]"),
        "lat": np.asarray([-45.0, -35.0]),
        "lon": np.asarray([170.0, 170.25]),
    }
    unit = {"t850": "K", "q850": "kg kg-1"}
    forecast_arrays, truth_arrays, initial_arrays = {}, {}, {}
    default_forecast = np.empty((2, 3, 2, 2, 2), dtype="float32")
    default_forecast[:, :, 0, :, :] = np.asarray([1.0, 2.0, 3.0])[None, :, None, None]
    default_forecast[:, :, 1, :, :] = np.asarray([2.0, 3.0, 4.0])[None, :, None, None]
    default_truth = np.empty((2, 2, 2, 2), dtype="float32")
    default_truth[:, 0] = 2.0
    default_truth[:, 1] = 3.0
    default_initial = np.zeros((2, 2, 2), dtype="float32")
    for index, variable in enumerate(variables):
        scale = float(index + 1)
        predicted = default_forecast * scale if forecast_values is None else forecast_values
        observed = default_truth * scale if truth_values is None else truth_values
        initial = default_initial if initial_values is None else initial_values
        forecast_arrays[variable] = xr.DataArray(
            da.from_array(predicted, chunks=(1, 3, 1, 1, 2)),
            dims=("time", "ensemble", "lead_time", "lat", "lon"),
            attrs={"units": unit[variable]})
        truth_arrays[variable] = xr.DataArray(
            da.from_array(observed, chunks=(1, 1, 1, 2)),
            dims=("time", "lead_time", "lat", "lon"), attrs={"units": unit[variable]})
        initial_arrays[variable] = xr.DataArray(
            da.from_array(initial, chunks=(1, 1, 2)), dims=("time", "lat", "lon"),
            attrs={"units": unit[variable]})
    forecast_ds = xr.Dataset(forecast_arrays, coords=coords)
    grid_hash = grid_coordinates_sha256(coords["lat"], coords["lon"])
    regional = RegionalForecastCube(forecast_ds, {
        "schema": "regional-external-forecast-cube/fcn3-v1",
        "request_sha256": SHA, "result_sha256": SHA, "artifact_sha256": SHA,
        "validation_sha256": SHA, "grid_shape": [2, 2],
        "grid_coordinates_sha256": grid_hash,
    })
    common = {"source_sha256": SHA, "split": "test", "write_complete": "true"}
    truth = xr.Dataset(truth_arrays, coords={name: coords[name]
                       for name in ("time", "lead_time", "lat", "lon")},
                       attrs={"schema": MATCHED_TRUTH_SCHEMA, **common})
    initial = xr.Dataset(initial_arrays, coords={name: coords[name]
                         for name in ("time", "lat", "lon")},
                         attrs={"schema": INITIAL_STATE_SCHEMA, **common})
    return regional, truth, initial


def test_exact_member_mean_persistence_crps_spread_and_rank_diagnostics():
    forecast, truth, initial = _fixture()
    result = evaluate_regional_ensemble(forecast, truth, initial, max_tile_bytes=256)
    metric = result.metrics["6.0"]["t850"]
    assert metric["member_errors"]["11"]["rmse"] == pytest.approx(1.0)
    assert metric["member_errors"]["22"]["rmse"] == pytest.approx(0.0)
    assert metric["ensemble_mean"]["rmse"] == pytest.approx(0.0)
    assert metric["ensemble_mean"]["mse_skill_score_vs_persistence"] == pytest.approx(1.0)
    assert metric["persistence"]["rmse"] == pytest.approx(2.0)
    assert metric["crps"] == pytest.approx(2.0 / 9.0)
    assert metric["spread_rms"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert metric["spread_skill_ratio"] is None
    assert metric["rank_histogram"]["fractional_tie_counts"] == pytest.approx([0, 4, 4, 0])
    assert sum(metric["rank_histogram"]["area_weighted_frequency"]) == pytest.approx(1.0)
    assert metric["rank_histogram"]["diagnostic_only"] is True
    assert result.maximum_observed_tile_bytes <= 256
    assert result.claim_boundary.startswith("Deterministic matched-sample")


def test_keeps_units_separate_and_uses_cosine_latitude_area_weights():
    forecast, truth, initial = _fixture(variables=("t850", "q850"))
    # Give both longitudes/times the same latitude-dependent error so the analytic area-weighted
    # answer is independent of sample multiplicity: 1 K at -45 degrees and 3 K at -35 degrees.
    truth["t850"] = xr.zeros_like(truth.t850)
    truth.t850.attrs["units"] = "K"
    weighted_values = np.ones((2, 3, 2, 2, 2), dtype="float32")
    weighted_values[:, :, :, 1, :] = 3.0
    forecast.dataset["t850"] = xr.DataArray(
        da.from_array(weighted_values, chunks=(1, 3, 1, 1, 2)),
        dims=("time", "ensemble", "lead_time", "lat", "lon"), attrs={"units": "K"})
    result = evaluate_regional_ensemble(forecast, truth, initial)
    assert result.metrics["6.0"]["t850"]["unit"] == "K"
    assert result.metrics["6.0"]["q850"]["unit"] == "kg kg-1"
    assert result.metrics["6.0"]["q850"]["member_errors"]["11"]["rmse"] == pytest.approx(2.0)
    weights = np.cos(np.deg2rad([-45.0, -35.0]))
    expected_rmse = np.sqrt((weights[0] * 1.0 + weights[1] * 9.0) / weights.sum())
    assert result.metrics["6.0"]["t850"]["ensemble_mean"]["rmse"] == pytest.approx(expected_rmse)
    assert "aggregate" not in result.to_mapping()
    assert result.area_weighting.startswith("cos(latitude)")


def test_result_is_deterministic_content_bound_and_lazy_input_survives():
    forecast, truth, initial = _fixture()
    first = evaluate_regional_ensemble(forecast, truth, initial, max_tile_bytes=256)
    second = evaluate_regional_ensemble(forecast, truth, initial, max_tile_bytes=256)
    assert first.fingerprint() == second.fingerprint()
    assert first.forecast_values_sha256 == second.forecast_values_sha256
    assert forecast.dataset.t850.chunks is not None
    assert truth.t850.chunks is not None


@pytest.mark.parametrize("target,mutation,match", [
    ("truth", lambda ds: ds.assign_coords(lon=[170.0, 170.5]), "coordinate 'lon'"),
    ("truth", lambda ds: ds.assign_coords(lead_time=np.asarray([12, 18], dtype="timedelta64[h]")),
     "coordinate 'lead_time'"),
    ("initial", lambda ds: ds.assign_coords(time=np.asarray(
        ["2020-01-02", "2020-01-03"], dtype="datetime64[ns]")), "coordinate 'time'"),
])
def test_refuses_date_lead_or_grid_mismatch(target, mutation, match):
    forecast, truth, initial = _fixture()
    if target == "truth":
        truth = mutation(truth)
    else:
        initial = mutation(initial)
    with pytest.raises(ForecastContractError, match=match):
        evaluate_regional_ensemble(forecast, truth, initial)


@pytest.mark.parametrize("target,mutation,match", [
    ("truth", lambda ds: ds.attrs.update(split="train"), "held out"),
    ("truth", lambda ds: ds.attrs.update(write_complete="false"), "write_complete"),
    ("truth", lambda ds: ds.t850.attrs.update(units="degC"), "units must be 'K'"),
    ("initial", lambda ds: ds.attrs.update(source_sha256="b" * 64), "same verifying source"),
    ("forecast", lambda ds: ds.provenance.update(grid_coordinates_sha256="b" * 64),
     "coordinates differ"),
])
def test_refuses_unheld_provenance_incomplete_data_units_or_lineage(target, mutation, match):
    forecast, truth, initial = _fixture()
    mutation({"forecast": forecast, "truth": truth, "initial": initial}[target])
    with pytest.raises(ForecastContractError, match=match):
        evaluate_regional_ensemble(forecast, truth, initial)


def test_refuses_nonfinite_truth_without_silent_deletion():
    forecast, truth, initial = _fixture()
    truth.t850.data = truth.t850.data.map_blocks(
        lambda value: np.full_like(value, np.nan), dtype="float32")
    with pytest.raises(ForecastContractError, match="missing-value deletion is refused"):
        evaluate_regional_ensemble(forecast, truth, initial)


def test_refuses_single_member_ensemble_and_impossibly_small_budget():
    forecast, truth, initial = _fixture()
    one = RegionalForecastCube(forecast.dataset.isel(ensemble=slice(0, 1)), forecast.provenance)
    with pytest.raises(ForecastContractError, match="at least two members"):
        evaluate_regional_ensemble(one, truth, initial)
    with pytest.raises(ForecastContractError, match="cannot hold one grid cell"):
        evaluate_regional_ensemble(forecast, truth, initial, max_tile_bytes=8)


def test_undefined_persistence_skill_is_null_not_infinite():
    forecast, truth, initial = _fixture()
    initial["t850"] = truth.t850.isel(lead_time=0, drop=True)
    initial.t850.attrs["units"] = "K"
    result = evaluate_regional_ensemble(forecast, truth, initial)
    assert result.metrics["6.0"]["t850"]["ensemble_mean"][
        "mse_skill_score_vs_persistence"] is None
