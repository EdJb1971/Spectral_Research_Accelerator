"""Acceptance tests for T5.6e reproducible external-evaluation orchestration."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest
import xarray as xr

import src.forecasting.evaluation_run as run_module
from src.data_layer.zarr_source import CropSpec, cache_path, manifest_path
from src.forecasting import (
    EVALUATION_RUN_CONFIG_SCHEMA,
    EVALUATION_RUN_RECEIPT_SCHEMA,
    EvaluationRunConfig,
    ForecastContractError,
    GeographicBounds,
    load_evaluation_run_receipt,
    run_external_evaluation,
)
from src.tests.test_external_forecast_cube import _request, _sealed, _write_cube


def _config(**overrides):
    values = dict(
        schema=EVALUATION_RUN_CONFIG_SCHEMA,
        bounds=GeographicBounds(
            south=-45.0, north=-44.75, west=170.0, east=170.25,
            longitude_convention="0..360", label="NZ orchestration fixture"),
        split="independent_test",
        split_start="2020-02-11T00:00:00",
        split_end="2020-02-11T06:00:00",
        evaluation_role="fresh_post_2019_holdout",
        max_forecast_chunk_bytes=2_000_000,
        max_evaluation_tile_bytes=256,
    )
    values.update(overrides)
    return EvaluationRunConfig(**values)


def _cached_era5(cache_dir):
    crop = CropSpec(
        store="fixture-era5", variables=("t",),
        time_start="2020-02-11T00:00:00", time_end="2020-02-11T06:00:00",
        lat_min=-45.0, lat_max=-44.75, lon_min=170.0, lon_max=170.25,
        levels=(850,), n_levels_analysis=1)
    times = np.asarray(["2020-02-11T00:00:00", "2020-02-11T06:00:00"],
                       dtype="datetime64[ns]")
    source = xr.Dataset(
        {"t": xr.DataArray(
            np.asarray([[[[0.0, 0.5], [1.0, 1.5]]],
                        [[[2.0, 2.5], [3.0, 3.5]]]], dtype="float32"),
            dims=("time", "level", "latitude", "longitude"), attrs={"units": "K"})},
        coords={"time": times, "level": [850], "latitude": [-44.75, -45.0],
                "longitude": [170.0, 170.25]})
    path = cache_path(crop, str(cache_dir))
    source.chunk({"time": 1, "level": 1, "latitude": 2, "longitude": 2}).to_zarr(
        path, mode="w", consolidated=True)
    manifest = {
        "content_key": crop.content_key(), "cache_path": path,
        "spec": crop.to_provenance(), "content_hash": "b" * 32,
        "source_route": "synthetic accepted ERA5 fixture",
    }
    with open(manifest_path(crop, str(cache_dir)), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    return crop


def _inputs(tmp_path):
    request = _request(variables=("t850",), steps=1, members=2)
    artifact = tmp_path / "forecast.zarr"
    _write_cube(artifact, request)
    result = _sealed(request, artifact)
    cache_dir = tmp_path / "era5-cache"
    crop = _cached_era5(cache_dir)
    return request, result, artifact, crop, cache_dir


def test_runs_complete_offline_pipeline_and_writes_verifiable_atomic_receipt(tmp_path):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    output = tmp_path / "receipts" / "evaluation.json"
    receipt = run_external_evaluation(
        forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
        era5_crop=crop, era5_cache_dir=cache_dir, config=_config(), receipt_path=output)

    saved = load_evaluation_run_receipt(output)
    assert receipt.schema == EVALUATION_RUN_RECEIPT_SCHEMA
    assert saved["receipt_sha256"] == receipt.fingerprint()
    assert saved["run_request_sha256"] == receipt.run_request_sha256
    assert saved["forecast_validation"]["artifact_sha256"] == result.artifact_sha256
    assert saved["matched_truth_provenance"]["evaluation_role"] == "fresh_post_2019_holdout"
    assert saved["evaluation_sha256"] == receipt.evaluation_sha256
    assert saved["evaluation"]["initialization_count"] == 1
    assert saved["evaluation"]["metrics"]["6.0"]["t850"]["persistence"]["rmse"] > 0
    assert not list(output.parent.glob("*.tmp"))


def test_refuses_overwrite_before_reopening_or_recomputing(tmp_path, monkeypatch):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("keep me", encoding="utf-8")
    monkeypatch.setattr(run_module, "open_canonical_forecast_cube",
                        lambda *args, **kwargs: pytest.fail("forecast must not open"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_external_evaluation(
            forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
            era5_crop=crop, era5_cache_dir=cache_dir, config=_config(), receipt_path=output)
    assert output.read_text(encoding="utf-8") == "keep me"


def test_tampered_saved_receipt_is_detected(tmp_path):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    output = tmp_path / "evaluation.json"
    run_external_evaluation(
        forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
        era5_crop=crop, era5_cache_dir=cache_dir, config=_config(), receipt_path=output)
    record = json.loads(output.read_text(encoding="utf-8"))
    record["evaluation"]["initialization_count"] = 999
    output.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ForecastContractError, match="SHA-256 mismatch"):
        load_evaluation_run_receipt(output)

    # Re-hashing the outer envelope cannot hide an inconsistent embedded result identity.
    record["receipt_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in record.items() if key != "receipt_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()
    output.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ForecastContractError, match="embedded ensemble evaluation"):
        load_evaluation_run_receipt(output)


def test_source_and_global_handles_close_when_matching_or_evaluation_fails(tmp_path, monkeypatch):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    source_closed = {"value": False}
    global_closed = {"value": False}

    real_close = xr.Dataset.close

    def tracked_close(dataset):
        if "ensemble" in dataset.dims:
            global_closed["value"] = True
        elif "level" in dataset.dims:
            source_closed["value"] = True
        return real_close(dataset)

    monkeypatch.setattr(xr.Dataset, "close", tracked_close)
    monkeypatch.setattr(
        run_module, "evaluate_regional_ensemble",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")))
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_external_evaluation(
            forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
            era5_crop=crop, era5_cache_dir=cache_dir, config=_config(),
            receipt_path=tmp_path / "never-written.json")
    assert source_closed["value"] is True and global_closed["value"] is True
    assert not (tmp_path / "never-written.json").exists()


@pytest.mark.parametrize("change,match", [
    ({"schema": "wrong"}, "schema"),
    ({"split": "train"}, "held out"),
    ({"evaluation_role": "skill_claim"}, "evaluation_role"),
    ({"level_hpa": 700}, "850-hPa"),
    ({"max_evaluation_tile_bytes": 0}, "positive integer"),
])
def test_config_refuses_ambiguous_or_scientifically_invalid_controls(change, match):
    with pytest.raises(ForecastContractError, match=match):
        _config(**change)


def test_config_is_versioned_hashable_and_budget_sensitive():
    first = _config()
    same = _config()
    changed = _config(max_evaluation_tile_bytes=512)
    assert first == same and hash(first) == hash(same)
    assert first.fingerprint() == same.fingerprint()
    assert first.fingerprint() != changed.fingerprint()
