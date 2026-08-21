"""Acceptance tests for the T5.2 training-native regional forecast dataset."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.core.errors import InvalidParameterError
from src.data_layer.regional_forecast import (
    RegionalForecastConfig, assess_manifest_readiness, cross_check_era5_overlap,
    prepare_cached_regional_forecast, prepare_regional_forecast_datasets)
from src.data_layer.zarr_source import CropSpec, materialise

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


LONG_NAMES = {
    "t": "temperature", "q": "specific_humidity", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "z": "geopotential",
}


def _dataset(nt=50, ny=4, nx=5):
    times = np.arange("2020-01-01", "2020-02-01", np.timedelta64(6, "h"),
                      dtype="datetime64[ns]")[:nt]
    level = np.array([850, 500])
    lat = np.linspace(-34.0, -47.0, ny)
    lon = np.linspace(166.0, 179.0, nx)
    time_signal = np.arange(nt, dtype="float64")[:, None, None, None]
    lev_signal = np.arange(2, dtype="float64")[None, :, None, None]
    spatial = (np.arange(ny, dtype="float64")[None, None, :, None]
               + np.arange(nx, dtype="float64")[None, None, None, :] / 10)
    variables = {}
    for channel, (short, long_name) in enumerate(LONG_NAMES.items()):
        values = 10 * channel + time_signal + 100 * lev_signal + spatial
        variables[long_name] = (("time", "level", "latitude", "longitude"),
                                values.astype("float32"), {"units": "unit-%s" % short})
    return xr.Dataset(variables, coords={"time": times, "level": level,
                                         "latitude": lat, "longitude": lon})


def _manifest(dataset):
    digest = hashlib.sha256()
    for name in sorted(dataset.data_vars):
        digest.update(np.ascontiguousarray(dataset[name].values).tobytes())
    return {"content_hash": digest.hexdigest()[:32], "cache_chunking": {"time": 50},
            "spec": {"store": "independent-test-fixture", "levels": [850],
                     "variables": list(dataset.data_vars)}}


def _config(**overrides):
    values = dict(history_frames=3, lead_frames=(1, 2), embargo_frames=2,
                  train_ratio=0.6, val_ratio=0.2)
    values.update(overrides)
    return RegionalForecastConfig(**values)


def test_shapes_aliases_and_default_collation_are_training_native():
    source = _dataset()
    bundle = prepare_regional_forecast_datasets(source, _config(), _manifest(source))
    item = bundle.train[0]
    assert item["inputs"].shape == (3, 5, 4, 5)
    assert item["targets"].shape == (2, 5, 4, 5)
    batch = next(iter(DataLoader(bundle.train, batch_size=2, shuffle=False)))
    assert batch["inputs"].shape == (2, 3, 5, 4, 5)
    assert batch["targets"].shape == (2, 2, 5, 4, 5)
    assert bundle.provenance["variables"]["resolved_names"] == LONG_NAMES


def test_split_then_sample_means_no_history_or_target_crosses_a_boundary_or_embargo():
    source = _dataset()
    bundle = prepare_regional_forecast_datasets(source, _config(), _manifest(source))
    all_used = {}
    for name, split in bundle.datasets().items():
        used = set()
        for item in split:
            inputs = set(item["input_frame_indices"].tolist())
            targets = set(item["target_frame_indices"].tolist())
            assert inputs <= set(split.frame_indices)
            assert targets <= set(split.frame_indices)
            used |= inputs | targets
        all_used[name] = used
    assert all_used["train"].isdisjoint(all_used["val"])
    assert all_used["val"].isdisjoint(all_used["test"])
    assert all_used["train"].isdisjoint(all_used["test"])
    split_record = bundle.provenance["temporal_split"]
    assert "embargo_train_val" in split_record and "embargo_val_test" in split_record
    assert set(bundle.train.frame_indices).isdisjoint(
        set(range(split_record["embargo_train_val"]["first_frame_index"],
                  split_record["embargo_train_val"]["last_frame_index"] + 1)))


def test_normalisation_is_fit_on_training_frames_only_and_reused_unchanged():
    source = _dataset()
    # Make held-out fields absurdly large: a full-record fit would be unmistakable.
    source = source.copy(deep=True)
    for name in source.data_vars:
        source[name].values[30:] += 1_000_000
    bundle = prepare_regional_forecast_datasets(source, _config(), _manifest(source))
    raw = torch.from_numpy(source["temperature"].sel(level=850).values[:30])
    assert bundle.normalisation.mean[0] == pytest.approx(float(raw.double().mean()))
    assert bundle.normalisation.mean[0] < 100
    assert bundle.normalisation.fitted_split == "train"
    assert bundle.train.normalisation is bundle.val.normalisation is bundle.test.normalisation
    assert bundle.provenance["normalisation"]["ddof"] == 0
    assert len(bundle.provenance["normalisation"]["artifact_hash"]) == 32


def test_provenance_fingerprints_source_crop_grid_time_split_and_normalisation():
    source = _dataset()
    manifest = _manifest(source)
    bundle = prepare_regional_forecast_datasets(source, _config(), manifest)
    record = bundle.provenance
    assert record["source_content_hash"] == manifest["content_hash"]
    assert record["source"]["cache_chunking"] == {"time": 50}
    assert record["level_hpa"] == 850
    assert record["config"]["variables"] == ["t", "q", "u", "v", "z"]
    assert len(record["timestamps"]["sha256"]) == 32
    assert len(record["grid"]["sha256"]) == 32
    assert record["construction_order"].startswith("split and embargo")
    assert "CUDA, ROCm or MPS" in record["device_policy"]


def test_scientific_refusals_are_explicit():
    source = _dataset()
    with pytest.raises(InvalidParameterError, match="longest lead"):
        _config(embargo_frames=1)
    with pytest.raises(InvalidParameterError, match="content_hash"):
        prepare_regional_forecast_datasets(source, _config(), {})
    broken = source.drop_vars("specific_humidity")
    with pytest.raises(InvalidParameterError, match="specific_humidity"):
        prepare_regional_forecast_datasets(broken, _config(), _manifest(broken))
    broken = source.copy(deep=True)
    broken["temperature"].values[0, 0, 0, 0] = np.nan
    with pytest.raises(InvalidParameterError, match="non-finite"):
        prepare_regional_forecast_datasets(broken, _config(), _manifest(broken))


def test_independent_overlap_cross_check_requires_exact_coordinates_and_reports_errors():
    primary = _dataset(nt=8)
    independent = primary.copy(deep=True)
    report = cross_check_era5_overlap(primary, independent, _config())
    assert report["passed"] and report["coordinates_exact"]
    assert all(v["max_abs_error"] == 0 for v in report["variables"].values())
    independent["temperature"].values[0, 0, 0, 0] += 1
    assert not cross_check_era5_overlap(primary, independent, _config())["passed"]
    shifted = primary.assign_coords(longitude=primary.longitude + 0.01)
    with pytest.raises(ValueError, match="interpolation is not allowed"):
        cross_check_era5_overlap(primary, shifted, _config())


def test_content_addressed_cache_has_the_same_one_call_local_or_hpc_interface(tmp_path):
    source = _dataset(nt=50)
    store = tmp_path / "source.zarr"
    source.chunk({"time": 1, "level": 2, "latitude": 4, "longitude": 5}).to_zarr(
        store, mode="w", consolidated=True)
    cache = tmp_path / "shared_hpc_or_local_cache"
    spec = CropSpec(store=str(store), variables=tuple(LONG_NAMES.values()),
                    time_start="2020-01-01", time_end="2020-01-13",
                    lat_min=-47.1, lat_max=-33.9, lon_min=165.9, lon_max=179.1,
                    levels=(850,), n_levels_analysis=1)
    manifest = materialise(spec, cache_dir=str(cache), time_chunk=16, check_size=False)
    bundle = prepare_cached_regional_forecast(spec, _config(), cache_dir=str(cache))
    assert manifest["cache_chunking"]["time"] == 16
    assert bundle.provenance["source"]["remote_bytes"] == 0
    assert bundle.train[0]["inputs"].device.type == "cpu"


def test_manifest_readiness_never_claims_value_checks_from_metadata():
    source = _dataset()
    manifest = _manifest(source)
    manifest["shape"] = {"time": 50, "level": 1, "latitude": 4, "longitude": 5}
    manifest["variables"] = list(source.data_vars)
    ready = assess_manifest_readiness(manifest, _config())
    assert ready["structurally_eligible"]
    assert ready["dataset_prepared"] is False
    assert ready["train_only_normalisation_verified"] is False
    assert ready["independent_era5_crosscheck"] == "NOT RUN"
    missing = dict(manifest, variables=["temperature"])
    missing["spec"] = dict(missing["spec"], variables=["temperature"])
    assert assess_manifest_readiness(missing, _config())["missing_variables"] == ["q", "u", "v", "z"]
