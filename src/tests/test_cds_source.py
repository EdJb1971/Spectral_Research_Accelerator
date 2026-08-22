"""Offline acceptance tests for T5.2c resumable CDS regional acquisition."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_source import (
    CDSRegionalRequest,
    acquire_cds_shards,
    estimate_cds_storage,
    materialise_cds,
    main,
    plan_monthly_shards,
    preflight_cds_storage,
    rematerialise_cds_from_provenance,
)
from src.data_layer.regional_forecast import (
    RegionalForecastConfig,
    prepare_cached_regional_forecast,
)
from src.data_layer.zarr_source import load_cached

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


class FakeCDSClient:
    """Writes deterministic NetCDF responses without contacting Copernicus."""

    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, request, target))
        times = [
            np.datetime64(datetime(int(request["year"][0]), int(request["month"][0]),
                                   int(day), int(hour[:2])), "ns")
            for day in request["day"] for hour in request["time"]
        ]
        levels = np.asarray([int(value) for value in request["pressure_level"]])
        north, west, south, east = [float(value) for value in request["area"]]
        grid = float(request["grid"][0])
        lat = np.arange(north, south - grid / 2, -grid)
        lon = np.arange(west, east + grid / 2, grid)
        shape = (len(times), len(levels), len(lat), len(lon))
        variables = {}
        for channel, name in enumerate(request["variable"]):
            base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            variables[name] = (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                base * np.float32(1e-5) + np.float32(channel + 1),
            )
        response = xr.Dataset(
            variables,
            coords={"valid_time": times, "pressure_level": levels,
                    "latitude": lat, "longitude": lon})
        response.to_netcdf(target, engine="h5netcdf")


def _request(**overrides):
    values = dict(
        variables=("t", "q", "u", "v", "z"),
        date_start="2020-01-01", date_end="2020-01-20",
        hours_utc=(0, 6, 12, 18),
        lat_min=-46.0, lat_max=-45.0, lon_min=170.0, lon_max=171.0,
        pressure_levels=(850,), n_levels_analysis=1)
    values.update(overrides)
    return CDSRegionalRequest(**values)


def test_monthly_plan_is_exact_hashable_and_uses_official_area_order():
    spec = _request(date_start="2020-01-30", date_end="2020-02-02")
    shards = plan_monthly_shards(spec)
    assert len(shards) == 2
    assert shards[0].days == (30, 31) and shards[1].days == (1, 2)
    assert shards[0].request["area"] == [-45.0, 170.0, -46.0, 171.0]
    assert shards[0].request["time"] == ["00:00", "06:00", "12:00", "18:00"]
    assert shards[0].request["pressure_level"] == ["850"]
    assert shards[0].request["variable"] == [
        "temperature", "specific_humidity", "u_component_of_wind",
        "v_component_of_wind", "geopotential"]
    assert plan_monthly_shards(spec) == shards
    assert len(spec.request_sha256()) == len(shards[0].request_sha256) == 64
    assert CDSRegionalRequest.from_provenance(spec.to_provenance()) == spec


@pytest.mark.parametrize("override,match", [
    ({"variables": ("t", "rain")}, "variables"),
    ({"hours_utc": (0, 24)}, "hours_utc"),
    ({"date_start": "2020-02-01", "date_end": "2020-01-01"}, "date_start/date_end"),
    ({"lon_min": 170.0, "lon_max": -170.0}, "longitude bounds"),
    ({"pressure_levels": (849,)}, "pressure_levels"),
])
def test_request_refuses_ambiguous_or_unsupported_selections(override, match):
    with pytest.raises(InvalidParameterError, match=match):
        _request(**override)


def test_acquisition_is_network_opt_in_even_with_an_injected_client(tmp_path):
    client = FakeCDSClient()
    with pytest.raises(DataSourceError, match="network-disabled"):
        acquire_cds_shards(_request(), tmp_path, client=client, allow_network=False)
    assert client.calls == []


def test_storage_preflight_is_conservative_and_combines_a_shared_volume(tmp_path, monkeypatch):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    estimate = estimate_cds_storage(spec)
    assert estimate["frames"] == 8
    assert estimate["compression_credit_assumed"] is False
    assert estimate["artifact_bytes_upper_bound"] > estimate["raw_value_bytes"] * 2
    monkeypatch.setattr(
        "src.data_layer.cds_source.shutil.disk_usage",
        lambda path: SimpleNamespace(total=10**12, used=0, free=10**12))
    report = preflight_cds_storage(
        spec, download_dir=tmp_path / "downloads", cache_dir=tmp_path / "cache",
        minimum_free_reserve_bytes=0)
    assert report["status"] == "READY"
    assert len(report["volumes"]) == 1
    assert report["volumes"][0]["roles"] == ["download", "cache"]
    assert report["volumes"][0]["working_bytes_required"] == (
        report["download_estimate"]["artifact_bytes_upper_bound"]
        + report["cache_estimate"]["artifact_bytes_upper_bound"])


def test_insufficient_storage_refuses_before_the_first_network_call(tmp_path, monkeypatch):
    client = FakeCDSClient()
    monkeypatch.setattr(
        "src.data_layer.cds_source.shutil.disk_usage",
        lambda path: SimpleNamespace(total=1, used=1, free=0))
    with pytest.raises(DataSourceError, match="storage preflight failed"):
        acquire_cds_shards(
            _request(), tmp_path / "downloads", client=client, allow_network=True)
    assert client.calls == []


def test_monthly_download_is_atomic_resumable_and_contains_no_credentials(tmp_path):
    client = FakeCDSClient()
    spec = _request(date_start="2020-01-30", date_end="2020-02-02")
    first = acquire_cds_shards(spec, tmp_path, client=client, allow_network=True)
    assert first["run"]["downloaded_shards"] == 2
    assert first["run"]["resumed_shards"] == 0
    assert len(client.calls) == 2
    assert not list(Path(tmp_path).glob("*.part"))

    second = acquire_cds_shards(spec, tmp_path, client=client, allow_network=True)
    assert second["run"]["downloaded_shards"] == 0
    assert second["run"]["resumed_shards"] == 2
    assert len(client.calls) == 2
    text = (Path(tmp_path) / "acquisition.json").read_text(encoding="utf-8").lower()
    assert "personal-access-token" not in text and '"key"' not in text
    assert json.loads(text)["request"]["credential_policy"].startswith("standard cds")


def test_resume_refuses_a_tampered_completed_shard(tmp_path):
    spec = _request()
    acquire_cds_shards(spec, tmp_path, client=FakeCDSClient(), allow_network=True)
    shard_path = Path(tmp_path) / plan_monthly_shards(spec)[0].filename
    with shard_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(DataSourceError, match="integrity verification"):
        acquire_cds_shards(spec, tmp_path, client=FakeCDSClient(), allow_network=True)


def test_cds_materialisation_enters_existing_lazy_dataset_interface(tmp_path):
    spec = _request()
    downloads, cache = tmp_path / "downloads", tmp_path / "cache"
    manifest = materialise_cds(
        spec, download_dir=downloads, cache_dir=str(cache), time_chunk=8,
        check_size=False, client=FakeCDSClient(), allow_network=True)
    assert manifest["source_route"] == "Copernicus Climate Data Store API"
    assert manifest["independent_overlap_check"] == "NOT RUN"
    assert manifest["shape"] == {
        "time": 80, "level": 1, "latitude": 5, "longitude": 5}
    assert manifest["cache_chunking"]["time"] == 8
    assert len(manifest["content_hash"]) == 32
    assert manifest["materialisation"] == {
        "strategy": "monthly shards appended in bounded time blocks",
        "maximum_source_frames_in_memory": 8,
        "requested_time_block_frames": 8,
        "full_record_loaded": False,
        "publication": "temporary sibling Zarr renamed only after complete validation",
        "content_hash": "logical variable-major stream; independent of Zarr chunking",
    }

    crop = spec.to_crop_spec()
    dataset, cached_manifest = load_cached(crop, cache_dir=str(cache))
    try:
        assert set(dataset.data_vars) == {"t", "q", "u", "v", "z"}
        assert np.array_equal(dataset.level.values, np.asarray([850]))
        assert cached_manifest["remote_bytes"] == 0
    finally:
        dataset.close()

    config = RegionalForecastConfig(
        variables=("t", "q", "u", "v", "z"), level_hpa=850,
        history_frames=2, lead_frames=(1, 2), embargo_frames=2,
        statistics_chunk_frames=8)
    with prepare_cached_regional_forecast(crop, config, cache_dir=str(cache)) as bundle:
        sample = bundle.train[0]
        assert sample["inputs"].shape == (2, 5, 5, 5)
        assert sample["targets"].shape == (2, 5, 5, 5)
        assert torch.isfinite(sample["inputs"]).all()
        assert bundle.provenance["source"]["source_route"].startswith("Copernicus")

    # A populated cache is entirely offline and does not re-enter CDS.
    repeat = materialise_cds(
        spec, download_dir=downloads, cache_dir=str(cache), time_chunk=8,
        check_size=False, client=None, allow_network=False)
    assert repeat["cache_hit"] is True and repeat["bytes_transferred"] == 0
    replay = rematerialise_cds_from_provenance(
        manifest, download_dir=downloads, cache_dir=str(cache), time_chunk=8,
        check_size=False, allow_network=False)
    assert replay["content_hash"] == manifest["content_hash"]


def test_materialisation_refuses_missing_requested_timestamp(tmp_path):
    class IncompleteClient(FakeCDSClient):
        def retrieve(self, dataset, request, target):
            broken = dict(request)
            broken["time"] = list(request["time"][:-1])
            super().retrieve(dataset, broken, target)

    with pytest.raises(DataSourceError, match="timestamps do not exactly match"):
        materialise_cds(
            _request(), download_dir=tmp_path / "downloads", cache_dir=str(tmp_path / "cache"),
            time_chunk=8, check_size=False, client=IncompleteClient(), allow_network=True)


def test_multimonth_materialisation_never_loads_a_full_shard(tmp_path, monkeypatch):
    """A real gate record is larger than RAM; boundedness must be executable evidence."""
    spec = _request(date_start="2020-01-30", date_end="2020-02-03")
    original_load = xr.Dataset.load
    loaded_frame_counts = []

    def guarded_load(dataset, *args, **kwargs):
        frames = int(dataset.sizes.get("time", 0))
        loaded_frame_counts.append(frames)
        if frames > 3:
            raise AssertionError("attempted to load more than the declared time block")
        return original_load(dataset, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", guarded_load)
    manifest = materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(tmp_path / "cache"),
        time_chunk=3, check_size=False, client=FakeCDSClient(), allow_network=True)
    assert len(plan_monthly_shards(spec)) == 2
    assert max(loaded_frame_counts) <= 3
    assert manifest["shape"]["time"] == 20
    assert manifest["materialisation"]["maximum_source_frames_in_memory"] == 3


def test_plan_cli_prints_exact_request_without_network(capsys):
    assert main([
        "plan", "--date-start", "2020-01-30", "--date-end", "2020-02-02",
        "--hours", "0,6,12,18", "--lat", "-46", "-45",
        "--lon", "170", "171", "--analysis-levels", "1",
    ]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["network_used"] is False
    assert len(result["monthly_shards"]) == 2
    assert result["request"]["hours_utc"] == [0, 6, 12, 18]
    assert result["storage_estimate"]["compression_credit_assumed"] is False
    assert "completed_shards" not in result
