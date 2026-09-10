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
from src.data_layer.era5_overlap import (
    encoding_step,
    load_overlap_receipt,
    main as overlap_main,
    overlap_receipt_path,
    validate_overlap_evidence,
    verify_cached_era5_overlap,
)
from src.data_layer.regional_forecast import (
    RegionalForecastConfig,
    prepare_cached_regional_forecast,
)
from src.data_layer.zarr_source import (
    CropSpec,
    cache_path,
    load_cached,
    manifest_path,
    streaming_content_hash,
)

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


class FakeCDSClient:
    """Writes deterministic NetCDF responses without contacting Copernicus."""

    def __init__(self, quantum=None) -> None:
        self.calls = []
        # When set, values are rounded onto a binary lattice, which is what a route delivering
        # GRIB-packed fields does. The default leaves them unpacked.
        self.quantum = quantum

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
        units = {
            "temperature": "K", "specific_humidity": "kg kg**-1",
            "u_component_of_wind": "m s**-1", "v_component_of_wind": "m s**-1",
            "geopotential": "m**2 s**-2",
        }
        for channel, name in enumerate(request["variable"]):
            base = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            values = base * np.float32(1e-5) + np.float32(channel + 1)
            if self.quantum:
                # A packed field has to vary across many steps within a single frame, because
                # that is what makes its lattice observable. The unpacked default spans under
                # two steps in total, which determines no lattice and is rightly refused.
                values = (np.float32(100 * (channel + 1)) + base * np.float32(0.05))
                values = (np.round(values / np.float32(self.quantum))
                          * np.float32(self.quantum)).astype(np.float32)
            variables[name] = (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                values,
                {"units": units[name]},
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
    ({"lat_max": -45.1}, "latitude bounds/grid"),
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


def test_materialisation_refuses_server_snapped_grid_bounds(tmp_path):
    class ShiftedGridClient(FakeCDSClient):
        def retrieve(self, dataset, request, target):
            super().retrieve(dataset, request, target)
            with xr.open_dataset(target) as opened:
                shifted = opened.load().assign_coords(latitude=opened.latitude + 0.125)
            shifted.to_netcdf(target, mode="w", engine="h5netcdf")

    with pytest.raises(DataSourceError, match="bounds do not exactly match"):
        materialise_cds(
            _request(date_start="2020-01-01", date_end="2020-01-02"),
            download_dir=tmp_path / "downloads", cache_dir=str(tmp_path / "cache"),
            time_chunk=8, check_size=False, client=ShiftedGridClient(), allow_network=True)


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


def _write_weatherbench_overlap(spec, primary_cache, independent_cache, *, perturb=False,
                                jitter=0.0):
    primary, _ = load_cached(spec.to_crop_spec(), cache_dir=str(primary_cache))
    try:
        overlap = primary.isel(time=slice(0, 4)).load().copy(deep=True)
    finally:
        primary.close()
    if perturb:
        overlap["t"].values[0, 0, 0, 0] += np.float32(1.0)
    if jitter:
        # A uniform offset every value shares, standing in for the second archive having
        # decoded the same field through its own pipeline.
        overlap["t"].values[...] = overlap["t"].values + np.float32(jitter)
    independent = CropSpec(
        store="era5_0p25_6h", variables=spec.variables,
        time_start=str(overlap.time.values[0]), time_end=str(overlap.time.values[-1]),
        lat_min=spec.lat_min, lat_max=spec.lat_max,
        lon_min=spec.lon_min, lon_max=spec.lon_max,
        levels=spec.pressure_levels, n_levels_analysis=1)
    target = Path(cache_path(independent, str(independent_cache)))
    target.parent.mkdir(parents=True, exist_ok=True)
    overlap = overlap.chunk({"time": 2, "level": 1, "latitude": 5, "longitude": 5})
    encoding = {}
    for name in overlap.data_vars:
        overlap[name].encoding.pop("chunks", None)
        overlap[name].encoding.pop("preferred_chunks", None)
        encoding[name] = {"chunks": (2, 1, 5, 5)}
    overlap.to_zarr(target, mode="w", consolidated=True, encoding=encoding)
    with xr.open_zarr(target, consolidated=True) as opened:
        content_hash = streaming_content_hash(opened, time_block=2)
    Path(manifest_path(independent, str(independent_cache))).write_text(json.dumps({
        "content_key": independent.content_key(),
        "spec": independent.to_provenance(),
        "content_hash": content_hash,
        "shape": {key: int(value) for key, value in overlap.sizes.items()},
        "variables": sorted(overlap.data_vars),
        "cache_chunking": {"time": 2, "level": 1, "latitude": 5, "longitude": 5},
    }), encoding="utf-8")
    return independent


def test_overlap_receipt_is_bounded_content_bound_atomic_and_gate_verifiable(tmp_path, capsys):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(), allow_network=True)
    independent = _write_weatherbench_overlap(spec, primary_cache, independent_cache)
    receipt = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2)
    assert receipt["passed"] and receipt["coordinates_exact"]
    assert receipt["bounded_execution"] == {
        "network_used": False, "block_frames": 2,
        "maximum_frames_per_source_in_memory": 2, "full_overlap_loaded": False}
    assert all(record["mismatch_count"] == 0 for record in receipt["variables"].values())
    stored = load_overlap_receipt(overlap_receipt_path(spec.to_crop_spec(), str(primary_cache)))
    assert stored["receipt_sha256"] == receipt["receipt_sha256"]
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))
    validate_overlap_evidence(manifest, variable="t", level_hpa=850)

    # Re-entry recovers/attaches the same immutable evidence; it cannot publish another result.
    repeated = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2)
    assert repeated["receipt_sha256"] == receipt["receipt_sha256"]
    assert overlap_main([
        "--primary-manifest", manifest_path(spec.to_crop_spec(), str(primary_cache)),
        "--independent-manifest", manifest_path(independent, str(independent_cache)),
        "--variables", "t", "--level-hpa", "850", "--block-frames", "2",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["receipt_sha256"] == receipt["receipt_sha256"]
    manifest["independent_overlap_receipt"]["passed"] = False
    with pytest.raises(DataSourceError, match="missing or tampered"):
        validate_overlap_evidence(manifest, variable="t", level_hpa=850)


def test_failed_overlap_is_recorded_but_cannot_authorise_the_gate(tmp_path):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(), allow_network=True)
    independent = _write_weatherbench_overlap(
        spec, primary_cache, independent_cache, perturb=True)
    receipt = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2)
    assert not receipt["passed"]
    assert receipt["variables"]["t"]["mismatch_count"] == 1
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))
    assert manifest["independent_overlap_check"] == "FAIL"
    with pytest.raises(DataSourceError, match="not a recorded PASS"):
        validate_overlap_evidence(manifest, variable="t", level_hpa=850)


def test_a_record_is_admitted_only_under_the_criterion_that_judged_it(tmp_path):
    """D87. The gate refused the real record because the criterion was assumed, not named.

    A manifest carries one set of overlap fields per criterion that has judged it. Reading the
    unsuffixed fields regardless meant an encoding-relative PASS was invisible, so a record
    holding exactly the evidence its campaign designed for could not be admitted -- and,
    symmetrically, an absolute PASS would have admitted a record whose campaign declared a
    different rule.
    """
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=2.0 ** -10),
        allow_network=True)
    independent = _write_weatherbench_overlap(spec, primary_cache, independent_cache)
    verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2, criterion="encoding_relative", steps_allowed=1.0)
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))

    admitted = validate_overlap_evidence(
        manifest, variable="t", level_hpa=850, criterion="encoding_relative")
    assert admitted["passed"]
    assert admitted["criterion"]["name"] == "encoding_relative"
    # The absolute criterion never ran here, so it must not admit this record.
    assert manifest["independent_overlap_check"] == "NOT RUN"
    with pytest.raises(DataSourceError, match="not a recorded PASS under the absolute"):
        validate_overlap_evidence(manifest, variable="t", level_hpa=850)
    with pytest.raises(InvalidParameterError):
        validate_overlap_evidence(
            manifest, variable="t", level_hpa=850, criterion="whatever_passes")


def test_an_encoding_pass_cannot_be_relabelled_as_a_different_criterion(tmp_path):
    """The receipt's own declared name is checked, not just the manifest field it sits under."""
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=2.0 ** -10),
        allow_network=True)
    independent = _write_weatherbench_overlap(spec, primary_cache, independent_cache)
    verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2, criterion="encoding_relative", steps_allowed=1.0)
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))
    manifest["independent_overlap_receipt_encoding_relative"]["criterion"]["name"] = "absolute"
    with pytest.raises(DataSourceError, match="not produced under the encoding_relative"):
        validate_overlap_evidence(
            manifest, variable="t", level_hpa=850, criterion="encoding_relative")


def test_an_audit_window_cannot_become_the_authorisation(tmp_path):
    """T4C.5n. A second window compared after the fact is evidence, not authorisation.

    The frozen campaign names exactly one overlap window and that window is what admitted the
    record. A later window closes a real evidentiary gap, but if it could be read back as the
    authorising receipt then the evidence that admitted a record could be chosen after the
    record was already in hand -- the same failure D86 exists to prevent, arriving by a
    different door.
    """
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=2.0 ** -10),
        allow_network=True)
    independent = _write_weatherbench_overlap(spec, primary_cache, independent_cache)

    audit = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2, criterion="encoding_relative", steps_allowed=1.0,
        label="midrecord")
    assert audit["passed"] and audit["criterion"]["role"] == "audit"
    assert audit["criterion"]["label"] == "midrecord"
    assert audit["criterion"]["authorises"].startswith("nothing")

    # It lands in its own receipt, beside rather than over the authorising one.
    assert overlap_receipt_path(
        spec.to_crop_spec(), str(primary_cache), "encoding_relative", "midrecord").exists()
    assert not overlap_receipt_path(
        spec.to_crop_spec(), str(primary_cache), "encoding_relative").exists()

    # And the gate cannot read it: the audit's manifest fields carry the label, and the only
    # names `validate_overlap_evidence` will accept are the two criteria.
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))
    assert "independent_overlap_check_encoding_relative_midrecord" in manifest
    with pytest.raises(DataSourceError, match="not a recorded PASS"):
        validate_overlap_evidence(
            manifest, variable="t", level_hpa=850, criterion="encoding_relative")
    with pytest.raises(InvalidParameterError):
        validate_overlap_evidence(
            manifest, variable="t", level_hpa=850,
            criterion="encoding_relative_midrecord")


def test_a_label_that_could_pass_for_a_criterion_is_refused(tmp_path):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    for bad in ("absolute", "encoding_relative", "Mid Record", "mid.record", ""):
        with pytest.raises(InvalidParameterError):
            overlap_receipt_path(spec.to_crop_spec(), None, "encoding_relative", bad)


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


# ====================================================== D86: what "the routes agree" may mean
#
# ERA5 arrives through CDS packed per field, so the resolution at which that route can express
# an agreement is a property of the frame rather than a constant in Kelvin. These cover the
# criterion that says so, and the refusals that stop it being applied where it means nothing.

QUANTUM = 2.0 ** -10


def test_a_packed_frame_reveals_its_step_and_an_unpacked_one_refuses_to_pretend():
    packed = np.round(np.linspace(250.0, 260.0, 4096) / QUANTUM) * QUANTUM
    assert encoding_step(packed) == QUANTUM
    coarse = np.round(np.linspace(250.0, 260.0, 4096) / (2.0 ** -6)) * (2.0 ** -6)
    # The largest step every value sits on, not merely one they happen to be divisible by.
    assert encoding_step(coarse) == 2.0 ** -6
    # Values off any plausible binary lattice are not packed, and saying so is the point: a
    # step guessed for them would make the comparison judge nothing.
    assert encoding_step(np.linspace(250.0, 260.0, 4096) + 1e-9) is None
    assert encoding_step(np.asarray([250.0])) is None


def test_a_tolerance_finer_than_the_route_can_express_fails_a_pair_that_agrees(tmp_path):
    """This is D86 itself, reproduced: same numbers, two verdicts, and only one of them means
    anything. The routes differ by 0.4 of a packing step - agreement as close as the coarser
    of them can represent - and the absolute criterion still calls it a disagreement."""
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=QUANTUM),
        allow_network=True)
    independent = _write_weatherbench_overlap(
        spec, primary_cache, independent_cache, jitter=0.4 * QUANTUM)

    absolute = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2)
    assert absolute["passed"] is False
    assert absolute["variables"]["t"]["mismatch_count"] > 0
    # The declared tolerance is finer than one step, which is why no pair could satisfy it.
    assert absolute["variables"]["t"]["atol"] < QUANTUM

    relative = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2, criterion="encoding_relative")
    assert relative["passed"] is True
    assert relative["variables"]["t"]["mismatch_count"] == 0
    assert relative["variables"]["t"]["encoding_steps"] == [QUANTUM]
    assert relative["variables"]["t"]["max_error_in_steps"] == pytest.approx(0.4, abs=0.01)
    assert relative["criterion"]["steps_allowed"] == 1.0

    # Both verdicts survive. The failing one is not a mistake to be overwritten: it is what the
    # criterion in force at the time returned, and erasing it would erase why v3 exists.
    assert (overlap_receipt_path(spec.to_crop_spec(), str(primary_cache)).exists()
            and overlap_receipt_path(spec.to_crop_spec(), str(primary_cache),
                                     "encoding_relative").exists())
    assert absolute["receipt_sha256"] != relative["receipt_sha256"]
    manifest = json.loads(Path(manifest_path(
        spec.to_crop_spec(), str(primary_cache))).read_text(encoding="utf-8"))
    assert manifest["independent_overlap_check"] == "FAIL"
    assert manifest["independent_overlap_check_encoding_relative"] == "PASS"


def test_the_encoding_criterion_still_fails_a_pair_that_actually_disagrees(tmp_path):
    """A criterion that cannot fail is decoration. 1.4 steps is a real disagreement."""
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=QUANTUM),
        allow_network=True)
    independent = _write_weatherbench_overlap(
        spec, primary_cache, independent_cache, jitter=1.4 * QUANTUM)
    receipt = verify_cached_era5_overlap(
        spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
        independent_cache_dir=str(independent_cache), variables=("t",),
        level_hpa=850, block_frames=2, criterion="encoding_relative")
    assert receipt["passed"] is False
    assert receipt["variables"]["t"]["max_error_in_steps"] > 1.0


def test_an_unpacked_primary_is_refused_rather_than_judged_against_an_invented_step(tmp_path):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(), allow_network=True)
    independent = _write_weatherbench_overlap(spec, primary_cache, independent_cache)
    with pytest.raises(DataSourceError, match="does not lie on a binary lattice"):
        verify_cached_era5_overlap(
            spec.to_crop_spec(), independent, primary_cache_dir=str(primary_cache),
            independent_cache_dir=str(independent_cache), variables=("t",),
            level_hpa=850, block_frames=2, criterion="encoding_relative")


def test_the_two_criteria_cannot_be_confused_for_one_another(tmp_path):
    spec = _request(date_start="2020-01-01", date_end="2020-01-02")
    primary_cache, independent_cache = tmp_path / "primary", tmp_path / "independent"
    materialise_cds(
        spec, download_dir=tmp_path / "downloads", cache_dir=str(primary_cache),
        time_chunk=3, check_size=False, client=FakeCDSClient(quantum=QUANTUM),
        allow_network=True)
    independent = _write_weatherbench_overlap(
        spec, primary_cache, independent_cache, jitter=0.4 * QUANTUM)
    common = dict(primary_cache_dir=str(primary_cache),
                  independent_cache_dir=str(independent_cache), variables=("t",),
                  level_hpa=850, block_frames=2)
    with pytest.raises(InvalidParameterError, match="steps_allowed"):
        verify_cached_era5_overlap(spec.to_crop_spec(), independent,
                                   steps_allowed=1.0, **common)
    with pytest.raises(InvalidParameterError, match="absolute or encoding_relative"):
        verify_cached_era5_overlap(spec.to_crop_spec(), independent,
                                   criterion="whatever_passes", **common)
    # A receipt already published under one bound may not be re-read as another's.
    verify_cached_era5_overlap(spec.to_crop_spec(), independent,
                               criterion="encoding_relative", steps_allowed=1.0, **common)
    with pytest.raises(DataSourceError, match="different comparison design"):
        verify_cached_era5_overlap(spec.to_crop_spec(), independent,
                                   criterion="encoding_relative", steps_allowed=2.0, **common)


def test_the_lattice_search_survives_the_magnitudes_the_other_variables_live_at():
    """Geopotential runs to ~5e4 m2/s2, where float64 spacing is 7e-12. A fixed residual
    tolerance of 1e-12 would refuse a genuinely packed z field - a false refusal that would
    look exactly like an unpacked route."""
    for magnitude, step in ((300.0, 2.0 ** -10),       # temperature
                            (5.0e4, 2.0 ** -6),        # geopotential
                            (1.0e-3, 2.0 ** -24)):     # specific humidity
        packed = np.round(np.linspace(magnitude, magnitude * 1.02, 4096) / step) * step
        assert encoding_step(packed) == step, magnitude


# ---------------- T4E.18: relative vorticity, and the sign it must be read with
#
# The identity path cannot join an external cyclone catalogue to a record of 850 hPa
# temperature: features there sit a median 153.9 km from a catalogue centre against a catalogue
# radius of 15.2 km. A cyclone IS a compact extremum in relative vorticity, which is why the
# variable was added. What it is NOT is a maximum, in this hemisphere.


def test_relative_vorticity_can_be_requested_and_is_not_potential_vorticity():
    """`vorticity` and `potential_vorticity` are different CDS variables, and only one is meant."""
    from src.data_layer.cds_source import CDS_VARIABLES

    assert CDS_VARIABLES["vo"] == "vorticity"
    assert "potential_vorticity" not in CDS_VARIABLES.values()


def test_the_southern_hemisphere_sign_convention_is_recorded_where_a_reader_will_meet_it():
    """A maximum-finder on raw vorticity in a southern crop locates ANTICYCLONES.

    The convention that repairs it -- negate before extraction -- is a transformation this
    programme chose, not a property of the data. It is declared before the record exists so it
    cannot become a knob turned after a disappointing result, and it is written beside the
    variable so that whoever requests it next cannot miss it.
    """
    import json
    from pathlib import Path

    import src.data_layer.cds_source as cds

    source = Path("src/data_layer/cds_source.py").read_text(encoding="utf-8")
    marker = source[source.index('"z": "geopotential"'):source.index('"vo": "vorticity"')]
    assert "NEGATIVE vorticity" in marker
    assert "negate the" in marker

    design = json.loads(Path(
        "data/identity_calibration/t4e18-vorticity-acquisition-design.json"
    ).read_text(encoding="utf-8"))
    convention = design["the_sign_convention_is_declared_before_the_data_exists"]
    assert "locate ANTICYCLONES" in convention["the_problem"]
    assert "negated before extraction" in convention["the_convention"]
    assert "not a property of the data" in convention["why_this_is_declared_and_not_slipped_in"]
    assert cds.CDS_VARIABLES["vo"] == "vorticity"


def test_the_acquisition_window_stops_before_the_forecast_test_period():
    """Not acquiring 2022-2023 makes the reservation physical rather than a matter of policy."""
    import json
    from pathlib import Path

    design = json.loads(Path(
        "data/identity_calibration/t4e18-vorticity-acquisition-design.json"
    ).read_text(encoding="utf-8"))

    assert design["the_request"]["date_end"] == "2021-12-31"
    why = design["why_the_window_stops_at_2021"]
    assert "cannot be opened by accident" in why["the_reason"]
    assert "second request with its own queue time" in why["what_that_costs"]


def test_the_acquisition_declares_what_would_make_the_record_adequate():
    """The question T4E.17 never asked, asked in advance this time."""
    import json
    from pathlib import Path

    design = json.loads(Path(
        "data/identity_calibration/t4e18-vorticity-acquisition-design.json"
    ).read_text(encoding="utf-8"))
    acceptance = design["acceptance_for_the_acquired_record"]

    assert "was never checked for whether the record's variable could SEE" in (
        acceptance["why_acceptance_is_declared_before_the_data_arrives"])
    assert "153.9 km" in acceptance["condition_1_the_join_must_close"]
    assert "cardinality 3" in acceptance["condition_2_a_configuration_must_exist"]
    assert "as unusable as one yielding none" in (
        acceptance["condition_3_the_extractor_must_not_be_swamped"])
    assert "MSLP" in acceptance["what_failure_would_mean"]


def test_credentials_are_the_maintainers_and_are_never_recorded():
    """The layer builds a standard client; the key belongs to the maintainer and to no file here."""
    import json
    from pathlib import Path

    design = json.loads(Path(
        "data/identity_calibration/t4e18-vorticity-acquisition-design.json"
    ).read_text(encoding="utf-8"))
    prerequisites = design["operational_prerequisites_the_maintainer_must_supply"]

    assert "never to be pasted into this conversation" in prerequisites["credentials"]
    assert "allow_network" in prerequisites["network_consent"]
    assert not list(Path("data/identity_calibration").glob("*cdsapirc*"))


# ---------------- The vorticity probe: the variable works, the crop does not


def _t4e18_probe():
    import json
    from pathlib import Path

    return json.loads(Path(
        "measurements/t4e18_vorticity_probe.json").read_text(encoding="utf-8"))


def test_the_variable_was_confirmed_before_the_full_request_was_spent():
    """A cyclone is the most cyclonic 0.02 per cent of its frame, one cell from the catalogue."""
    works = _t4e18_probe()["the_variable_works"]

    assert works["verdict"].startswith("CONFIRMED")
    assert works["gita_2018_02_13_12"]["percentile_of_frame"] == 0.02
    assert works["gita_2018_02_13_12"]["distance_from_storm_to_frame_minimum_cells"] == 1
    assert "sign convention declared before acquisition is correct" in works["reading"]


def test_the_probe_failed_for_the_crop_and_the_record_says_which():
    """77 per cent of the cyclone population is north of the crop, and the rest is at its edge."""
    failure = _t4e18_probe()["but_the_join_still_fails_and_the_reason_is_the_crop"]

    assert failure["storms_with_a_feature_inside_their_catalogue_radius"] == "0 of 4"
    assert "707 lie NORTH of -20" in failure["the_crop_is_wrong_for_this_purpose"]
    assert "outside_valid_interior" in failure["the_cause"]


def test_the_trade_off_is_recorded_and_not_resolved():
    """Moving north buys 30 storms and costs the kind label. Picking the flattering box is the
    horse race every declaration in this sequence forbids.
    """
    alternative = _t4e18_probe()["the_alternative_and_its_cost"]

    boxes = {b["lat"]: b for b in alternative["candidate_boxes"]}
    assert boxes["-60..-20 (current)"]["nature_base_rate"] == 0.571
    assert boxes["-45..-5"]["nature_base_rate"] == 0.872
    assert boxes["-45..-5"]["storms"] == 30
    assert "close to degenerate" in alternative["THE_KIND_LABEL_DEGRADES"]
    assert "horse race" in alternative["this_is_a_decision_and_not_an_optimisation"]


def test_changing_the_crop_would_void_the_signature_already_given():
    """T4E.17 was signed on terms naming this crop and a base rate of 0.571."""
    probe = _t4e18_probe()

    assert "would have to be re-given" in probe["what_this_costs_the_signed_catalogue"]
    refused = " ".join(probe["what_was_NOT_done"])
    assert "full 2018-2021 acquisition was not run" in refused
    assert "No frame of the 2022-2023 forecast-test period" in refused


# ---------------- The acquired record, and the acceptance it failed


def _t4e18_acceptance():
    import json
    from pathlib import Path

    return json.loads(Path("measurements/t4e18_acceptance.json").read_text(encoding="utf-8"))


def test_the_acceptance_failed_on_conditions_declared_before_the_data_existed():
    """2 of 18 and 1 of 18 against a bar of 9. Declared in advance, so it is a result."""
    conditions = _t4e18_acceptance()["conditions_as_declared_before_the_data_existed"]

    assert "MET BY 2 OF 18" in conditions["condition_1"]
    assert "MET BY 1 OF 18" in conditions["condition_2"]
    assert conditions["condition_3"].startswith("features per frame usable -- MET")


def test_the_variable_is_cleared_and_so_are_the_crop_and_the_representation():
    """The signal is present, correctly signed, and nothing extracted it."""
    established = _t4e18_acceptance()["what_is_established_and_is_not_in_doubt"]

    assert "0.02nd percentile" in established["the_variable_carries_the_signal"]
    assert "sampling artefact" in established["the_crop_is_no_longer_the_problem"]
    assert "3 to 7 features per frame" in established["the_representation_is_not_the_problem_either"]


def test_the_diagnosis_names_the_extractors_calibration():
    """Phase randomisation preserves the spectrum, so the surrogate maxima match the field's."""
    points_at = _t4e18_acceptance()["what_this_points_at"]

    assert "306.74 K against a field maximum of 302.5 K" in points_at["the_extractor_s_calibration"]
    assert "second registered extractor" in points_at["the_extractor_s_declared_assumption"]
    assert "property of the instrument" in points_at["the_finding"]


def test_the_failure_does_not_condemn_the_variable_or_authorise_a_rewrite():
    """A second extractor is untested and would be its own declared work."""
    record = _t4e18_acceptance()

    refused = " ".join(record["what_this_does_NOT_establish"])
    assert "Not that relative vorticity is the wrong variable" in refused
    assert "Not that the acquisition was wasted" in refused
    assert "Not that a second extractor would succeed" in refused
    assert "neither is authorised by this measurement" in record["what_would_be_needed_next"]


# ---------------- The diagnosis was wrong, and the correction is part of the record


def test_the_first_diagnosis_was_wrong_and_is_superseded_not_edited_away():
    """It claimed the calibration rejects the cyclone. Measurement says it clears by six-fold."""
    correction = _t4e18_acceptance()["CORRECTION_2026_09_10"]

    assert "BOTH CLAIMS ARE WRONG" in correction["what_was_wrong"]
    assert "generalised to vorticity without measuring it" in correction["how_the_error_was_made"]
    assert "11 of 18 are ACCEPTED" in correction["the_calibration_actually_clears_comfortably"]


def test_four_of_five_surrogate_methods_have_no_power_against_a_frame_maximum():
    """They preserve the marginal, so the surrogate maximum equals the observation's.

    Worth pinning before any of them is proposed as a replacement null.
    """
    import numpy as np

    from src.statistics import surrogates

    rng = np.random.default_rng(3)
    field = rng.normal(0.0, 1.0, (48, 48))
    field[24, 24] += 30.0
    for method in ("aaft", "iaaft", "circular_shift"):
        member = np.asarray(surrogates.generate(field, method=method, n=1, seed=5)["members"][0])
        assert abs(float(member.max()) - float(field.max())) < 1e-9, method
    spectral = np.asarray(
        surrogates.generate(field, method="phase_randomise", n=1, seed=5)["members"][0])
    assert float(spectral.max()) != float(field.max())


def test_the_two_real_failure_modes_are_named_with_their_numbers():
    """The dateline edge, and a one-to-two-cell offset. Neither is the calibration."""
    modes = _t4e18_acceptance()["CORRECTION_2026_09_10"]["the_two_real_failure_modes"]

    assert "179.0 to 179.8" in modes["the_dateline_edge"]
    assert "refuses dateline-crossing requests by design" in modes["the_dateline_edge"]
    assert "factor of two to three, not the order of magnitude" in (
        modes["a_systematic_positional_offset"])


def test_raw_extraction_beats_the_swt_planes_for_this_purpose():
    """The reverse of what a four-storm comparison suggested, and measured across eighteen."""
    shows = _t4e18_acceptance()["CORRECTION_2026_09_10"]["what_the_measurement_actually_shows"]

    assert shows["raw_field_extraction"]["nearest_km_median"] == 52.1
    assert shows["swt_plane_extraction"]["nearest_km_median"] == 127.1
    assert "markedly BETTER" in shows["reading"]
    assert "Acceptance still fails either way" in shows["reading"]


# ---------------- T4E.19: the offset decomposed, with the declared negative result intact


def _t4e19():
    import json
    from pathlib import Path

    return json.loads(
        Path("measurements/t4e19_positional_error.json").read_text(encoding="utf-8"))


def test_the_declared_answer_is_that_the_causes_are_not_separated():
    """Condition 3 anticipated this and required it to be reported rather than resolved."""
    record = _t4e19()

    assert record["VERDICT"].startswith("NONE OF THE THREE DECLARED CAUSES IS ESTABLISHED")
    assert "rather than resolved by picking the most plausible" in record["VERDICT"]


def test_the_catalogue_uncertainty_cause_is_ruled_out_by_its_own_prediction():
    """It predicted the offset would track the agencies' disagreement. It does not.

    That was the outcome that would have needed no code at all, so ruling it out costs
    something and is worth stating precisely.
    """
    cause = _t4e19()["prediction_tests"]["cause_B_catalogue_uncertainty"]

    assert "+0.020" in cause["measured"]
    assert "RULED OUT" in cause["reading"]
    assert "not a case of comparing at the wrong tolerance" in cause["reading"]


def test_the_physical_cause_fails_its_sharpest_test():
    """Storm type. A transitioning system offsets like a tropical one to within 3.5 km."""
    cause = _t4e19()["prediction_tests"]["cause_C_a_real_physical_offset"]

    assert "ET 36.2 km" in cause["measured_by_storm_type"]
    assert "OPPOSITE SIGN to the prediction" in cause["measured_latitude"]
    assert cause["reading"].startswith("NOT SUPPORTED")


def test_the_surviving_cause_is_reported_as_too_weak_to_carry_the_explanation():
    """Sign only. About 2 per cent of the variance is not a cause."""
    cause = _t4e19()["prediction_tests"]["cause_A_estimator_bias"]

    assert "+0.143" in cause["measured"]
    assert "nowhere near enough to call it the cause" in cause["reading"]


def test_the_post_hoc_pattern_is_fenced_off_from_the_declared_result():
    """It was noticed in the data that would have to test it, so it is a hypothesis, not a finding."""
    post = _t4e19()["post_hoc_and_NOT_adjudicated"]

    assert "cannot be claimed from this measurement" in post["why_this_is_fenced_off"]
    assert "0.76 cells" in post["what_was_seen"]
    assert "Not a fixed coordinate shift" in post["what_it_is_not"]
    assert "none of it is authorised here" in post["what_would_test_it"]


def test_the_unit_of_independence_is_the_storm_not_the_observation():
    """154 observations, 16 storms, and every interval is leave-one-storm-out."""
    record = _t4e19()

    assert record["population"]["core_excluding_dateline"] == 154
    assert record["population"]["core_storms"] == 16
    assert "The unit of independence is the storm: 16, not 154" in (
        record["prediction_tests"]["note"])


# ---------------- T4E.20: the synthetic centre test, and the gate that failed by its own rule


def _t4e20():
    import json
    from pathlib import Path

    return json.loads(
        Path("measurements/t4e20_synthetic_centre.json").read_text(encoding="utf-8"))


def test_the_gate_failed_by_the_declared_criterion_and_the_code_said_otherwise():
    """The declaration disqualified a background yielding 'hundreds, OR NONE'. It yields none.

    The coded check tested only an upper bound. Recording the mismatch is the point: a validity
    gate whose code does not match its declaration is not a gate.
    """
    correction = _t4e20()["GATE_CORRECTION"]

    assert "never implemented the 'or none' half" in correction["what_happened"]
    assert "EASIER than the real one" in correction["why_it_matters"]
    assert "Toward the conclusion, not away from it" in correction["which_way_it_cuts"]


def test_the_directional_prediction_is_confirmed_monotonically():
    """45 degrees at symmetry, 19 at 2.5x stretch. Declared before the measurement."""
    results = _t4e20()["results"]["cause_A_directional_prediction_CONFIRMED"]

    assert "45.3 degrees" in results["stretch_1.0"]
    assert "18.9 degrees" in results["stretch_2.5"]
    assert "toward the broader side" in results["reading"]


def test_the_estimator_sizes_correctly_while_mislocating():
    """A centroid problem, not a scale one -- which rules out the diverged-scale defect."""
    scale = _t4e20()["results"]["scale_itself_is_recovered_accurately"]

    assert scale["planted_vs_recovered"]["9.00"] == 8.93
    assert "centroid problem, not a sizing one" in scale["reading"]


def test_the_quantitative_claim_is_refused_while_the_mechanism_is_kept():
    """10.2 km synthetic against 33.8 real, with a failed gate between them."""
    record = _t4e20()

    assert "DEMONSTRATED as a real mechanism" in record["VERDICT"]
    assert "NOT established that this accounts for all" in record["VERDICT"]
    refused = " ".join(record["what_this_does_not_settle"])
    assert "10.2 km against 33.8 observed" in refused
    assert "That the extractor should be changed" in refused
    assert "cannot see a fixed geographic bearing" in refused
