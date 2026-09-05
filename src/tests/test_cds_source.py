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
