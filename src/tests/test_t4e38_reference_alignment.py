import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmarks.reference_alignment import (
    DEFAULT_DECLARATION,
    acquire_reference_alignment,
    basin_walk,
    join_mslp_segments,
    plan_reference_alignment,
    reference_alignment_decision,
)
from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import DataSourceError, InvalidParameterError
from src.data_layer.cds_single_level import (
    CDSSingleLevelExactRequest,
    SINGLE_LEVEL_DATASET,
    acquire_single_level_shards,
    estimate_single_level_storage,
    plan_exact_timestamp_shards,
)

xr = pytest.importorskip("xarray")


def test_t4e38_plan_binds_sources_and_exactly_36_offline_timestamp_shards():
    plan = plan_reference_alignment()
    declaration = json.loads(DEFAULT_DECLARATION.read_text(encoding="utf-8"))

    assert plan["status"] == "DRAFTED_NOT_ADOPTED"
    assert len(plan["timestamps"]) == len(set(plan["timestamps"])) == 18
    assert [segment["name"] for segment in plan["segments"]] == ["parent", "complement"]
    assert [len(segment["shards"]) for segment in plan["segments"]] == [18, 18]
    assert plan["total_shards"] == declaration["single_level_request"]["expected_total_shards"] == 36
    assert plan["total_raw_value_bytes"] == 3_732_624
    assert plan["record_opened"] is False and plan["network_used"] is False
    assert plan["declaration_sha256"] == hashlib.sha256(DEFAULT_DECLARATION.read_bytes()).hexdigest()


def test_each_shard_requests_one_time_and_never_invents_a_pressure_level():
    plan = plan_reference_alignment()
    for segment in plan["segments"]:
        for timestamp, shard in zip(plan["timestamps"], segment["shards"]):
            request = shard["request"]
            assert request["variable"] == ["mean_sea_level_pressure"]
            assert len(request["year"]) == len(request["month"]) == 1
            assert len(request["day"]) == len(request["time"]) == 1
            assert "pressure_level" not in request
            assert shard["timestamp"] == timestamp


def test_single_level_request_refuses_wrong_product_variable_time_and_geometry():
    good = dict(
        variables=("msl",), timestamps=("2020-01-01 00:00:00",),
        lat_min=-2.0, lat_max=-1.0, lon_min=179.0, lon_max=180.0)
    with pytest.raises(InvalidParameterError, match="single-level product"):
        CDSSingleLevelExactRequest(**good, dataset="reanalysis-era5-pressure-levels")
    with pytest.raises(InvalidParameterError, match="canonical variable"):
        CDSSingleLevelExactRequest(**{**good, "variables": ("vo",)})
    with pytest.raises(InvalidParameterError, match="whole UTC hour"):
        CDSSingleLevelExactRequest(**{**good, "timestamps": ("2020-01-01 00:30:00",)})
    with pytest.raises(InvalidParameterError, match="split a dateline crossing"):
        CDSSingleLevelExactRequest(**{**good, "lon_min": 140.0, "lon_max": -140.0})


def test_storage_estimate_counts_only_the_exact_frames_without_compression_credit():
    spec = CDSSingleLevelExactRequest(
        variables=("msl",),
        timestamps=("2020-01-01 00:00:00", "2020-01-01 06:00:00"),
        lat_min=-2.0, lat_max=-1.0, lon_min=179.0, lon_max=180.0,
        grid_degrees=0.5)
    estimate = estimate_single_level_storage(spec)
    assert estimate["frames"] == estimate["shards"] == 2
    assert estimate["raw_value_bytes"] == 2 * 3 * 3 * 4
    assert estimate["compression_credit_assumed"] is False


def test_adoption_and_both_network_gates_stop_before_a_client_call(tmp_path, monkeypatch):
    declaration = tmp_path / DEFAULT_DECLARATION.name
    declaration.write_bytes(DEFAULT_DECLARATION.read_bytes())
    calls = []

    class Client:
        def retrieve(self, *args):
            calls.append(args)

    with pytest.raises(DataSourceError, match="has not been adopted"):
        acquire_reference_alignment(
            declaration, download_root=tmp_path / "downloads",
            explicit_network_authorisation=True, client=Client())

    sign_declaration(
        tmp_path, declaration.name, adopted_by="Test Human Reviewer",
        adopted_as="ADOPTED_FOR_TEST",
        what_was_adopted="the copied T4E.38 declaration",
        affirmation=REQUIRED_AFFIRMATION)
    with pytest.raises(DataSourceError, match="experiment-specific"):
        acquire_reference_alignment(
            declaration, download_root=tmp_path / "downloads", client=Client())
    monkeypatch.delenv("SPECTRALEARTH_ALLOW_NETWORK", raising=False)
    with pytest.raises(DataSourceError, match="network-disabled"):
        acquire_reference_alignment(
            declaration, download_root=tmp_path / "downloads",
            explicit_network_authorisation=True, client=Client())
    assert calls == []


def test_one_exact_shard_is_atomic_content_checked_and_resumable(tmp_path, monkeypatch):
    spec = CDSSingleLevelExactRequest(
        variables=("msl",), timestamps=("2020-01-01 00:00:00",),
        lat_min=-2.0, lat_max=-1.0, lon_min=179.0, lon_max=180.0,
        grid_degrees=0.5)
    calls = []

    class Client:
        def retrieve(self, dataset, request, target):
            calls.append((dataset, request))
            xr.Dataset(
                {"msl": (("time", "latitude", "longitude"),
                         np.ones((1, 3, 3), dtype=np.float32) * 100_000.0)},
                coords={"time": [np.datetime64("2020-01-01T00:00:00")],
                        "latitude": [-1.0, -1.5, -2.0],
                        "longitude": [179.0, 179.5, 180.0]},
            ).to_netcdf(target, engine="scipy")

    monkeypatch.setenv("SPECTRALEARTH_ALLOW_NETWORK", "1")
    monkeypatch.setattr(
        "src.data_layer.cds_single_level.shutil.disk_usage",
        lambda path: shutil_usage(total=20 * 1024**3, used=0, free=20 * 1024**3))
    first = acquire_single_level_shards(
        spec, tmp_path / "download", explicit_network_authorisation=True, client=Client())
    second = acquire_single_level_shards(
        spec, tmp_path / "download", explicit_network_authorisation=True, client=Client())

    assert calls[0][0] == SINGLE_LEVEL_DATASET and len(calls) == 1
    assert first["run"]["downloaded_shards"] == 1
    assert second["run"]["resumed_shards"] == 1
    shard = plan_exact_timestamp_shards(spec)[0]
    recorded = second["completed_shards"][shard.filename]
    assert recorded["request_sha256"] == shard.request_sha256
    assert recorded["bytes"] > 0 and len(recorded["sha256"]) == 64


def shutil_usage(*, total, used, free):
    return type("usage", (), {"total": total, "used": used, "free": free})()


def _segment_dataset(spec, timestamp, values):
    return xr.Dataset(
        {"msl": (("valid_time", "latitude", "longitude"), values[None, :, :])},
        coords={
            "valid_time": [np.datetime64(timestamp.replace(" ", "T"))],
            "latitude": np.linspace(spec.lat_max, spec.lat_min, values.shape[0]),
            "longitude": np.linspace(spec.lon_min, spec.lon_max, values.shape[1]),
        })


def test_exact_pair_materialisation_measures_and_deduplicates_the_seam():
    plan = plan_reference_alignment()
    specs = []
    for segment in plan["segments"]:
        request = segment["request"]
        specs.append(CDSSingleLevelExactRequest(
            variables=tuple(request["variables"]), timestamps=tuple(request["timestamps"]),
            lat_min=request["lat_min"], lat_max=request["lat_max"],
            lon_min=request["lon_min"], lon_max=request["lon_max"],
            grid_degrees=request["grid_degrees"]))
    stamp = plan["timestamps"][0]
    left_values = (100_000 + np.arange(161 * 161).reshape(161, 161)).astype(np.float32)
    right_values = (200_000 + np.arange(161 * 161).reshape(161, 161)).astype(np.float32)
    right_values[:, 0] = left_values[:, -1]
    joined, seam = join_mslp_segments(
        _segment_dataset(specs[0], stamp, left_values),
        _segment_dataset(specs[1], stamp, right_values), specs[0], specs[1], stamp)

    assert dict(joined.sizes) == {"time": 1, "latitude": 161, "longitude": 321}
    assert joined.longitude.values[160] == 180.0
    assert seam["passed"] is True and seam["maximum_absolute_difference"] == 0.0
    assert seam["complement_duplicate_removed"] is True


def test_exact_pair_materialisation_refuses_a_seam_over_the_measured_step():
    plan = plan_reference_alignment()
    specs = []
    for segment in plan["segments"]:
        request = segment["request"]
        specs.append(CDSSingleLevelExactRequest(
            variables=tuple(request["variables"]), timestamps=tuple(request["timestamps"]),
            lat_min=request["lat_min"], lat_max=request["lat_max"],
            lon_min=request["lon_min"], lon_max=request["lon_max"],
            grid_degrees=request["grid_degrees"]))
    stamp = plan["timestamps"][0]
    left = (100_000 + np.arange(161 * 161).reshape(161, 161)).astype(np.float32)
    right = left.copy()
    right[:, 0] = left[:, -1] + 1.0
    with pytest.raises(DataSourceError, match="seam mismatch"):
        join_mslp_segments(
            _segment_dataset(specs[0], stamp, left),
            _segment_dataset(specs[1], stamp, right), specs[0], specs[1], stamp)


def test_basin_walk_uses_catalogue_distance_then_lower_indices_for_ties():
    latitudes = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    longitudes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = np.full((5, 5), 10.0)
    values[1, 2] = values[2, 1] = 0.0
    walk = basin_walk(
        values, latitudes, longitudes, (0.0, 0.0), direction="minimum")
    assert walk["status"] == "CENTRE_FOUND"
    assert walk["seed_grid_index"] == [2, 2]
    assert walk["centre"]["grid_index"] == [1, 2]
    assert walk["steps"] == 1


def test_basin_walk_refuses_when_the_declared_path_reaches_the_boundary():
    coordinates = np.arange(5, dtype=float)
    values = np.tile(np.arange(5, dtype=float), (5, 1))
    walk = basin_walk(
        values, coordinates, coordinates, (2.0, 2.0), direction="minimum")
    assert walk["status"] == "REFUSED_BOUNDARY"
    assert walk["centre"] is None


def test_reference_decision_uses_ten_of_all_eighteen_and_never_accepts():
    vertical = [{"classification": "VERTICAL_QUANTITY_SEPARATION_CANDIDATE"}] * 10
    other = [{"classification": "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE"}] * 7
    refused = [{"classification": "REFUSED"}]
    decision = reference_alignment_decision(vertical + other + refused)
    assert decision["outcome"] == "VERTICAL_QUANTITY_SEPARATION_DOMINANT"
    assert decision["population_denominator"] == 18
    assert decision["is_acceptance_verdict"] is False
