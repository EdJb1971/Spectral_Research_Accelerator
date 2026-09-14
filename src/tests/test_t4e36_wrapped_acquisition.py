import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import DataSourceError
from src.data_layer.cds_source import (
    CDSRegionalRequest,
    acquire_cds_shards,
    estimate_cds_storage,
    plan_monthly_shards,
)
from src.data_layer.wrapped_record import (
    WRAPPED_RECORD_SCHEMA,
    acquire_wrapped_complement,
    join_wrapped_segments,
    materialise_wrapped_record,
    require_current_adoption,
)
from src.benchmarks.wrapped_join import evaluate_wrapped_join

xr = pytest.importorskip("xarray")


DECLARATION = Path(
    "data/identity_calibration/t4e36-wrapped-acquisition-declaration.json")


def test_t4e36_declaration_is_an_executable_complement_without_adopting_it():
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    request = dict(declaration["complementary_request"])
    expected_shards = request.pop("expected_monthly_shards")
    expected_frames = request.pop("expected_frames")
    expected_shape = request.pop("expected_segment_shape")
    spec = CDSRegionalRequest(
        **{**request, "variables": tuple(request["variables"]),
           "hours_utc": tuple(request["hours_utc"]),
           "pressure_levels": tuple(request["pressure_levels"])})

    shards = plan_monthly_shards(spec)
    estimate = estimate_cds_storage(spec, shards=shards)
    geometry = declaration["wrapped_geometry"]

    assert declaration["status"] == "DRAFTED_NOT_ADOPTED"
    assert len(shards) == expected_shards == 48
    assert estimate["frames"] == expected_frames == 5844
    assert [estimate["latitude_points_upper_bound"],
            estimate["longitude_points_upper_bound"]] == expected_shape == [161, 161]
    assert shards[0].request["area"] == [-18.0, -180.0, -58.0, -140.0]
    assert geometry["parent_longitude"] == [140.0, 180.0]
    assert geometry["complement_longitude_after_wrapping"] == [180.0, 220.0]
    assert geometry["expected_joined_longitude_points"] == 321

    frozen = declaration["frozen_extraction"]
    gate = declaration["gate_declared_before_acquisition"]
    assert frozen["n_surrogates"] == 99 and frozen["seed"] == 1234
    assert "raw_field" in gate["primary_scientific_path"]
    assert "9 of the fixed 18" in gate["condition_1"]
    assert "9 of the fixed 18" in gate["condition_2"]
    assert "SWT-plane counts cannot rescue" in gate["decision"]


def _copy_and_adopt(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    target = tmp_path / DECLARATION.name
    target.write_bytes(DECLARATION.read_bytes())
    sign_declaration(
        tmp_path, target.name,
        adopted_by="Test Human Reviewer", adopted_as="ADOPTED_FOR_TEST",
        what_was_adopted="the copied T4E.36 declaration",
        affirmation=REQUIRED_AFFIRMATION)
    return target


def test_executor_requires_current_adoption_and_separate_network_consent(tmp_path):
    declaration = tmp_path / DECLARATION.name
    declaration.write_bytes(DECLARATION.read_bytes())
    with pytest.raises(DataSourceError, match="has not been adopted"):
        require_current_adoption(declaration)

    declaration = _copy_and_adopt(tmp_path / "adopted")
    calls = []

    class Client:
        def retrieve(self, *args):
            calls.append(args)

    with pytest.raises(DataSourceError, match="experiment-specific explicit"):
        acquire_wrapped_complement(
            declaration, tmp_path / "downloads", client=Client(),
            explicit_network_authorisation=False)
    assert calls == []

    declaration.write_bytes(declaration.read_bytes() + b"\n")
    with pytest.raises(DataSourceError, match="does not bind the current declaration"):
        acquire_wrapped_complement(
            declaration, tmp_path / "downloads", client=Client(),
            explicit_network_authorisation=True)
    assert calls == []


def _segment(spec, longitudes, seam_offset=0.0):
    times = np.asarray([np.datetime64("2020-01-01T00:00:00", "ns")])
    latitudes = np.asarray([-1.0, -1.5, -2.0])
    values = np.empty((1, 1, 3, len(longitudes)), dtype=np.float32)
    for column in range(len(longitudes)):
        values[0, 0, :, column] = np.asarray([1.0, 1.25, 1.5]) + column * 0.25
    if seam_offset:
        values[:, :, :, 0] += np.float32(seam_offset)
    return xr.Dataset(
        {"vo": (("valid_time", "pressure_level", "latitude", "longitude"), values)},
        coords={"valid_time": times, "pressure_level": [850],
                "latitude": latitudes, "longitude": longitudes})


def test_wrapped_join_checks_and_deduplicates_the_180_degree_seam():
    common = dict(
        variables=("vo",), date_start="2020-01-01", date_end="2020-01-01",
        hours_utc=(0,), lat_min=-2.0, lat_max=-1.0,
        pressure_levels=(850,), grid_degrees=0.5, n_levels_analysis=1)
    parent_spec = CDSRegionalRequest(**common, lon_min=179.0, lon_max=180.0)
    complement_spec = CDSRegionalRequest(**common, lon_min=-180.0, lon_max=-179.0)
    parent = _segment(parent_spec, [179.0, 179.5, 180.0])
    complement = _segment(complement_spec, [-180.0, -179.5, -179.0])
    # Make the two physical seam columns equal while leaving both fields on a 0.25 lattice.
    complement["vo"].values[:, :, :, 0] = parent["vo"].values[:, :, :, -1]

    joined, seam = join_wrapped_segments(
        parent, complement, parent_spec, complement_spec,
        expected_times=[datetime(2020, 1, 1, 0)])
    assert joined.longitude.values.tolist() == [179.0, 179.5, 180.0, 180.5, 181.0]
    assert joined.sizes["longitude"] == 5
    assert seam["passed"] and seam["complement_duplicate_removed"]
    assert seam["maximum_absolute_difference"]["vo"] == 0.0


def test_wrapped_join_refuses_a_seam_mismatch_and_an_unmeasurable_tolerance():
    common = dict(
        variables=("vo",), date_start="2020-01-01", date_end="2020-01-01",
        hours_utc=(0,), lat_min=-2.0, lat_max=-1.0,
        pressure_levels=(850,), grid_degrees=0.5, n_levels_analysis=1)
    parent_spec = CDSRegionalRequest(**common, lon_min=179.0, lon_max=180.0)
    complement_spec = CDSRegionalRequest(**common, lon_min=-180.0, lon_max=-179.0)
    parent = _segment(parent_spec, [179.0, 179.5, 180.0])
    complement = _segment(complement_spec, [-180.0, -179.5, -179.0])
    complement["vo"].values[:, :, :, 0] = parent["vo"].values[:, :, :, -1] + 1.0
    with pytest.raises(DataSourceError, match="seam mismatch"):
        join_wrapped_segments(
            parent, complement, parent_spec, complement_spec,
            expected_times=[datetime(2020, 1, 1, 0)])

    constant_parent = parent.copy(deep=True)
    constant_complement = complement.copy(deep=True)
    constant_parent["vo"].values.fill(1.0)
    constant_complement["vo"].values.fill(1.0)
    with pytest.raises(DataSourceError, match="tolerance is not measurable"):
        join_wrapped_segments(
            constant_parent, constant_complement, parent_spec, complement_spec,
            expected_times=[datetime(2020, 1, 1, 0)])


def test_materialiser_publishes_a_content_addressed_union_without_changing_parent(tmp_path):
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    declaration["complementary_request"] = {
        "variables": ["vo"], "date_start": "2020-01-01", "date_end": "2020-01-01",
        "hours_utc": [0], "lat_min": -2.0, "lat_max": -1.0,
        "lon_min": -180.0, "lon_max": -179.0, "pressure_levels": [850],
        "grid_degrees": 0.5, "n_levels_analysis": 1,
        "expected_monthly_shards": 1, "expected_frames": 1,
        "expected_segment_shape": [3, 3],
    }
    declaration["wrapped_geometry"].update({
        "parent_longitude": [179.0, 180.0],
        "complement_longitude_as_requested": [-180.0, -179.0],
        "complement_longitude_after_wrapping": [180.0, 181.0],
        "canonical_joined_longitude": [179.0, 181.0],
        "expected_joined_longitude_points": 5,
    })
    declaration_path = tmp_path / DECLARATION.name
    declaration_path.write_text(json.dumps(declaration), encoding="utf-8")
    sign_declaration(
        tmp_path, declaration_path.name, adopted_by="Test Human Reviewer",
        adopted_as="ADOPTED_FOR_TEST", what_was_adopted="the small fixture declaration",
        affirmation=REQUIRED_AFFIRMATION)

    class Client:
        def retrieve(self, dataset, request, target):
            north, west, south, east = request["area"]
            grid = request["grid"][0]
            lat = np.arange(north, south - grid / 2, -grid)
            lon = np.arange(west, east + grid / 2, grid)
            canonical_lon = np.where(lon < 0, lon + 360.0, lon)
            values = np.empty((1, 1, len(lat), len(lon)), dtype=np.float32)
            for column, longitude in enumerate(canonical_lon):
                values[0, 0, :, column] = (
                    np.asarray([1.0, 1.25, 1.5], dtype=np.float32)
                    + np.float32((longitude - 179.0) * 0.5))
            xr.Dataset(
                {"vorticity": (("valid_time", "pressure_level", "latitude", "longitude"),
                                values)},
                coords={"valid_time": [np.datetime64("2020-01-01T00:00:00", "ns")],
                        "pressure_level": [850], "latitude": lat, "longitude": lon},
            ).to_netcdf(target, engine="h5netcdf")

    complement_spec = CDSRegionalRequest(
        variables=("vo",), date_start="2020-01-01", date_end="2020-01-01",
        hours_utc=(0,), lat_min=-2.0, lat_max=-1.0, lon_min=-180.0, lon_max=-179.0,
        pressure_levels=(850,), grid_degrees=0.5, n_levels_analysis=1)
    parent_spec = CDSRegionalRequest(
        variables=("vo",), date_start="2020-01-01", date_end="2020-01-01",
        hours_utc=(0,), lat_min=-2.0, lat_max=-1.0, lon_min=179.0, lon_max=180.0,
        pressure_levels=(850,), grid_degrees=0.5, n_levels_analysis=4)
    parent_dir, complement_dir = tmp_path / "parent", tmp_path / "complement"
    acquire_cds_shards(parent_spec, parent_dir, client=Client(), allow_network=True)
    acquire_cds_shards(complement_spec, complement_dir, client=Client(), allow_network=True)
    parent_before = {path.name: path.read_bytes() for path in parent_dir.glob("*.nc")}

    receipt = materialise_wrapped_record(
        declaration_path, parent_dir=parent_dir, complement_dir=complement_dir,
        record_dir=tmp_path / "records", time_chunk=1)
    assert receipt["shape"] == {
        "time": 1, "level": 1, "latitude": 3, "longitude": 5}
    assert receipt["parent"]["preserved_byte_for_byte"] is True
    assert len(receipt["record_sha256"]) == len(receipt["receipt_sha256"]) == 64
    assert Path(receipt["record_path"]).name == receipt["record_sha256"] + ".zarr"
    assert parent_before == {path.name: path.read_bytes() for path in parent_dir.glob("*.nc")}


def test_evaluator_uses_fixed_18_rows_and_swt_cannot_rescue_raw_failure(tmp_path, monkeypatch):
    declaration_path = _copy_and_adopt(tmp_path / "adopted-evaluation")
    baseline = json.loads(
        Path("measurements/t4e28_join_rerun.json").read_text(encoding="utf-8"))
    rows = baseline["paths"]["raw_field"]["rows"]
    times = np.asarray([
        np.datetime64(row["time"].replace(" ", "T"), "ns") for row in rows])
    record_path = tmp_path / "fixture.zarr"
    xr.Dataset(
        {"vo": (("time", "level", "latitude", "longitude"),
                np.zeros((18, 1, 3, 5), dtype=np.float32))},
        coords={"time": times, "level": [850], "latitude": [-1.0, -1.5, -2.0],
                "longitude": [179.0, 179.5, 180.0, 180.5, 181.0]},
    ).chunk({"time": 1}).to_zarr(record_path, mode="w", consolidated=True)
    declaration_sha = hashlib.sha256(declaration_path.read_bytes()).hexdigest()
    unsigned = {
        "schema": WRAPPED_RECORD_SCHEMA, "record_sha256": "fixture-content",
        "record_path": str(record_path), "declaration_sha256": declaration_sha,
    }
    canonical = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str).encode("utf-8")
    receipt = {**unsigned, "receipt_sha256": hashlib.sha256(canonical).hexdigest()}
    receipt_path = tmp_path / "record-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    monkeypatch.setattr(
        "src.benchmarks.wrapped_join.streaming_content_sha256",
        lambda dataset, time_block: "fixture-content")
    monkeypatch.setattr("src.benchmarks.wrapped_join._raw_positions", lambda *args: [])
    # Diagnostic SWT is intentionally made very feature-rich; it still cannot change FAIL.
    monkeypatch.setattr(
        "src.benchmarks.wrapped_join._swt_positions",
        lambda field, latitudes, longitudes: [(row["lat"], row["lon"]) for row in rows])
    output = tmp_path / "measurement.json"
    measured = evaluate_wrapped_join(
        declaration_path, receipt_path,
        baseline_path="measurements/t4e28_join_rerun.json", output_path=output)
    assert measured["fixed_population"] == {
        "source": "measurements/t4e28_join_rerun.json",
        "source_sha256": hashlib.sha256(
            Path("measurements/t4e28_join_rerun.json").read_bytes()).hexdigest(),
        "rows": 18, "catalogue_census_rerun": False,
    }
    assert measured["paths"]["raw_field"]["aggregates"]["storms"] == 18
    assert measured["paths"]["swt_planes"]["aggregates"][
        "condition_1_at_least_one_inside_radius"]["met"] >= 9
    assert measured["gate"]["verdict"] == "FAIL"
    assert measured["gate"]["swt_can_rescue_raw_failure"] is False
