"""ERA5-over-Zarr adapter tests (roadmap T3.5.18).

**These tests do not touch the network, and that is deliberate.** A suite whose result depends
on connectivity is a suite that cannot be trusted to tell you about your code, and a stray
bounding box against a 0.25 degree store can move tens of gigabytes. So every mechanism -
chunk reporting, hostility prediction, byte counting, the cache, the provenance round trip - is
exercised against **synthetic local Zarr stores built with the same pathological layout as the
real archive**: one timestep x all levels x whole globe per chunk.

The one thing a local store cannot verify is the GCS transport itself. That was verified
separately and by hand against the live WeatherBench 2 bucket, with the numbers recorded in
`VERIFICATION.md` (slice 11); the single network test here is opt-in via
``SPECTRALEARTH_ALLOW_NETWORK=1`` so that verification is repeatable without becoming a
dependency of the suite.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from src.core.errors import DataSourceError, FieldTooSmallError, InvalidParameterError
from src.data_layer import builtin_sources  # noqa: F401  (registers the other sources)
from src.data_layer import zarr_source as zs
from src.transform_engine.stationary import valid_interior_halfwidth

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


# --------------------------------------------------------------------------- fixtures

def _build_store(path, *, nt=24, nl=8, ny=128, nx=256, time_chunk=1,
                 descending_lat=True, lon_360=True):
    """Write a small Zarr store with a controllable chunk layout.

    ``time_chunk=1`` reproduces the WeatherBench 2 0.25 degree layout (chunk-hostile for a
    regional crop); a large ``time_chunk`` reproduces the coarse stores, which are not.
    """
    rng = np.random.default_rng(1234)
    lat = np.linspace(90, -90, ny) if descending_lat else np.linspace(-90, 90, ny)
    lon = np.linspace(0, 359, nx) if lon_360 else np.linspace(-180, 179, nx)
    dataset = xr.Dataset(
        {
            "temperature": (("time", "level", "latitude", "longitude"),
                            rng.standard_normal((nt, nl, ny, nx)).astype("float32")),
            "geopotential": (("time", "level", "latitude", "longitude"),
                             rng.standard_normal((nt, nl, ny, nx)).astype("float32")),
        },
        coords={
            "time": np.arange("2020-01-01", "2020-01-31",
                              np.timedelta64(6, "h"), dtype="datetime64[ns]")[:nt],
            "level": [1000, 925, 850, 700, 500, 300, 200, 100][:nl],
            "latitude": lat,
            "longitude": lon,
        },
    )
    dataset.chunk({"time": time_chunk, "level": nl, "latitude": ny,
                   "longitude": nx}).to_zarr(str(path), mode="w", consolidated=True)
    return str(path)


@pytest.fixture()
def hostile_store(tmp_path):
    return _build_store(tmp_path / "hostile.zarr", time_chunk=1)


@pytest.fixture()
def friendly_store(tmp_path):
    """Time-contiguous chunks: the layout the local cache is rechunked *into*."""
    return _build_store(tmp_path / "friendly.zarr", time_chunk=24)


def _spec(store, **overrides):
    kwargs = dict(
        store=store, variables=("temperature",),
        time_start="2020-01-01", time_end="2020-01-03",
        lat_min=0.0, lat_max=32.0, lon_min=0.0, lon_max=32.0,
        levels=(850, 700, 500, 300), n_levels_analysis=4,
    )
    kwargs.update(overrides)
    return zs.CropSpec(**kwargs)


# ======================================================== R13 crop geometry

def test_edge_exclusion_reproduces_the_r13_table():
    """R13 uses the conservative radius and agrees with the transform implementation."""
    margins = [zs.edge_exclusion(j, taps=14) for j in (1, 2, 3, 4)]
    assert margins == [7, 13, 26, 52]

    # Exercise the real shared family as well as the published 14-tap sizing convention.
    assert [zs.edge_exclusion(j, taps=4) for j in (1, 2, 3)] == [
        valid_interior_halfwidth("db2", j) for j in (1, 2, 3)
    ]


def test_valid_interior_reproduces_the_r13_table():
    # R13's corrected table: N=64 has 50 valid px at level 1 and none at level 4.
    assert zs.valid_interior(64, 1) == 50
    assert zs.valid_interior(64, 4) <= 0
    assert zs.valid_interior(256, 4) == 152
    assert zs.valid_interior(512, 4) == 408


def test_minimum_crop_size_matches_the_roadmaps_numbers():
    """R13: practical minimum 256 for four dyadic levels, 512 for five."""
    assert zs.minimum_crop_size(4) == 256
    assert zs.minimum_crop_size(5) == 512


def test_crop_too_small_is_refused_and_names_the_minimum():
    with pytest.raises(FieldTooSmallError) as excinfo:
        zs.check_crop_size(64, 64, levels=4)
    message = str(excinfo.value)
    assert "256" in message, "the refusal must name the minimum size"
    assert "52 px per side" in message
    # And it must say which dimension to give up instead - R13 is explicit that the grid is
    # never the thing to shrink.
    assert "never the grid" in message


def test_crop_at_the_floor_is_accepted_and_reports_its_interior():
    report = zs.check_crop_size(256, 256, levels=4)
    assert report["ok"] is True
    assert report["minimum_size"] == 256
    assert report["valid_interior_by_level"][4] == 152


def test_a_crop_that_passes_four_levels_can_still_fail_five():
    zs.check_crop_size(256, 256, levels=4)
    with pytest.raises(FieldTooSmallError):
        zs.check_crop_size(256, 256, levels=5)


# ======================================================== crop specification

def test_content_key_is_deterministic_and_order_independent():
    a = _spec("store", variables=("temperature", "geopotential"))
    b = _spec("store", variables=("geopotential", "temperature"))
    assert a.content_key() == b.content_key(), (
        "the key must describe the request, not the order it was typed in")
    assert len(a.content_key()) == 16


def test_content_key_changes_with_any_part_of_the_request():
    base = _spec("store")
    for change in ({"lat_min": 1.0}, {"time_end": "2020-01-04"}, {"levels": (850,)},
                   {"variables": ("geopotential",)}, {"store": "other"}):
        kwargs = dict(change)
        store = kwargs.pop("store", "store")
        assert _spec(store, **kwargs).content_key() != base.content_key(), change


def test_analysis_depth_is_not_part_of_the_key():
    """`n_levels_analysis` constrains the *check*, not the data - so it must not fork the cache."""
    assert (_spec("s", n_levels_analysis=4).content_key()
            == _spec("s", n_levels_analysis=5).content_key())


def test_provenance_round_trip_is_exact():
    spec = _spec("era5_0p25_6h")
    assert zs.CropSpec.from_provenance(spec.to_provenance()) == spec


def test_provenance_round_trip_tolerates_added_fields():
    """A lineage record grows over time; replay must not break when it does."""
    record = _spec("era5_0p25_6h").to_provenance()
    record["recorded_at"] = "2026-08-20T12:00:00Z"
    record["bytes_transferred"] = 12345
    assert zs.CropSpec.from_provenance(record) == _spec("era5_0p25_6h")


def test_provenance_missing_a_required_field_is_refused():
    record = _spec("era5_0p25_6h").to_provenance()
    del record["lat_min"]
    with pytest.raises(InvalidParameterError):
        zs.CropSpec.from_provenance(record)


def test_inverted_bounding_box_is_refused_at_construction():
    with pytest.raises(InvalidParameterError):
        _spec("s", lat_min=40.0, lat_max=10.0)
    with pytest.raises(InvalidParameterError):
        _spec("s", lon_min=40.0, lon_max=10.0)


def test_catalogue_entries_resolve_to_gcs_uris():
    assert zs.CATALOGUE, "the catalogue must not be empty"
    for name, entry in zs.CATALOGUE.items():
        assert entry["uri"].startswith("gs://weatherbench2/"), name
        assert entry["note"], "every store needs a note saying what it is good and bad for"
    # A raw URI must pass through unchanged, so an unlisted store is still usable.
    assert _spec("gs://elsewhere/x.zarr").uri == "gs://elsewhere/x.zarr"


# ======================================================== chunk reporting

def test_describe_store_reports_chunk_shape_and_bytes(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    report = zs.describe_store(dataset, ["temperature"])
    entry = report["variables"]["temperature"]
    assert entry["chunks"] == [1, 8, 128, 256]
    # Shape alone is not actionable; the megabyte figure is what makes it obvious.
    assert entry["chunk_megabytes"] == pytest.approx(1.049, abs=0.01)
    assert entry["n_chunks"] == 24
    assert report["dimensions"]["time"] == 24


def test_hostile_layout_is_detected_and_explained(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    assessment = zs.assess_access_pattern(dataset, _spec(hostile_store))
    assert assessment["chunk_hostile"] is True
    assert assessment["amplification"] > 100
    assert "CHUNK-HOSTILE" in assessment["warning"]
    # The advice must name the specific trap: subsetting these axes saves nothing.
    assert any("saves nothing over the network" in a for a in assessment["advice"])


def test_friendly_layout_is_not_flagged(friendly_store):
    """A time-contiguous store serving a whole-globe crop is exactly what we rechunk *to*."""
    spec = _spec(friendly_store, lat_min=-90.0, lat_max=90.0,
                 lon_min=0.0, lon_max=359.0, time_start="2020-01-01", time_end="2020-01-07",
                 levels=(1000, 925, 850, 700, 500, 300, 200, 100))
    dataset, _counter = zs.open_dataset(friendly_store, chunks={})
    assessment = zs.assess_access_pattern(dataset, spec)
    assert assessment["chunk_hostile"] is False
    assert assessment["amplification"] == pytest.approx(1.0, abs=0.01)


def test_half_a_chunk_still_costs_a_whole_chunk(friendly_store):
    """Not a defect: asking for 12 of a 24-step chunk fetches all 24, so amplification is 2.

    Worth an explicit test because it is the mechanism behind the 51x figure on the real
    archive, seen at a size small enough to check by hand.
    """
    spec = _spec(friendly_store, lat_min=-90.0, lat_max=90.0, lon_min=0.0, lon_max=359.0,
                 levels=(1000, 925, 850, 700, 500, 300, 200, 100))
    dataset, _counter = zs.open_dataset(friendly_store, chunks={})
    assessment = zs.assess_access_pattern(dataset, spec)
    assert assessment["selection"]["time"] == 12
    assert assessment["amplification"] == pytest.approx(2.0, abs=0.01)


def test_prediction_uses_the_same_selection_as_the_fetch(hostile_store):
    """The defect this closes, asserted.

    The first version of `assess_access_pattern` counted timestamps in ``[start, end]`` with
    numpy while `select` handed the same strings to ``xarray.sel(slice(...))``, which treats a
    partial date as the whole day. They disagreed by 3 of 12 timesteps, so the hostility
    warning under-stated the transfer by 25%. A warning that under-states the cost is worse
    than no warning at all, so the predictor now derives its sizes from `select` itself.
    """
    spec = _spec(hostile_store, time_start="2020-01-01", time_end="2020-01-03")
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    predicted = zs.assess_access_pattern(dataset, spec)["selection"]
    actual = {str(k): int(v) for k, v in zs.select(dataset, spec).sizes.items()}
    assert predicted == actual
    assert actual["time"] == 12, "a partial end date covers the whole day"


def test_estimate_is_an_upper_bound_on_wire_bytes(hostile_store, tmp_path):
    """Estimates are uncompressed; the wire is compressed. The estimate must not undershoot."""
    spec = _spec(hostile_store)
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    predicted = zs.assess_access_pattern(dataset, spec)["bytes_fetched_estimate"]
    manifest = zs.materialise(spec, cache_dir=str(tmp_path / "c"), check_size=False)
    measured = manifest["bytes_transferred"]
    assert measured <= predicted, (
        "the estimate must bound the transfer: predicted %d, measured %d"
        % (predicted, measured))
    # And it must not be a vacuous bound - random float32 compresses barely at all.
    assert measured > 0.8 * predicted


# ======================================================== byte counting

def test_counting_store_includes_metadata_reads(hostile_store):
    """A data-only measurement would understate the transfer; this counts .zmetadata too."""
    store = zs.open_store(hostile_store)
    assert store.bytes_read == 0
    dataset = xr.open_zarr(store, chunks={}, decode_timedelta=False, consolidated=None)
    assert store.bytes_read > 0, "opening a store reads its metadata"
    assert store.keys_read > 0
    dataset.close()


def test_counting_store_reports_zero_before_any_read(hostile_store):
    store = zs.open_store(hostile_store)
    assert store.report() == {"bytes_read": 0, "megabytes_read": 0.0, "chunks_read": 0}


def test_missing_local_store_is_explained():
    with pytest.raises(DataSourceError) as excinfo:
        zs.open_store("no/such/store.zarr")
    assert "directory" in str(excinfo.value), (
        "the error must say a Zarr store is a directory, not a file")


# ======================================================== selection traps

def test_descending_latitude_axis_is_handled(hostile_store):
    """ERA5 stores latitude 90 -> -90. An ascending slice against it selects nothing."""
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    assert float(dataset.latitude.values[0]) > float(dataset.latitude.values[-1])
    subset = zs.select(dataset, _spec(hostile_store))
    assert subset.sizes["latitude"] > 0, "a descending axis must not silently select nothing"


def test_ascending_latitude_axis_is_also_handled(tmp_path):
    path = _build_store(tmp_path / "asc.zarr", descending_lat=False)
    dataset, _counter = zs.open_dataset(path, chunks={})
    assert zs.select(dataset, _spec(path)).sizes["latitude"] > 0


def test_negative_longitude_against_a_0_360_store_is_refused(hostile_store):
    """Guessing a conversion here would select nothing; the wrap needs two crops."""
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    with pytest.raises(InvalidParameterError) as excinfo:
        zs.select(dataset, _spec(hostile_store, lon_min=-20.0, lon_max=20.0))
    assert "prime meridian" in str(excinfo.value)


def test_unknown_variable_lists_what_is_available(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    with pytest.raises(DataSourceError) as excinfo:
        zs.select(dataset, _spec(hostile_store, variables=("vorticity",)))
    assert "vorticity" in str(excinfo.value)
    assert "temperature" in str(excinfo.value)


def test_unknown_level_lists_what_is_available(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    with pytest.raises(InvalidParameterError) as excinfo:
        zs.select(dataset, _spec(hostile_store, levels=(777,)))
    assert "777" in str(excinfo.value)


def test_time_window_outside_coverage_names_the_coverage(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    with pytest.raises(InvalidParameterError) as excinfo:
        zs.select(dataset, _spec(hostile_store, time_start="1999-01-01",
                                 time_end="1999-02-01"))
    assert "2020-01-01" in str(excinfo.value)


def test_region_outside_the_grid_is_refused(hostile_store):
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    with pytest.raises(InvalidParameterError):
        zs.select(dataset, _spec(hostile_store, lat_min=95.0, lat_max=99.0))


# ======================================================== materialisation and cache

def test_materialise_writes_a_cache_and_a_manifest(hostile_store, tmp_path):
    cache = str(tmp_path / "cache")
    spec = _spec(hostile_store)
    manifest = zs.materialise(spec, cache_dir=cache, check_size=False)
    assert manifest["cache_hit"] is False
    assert os.path.isdir(manifest["cache_path"])
    assert os.path.exists(zs.manifest_path(spec, cache))
    assert manifest["bytes_transferred"] > 0
    assert manifest["content_hash"]
    assert manifest["shape"]["time"] == 12
    assert manifest["shape"]["level"] == 4
    # The manifest must carry the evidence, not just the data.
    assert manifest["remote_chunk_structure"]["variables"]["temperature"]["chunks"]
    assert manifest["access_assessment"]["chunk_hostile"] is True


def test_cache_is_rechunked_time_contiguous(hostile_store, tmp_path):
    """The inversion is the whole justification for keeping a second copy."""
    cache = str(tmp_path / "cache")
    spec = _spec(hostile_store)
    manifest = zs.materialise(spec, cache_dir=cache, check_size=False)
    assert manifest["cache_chunking"]["time"] == manifest["shape"]["time"]
    cached, _m = zs.load_cached(spec, cache)
    assert cached["temperature"].encoding["chunks"][0] == manifest["shape"]["time"], (
        "the remote store gives one timestep per chunk; the cache must give all of them")


def test_repeat_request_transfers_zero_bytes(hostile_store, tmp_path):
    """The acceptance criterion, asserted as zero rather than as 'fast'."""
    cache = str(tmp_path / "cache")
    spec = _spec(hostile_store)
    first = zs.materialise(spec, cache_dir=cache, check_size=False)
    second = zs.materialise(spec, cache_dir=cache, check_size=False)
    assert first["bytes_transferred"] > 0
    assert second["cache_hit"] is True
    assert second["bytes_transferred"] == 0
    _dataset, manifest = zs.load_cached(spec, cache)
    assert manifest["remote_bytes"] == 0
    assert manifest["local_read"]["bytes_read"] > 0, "it did read - locally"


def test_cache_is_shared_by_equivalent_specs(hostile_store, tmp_path):
    """Two specs differing only in typing order must hit one cache entry, not two."""
    cache = str(tmp_path / "cache")
    a = _spec(hostile_store, variables=("temperature", "geopotential"))
    b = _spec(hostile_store, variables=("geopotential", "temperature"))
    zs.materialise(a, cache_dir=cache, check_size=False)
    assert zs.materialise(b, cache_dir=cache, check_size=False)["cache_hit"] is True
    assert len(zs.cached_crops(cache)) == 1


def test_load_cached_refuses_to_silently_fetch(hostile_store, tmp_path):
    with pytest.raises(DataSourceError) as excinfo:
        zs.load_cached(_spec(hostile_store), str(tmp_path / "empty"))
    assert "materialise() first" in str(excinfo.value)
    assert "four orders of magnitude" in str(excinfo.value)


def test_rematerialise_from_provenance_reproduces_the_crop(hostile_store, tmp_path):
    """The last acceptance criterion: rebuild the exact input from the lineage record alone."""
    original = zs.materialise(_spec(hostile_store), cache_dir=str(tmp_path / "a"),
                              check_size=False)
    # Simulate reading the record back out of the database.
    record = json.loads(json.dumps(original))
    replayed = zs.rematerialise_from_provenance(record, cache_dir=str(tmp_path / "b"),
                                                check_size=False)
    assert replayed["content_key"] == original["content_key"]
    assert replayed["content_hash"] == original["content_hash"]


def test_content_hash_is_independent_of_cache_chunking(hostile_store, tmp_path):
    """The hash must describe the data, not the layout it happens to be stored in."""
    spec = _spec(hostile_store)
    a = zs.materialise(spec, cache_dir=str(tmp_path / "a"), check_size=False, time_chunk=1)
    b = zs.materialise(spec, cache_dir=str(tmp_path / "b"), check_size=False, time_chunk=12)
    assert a["cache_chunking"]["time"] != b["cache_chunking"]["time"]
    assert a["content_hash"] == b["content_hash"]


def test_size_check_is_enforced_during_materialisation(hostile_store, tmp_path):
    """A 32-degree crop of this 128x256 grid is far below the four-level floor."""
    with pytest.raises(FieldTooSmallError):
        zs.materialise(_spec(hostile_store), cache_dir=str(tmp_path / "c"), check_size=True)


def test_cached_crops_lists_manifests(hostile_store, tmp_path):
    cache = str(tmp_path / "cache")
    assert zs.cached_crops(cache) == []
    zs.materialise(_spec(hostile_store), cache_dir=cache, check_size=False)
    listing = zs.cached_crops(cache)
    assert len(listing) == 1
    assert listing[0]["spec"]["variables"] == ["temperature"]


def test_a_corrupt_manifest_does_not_break_the_listing(hostile_store, tmp_path):
    cache = str(tmp_path / "cache")
    zs.materialise(_spec(hostile_store), cache_dir=cache, check_size=False)
    with open(os.path.join(cache, "broken.json"), "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert len(zs.cached_crops(cache)) == 1


# ======================================================== the source seam

def test_source_is_registered_between_local_files_and_the_simulator():
    from src.data_layer.sources import SOURCES

    spec = SOURCES.get("era5_zarr")
    assert spec.priority == 20
    assert SOURCES.get("netcdf_local").priority < spec.priority
    assert spec.priority < SOURCES.get("simulated").priority
    assert spec.is_simulated is False
    assert spec.kind == "zarr"
    capabilities = SOURCES.entry("era5_zarr").capabilities
    assert capabilities["streaming"] is True
    assert capabilities["observational"] is True
    # Not `dataset_ids`: see the comment on the registration. A crop family has no concrete
    # extent, so it must not be advertised as a listable dataset.
    assert "dataset_ids" not in capabilities
    assert capabilities["crop_dataset_ids"] == [zs.ZARR_DATASET_ID]


def test_the_streaming_capability_query_now_returns_something():
    """`sources.py` promised this seam before anything could satisfy it.

    Its docstring said `SOURCES.with_capability("streaming")` is how the Zarr adapter "will
    be selected without anyone editing a dispatch chain". Until this slice that query returned
    an empty list, so the seam was a claim rather than a fact.
    """
    from src.data_layer.sources import SOURCES

    streaming = [e.name for e in SOURCES.with_capability("streaming")]
    assert streaming == ["era5_zarr"]


def test_network_is_disabled_by_default(monkeypatch):
    """Reaching the internet must never be a side effect of running something."""
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    assert zs.network_enabled() is False
    monkeypatch.setenv(zs.NETWORK_ENV_VAR, "1")
    assert zs.network_enabled() is True


def test_source_declines_offline_with_an_actionable_reason(monkeypatch, tmp_path):
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    monkeypatch.setattr(zs, "DEFAULT_CACHE_DIR", str(tmp_path / "empty"))
    assert zs.ERA5ZarrSource.can_serve(zs.ZARR_DATASET_ID) is False
    reason = zs.ERA5ZarrSource.why_not(zs.ZARR_DATASET_ID)
    assert zs.NETWORK_ENV_VAR in reason
    assert "materialised" in reason


def test_source_serves_a_cached_crop_with_no_network(monkeypatch, hostile_store, tmp_path):
    """A materialised crop makes this source usable offline - the point of stage 2."""
    cache = str(tmp_path / "cache")
    spec = _spec(hostile_store)
    zs.materialise(spec, cache_dir=cache, check_size=False)
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    monkeypatch.setattr(zs, "DEFAULT_CACHE_DIR", cache)
    assert zs.ERA5ZarrSource.can_serve(zs.ZARR_DATASET_ID) is True
    dataset = zs.ERA5ZarrSource.fetch(zs.ZARR_DATASET_ID,
                                      crop=spec.to_provenance(), cache_dir=cache)
    assert "temperature" in dataset.data_vars

    # D42: real crops must reach the Phase 4 sequence action, not stop at the Zarr tab.
    from src.artifact_store import store as store_module
    from src.artifact_store.store import ArtifactStore
    from src.data_layer.adapters import MeteorologicalDataAdapter
    from src.experiment_engine import actions

    monkeypatch.setattr(store_module, "_DEFAULT", ArtifactStore(str(tmp_path / "artifacts")))
    source = {"crop": spec.to_provenance(), "cache_dir": cache}
    sequence = MeteorologicalDataAdapter.slice_sequence(
        zs.ZARR_DATASET_ID, "temperature", level=850, source_options=source)
    assert len(sequence) > 1
    assert sequence.metadata["is_simulated"] is False
    assert sequence.metadata["source_kind"] == "zarr"
    assert sequence.grid.kind == "latlon"

    result = actions.execute("slice_sequence", {
        "dataset_id": zs.ZARR_DATASET_ID,
        "variable": "temperature",
        "level": 850,
        "crop": spec.to_provenance(),
        "cache_dir": cache,
    }, torch.device("cpu"))
    assert result["sequence_ref"].startswith("artifact://")
    assert result["summary"]["metadata"]["is_simulated"] is False


def test_source_without_a_crop_explains_why_it_cannot_guess():
    with pytest.raises(DataSourceError) as excinfo:
        zs.ERA5ZarrSource.fetch(zs.ZARR_DATASET_ID)
    assert "64 years of the whole planet" in str(excinfo.value)


def test_source_refuses_an_uncached_crop_when_offline(monkeypatch, hostile_store, tmp_path):
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    with pytest.raises(DataSourceError) as excinfo:
        zs.ERA5ZarrSource.fetch(zs.ZARR_DATASET_ID,
                                crop=_spec(hostile_store).to_provenance(),
                                cache_dir=str(tmp_path / "empty"))
    assert zs.NETWORK_ENV_VAR in str(excinfo.value)


def test_source_declines_unknown_dataset_ids():
    assert zs.ERA5ZarrSource.can_serve("era5_reanalysis") is False
    assert "only serves" in zs.ERA5ZarrSource.why_not("era5_reanalysis")


# ======================================================== CLI

def test_cli_catalogue_emits_valid_json(capsys):
    assert zs._main(["catalogue"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(zs.CATALOGUE)


def test_cli_inspect_reports_without_transferring_data(hostile_store, capsys):
    """`inspect` is what a researcher runs *before* committing to a download."""
    assert zs._main([
        "inspect", "--store", hostile_store, "--variables", "temperature",
        "--start", "2020-01-01", "--end", "2020-01-03",
        "--lat", "0", "32", "--lon", "0", "32", "--levels", "850,700,500,300",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["assessment"]["chunk_hostile"] is True
    assert report["structure"]["variables"]["temperature"]["chunk_megabytes"] > 0


# ======================================================== the NetCDF engine (D33)

def test_a_netcdf4_hdf5_file_can_actually_be_opened(tmp_path):
    """Defect D33, asserted so it cannot regress by dependency drift.

    `data/README.md` invites the researcher to drop an ERA5 `.nc` file into `data/`, and
    `LocalNetCDFSource` is the highest-priority source. But `requirements.txt` declared
    `xarray` and **no NetCDF engine**, so on a clean install the only engines present were
    `scipy` (NetCDF3 classic only) and `zarr`. A modern ERA5 download is NetCDF4/HDF5, and
    opening one failed with *"found the following matches with the input file in xarray's IO
    backends: ['netcdf4', 'h5netcdf']. But their dependencies may not be installed"* - a
    documented feature that could not work. The engine is now declared; this proves it reads.
    """
    path = tmp_path / "era5_like.nc"
    original = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"),
                 np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4))},
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
                "latitude": [10.0, 5.0, 0.0], "longitude": [0.0, 1.0, 2.0, 3.0]},
    )
    original.to_netcdf(str(path), engine="h5netcdf")
    # Opened with no explicit engine, exactly as LocalNetCDFSource.fetch does: xarray must
    # sniff the HDF5 magic bytes and find an installed backend for them.
    reopened = xr.open_dataset(str(path))
    try:
        assert np.array_equal(reopened["t2m"].values, original["t2m"].values)
        assert list(reopened.sizes) and reopened.sizes["time"] == 2
    finally:
        reopened.close()


def test_an_hdf5_engine_is_registered():
    """The dependency itself, named, so the failure message is about the cause."""
    engines = set(xr.backends.list_engines())
    assert engines & {"h5netcdf", "netcdf4"}, (
        "no NetCDF4/HDF5 engine is installed - only %s. `pip install h5netcdf h5py`; "
        "without one, every real ERA5 .nc file is unreadable (D33)." % sorted(engines))


# ======================================================== HTTP surface

def test_catalogue_endpoint_reports_stores_and_the_network_gate(client):
    response = client.get("/api/v1/data/zarr/catalogue")
    assert response.status_code == 200
    body = response.json()
    assert set(body["stores"]) == set(zs.CATALOGUE)
    assert body["network_env_var"] == zs.NETWORK_ENV_VAR
    assert body["missing_dependencies"] == []
    # The R13 floor is published, so a caller can size a crop before requesting one.
    assert body["r13_minimum_crop"]["4"] == 256
    assert body["r13_minimum_crop"]["5"] == 512


def test_inspect_endpoint_reports_hostility_without_transferring_data(client, hostile_store):
    response = client.post("/api/v1/data/zarr/inspect", json={
        "store": hostile_store, "variables": ["temperature"],
        "time_start": "2020-01-01", "time_end": "2020-01-03",
        "lat_min": 0.0, "lat_max": 32.0, "lon_min": 0.0, "lon_max": 32.0,
        "levels": [850, 700, 500, 300], "n_levels_analysis": 4,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["assessment"]["chunk_hostile"] is True
    assert "CHUNK-HOSTILE" in body["assessment"]["warning"]
    assert body["cached"] is False
    assert body["spec"]["content_key"]
    # A crop below the R13 floor is *reported*, not raised: the caller asked what this crop
    # would cost, and "too small, minimum 256x256" is the answer to that question.
    assert body["geometry"]["ok"] is False
    assert body["geometry"]["minimum_size"] == 256
    # And the response hands back the command that would do it for real.
    assert "materialise" in body["cli"]


def test_inspect_endpoint_refuses_a_remote_store_when_the_network_is_off(client, monkeypatch):
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    response = client.post("/api/v1/data/zarr/inspect", json={
        "store": "era5_0p25_6h", "variables": ["temperature"],
        "time_start": "2020-01-01", "time_end": "2020-01-02",
        "lat_min": -4.0, "lat_max": 60.0, "lon_min": 0.0, "lon_max": 64.0,
        "levels": [850, 700, 500, 300],
    })
    assert response.status_code == 409
    assert zs.NETWORK_ENV_VAR in response.json()["detail"]


def test_inspect_endpoint_rejects_an_inverted_bounding_box(client, hostile_store):
    response = client.post("/api/v1/data/zarr/inspect", json={
        "store": hostile_store, "variables": ["temperature"],
        "time_start": "2020-01-01", "time_end": "2020-01-03",
        "lat_min": 40.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 32.0,
        "levels": [850],
    })
    assert response.status_code == 400


def test_cached_endpoint_lists_materialised_crops(client, hostile_store, tmp_path,
                                                  monkeypatch):
    cache = str(tmp_path / "cache")
    zs.materialise(_spec(hostile_store), cache_dir=cache, check_size=False)
    monkeypatch.setattr(zs, "DEFAULT_CACHE_DIR", cache)
    body = client.get("/api/v1/data/zarr/cached").json()
    assert body["count"] == 1
    assert body["crops"][0]["content_hash"]
    assert body["crops"][0]["spec"]["variables"] == ["temperature"]


# ======================================================== opt-in live check

@pytest.mark.skipif(not zs.network_enabled(),
                    reason="set SPECTRALEARTH_ALLOW_NETWORK=1 to check the live GCS store")
def test_live_weatherbench_store_matches_the_recorded_structure():
    """The one thing a local store cannot verify: that the real archive is as documented.

    Metadata only - a few hundred kilobytes - so it is cheap enough to run deliberately. The
    figures asserted here are the ones `CATALOGUE` and the module docstring quote, so if
    WeatherBench 2 rechunks its archive this test says so rather than the documentation
    quietly becoming false.
    """
    entry = zs.CATALOGUE["era5_0p25_6h"]
    dataset, _counter = zs.open_dataset(entry["uri"], chunks={})
    try:
        report = zs.describe_store(dataset, ["temperature"])
        assert report["dimensions"]["latitude"] == 721
        assert report["dimensions"]["longitude"] == 1440
        assert report["dimensions"]["level"] == 13
        chunks = report["variables"]["temperature"]["chunks"]
        assert chunks == [1, 13, 721, 1440], (
            "the documented 54 MB/chunk hostile layout has changed: %s" % chunks)
        spec = zs.CropSpec(
            store="era5_0p25_6h", variables=("temperature",),
            time_start="2020-01-01", time_end="2020-12-31",
            lat_min=-4.0, lat_max=60.0, lon_min=0.0, lon_max=64.0,
            levels=(850, 700, 500, 300))
        assessment = zs.assess_access_pattern(dataset, spec)
        assert assessment["chunk_hostile"] is True
        assert assessment["amplification"] == pytest.approx(51.1, abs=0.5)
    finally:
        dataset.close()
