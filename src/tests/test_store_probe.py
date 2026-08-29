"""Probing a store as a recorded act (roadmap TG10.3, standards E1/E5).

**No test here touches the network**, for the reason `test_zarr_source.py` states: a suite
whose result depends on connectivity cannot tell you about your code. Every probe runs against
a local Zarr store built with the same pathological layout as the real archive, and the one
opt-in live check at the bottom is skipped unless `SPECTRALEARTH_ALLOW_NETWORK=1`.

The slice's acceptance criterion is *"a deliberately hostile store is characterised as hostile
before anyone crops it"*, and that word **before** is what most of this file is about: the
probe must reach its verdict from chunk metadata, with nothing materialised and no cache entry
created. The other half of the criterion - reproducing the four ERA5 stores' recorded notes -
is offline only as far as the transcriptions go, and is recorded as NOT RUN live.

The registration checks are the ones with teeth. A store that claims a measurement must cite
the probe that made it; the probe must be of *that store's* URI; and the entry may not quote a
figure the cited probe never saw. Defect D62 was a per-chunk size written into a field no
inspection had filled, and a wrong number with a citation attached is that defect improved only
in appearance.
"""

from __future__ import annotations

import json
import os

import pytest

from src.core.errors import InvalidParameterError
from src.core.registry import restore, snapshot
from src.data_layer import store_probe as sp
from src.data_layer import stores as st
from src.data_layer import zarr_source as zs
from src.tests.test_zarr_source import _build_store

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")


@pytest.fixture
def clean_ledgers():
    """Snapshot both registries so a test may probe and register temporarily."""
    saved_probes = snapshot(sp.STORE_PROBES)
    saved_stores = snapshot(st.GRIDDED_STORES)
    yield
    restore(sp.STORE_PROBES, saved_probes)
    restore(st.GRIDDED_STORES, saved_stores)


@pytest.fixture(scope="module")
def hostile_store(tmp_path_factory):
    """One timestep per chunk, every level, the whole globe - the WeatherBench 0.25 layout."""
    return _build_store(tmp_path_factory.mktemp("store-probe-stores") / "hostile.zarr",
                        time_chunk=1)


@pytest.fixture(scope="module")
def friendly_store(tmp_path_factory):
    """Chunked small in space as well as in time - the layout a regional crop wants.

    `test_zarr_source.py`'s "friendly" store is time-contiguous but still spans the globe in
    each chunk, which is friendly to a whole-globe read and emphatically not to a 32-degree
    box: probed with a regional crop it amplifies 247x. That is a true fact about that layout
    and the wrong fixture for this test, so this one is chunked the way the local cache is
    written rather than the way the archive is.
    """
    import numpy as np

    path = tmp_path_factory.mktemp("store-probe-stores") / "friendly.zarr"
    rng = np.random.default_rng(99)
    dataset = xr.Dataset(
        {"temperature": (("time", "level", "latitude", "longitude"),
                         rng.standard_normal((24, 8, 128, 256)).astype("float32"))},
        coords={"time": np.arange("2020-01-01", "2020-01-31", np.timedelta64(6, "h"),
                                  dtype="datetime64[ns]")[:24],
                "level": [1000, 925, 850, 700, 500, 300, 200, 100],
                "latitude": np.linspace(90, -90, 128),
                "longitude": np.linspace(0, 359, 256)})
    dataset.chunk({"time": 24, "level": 8, "latitude": 32, "longitude": 32}).to_zarr(
        str(path), mode="w", consolidated=True)
    return str(path)


def _crop(store, **overrides):
    kwargs = dict(store=store, variables=("temperature",),
                  time_start="2020-01-01", time_end="2020-01-03",
                  lat_min=0.0, lat_max=32.0, lon_min=0.0, lon_max=32.0,
                  levels=(850, 700, 500, 300), n_levels_analysis=4)
    kwargs.update(overrides)
    return zs.CropSpec(**kwargs)


def _store(**overrides) -> st.GriddedStore:
    kwargs = dict(
        name="probe_fixture", uri="gs://bucket/thing.zarr", domain="reanalysis",
        access="anonymous", vertical_dim="level", grid=(8, 8), resolution_deg=1.0,
        cadence_hours=6, levels=1, note="A fixture store, good for nothing in particular.",
        chunks=st.ChunkFacts(megabytes_per_chunk=None, method="not measured"))
    kwargs.update(overrides)
    return st.GriddedStore(**kwargs)


# ============================================================== probing a real local store

def test_a_probe_reads_a_stores_structure_without_transferring_data(hostile_store):
    probe = sp.probe_store(hostile_store, variables=["temperature"])
    assert probe.outcome == "described"
    assert probe.evidence == "probe run"
    assert probe.dimensions["latitude"] == 128 and probe.dimensions["time"] == 24
    assert probe.variables == ("temperature",)
    entry = probe.variable_structure["temperature"]
    assert entry["chunks"] == [1, 8, 128, 256]
    assert probe.megabytes_per_chunk == pytest.approx(1.049, abs=0.01)


def test_the_probe_reports_the_worst_chunk_not_the_average(tmp_path):
    """A store is as expensive to crop as its most expensive variable.

    The two variables carry **different dtypes** on purpose. `_build_store`'s pair are both
    float32 and therefore identically sized, so a mutation replacing the maximum with the mean
    passed against it - the fixture could not tell the two apart, which made the assertion
    decorative. Averaging hides the variable that hurts, and this is what says so.
    """
    import numpy as np

    path = tmp_path / "mixed_widths.zarr"
    rng = np.random.default_rng(7)
    dataset = xr.Dataset(
        {"narrow": (("time", "latitude", "longitude"),
                    rng.standard_normal((4, 64, 64)).astype("float32")),
         "wide": (("time", "latitude", "longitude"),
                  rng.standard_normal((4, 64, 64)).astype("float64"))},
        coords={"time": np.arange("2020-01-01", "2020-01-05",
                                  np.timedelta64(1, "D"), dtype="datetime64[ns]"),
                "latitude": np.linspace(90, -90, 64),
                "longitude": np.linspace(0, 359, 64)})
    dataset.chunk({"time": 1, "latitude": 64, "longitude": 64}).to_zarr(
        str(path), mode="w", consolidated=True)

    probe = sp.probe_store(str(path))
    sizes = [v["chunk_megabytes"] for v in probe.variable_structure.values()]
    assert len(sizes) == 2 and min(sizes) < max(sizes), (
        "the fixture must contain variables of different sizes or it proves nothing")
    assert probe.megabytes_per_chunk == pytest.approx(max(sizes))
    assert probe.megabytes_per_chunk != pytest.approx(sum(sizes) / len(sizes))


def test_acceptance_a_hostile_store_is_characterised_as_hostile_before_anyone_crops_it(
        hostile_store, tmp_path):
    """TG10.3's acceptance criterion. The load-bearing word is *before*."""
    cache = tmp_path / "cache"
    crop = _crop(hostile_store)
    probe = sp.probe_store(hostile_store, variables=["temperature"], crop=crop)

    assert probe.chunk_hostile is True
    assert probe.amplification >= sp.HOSTILE_AMPLIFICATION
    assert probe.crop is not None, "an amplification with no crop attached is not a fact"
    assert probe.crop["content_key"] == crop.content_key()

    # Nothing was materialised: no cache entry exists, and the figure agrees with the
    # prediction made from chunk metadata alone.
    assert not cache.exists()
    assert not zs.is_cached(crop, str(cache))
    dataset, _counter = zs.open_dataset(hostile_store, chunks={})
    try:
        predicted = zs.assess_access_pattern(dataset, crop)
    finally:
        dataset.close()
    assert probe.amplification == pytest.approx(predicted["amplification"], rel=1e-9)


def test_a_friendly_store_is_not_called_hostile(friendly_store):
    """A layout the request lines up with, which is the only thing "friendly" can mean.

    Hostility is a property of a *pairing*, never of a store alone: this same store amplifies
    30x for a request that straddles its chunk boundaries and asks for half the levels. That is
    why an amplification is refused unless the crop it was computed for travels with it.
    """
    aligned = _crop(friendly_store, time_start="2020-01-01", time_end="2020-01-07",
                    lat_min=45.0, lat_max=90.0, lon_min=0.0, lon_max=44.0, levels=())
    probe = sp.probe_store(friendly_store, variables=["temperature"], crop=aligned)
    assert probe.chunk_hostile is False
    assert probe.amplification < sp.HOSTILE_AMPLIFICATION


def test_hostility_is_three_valued_and_unknown_is_never_folded_into_fine(hostile_store):
    """The two states D43 confused: "nobody measured" and "it is fine"."""
    probe = sp.probe_store(hostile_store, variables=["temperature"])
    assert probe.amplification is None
    assert probe.chunk_hostile is None
    assert probe.chunk_hostile is not False


# ============================================================== a refusal is a result

def test_network_being_off_is_recorded_rather_than_raised(monkeypatch):
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    probe = sp.probe_store("gs://weatherbench2/datasets/era5/anything.zarr")
    assert probe.outcome == "network_disabled"
    assert zs.NETWORK_ENV_VAR in probe.refusal_detail
    assert "not about the store" in probe.refusal_detail


def test_an_unopenable_store_is_recorded_with_its_reason(tmp_path):
    probe = sp.probe_store(str(tmp_path / "nothing_here.zarr"))
    assert probe.outcome == "unreachable"
    assert probe.refusal_detail, "a refusal without its reason cannot be told from a typo"
    assert "nothing_here" in probe.refusal_detail


def test_a_store_that_is_not_zarr_is_recorded_as_not_readable(tmp_path):
    directory = tmp_path / "not_zarr.zarr"
    directory.mkdir()
    (directory / "readme.txt").write_text("this is not a zarr store", encoding="utf-8")
    probe = sp.probe_store(str(directory))
    assert probe.outcome in ("not_readable", "unreachable")
    assert probe.refusal_detail


@pytest.mark.parametrize("message, expected", [
    ("Forbidden: anonymous caller does not have storage.objects.get", "needs_credentials"),
    ("HTTP Error 401: Unauthorized", "needs_credentials"),
    ("Access Denied", "needs_credentials"),
    ("could not read it as Zarr (KeyError: .zgroup)", "not_readable"),
    ("Name or service not known", "unreachable"),
])
def test_an_open_failure_is_classified_and_its_text_kept_verbatim(message, expected):
    """The difference between "no such bucket" and "403" is a typo versus an account."""
    outcome, detail = sp._classify(RuntimeError(message))
    assert outcome == expected
    assert message in detail and "RuntimeError" in detail


def test_a_refusal_recorded_without_a_reason_is_refused():
    with pytest.raises(InvalidParameterError):
        sp.StoreProbe(uri="gs://x/y.zarr", outcome="unreachable", evidence="probe run")


# ============================================================== the record itself

@pytest.mark.parametrize("kwargs", [
    {"uri": " ", "outcome": "described", "evidence": "probe run",
     "megabytes_per_chunk": 1.0},
    {"uri": "gs://x/y.zarr", "outcome": "went_fine", "evidence": "probe run"},
    {"uri": "gs://x/y.zarr", "outcome": "described", "evidence": "somebody said so",
     "megabytes_per_chunk": 1.0},
    {"uri": "gs://x/y.zarr", "outcome": "described", "evidence": "probe run",
     "probed_on": "  ", "megabytes_per_chunk": 1.0},
    {"uri": "gs://x/y.zarr", "outcome": "described", "evidence": "probe run"},
    {"uri": "gs://x/y.zarr", "outcome": "described", "evidence": "probe run",
     "megabytes_per_chunk": 1.0, "amplification": 3.0},
    {"uri": "gs://x/y.zarr", "outcome": "described", "evidence": "probe run",
     "megabytes_per_chunk": -1.0},
])
def test_an_incoherent_probe_record_is_refused(kwargs):
    with pytest.raises(InvalidParameterError):
        sp.StoreProbe(**kwargs)


def test_an_amplification_must_name_the_crop_it_is_the_ratio_for():
    with pytest.raises(InvalidParameterError) as excinfo:
        sp.StoreProbe(uri="gs://x/y.zarr", outcome="described", evidence="probe run",
                      dimensions={"time": 4}, amplification=51.1)
    assert "one access pattern" in str(excinfo.value)


def test_the_digest_describes_the_observation_and_not_the_day_it_was_taken():
    """Otherwise every look produces a new record and the ledger becomes a log."""
    common = dict(uri="gs://x/y.zarr", outcome="described", dimensions={"time": 4})
    monday = sp.StoreProbe(evidence="probe run", probed_on="2026-08-01", **common)
    friday = sp.StoreProbe(evidence="prior recorded inspection", probed_on="2026-08-05",
                           **common)
    assert monday.digest() == friday.digest()
    different = sp.StoreProbe(evidence="probe run", probed_on="2026-08-01",
                              uri="gs://x/y.zarr", outcome="described",
                              dimensions={"time": 5})
    assert different.digest() != monday.digest()


def test_a_probe_round_trips_through_its_dictionary_form(hostile_store):
    probe = sp.probe_store(hostile_store, variables=["temperature"])
    rebuilt = sp.StoreProbe.from_dict(probe.to_dict())
    assert rebuilt.digest() == probe.digest()
    assert rebuilt.probed_on == probe.probed_on and rebuilt.evidence == probe.evidence


# ============================================================== the ledger

def test_recording_the_same_observation_twice_is_one_entry(clean_ledgers, hostile_store):
    first = sp.probe_and_record(hostile_store, variables=["temperature"])
    before = len(sp.STORE_PROBES)
    second = sp.probe_and_record(hostile_store, variables=["temperature"])
    assert first.digest() == second.digest()
    assert len(sp.STORE_PROBES) == before


def test_the_ledger_finds_probes_by_uri_newest_first(clean_ledgers):
    old = sp.StoreProbe(uri="gs://x/y.zarr", outcome="described", evidence="probe run",
                        probed_on="2026-01-01", dimensions={"time": 1})
    new = sp.StoreProbe(uri="gs://x/y.zarr", outcome="described", evidence="probe run",
                        probed_on="2026-08-01", dimensions={"time": 2})
    other = sp.StoreProbe(uri="gs://x/z.zarr", outcome="unreachable", evidence="probe run",
                          probed_on="2026-08-02", refusal_detail="no such bucket")
    for probe in (old, new, other):
        sp.record_probe(probe)
    found = sp.probes_for_uri("gs://x/y.zarr")
    assert [p.probed_on for p in found] == ["2026-08-01", "2026-01-01"]
    assert sp.latest_probe_for("gs://x/y.zarr").digest() == new.digest()
    assert sp.latest_probe_for("gs://x/absent.zarr") is None
    assert sp.probe_by_digest(other.digest()).outcome == "unreachable"


def test_an_earlier_probe_is_not_erased_by_a_later_one(clean_ledgers):
    """A store that was hostile last year was hostile last year."""
    old = sp.StoreProbe(uri="gs://x/y.zarr", outcome="described", evidence="probe run",
                        probed_on="2026-01-01", crop={"description": "a crop"},
                        amplification=51.1)
    new = sp.StoreProbe(uri="gs://x/y.zarr", outcome="described", evidence="probe run",
                        probed_on="2026-08-01", crop={"description": "a crop"},
                        amplification=1.2)
    sp.record_probe(old)
    sp.record_probe(new)
    assert len(sp.probes_for_uri("gs://x/y.zarr")) == 2
    assert sp.latest_probe_for("gs://x/y.zarr").chunk_hostile is False


# ============================================================== persistence

def test_a_probe_is_written_atomically_content_addressed_and_never_overwritten(
        tmp_path, hostile_store):
    probe = sp.probe_store(hostile_store, variables=["temperature"])
    path = sp.save_probe(probe, str(tmp_path))
    assert os.path.basename(path) == "%s.json" % probe.digest()
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["digest"] == probe.digest()
    assert payload["schema"] == sp.PROBE_SCHEMA

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(" ")
    tampered = open(path, encoding="utf-8").read()
    assert sp.save_probe(probe, str(tmp_path)) == path
    assert open(path, encoding="utf-8").read() == tampered, (
        "an existing record must not be silently rewritten")
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]


def test_persisted_probes_reload_into_the_ledger(clean_ledgers, tmp_path, hostile_store,
                                                 friendly_store):
    for store in (hostile_store, friendly_store):
        sp.save_probe(sp.probe_store(store, variables=["temperature"]), str(tmp_path))
    restore(sp.STORE_PROBES, {})
    loaded = sp.load_probes(str(tmp_path))
    assert len(loaded) == 2
    assert len(sp.STORE_PROBES) == 2
    assert sp.load_probes(str(tmp_path / "absent")) == []


# ============================================================== the registration gate

def test_a_store_claiming_a_measurement_must_cite_the_probe_that_made_it(clean_ledgers):
    claimed = st.ChunkFacts(megabytes_per_chunk=54.0, method="store metadata")
    with pytest.raises(InvalidParameterError) as excinfo:
        st.register_store(_store(name="uncited", chunks=claimed))
    message = str(excinfo.value)
    assert "probe_digest" in message
    # The message must tell the author what to do, and it must name the method that put them
    # here. Asserting only that *something* raised let a mutation removing this check pass:
    # the next check happened to raise too, with an error about a ledger lookup that would
    # send an author looking in the wrong place entirely.
    assert "store metadata" in message, "the message must name the claim being made"
    assert "probe_and_record" in message, "the message must say how to satisfy the rule"
    assert "not measured" in message, "and what the honest alternative is"
    assert "uncited" not in st.GRIDDED_STORES


def test_a_store_may_not_both_cite_a_look_and_say_nobody_looked(clean_ledgers):
    probe = sp.record_probe(sp.StoreProbe(
        uri="gs://bucket/thing.zarr", outcome="described", evidence="probe run",
        dimensions={"time": 4}))
    with pytest.raises(InvalidParameterError):
        st.register_store(_store(name="contradictory", probe_digest=probe.digest()))


def test_a_probe_of_a_different_store_does_not_license_this_one(clean_ledgers):
    """Otherwise a probe of a friendly store licenses a hostile one, checkably."""
    elsewhere = sp.record_probe(sp.StoreProbe(
        uri="gs://bucket/somewhere_else.zarr", outcome="described", evidence="probe run",
        megabytes_per_chunk=1.0))
    with pytest.raises(InvalidParameterError) as excinfo:
        st.register_store(_store(
            name="borrowed", probe_digest=elsewhere.digest(),
            chunks=st.ChunkFacts(megabytes_per_chunk=1.0, method="store metadata")))
    assert "somewhere_else" in str(excinfo.value)


def test_an_unrecorded_digest_is_refused(clean_ledgers):
    with pytest.raises(InvalidParameterError) as excinfo:
        st.register_store(_store(
            name="invented", probe_digest="0" * 16,
            chunks=st.ChunkFacts(megabytes_per_chunk=1.0, method="store metadata")))
    assert "probe ledger" in str(excinfo.value)


def test_an_entry_may_not_quote_a_figure_the_cited_probe_never_saw(clean_ledgers):
    """D62 with a citation attached is D62 improved only in appearance."""
    probe = sp.record_probe(sp.StoreProbe(
        uri="gs://bucket/thing.zarr", outcome="described", evidence="probe run",
        megabytes_per_chunk=54.0, crop={"description": "a crop"}, amplification=51.1))
    with pytest.raises(InvalidParameterError) as excinfo:
        st.register_store(_store(
            name="overstated", probe_digest=probe.digest(),
            chunks=st.ChunkFacts(megabytes_per_chunk=8.0, method="store metadata")))
    assert "54.0" in str(excinfo.value)
    with pytest.raises(InvalidParameterError):
        st.register_store(_store(
            name="overstated_amp", probe_digest=probe.digest(),
            chunks=st.ChunkFacts(megabytes_per_chunk=54.0, method="store metadata",
                                 regional_amplification=2.0)))
    st.register_store(_store(
        name="honest", probe_digest=probe.digest(),
        chunks=st.ChunkFacts(megabytes_per_chunk=54.0, method="store metadata",
                             regional_amplification=51.1)))
    assert st.store_for("honest").probe_digest == probe.digest()


def test_a_probed_store_registers_end_to_end(clean_ledgers, hostile_store):
    """The recipe the docstring gives an author: probe, record, cite."""
    probe = sp.probe_and_record(hostile_store, variables=["temperature"],
                                crop=_crop(hostile_store))
    st.register_store(st.GriddedStore(
        name="freshly_probed", uri=hostile_store, domain="reanalysis", access="local",
        vertical_dim="level", grid=(128, 256), resolution_deg=1.4, cadence_hours=6, levels=8,
        note="A local fixture store, probed before it was registered.",
        probe_digest=probe.digest(),
        chunks=st.ChunkFacts(
            megabytes_per_chunk=probe.megabytes_per_chunk, method="live inspection",
            measured_on=probe.probed_on, regional_amplification=probe.amplification)))
    entry = st.store_for("freshly_probed")
    assert entry.probe_digest == probe.digest()
    assert sp.probe_by_digest(entry.probe_digest).chunk_hostile is True


# ============================================================== the built-in transcriptions

def test_every_registered_store_that_claims_a_measurement_cites_a_recorded_probe():
    for entry in st.GRIDDED_STORES.entries():
        store = entry.value
        if store.chunks.method == "not measured":
            assert store.probe_digest is None, entry.name
            continue
        assert store.probe_digest in sp.STORE_PROBES, entry.name
        assert sp.probe_by_digest(store.probe_digest).uri == store.uri, entry.name


def test_the_four_era5_records_are_transcriptions_and_are_counted_as_such():
    """The debt is published rather than hidden, so it can only fall where anyone can see."""
    transcribed = {p.uri for p in sp.transcribed_probes()}
    assert transcribed == {store.uri for store in st.BUILTIN_STORES}, (
        "every built-in figure is transcribed from an inspection this code did not run; "
        "when one is re-probed live, this set shrinks and the change is visible")
    for probe in st.BUILTIN_PROBES:
        assert probe.evidence == "prior recorded inspection"
        assert probe.outcome == "described"
        assert probe.note, "a transcription must say what it was transcribed from"


def test_the_transcriptions_carry_exactly_what_was_recorded_and_no_more():
    """Two of the four have a size and no shape, and that is left as it stands (D62)."""
    by_uri = {p.uri: p for p in st.BUILTIN_PROBES}
    hostile = by_uri[st.store_for("era5_0p25_6h").uri]
    assert hostile.megabytes_per_chunk == pytest.approx(54.0)
    assert hostile.amplification == pytest.approx(51.1)
    assert hostile.variable_structure["temperature"]["chunks"] == [1, 13, 721, 1440]
    assert hostile.chunk_hostile is True

    middle = by_uri[st.store_for("era5_0p7_6h").uri]
    assert middle.megabytes_per_chunk is None, "no inspection recorded a per-chunk size"
    assert middle.amplification == pytest.approx(26.2)
    assert "D43" in middle.note

    partial = by_uri[st.store_for("era5_1p5_6h").uri]
    assert partial.megabytes_per_chunk == pytest.approx(12.1)
    assert partial.variable_structure == {} and partial.dimensions == {}
    assert partial.amplification is None and partial.chunk_hostile is None


# ============================================================== the HTTP surface

def test_the_probe_ledger_route_publishes_the_transcription_debt(client):
    body = client.get("/api/v1/data/zarr/probes").json()
    assert body["count"] == len(sp.STORE_PROBES)
    assert body["transcribed"] == len(sp.transcribed_probes())
    assert set(body["outcomes"]) == set(sp.PROBE_OUTCOMES)
    assert set(body["evidence_kinds"]) == set(sp.EVIDENCE_KINDS)
    dates = [p["probed_on"] for p in body["probes"]]
    assert dates == sorted(dates, reverse=True)


def test_probing_with_the_network_off_answers_a_recorded_result_not_an_error(
        client, monkeypatch):
    """Unlike `/inspect`, which returns 409. A deployment that cannot reach a store needs
    that written down, not raised."""
    monkeypatch.delenv(zs.NETWORK_ENV_VAR, raising=False)
    response = client.post("/api/v1/data/zarr/probe",
                           json={"uri": "era5_0p25_6h", "persist": False})
    assert response.status_code == 200
    body = response.json()
    assert body["probe"]["outcome"] == "network_disabled"
    assert body["resolved_uri"].startswith("gs://weatherbench2/")
    assert body["requested"] == "era5_0p25_6h"
    assert body["saved_to"] is None


def test_probing_a_local_store_over_http_describes_it_and_persists_the_record(
        client, clean_ledgers, hostile_store, tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "DEFAULT_PROBE_DIR", str(tmp_path / "probes"))
    response = client.post("/api/v1/data/zarr/probe",
                           json={"uri": hostile_store, "variables": ["temperature"]})
    assert response.status_code == 200
    body = response.json()
    assert body["probe"]["outcome"] == "described"
    assert body["probe"]["variable_structure"]["temperature"]["chunks"] == [1, 8, 128, 256]
    assert os.path.exists(body["saved_to"])
    assert body["probe"]["digest"] in sp.STORE_PROBES


def test_a_probe_request_with_no_uri_is_a_client_error(client):
    """The one thing that stays an error: a fault in the request, not a fact about a store."""
    response = client.post("/api/v1/data/zarr/probe", json={"uri": "   "})
    assert response.status_code >= 400


def test_probing_over_http_costs_the_crop_when_one_is_stated(client, clean_ledgers,
                                                             hostile_store):
    response = client.post("/api/v1/data/zarr/probe", json={
        "uri": hostile_store, "variables": ["temperature"], "persist": False,
        "crop": {"store": hostile_store, "variables": ["temperature"],
                 "time_start": "2020-01-01", "time_end": "2020-01-03",
                 "lat_min": 0.0, "lat_max": 32.0, "lon_min": 0.0, "lon_max": 32.0,
                 "levels": [850, 700, 500, 300], "n_levels_analysis": 4},
    })
    assert response.status_code == 200
    probe = response.json()["probe"]
    assert probe["chunk_hostile"] is True
    assert probe["crop"]["content_key"] == _crop(hostile_store).content_key()


# ============================================================== opt-in live check

@pytest.mark.skipif(not zs.network_enabled(),
                    reason="set SPECTRALEARTH_ALLOW_NETWORK=1 to probe the live GCS stores")
def test_live_probe_reproduces_the_transcribed_era5_record():
    """The half of the acceptance criterion a local store cannot check.

    Recorded as NOT RUN in `VERIFICATION.md` until somebody runs it deliberately. When it does
    run, it says whether the transcription is still true - and if WeatherBench 2 has rechunked,
    that is the point rather than a nuisance.
    """
    store = st.store_for("era5_0p25_6h")
    probe = sp.probe_store(store.uri, variables=["temperature"])
    assert probe.outcome == "described", probe.refusal_detail
    assert probe.variable_structure["temperature"]["chunks"] == [1, 13, 721, 1440], (
        "the transcribed 54 MB/chunk hostile layout has changed")
    assert probe.megabytes_per_chunk == pytest.approx(54.0, abs=0.5)
