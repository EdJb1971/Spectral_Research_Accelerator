"""The gridded-store registry (roadmap TG10.1, standard E1).

Three things are under test and they are not equally interesting.

The dull half is that four ERA5 stores still resolve, the catalogue route still answers, and
the content key of an existing crop has not moved. That last one is not dull at all in
consequence: a change to `CropSpec.canonical` would orphan every materialised crop in the
local cache and make every recorded provenance record name a key that no longer resolves, so
one key is **pinned to a literal here**. If that assertion fails, the cache is the casualty,
and the failure should be read as a decision to take rather than a number to update.

The interesting half is what the registry now refuses. A store must name a domain something
has declared, must say how it is reached, must declare its vertical axis rather than have
`level` assumed of it (E14), and must say how its chunk figures were obtained. An undated
"live inspection" is refused outright, because defect D43 is exactly what an unmeasured store
treated as a known quantity costs.

The acceptance criterion is at the bottom: a fifth store, on a `depth` axis, added in a file
no core module knows about, reaching the HTTP catalogue with `zarr_source.py` and `main.py`
byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
import importlib
import json

import numpy as np
import pytest
import xarray as xr

from src.core.domain import DOMAIN_DECLARATIONS
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.registry import restore, snapshot
from src.data_layer import zarr_source as zs
from src.data_layer.stores import (ACCESS_REQUIREMENTS, BUILTIN_STORES, GRIDDED_STORES,
                                   MEASUREMENT_METHODS, CatalogueView, ChunkFacts,
                                   GriddedStore, catalogue_payload, domains_with_stores,
                                   register_builtin_stores, register_store, store_for,
                                   stores_for_domain, uri_for)

CORE_FILES = ("src/data_layer/zarr_source.py", "src/api/main.py")

#: The content key of one ordinary ERA5 crop, taken by running the **pre-TG10.1 module**
#: (`git show HEAD:src/data_layer/zarr_source.py`) against this spec, rather than by writing
#: down whatever the new code produced. A pin copied from the code it guards guards nothing.
PINNED_KEY_SPEC = dict(
    store="era5_0p25_6h", variables=("t",), time_start="2020-01-01", time_end="2020-01-02",
    lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0, levels=(850,))
PINNED_KEY = "09d0e1b7cacc25b0"


def _digest(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


@pytest.fixture
def clean_stores():
    """Snapshot the store registry so a test may register temporarily."""
    saved = snapshot(GRIDDED_STORES)
    yield
    restore(GRIDDED_STORES, saved)


def _store(**overrides) -> GriddedStore:
    kwargs = dict(
        name="probe_fixture", uri="gs://bucket/thing.zarr", domain="reanalysis",
        access="anonymous", vertical_dim="level", grid=(8, 8), resolution_deg=1.0,
        cadence_hours=6, levels=1, note="A fixture store, good for nothing in particular.",
        chunks=ChunkFacts(megabytes_per_chunk=None, method="not measured"))
    kwargs.update(overrides)
    return GriddedStore(**kwargs)


# ============================================================== the registry itself

def test_the_four_era5_stores_are_registered_and_declare_the_reanalysis_domain():
    assert len(GRIDDED_STORES) >= len(BUILTIN_STORES)
    for store in BUILTIN_STORES:
        entry = store_for(store.name)
        assert entry.uri.startswith("gs://weatherbench2/"), store.name
        assert entry.domain == "reanalysis", store.name
        assert entry.declaration().name == "reanalysis"
        assert entry.note.strip(), "every store says what it is good and bad FOR"


def test_registering_the_builtins_again_changes_nothing():
    """D35 in miniature: a catalogue whose contents depend on who imported first."""
    before = catalogue_payload()
    assert set(register_builtin_stores()) == {s.name for s in BUILTIN_STORES}
    assert catalogue_payload() == before


def test_a_store_is_discoverable_by_domain_and_by_capability(clean_stores):
    """Standard E2: ask what a store can do, do not hard-code which stores exist."""
    assert domains_with_stores() == ["reanalysis"]
    assert {s.name for s in stores_for_domain("reanalysis")} >= {s.name for s in BUILTIN_STORES}
    assert stores_for_domain("order_book") == []

    assert GRIDDED_STORES.with_capability("requires_credentials", True) == []
    register_store(_store(name="paid_store", access="credentials"))
    named = [e.name for e in GRIDDED_STORES.with_capability("requires_credentials", True)]
    assert named == ["paid_store"]
    measured = [e.name for e in GRIDDED_STORES.with_capability("chunk_measured", True)]
    assert set(measured) == {s.name for s in BUILTIN_STORES}, (
        "every ERA5 entry carries a measured chunk figure; the fixture store does not")
    assert "paid_store" not in measured


def test_every_registered_store_names_a_domain_that_is_actually_declared():
    """The check that stops a store being selectable and then failing at its refusals."""
    for entry in GRIDDED_STORES.entries():
        assert entry.value.domain in DOMAIN_DECLARATIONS, entry.name


def test_a_store_naming_an_undeclared_domain_is_refused_with_the_valid_names(clean_stores):
    with pytest.raises(UnknownNameError) as excinfo:
        register_store(_store(name="orphan", domain="oceanography"))
    message = str(excinfo.value)
    assert "reanalysis" in message and "order_book" in message
    assert "orphan" not in GRIDDED_STORES, "a refused store must not be half-registered"


def test_duplicate_registration_raises_unless_replacement_is_explicit(clean_stores):
    register_store(_store(name="twice"))
    with pytest.raises(Exception):
        register_store(_store(name="twice"))
    register_store(_store(name="twice", levels=3), replace=True)
    assert store_for("twice").levels == 3


# ============================================================== what an entry must say

@pytest.mark.parametrize("overrides, because", [
    ({"access": "public"}, "an access requirement that is not one of the declared three"),
    ({"note": "   "}, "a store with no note saying what it is good and bad for"),
    ({"grid": (0, 8)}, "a grid with a non-positive extent"),
    ({"grid": (8,)}, "a grid that is not a (rows, columns) pair"),
    ({"vertical_dim": None, "levels": 3}, "levels on an axis the store says it has not got"),
    ({"vertical_dim": "  "}, "a blank vertical axis name"),
    ({"uri": "bucket/thing.zarr"}, "a network store with no URI scheme"),
    ({"name": " "}, "an unnamed store"),
])
def test_a_store_entry_is_refused_when_it_does_not_say_what_it_must(overrides, because):
    with pytest.raises(InvalidParameterError):
        _store(**overrides)


def test_a_local_store_may_have_no_uri_scheme_because_a_path_has_none():
    """The exemption is narrow and deliberate: `local` means nothing crosses the network."""
    entry = _store(name="fixture", uri="fixture-era5", access="local")
    assert entry.uri == "fixture-era5"
    with pytest.raises(InvalidParameterError):
        _store(uri="fixture-era5", access="anonymous")


def test_a_store_with_no_vertical_axis_may_declare_none():
    entry = _store(vertical_dim=None, levels=0)
    assert entry.vertical_dim is None
    assert entry.to_dict()["vertical_dim"] is None


# ============================================================== chunk facts

def test_an_undated_live_inspection_is_refused():
    """An undated measurement of an archive that may rechunk describes nothing."""
    with pytest.raises(InvalidParameterError) as excinfo:
        ChunkFacts(megabytes_per_chunk=54.0, method="live inspection")
    assert "measured_on" in str(excinfo.value)
    dated = ChunkFacts(megabytes_per_chunk=54.0, method="live inspection",
                       measured_on="2026-08-20")
    assert dated.measured_on == "2026-08-20"


def test_an_unmeasured_store_may_not_quote_a_chunk_size():
    """The failure mode being blocked: a guess that reads exactly like a measurement."""
    with pytest.raises(InvalidParameterError):
        ChunkFacts(megabytes_per_chunk=12.0, method="not measured")
    admitted = ChunkFacts(megabytes_per_chunk=None, method="not measured")
    assert admitted.megabytes_per_chunk is None
    assert "nobody has looked" in admitted.to_dict()["method_means"]


@pytest.mark.parametrize("kwargs", [
    {"megabytes_per_chunk": 1.0, "method": "guessed"},
    {"megabytes_per_chunk": 0.0, "method": "store metadata"},
    {"megabytes_per_chunk": None, "method": "store metadata"},
    {"megabytes_per_chunk": 1.0, "method": "store metadata",
     "shape": (1, 2, 3), "dims": ("time", "latitude")},
])
def test_chunk_facts_refuse_an_incoherent_record(kwargs):
    with pytest.raises(InvalidParameterError):
        ChunkFacts(**kwargs)


def test_the_measured_era5_figures_survived_the_move_from_prose_to_fields():
    """The 51.1x amplification is the figure the module docstring and VERIFICATION.md quote."""
    hostile = store_for("era5_0p25_6h").chunks
    assert hostile.megabytes_per_chunk == pytest.approx(54.0)
    assert hostile.shape == (1, 13, 721, 1440)
    assert hostile.dims == ("time", "level", "latitude", "longitude")
    assert hostile.regional_amplification == pytest.approx(51.1)
    assert hostile.method == "live inspection" and hostile.measured_on == "2026-08-20"

    middle = store_for("era5_0p7_6h").chunks
    assert middle.regional_amplification == pytest.approx(26.2)
    assert "D43" in middle.note, "the open defect must be attached to the store it is about"

    unmeasured = store_for("era5_0p25_1h_full37").chunks
    assert unmeasured.method == "store metadata" and unmeasured.shape is None, (
        "a chunk shape nobody recorded must stay unstated rather than be reconstructed")


def test_the_declared_vocabularies_are_small_and_described():
    for table in (ACCESS_REQUIREMENTS, MEASUREMENT_METHODS):
        assert all(text.strip() for text in table.values())


# ============================================================== the catalogue view

def test_the_catalogue_view_reads_through_to_the_registry(clean_stores):
    assert isinstance(zs.CATALOGUE, CatalogueView)
    assert set(zs.CATALOGUE) == set(GRIDDED_STORES.names())
    register_store(_store(name="late_arrival"))
    assert "late_arrival" in zs.CATALOGUE, "the view is a window, not a copy"
    assert zs.CATALOGUE["late_arrival"]["domain"] == "reanalysis"


def test_the_catalogue_view_keeps_every_key_the_dictionary_carried():
    entry = zs.CATALOGUE["era5_0p25_6h"]
    for key in ("uri", "resolution_deg", "cadence_hours", "grid", "levels", "note"):
        assert key in entry, key
    assert entry["grid"] == [721, 1440]
    assert json.dumps(dict(zs.CATALOGUE)), "the view must serialise for the HTTP layer"


def test_the_catalogue_view_cannot_be_written_to():
    """A write would put a store in the catalogue with no domain and no chunk record."""
    with pytest.raises(TypeError):
        zs.CATALOGUE["invented"] = {"uri": "gs://nowhere"}
    with pytest.raises(AttributeError):
        zs.CATALOGUE.pop("era5_0p25_6h")


def test_an_uncatalogued_name_answers_no_rather_than_raising():
    """`CropSpec.uri` asks this question of every raw URI, and 'no' is a supported answer."""
    assert zs.CATALOGUE.get("not_a_store") is None
    assert "not_a_store" not in zs.CATALOGUE
    assert uri_for("not_a_store") is None
    with pytest.raises(KeyError):
        zs.CATALOGUE["not_a_store"]


# ============================================================== CropSpec

def test_the_content_key_of_an_existing_crop_has_not_moved():
    """Pinned. A drift here orphans the local cache and every recorded provenance record."""
    assert zs.CropSpec(**PINNED_KEY_SPEC).content_key() == PINNED_KEY


def test_the_vertical_axis_is_declared_and_separates_two_otherwise_identical_crops():
    pressure = zs.CropSpec(**PINNED_KEY_SPEC)
    depth = zs.CropSpec(**dict(PINNED_KEY_SPEC, vertical_dim="depth"))
    assert pressure.vertical_dim == "level"
    assert depth.content_key() != pressure.content_key()
    assert "vertical_dim" not in pressure.canonical()
    assert depth.canonical()["vertical_dim"] == "depth"


def test_the_vertical_axis_round_trips_through_a_provenance_record():
    spec = zs.CropSpec(**dict(PINNED_KEY_SPEC, vertical_dim="depth"))
    record = spec.to_provenance()
    assert record["vertical_dim"] == "depth"
    assert zs.CropSpec.from_provenance(record) == spec
    # An older record predating TG10.1 has no such key, and must still replay.
    legacy = {k: v for k, v in zs.CropSpec(**PINNED_KEY_SPEC).to_provenance().items()
              if k != "vertical_dim"}
    assert zs.CropSpec.from_provenance(legacy).vertical_dim == "level"


def test_a_blank_vertical_axis_name_is_refused_at_construction():
    with pytest.raises(InvalidParameterError):
        zs.CropSpec(**dict(PINNED_KEY_SPEC, vertical_dim=" "))


def test_a_spec_built_for_a_store_takes_that_stores_declared_axis(clean_stores):
    register_store(_store(name="ocean_fixture", vertical_dim="depth"))
    assert zs.vertical_dim_for_store("ocean_fixture") == "depth"
    assert zs.vertical_dim_for_store("era5_0p25_6h") == "level"
    # A raw URI declares nothing, so the caller's default stands rather than being guessed.
    assert zs.vertical_dim_for_store("gs://elsewhere/x.zarr") == "level"
    spec = zs.crop_for_store("ocean_fixture", variables=("thetao",),
                             time_start="2020-01-01", time_end="2020-01-02",
                             lat_min=0.0, lat_max=1.0, lon_min=0.0, lon_max=1.0)
    assert spec.vertical_dim == "depth"


# ============================================================== selection on a depth axis

def _depth_dataset() -> xr.Dataset:
    times = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    depth = np.array([0, 50, 200])
    lat = np.linspace(0.0, 3.0, 4)
    lon = np.linspace(0.0, 3.0, 4)
    values = np.arange(2 * 3 * 4 * 4, dtype="float32").reshape(2, 3, 4, 4)
    return xr.Dataset(
        {"thetao": (("time", "depth", "latitude", "longitude"), values)},
        coords={"time": times, "depth": depth, "latitude": lat, "longitude": lon})


def _depth_spec(**overrides) -> zs.CropSpec:
    kwargs = dict(store="gs://example/ocean.zarr", variables=("thetao",),
                  time_start="2020-01-01", time_end="2020-01-02",
                  lat_min=0.0, lat_max=2.0, lon_min=0.0, lon_max=2.0,
                  levels=(0, 50), vertical_dim="depth")
    kwargs.update(overrides)
    return zs.CropSpec(**kwargs)


def test_select_applies_the_vertical_selection_to_the_declared_axis():
    subset = zs.select(_depth_dataset(), _depth_spec())
    assert int(subset.sizes["depth"]) == 2
    assert subset.depth.values.tolist() == [0, 50]


def test_select_names_the_declared_axis_when_a_requested_value_is_absent():
    with pytest.raises(InvalidParameterError) as excinfo:
        zs.select(_depth_dataset(), _depth_spec(levels=(0, 999)))
    message = str(excinfo.value)
    assert "depth" in message and "999" in message


def test_a_depth_store_read_as_though_it_were_era5_selects_no_vertical_subset():
    """The bug TG10.1 removes, demonstrated: `level` is simply absent, so nothing is cut.

    Before the generalisation this was silent - a caller asking for two of three depths got
    all three and no error, because the hard-coded `"level" in subset.coords` was false. The
    full depth axis then flowed into the cache and the manifest recorded the request rather
    than what arrived.
    """
    subset = zs.select(_depth_dataset(), _depth_spec(vertical_dim="level"))
    assert int(subset.sizes["depth"]) == 3, (
        "with the wrong axis name the vertical selection quietly does nothing")


# ============================================================== the HTTP surface

def test_the_catalogue_route_is_generated_from_the_registry(client):
    body = client.get("/api/v1/data/zarr/catalogue").json()
    assert set(body["stores"]) == set(GRIDDED_STORES.names())
    assert body["store_domains"] == domains_with_stores()
    assert set(body["access_requirements"]) == set(ACCESS_REQUIREMENTS)
    hostile = body["stores"]["era5_0p25_6h"]
    assert hostile["domain"] == "reanalysis"
    assert hostile["vertical_dim"] == "level"
    assert hostile["chunks"]["regional_amplification"] == pytest.approx(51.1)
    assert hostile["access_means"] == ACCESS_REQUIREMENTS["anonymous"]


def test_the_cli_catalogue_still_emits_valid_json(capsys):
    assert zs._main(["catalogue"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(GRIDDED_STORES.names())


# ============================================================== the acceptance criterion

def test_acceptance_a_fifth_store_needs_no_core_edits(clean_stores, client):
    """TG10.1's acceptance criterion, executed literally.

    "A fifth store registers from a file no core module knows about, declares a vertical axis
    that is not ERA5's, and reaches the catalogue route - with zero edits to zarr_source.py
    or main.py."

    The "zero edits" half is verified by hashing the two core files before and after, the
    method `test_registries.py` uses for the data-source seam, because a claim about not
    editing files is checkable and should be checked.
    """
    before = {path: _digest(path) for path in CORE_FILES}
    baseline = set(GRIDDED_STORES.names())

    from src.tests import store_plugin_example  # noqa: F401  the whole point: just an import
    if store_plugin_example.STORE_NAME not in GRIDDED_STORES:
        # A module imported earlier in the session does not re-run its registration, and
        # `clean_stores` will have restored the registry to before it. Reloading re-runs it.
        importlib.reload(store_plugin_example)

    name = store_plugin_example.STORE_NAME
    assert name in GRIDDED_STORES
    assert set(GRIDDED_STORES.names()) == baseline | {name}

    entry = store_for(name)
    assert entry.vertical_dim == "depth", "the axis ERA5's code path could not assume"
    assert entry.domain == "reanalysis", (
        "R17: a gridded store that breaks no inherited assumption is a source, not a domain")
    assert entry.chunks.method == "not measured"

    listed = client.get("/api/v1/data/zarr/catalogue").json()["stores"]
    assert name in listed, "a store added in a new file must reach the catalogue route"
    assert listed[name]["vertical_dim"] == "depth"
    assert listed[name]["chunks"]["megabytes_per_chunk"] is None

    # And a crop built for it selects on `depth` without anyone naming the axis by hand.
    assert zs.vertical_dim_for_store(name) == "depth"

    after = {path: _digest(path) for path in CORE_FILES}
    assert before == after, (
        "adding a store modified a core file: %s"
        % [path for path in CORE_FILES if before[path] != after[path]])
