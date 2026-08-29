"""TG1.1: an axis has the role it was given, not the role its name suggests (standard E14).

The assumption being localised is small enough to state in one line and consequential enough
to have survived unexamined for the whole atmospheric line: *the last two dimensions are
space, unless one of them is called something like `lat`*. Both halves of that are ERA5
conventions. Neither is a fact about data.

Three properties are asserted here, and the third is the one that matters for the fork:

1.  **The atmospheric line is unchanged.** Every dimension arrangement the importer met before
    resolves to the same spatial pair, so no receipt moves.
2.  **A guess now leaves a trace.** `basis` records `declared`, `name` or `position` per axis,
    and it travels into the import provenance - so "the spatial axes were chosen by position"
    becomes a readable caveat on every orientation statistic instead of an invisible one.
3.  **A declaration cannot be overruled by a name, and inference can be refused outright.**
    That refusal is what a non-atmospheric adapter needs: a sensor archive with columns called
    `x` and `y` must not silently acquire a geometry, because a geometry is what licenses
    per-metre reporting, and this domain has no metre.

One deliberate divergence from the code replaced, in
`test_half_a_spatial_pair_is_not_a_pair`. It is a refusal where there used to be a wrong
answer, and it is asserted rather than tolerated.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.axes import (AXIS_NAME_HINTS, AxisResolution, AxisRoleNotDeclaredError, NameHint,
                           find_coordinate, resolve_axis_roles)
from src.core.domain import AxisSpec, DomainDeclaration
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.registry import restore, snapshot
from src.data_layer import exporters, importers

xr = pytest.importorskip("xarray")


# ============================================================ the atmospheric line is intact

@pytest.mark.parametrize("dims,expected", [
    (("latitude", "longitude", "time"), ("latitude", "longitude")),
    (("time", "lat", "lon"), ("lat", "lon")),
    (("time", "level", "latitude", "longitude"), ("latitude", "longitude")),
    (("t", "rows", "cols"), ("rows", "cols")),
    (("y", "x"), ("y", "x")),
])
def test_every_arrangement_the_importer_met_resolves_as_it_did_before(dims, expected):
    assert importers._spatial_dims(dims) == expected
    assert resolve_axis_roles(dims).spatial_pair() == expected


def test_a_transposed_file_is_put_back_the_right_way_round():
    """`(lon, lat)` must come back `(lat, lon)`.

    A transposed field still looks like a field, so every anisotropy and orientation statistic
    computed from it would be wrong in a way nothing downstream can detect.
    """
    assert resolve_axis_roles(("lon", "lat")).spatial_pair() == ("lat", "lon")
    assert resolve_axis_roles(("time", "longitude", "latitude")).spatial_pair() == (
        "latitude", "longitude")


# ==================================================================== the guess leaves a trace

def test_a_named_axis_is_recorded_as_a_hint_not_as_a_fact():
    resolution = resolve_axis_roles(("time", "latitude", "longitude"))
    assert resolution.roles == {"time": "time", "latitude": "space", "longitude": "space"}
    assert set(resolution.basis.values()) == {"name"}
    assert resolution.fully_declared is False


def test_the_positional_fallback_says_that_it_guessed():
    resolution = resolve_axis_roles(("t", "rows", "cols"))
    assert resolution.basis["rows"] == "position"
    assert resolution.basis["cols"] == "position"
    assert resolution.describe()["unresolved"] == ["t"]
    assert resolution.guessed_axes() == ("rows", "cols")


def test_a_declared_axis_is_recorded_as_declared_and_the_resolution_is_clean():
    resolution = resolve_axis_roles(("station", "reading"),
                                    {"station": "category", "reading": "time"})
    assert resolution.basis == {"station": "declared", "reading": "declared"}
    assert resolution.fully_declared is True
    resolution.require_declared("a test that should not raise")


# ============================================ a declaration cannot be overruled by a name

def test_a_declared_role_beats_the_name_that_contradicts_it():
    """The whole of E14 in one assertion: `y` is a category here because somebody said so."""
    resolution = resolve_axis_roles(("y", "x"), {"y": "category", "x": "time"})
    assert resolution.roles == {"y": "category", "x": "time"}
    assert resolution.spatial_pair() == (None, None)
    assert set(resolution.basis.values()) == {"declared"}


def test_a_declaration_stops_the_positional_fallback_from_inventing_a_grid():
    resolution = resolve_axis_roles(("t", "rows", "cols"),
                                    {"t": "time", "rows": "category", "cols": "category"})
    assert resolution.spatial_pair() == (None, None)
    assert "position" not in set(resolution.basis.values())


def test_declared_ordinals_decide_which_spatial_axis_is_the_row():
    resolution = resolve_axis_roles(
        ("a", "b"), {"a": AxisSpec("a", "space", ordinal=1),
                     "b": AxisSpec("b", "space", ordinal=0)})
    assert resolution.spatial_pair() == ("b", "a")


def test_an_axis_spec_carries_its_role_straight_into_the_resolution():
    axes = {"minute": AxisSpec("minute", "time", units="s"),
            "sensor": AxisSpec("sensor", "category", ordered=False)}
    resolution = resolve_axis_roles(("minute", "sensor"), axes)
    assert resolution.dims_with_role("time") == ("minute",)
    assert resolution.dims_with_role("category") == ("sensor",)


# ================================================================== inference can be refused

def test_a_domain_general_adapter_can_switch_the_name_conventions_off():
    """`x` and `y` in a sensor archive are column headings, not a coordinate system."""
    resolution = resolve_axis_roles(("y", "x"), allow_name_inference=False,
                                    allow_positional_inference=False)
    assert resolution.roles == {}
    assert resolution.describe()["unresolved"] == ["y", "x"]


def test_requiring_declared_roles_refuses_a_positional_guess_and_says_why():
    resolution = resolve_axis_roles(("t", "rows", "cols"))
    with pytest.raises(AxisRoleNotDeclaredError) as caught:
        resolution.require_declared("this domain has no metre")
    message = str(caught.value)
    assert "this domain has no metre" in message
    assert "rows (by position)" in message
    assert "per-metre" in message


def test_requiring_declared_roles_refuses_a_name_match_too():
    """A name match is a strong hint. A hint is still not a declaration."""
    with pytest.raises(AxisRoleNotDeclaredError):
        resolve_axis_roles(("latitude", "longitude")).require_declared("E14")


def test_an_axis_nobody_declared_is_not_declared_by_omission():
    resolution = resolve_axis_roles(("time", "extra"), {"time": "time"},
                                    allow_name_inference=False,
                                    allow_positional_inference=False)
    assert resolution.fully_declared is False
    with pytest.raises(AxisRoleNotDeclaredError):
        resolution.require_declared("every axis must be accounted for")


# ====================================================================== deliberate divergence

def test_half_a_spatial_pair_is_not_a_pair():
    """`(lat, time, cols)` used to import a (time x cols) slab and call it a field.

    The code replaced here searched for a latitude name and a longitude name independently,
    and fell back to the trailing two axes when either was missing - which pairs the *clock*
    with the columns and calls the result spatial. That is not a transposition, it is a
    different field, and no downstream statistic can detect it.

    A single resolved spatial axis is therefore discarded rather than half-trusted, and the
    import refuses instead. This is a behaviour change, and it is the only one in TG1.1.
    """
    assert importers._spatial_dims(("lat", "time", "cols")) == (None, None)
    resolution = resolve_axis_roles(("lat", "time", "cols"))
    assert resolution.roles == {"time": "time"}


# =================================================================== refusals on the inputs

def test_two_axes_with_one_name_are_refused():
    with pytest.raises(InvalidParameterError, match="distinct axis names"):
        resolve_axis_roles(("x", "x"))


def test_declaring_a_role_for_an_axis_that_is_not_there_is_refused():
    """A declaration describing different data is the likeliest reason a result looks wrong."""
    with pytest.raises(InvalidParameterError, match="axes that are present"):
        resolve_axis_roles(("time", "value"), {"tim": "time"})


def test_two_declared_axes_at_the_same_position_are_refused():
    with pytest.raises(InvalidParameterError, match="distinct ordinals"):
        resolve_axis_roles(("a", "b"), {"a": AxisSpec("a", "space", ordinal=0),
                                        "b": AxisSpec("b", "space", ordinal=0)})


def test_an_unknown_role_lists_the_known_ones():
    with pytest.raises(UnknownNameError) as caught:
        resolve_axis_roles(("a",), {"a": "spatial"})
    assert "space" in str(caught.value)


def test_a_negative_ordinal_is_refused_on_the_axis_spec():
    with pytest.raises(InvalidParameterError, match="non-negative"):
        AxisSpec("a", "space", ordinal=-1)


# ======================================================================= the hint registry

def test_a_new_naming_convention_registers_without_editing_src():
    """The plugin acceptance criterion, applied to axis vocabulary.

    An archive that spells its axes `northing` and `easting` is onboarded by registering, from
    outside `src/`, exactly as a transform is.
    """
    state = snapshot(AXIS_NAME_HINTS)
    try:
        AXIS_NAME_HINTS.add("space_y", NameHint("space", 0, ("northing",)), replace=True)
        AXIS_NAME_HINTS.add("space_x", NameHint("space", 1, ("easting",)), replace=True)
        resolution = resolve_axis_roles(("time", "easting", "northing"))
        assert resolution.spatial_pair() == ("northing", "easting")
        assert resolution.basis["northing"] == "name"
    finally:
        restore(AXIS_NAME_HINTS, state)
    # ...and the atmospheric vocabulary is back afterwards.
    assert resolve_axis_roles(("lat", "lon")).spatial_pair() == ("lat", "lon")


def test_two_hints_claiming_one_name_are_refused_rather_than_resolved_by_import_order():
    state = snapshot(AXIS_NAME_HINTS)
    try:
        AXIS_NAME_HINTS.add("depth_sounding", NameHint("level", 0, ("lat",)))
        with pytest.raises(InvalidParameterError, match="claimed by exactly one hint"):
            resolve_axis_roles(("lat", "lon"))
    finally:
        restore(AXIS_NAME_HINTS, state)


def test_a_hint_with_an_unknown_role_is_refused_at_construction():
    with pytest.raises(UnknownNameError):
        NameHint("spatial", 0, ("q",))


def test_find_coordinate_answers_a_spelling_question_and_nothing_more():
    assert find_coordinate(["time", "latitude", "longitude"], "space", 0) == "latitude"
    assert find_coordinate(["time", "lat", "lon"], "space", 1) == "lon"
    assert find_coordinate(["time", "lat"], "space", 1) is None
    assert find_coordinate(["station", "reading"], "space", 0) is None


# ================================================================ the domain declaration path

def _declaration(**overrides):
    kwargs = dict(
        name="sensor_archive", description="Two channels on a regular clock.",
        licence="CC-BY-4.0 (fixture)",
        axes=(AxisSpec("minute", "time", units="s"),
              AxisSpec("sensor", "category", ordered=False)),
        violations=("no_physical_metric", "no_propagation_speed"))
    kwargs.update(overrides)
    return DomainDeclaration(**kwargs)


def test_a_domain_resolves_its_own_axes_from_its_declaration():
    resolution = _declaration().resolve_axes(("minute", "sensor"))
    assert resolution.fully_declared is True
    assert resolution.dims_with_role("time") == ("minute",)


def test_a_domain_refuses_an_axis_it_never_declared_rather_than_guessing_it():
    """`y` would resolve to a spatial axis by name. The domain has no space, so it must not."""
    declaration = _declaration(
        axes=(AxisSpec("minute", "time", units="s"), AxisSpec("y", "category", ordered=False)))
    with pytest.raises(InvalidParameterError, match="axes that are present"):
        declaration.resolve_axes(("minute", "sensor"))


def test_a_declaration_describes_its_ordinals():
    axis = AxisSpec("row", "space", ordinal=0)
    assert axis.describe()["ordinal"] == 0
    assert _declaration().describe()["axes"][0]["ordinal"] is None


# =========================================================== the record reaches the importer

def _era5_like(tmp_path, dims=("time", "level", "latitude", "longitude"), shape=(2, 3, 4, 5)):
    values = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
    dataset = xr.Dataset(
        {"t": (dims, values, {"units": "K"})},
        coords={"time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
                "level": [850, 500, 300], "latitude": np.linspace(60, -4, shape[2]),
                "longitude": np.linspace(0, 64, shape[3])})
    path = tmp_path / "era5.nc"
    dataset.to_netcdf(str(path), engine="h5netcdf")
    return path.read_bytes()


def test_inspect_reports_how_each_axis_was_identified(tmp_path):
    record = importers.inspect(_era5_like(tmp_path), "era5.nc")
    axes = record["variables"]["t"]["axes"]
    assert axes["roles"]["latitude"] == "space"
    assert axes["basis"] == {"time": "name", "level": "name",
                             "latitude": "name", "longitude": "name"}
    assert axes["fully_declared"] is False


def test_declared_roles_reach_the_importer_and_are_believed(tmp_path):
    """Pinning `level` and `latitude` as the field leaves `time` and `longitude` to select."""
    payload = _era5_like(tmp_path)
    roles = {"time": "time", "longitude": "category", "level": "space", "latitude": "space"}
    record = importers.inspect(payload, "era5.nc", axis_roles=roles)
    assert record["variables"]["t"]["spatial_dims"] == ["level", "latitude"]
    assert sorted(record["variables"]["t"]["extra_dims"]) == ["longitude", "time"]

    result = importers.read_field(payload, "era5.nc", variable="t",
                                  selection={"time": 0, "longitude": 1}, axis_roles=roles)
    assert np.array(result["field_data"]).shape == (3, 4)
    assert set(result["provenance"]["axes"]["basis"].values()) == {"declared"}


def test_a_role_declared_for_an_absent_axis_is_a_typo_and_is_refused(tmp_path):
    with pytest.raises(InvalidParameterError, match="roles only for axes this file has"):
        importers.inspect(_era5_like(tmp_path), "era5.nc", axis_roles={"lattitude": "space"})


def test_an_imported_flat_grid_records_its_axes_as_declared():
    """A CSV grid's axes are built by the reader, so nothing about them is a guess."""
    payload = exporters.export_field([[1.0, 2.0], [3.0, 4.0]], "csv")
    provenance = importers.read_field(payload, "f.csv")["provenance"]
    assert provenance["axes"]["basis"] == {"y": "declared", "x": "declared"}
    assert provenance["axes"]["fully_declared"] is True


def test_the_import_provenance_carries_the_positional_caveat(tmp_path):
    """The reason this record exists: an orientation statistic computed from these axes rests
    on the trailing-axes convention, and the file that cites it now says so."""
    payload = _era5_like(tmp_path, dims=("time", "level", "rows", "cols"))
    result = importers.read_field(payload, "grid.nc", variable="t",
                                  selection={"time": 0, "level": 0})
    axes = result["provenance"]["axes"]
    assert axes["basis"]["rows"] == "position"
    assert axes["basis"]["cols"] == "position"
    assert axes["fully_declared"] is False


def test_a_resolution_is_immutable():
    resolution = resolve_axis_roles(("lat", "lon"))
    assert isinstance(resolution, AxisResolution)
    with pytest.raises(Exception):
        resolution.axes = ("x", "y")            # type: ignore[misc]


# =============================================== the field's own coordinate handling (D56)

torch = pytest.importorskip("torch")
from src.physical_core.field import PhysicalField                          # noqa: E402


def _field(coord_names, height=4, width=6, **kwargs):
    row, column = coord_names
    data = torch.arange(height * width, dtype=torch.float64).reshape(height, width)
    coords = {row: torch.linspace(60.0, 30.0, height, dtype=torch.float64),
              column: torch.linspace(0.0, 50.0, width, dtype=torch.float64)}
    return PhysicalField(data, coords=coords, **kwargs)


@pytest.mark.parametrize("names", [("lat", "lon"), ("y", "x")])
def test_the_short_coordinate_names_split_exactly_as_they_did(names):
    """The atmospheric line uses these two spellings, and neither may move."""
    row, column = names
    parts = _field(names).split_field(train_ratio=0.5, val_ratio=0.25)
    assert parts["train"].coords[column].numel() == 3
    assert parts["train"].coords[row].numel() == 4          # the row axis is not cut
    assert parts["test"].coords[column].numel() == 2


def test_a_full_length_coordinate_name_is_now_cut_by_the_split():
    """D56. `longitude` fell through the hardcoded `k in ["x","lon"]` test and was cloned.

    The split then returned a 3-column field carrying a 6-entry longitude vector - a
    coordinate that no longer describes its own data, on the exact spelling ERA5 and CF both
    use. Nothing raised, and `GridSpec.from_coords` would go on to read that vector.
    """
    parts = _field(("latitude", "longitude")).split_field(train_ratio=0.5, val_ratio=0.25)
    assert parts["train"].data.shape == (4, 3)
    assert parts["train"].coords["longitude"].numel() == 3
    assert parts["train"].coords["latitude"].numel() == 4


def test_a_full_length_coordinate_name_is_now_resized_along_the_right_axis():
    """The other half of D56: `latitude` was interpolated to the *column* count."""
    scaled = _field(("latitude", "longitude")).scale_resolution((8, 3))
    assert scaled.data.shape == (8, 3)
    assert scaled.coords["latitude"].numel() == 8
    assert scaled.coords["longitude"].numel() == 3


def test_a_declared_role_decides_which_coordinate_the_split_cuts():
    """Coordinates called `alpha` and `beta` mean nothing to any naming convention."""
    field = _field(("alpha", "beta"),
                   axis_roles={"alpha": "space", "beta": "space"})
    parts = field.split_field(train_ratio=0.5, val_ratio=0.25)
    assert parts["train"].coords["beta"].numel() == 3       # declared second: the column
    assert parts["train"].coords["alpha"].numel() == 4


def test_a_declaration_survives_a_split_and_a_resample():
    field = _field(("alpha", "beta"), axis_roles={"alpha": "space", "beta": "space"})
    assert field.split_field()["train"].axis_roles == {"alpha": "space", "beta": "space"}
    assert field.scale_resolution((8, 3)).axis_roles == {"alpha": "space", "beta": "space"}


def test_an_undeclared_unrecognised_coordinate_keeps_the_old_behaviour():
    """Preserved deliberately: the two name lists disagreed, and both quirks are kept.

    `scale_resolution` treated an unknown coordinate as a column and `split_field` treated it
    as a row. Reconciling them would move results computed before TG1.1 for no gain, so the
    inconsistency is recorded rather than quietly repaired.
    """
    field = _field(("alpha", "beta"))
    parts = field.split_field(train_ratio=0.5, val_ratio=0.25)
    assert parts["train"].coords["beta"].numel() == 6       # cloned, as before
    scaled = field.scale_resolution((8, 3))
    assert scaled.coords["alpha"].numel() == 3              # column length, as before
