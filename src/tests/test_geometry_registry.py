"""TG1.2: geometry is a registry, and operators ask what a geometry can do.

Three properties, in the order they matter.

**1. The atmospheric line does not move.** ``pixel``, ``cartesian`` and ``latlon`` were
transcribed into `geometry.py`, not rewritten. Every metric, area weight, crop and provenance
record is asserted against the value the pre-TG1.2 code produced, because a refactor that
changes a number is not a refactor.

**2. A fourth geometry registers from outside `src/`.** That is the roadmap's acceptance
criterion, and it is met here with a deliberately awkward geometry: `polar_scan`, a radar PPI
sweep whose rows are range gates and whose columns are azimuth. Its zonal metric grows with
range - ``r * dtheta`` - so it declares ``uniform_metric=False`` while declaring
``spherical=False``. A fourth geometry that is merely `cartesian` under another name would
prove nothing; this one is the case the old `if kind == "latlon": ... else: cartesian` shape
could not represent at all.

**3. The `else` branch has a precondition now.** Feeding `polar_scan` to `laplacian` under
the old code would have run the five-point Cartesian stencil at a single representative
spacing and returned a finite array labelled ``value per m^2``. It now refuses.
"""

import math

import pytest
import torch

from src.core.registry import restore, snapshot
from src.physical_core.field import PhysicalField
from src.physical_core.geometry import GEOMETRIES, Geometry, recognisers
from src.physical_core.grid import EARTH_RADIUS_M, GridError, GridSpec
from src.physical_core.operators import gradient, laplacian


# ===================================================================== 1. no movement

def test_the_three_builtin_geometries_are_registered_with_capabilities():
    assert set(GEOMETRIES.names()) >= {"pixel", "cartesian", "latlon"}
    caps = {name: GEOMETRIES.entry(name).capabilities for name in
            ("pixel", "cartesian", "latlon")}
    assert caps["pixel"]["physical_metric"] is False
    assert caps["cartesian"]["physical_metric"] is True
    assert caps["latlon"]["spherical"] is True
    # The one that carries the weight: latlon is the only builtin whose metric varies.
    assert [n for n in GEOMETRIES.names()
            if GEOMETRIES.entry(n).capabilities.get("uniform_metric") is False] == ["latlon"]


def test_is_physical_and_length_units_now_come_from_the_declaration():
    """These two read `kind != "pixel"` and `"pixel" if kind == "pixel" else "m"` before.

    Same answers, different source. The point is that a fourth geometry gets an answer at
    all, instead of being classified by not being one particular string.
    """
    assert GridSpec.pixel((4, 4)).is_physical is False
    assert GridSpec.pixel((4, 4)).length_units == "pixel"
    assert GridSpec.cartesian((4, 4), 100.0).is_physical is True
    assert GridSpec.cartesian((4, 4), 100.0).length_units == "m"
    assert GridSpec.latlon((4, 4), 40.0, 0.25, 170.0, 0.25).length_units == "m"


@pytest.mark.parametrize("spec", [
    GridSpec.pixel((8, 12)),
    GridSpec.cartesian((8, 12), 2500.0, 5000.0),
    GridSpec.latlon((8, 12), 40.0, -0.25, 170.0, 0.25),
])
def test_metrics_match_the_pre_registry_arithmetic(spec):
    """Recompute each metric inline, from the formulae the old `grid.py` held."""
    if spec.kind == "latlon":
        expected_dy = EARTH_RADIUS_M * math.radians(abs(spec.dy))
        lats = torch.tensor([spec.lat0 + spec.dy * i for i in range(spec.height)],
                            dtype=torch.float64)
        expected_dx = EARTH_RADIUS_M * torch.cos(torch.deg2rad(lats)) * math.radians(spec.dx)
        mean_lat = float(lats.mean())
        expected_rep_dx = EARTH_RADIUS_M * math.cos(math.radians(mean_lat)) * math.radians(spec.dx)
    else:
        expected_dy = abs(spec.dy)
        expected_dx = torch.full((spec.height,), abs(spec.dx), dtype=torch.float64)
        expected_rep_dx = abs(spec.dx)

    assert spec.representative_dy_metres() == pytest.approx(expected_dy)
    assert spec.representative_dx_metres() == pytest.approx(expected_rep_dx)
    assert torch.allclose(spec.dx_metres(), expected_dx)
    assert torch.allclose(spec.dy_metres(),
                          torch.full((spec.height,), expected_dy, dtype=torch.float64))


def test_cell_area_and_weights_are_unchanged_for_a_sphere():
    spec = GridSpec.latlon((6, 9), 60.0, -1.0, 0.0, 1.0)
    lats = spec.latitudes()
    north = torch.clamp(lats + 0.5, max=90.0)
    south = torch.clamp(lats - 0.5, min=-90.0)
    expected = torch.abs((EARTH_RADIUS_M ** 2) * math.radians(1.0) * (
        torch.sin(torch.deg2rad(north)) - torch.sin(torch.deg2rad(south))))
    assert torch.allclose(spec.cell_area()[:, 0], expected)
    assert float(spec.area_weights().sum()) == pytest.approx(1.0)


@pytest.mark.parametrize("spec", [
    GridSpec.pixel((8, 12)),
    GridSpec.cartesian((8, 12), 2500.0, 5000.0),
    GridSpec.latlon((8, 12), 40.0, -0.25, 170.0, 0.25),
])
def test_subset_and_resample_produce_the_same_grids_as_before(spec):
    crop = spec.subset(row_start=2, row_stop=6, col_start=3, col_stop=9)
    assert crop.shape == (4, 6)
    assert crop.kind == spec.kind
    if spec.kind == "pixel":
        assert (crop.dy, crop.dx) == (1.0, 1.0)
    else:
        assert (crop.dy, crop.dx) == (spec.dy, spec.dx)
    if spec.kind == "latlon":
        assert crop.lat0 == pytest.approx(spec.lat0 + spec.dy * 2)
        assert crop.lon0 == pytest.approx(spec.lon0 + spec.dx * 3)

    up = spec.resampled((16, 24))
    assert up.shape == (16, 24)
    assert up.kind == spec.kind
    if spec.kind == "pixel":
        assert (up.dy, up.dx) == (1.0, 1.0)
    else:
        # align_corners=True preserves endpoints, so spacing scales as (n-1)/(m-1).
        assert up.dy == pytest.approx(spec.dy * 7 / 15)
        assert up.dx == pytest.approx(spec.dx * 11 / 23)


@pytest.mark.parametrize("spec", [
    GridSpec.pixel((8, 12), variable_units="K"),
    GridSpec.cartesian((8, 12), 2500.0, 5000.0, variable_units="K"),
    GridSpec.latlon((8, 12), 40.0, -0.25, 170.0, 0.25, variable_units="K"),
])
def test_provenance_still_round_trips(spec):
    assert GridSpec.from_provenance(spec.to_provenance()) == spec


def test_from_coords_answers_exactly_as_it_did_for_the_names_it_already_knew():
    lat = torch.linspace(40.0, 30.0, 32)
    lon = torch.linspace(170.0, 180.0, 64)
    assert GridSpec.from_coords({"lat": lat, "lon": lon}, (32, 64)).kind == "latlon"

    y = torch.linspace(0.0, 31000.0, 32)
    x = torch.linspace(0.0, 63000.0, 64)
    assert GridSpec.from_coords({"y": y, "x": x}, (32, 64),
                                {"grid_units": "m"}).kind == "cartesian"
    # No grid_units, so no metric: still a pixel grid, not a guessed one.
    assert GridSpec.from_coords({"y": y, "x": x}, (32, 64)).kind == "pixel"
    assert GridSpec.from_coords({"a": y, "b": x}, (32, 64)).kind == "pixel"


def test_a_full_length_latitude_name_is_now_recognised_as_a_sphere():
    """Behaviour change, deliberate, and the same family as D56.

    ``latitude``/``longitude`` is how CF and ERA5 spell it. `from_coords` matched only the
    abbreviation, so such a field received a **pixel** grid: gradients per pixel, no cos(lat)
    area weighting, `wavelength_units` reporting per-pixel - a field with a perfectly good
    spherical metric analysed as an array of numbers. Nothing in this repository builds a
    `PhysicalField` with those spellings (the importers normalise to ``lat``/``lon``), so no
    existing result moves; a caller constructing one by hand now gets the right geometry.
    """
    lat = torch.linspace(40.0, 30.0, 32)
    lon = torch.linspace(170.0, 180.0, 64)
    spec = GridSpec.from_coords({"latitude": lat, "longitude": lon}, (32, 64))
    assert spec.kind == "latlon"
    assert spec.lat0 == pytest.approx(40.0)


def test_a_positionally_resolved_pair_is_not_evidence_of_a_sphere():
    """Every 2D field has a row axis and a column axis. That is not a latitude.

    `find_coordinate` is name-based by design, but the guard is asserted here because the
    consequence of getting it wrong is a spherical metric invented for a microscope image.
    """
    a = torch.linspace(0.0, 1.0, 8)
    b = torch.linspace(0.0, 1.0, 12)
    assert GridSpec.from_coords({"dim_0": a, "dim_1": b}, (8, 12)).kind == "pixel"


def test_an_unregistered_kind_still_raises_griderror_and_names_what_exists():
    with pytest.raises(GridError, match="GridSpec.pixel"):
        GridSpec(kind="mercator", shape=(4, 4))
    with pytest.raises(GridError) as excinfo:
        GridSpec(kind="mercator", shape=(4, 4))
    # The valid set is now read from the registry rather than repeated in the message, so a
    # plugin geometry appears in the error the moment it registers.
    assert "latlon" in str(excinfo.value)


# ===================================================================== 2. a fourth geometry

class PolarScanGeometry(Geometry):
    """A radar PPI sweep: rows are range gates, columns are azimuth.

    ``dy`` is the range-gate spacing in metres and ``dx`` is the azimuth step in degrees, so
    the along-column arc length at row ``i`` is ``(r0 + i*dy) * radians(dx)`` - a metric that
    varies from row to row, exactly like `latlon`'s ``cos(lat)`` and for a completely
    different reason. ``r0`` has no named field on `GridSpec`, which is the point of
    `params`.
    """

    name = "polar_scan"
    coord_priority = 50

    def validate(self, spec):
        if spec.param("r0") is None:
            raise GridError(
                "polar_scan needs params={'r0': <range to the first gate, m>}; without it "
                "the arc length per azimuth step is undefined")
        if spec.dy <= 0 or spec.dx <= 0:
            raise GridError("polar_scan needs positive range and azimuth spacing")

    def _ranges(self, spec, device=None, dtype=torch.float64):
        idx = torch.arange(spec.height, device=device, dtype=dtype)
        return spec.param("r0") + spec.dy * idx

    def dx_metres(self, spec, device, dtype):
        return self._ranges(spec, device, dtype) * math.radians(spec.dx)

    def representative_dx(self, spec):
        return float(self._ranges(spec).mean()) * math.radians(spec.dx)

    def cell_area(self, spec, device, dtype):
        arc = self.dx_metres(spec, device, dtype) * spec.dy
        return arc.unsqueeze(1).expand(spec.height, spec.width).contiguous()

    def subset(self, spec, row_start, col_start, shape):
        return {"shape": shape,
                "params": {"r0": spec.param("r0") + spec.dy * row_start}}

    def describe(self, spec):
        return "polar_scan %dx%d, r0=%.0f m, dr=%.0f m, dtheta=%.2f deg" % (
            spec.shape[0], spec.shape[1], spec.param("r0"), spec.dy, spec.dx)


@pytest.fixture
def polar_scan():
    """Register the fourth geometry for one test, then put the registry back."""
    state = snapshot(GEOMETRIES)
    GEOMETRIES.add(
        "polar_scan", PolarScanGeometry(),
        description="Radar PPI sweep: range gates by azimuth.",
        capabilities={"physical_metric": True, "uniform_metric": False, "spherical": False,
                      "has_latitude": False, "length_units": "m"},
        tags=["test"])
    try:
        yield GridSpec(kind="polar_scan", shape=(6, 12), dy=250.0, dx=1.0,
                       params={"r0": 1000.0}, variable_units="dBZ")
    finally:
        restore(GEOMETRIES, state)


def test_a_fourth_geometry_registers_without_editing_src(polar_scan):
    """The roadmap's acceptance criterion for TG1.2.

    Nothing in `src/` mentions `polar_scan`. It is constructed, validated, measured, cropped,
    resampled and serialised entirely through the seam.
    """
    assert polar_scan.kind == "polar_scan"
    assert polar_scan.is_physical is True
    assert polar_scan.length_units == "m"
    assert "polar_scan" in polar_scan.describe()

    # The metric varies with range: the outermost gate's arc is 2.25x the innermost.
    dx = polar_scan.dx_metres()
    assert float(dx[0]) == pytest.approx(1000.0 * math.radians(1.0))
    assert float(dx[-1]) == pytest.approx(2250.0 * math.radians(1.0))
    assert float(polar_scan.area_weights().sum()) == pytest.approx(1.0)


def test_the_fourth_geometry_survives_a_crop_and_a_provenance_round_trip(polar_scan):
    crop = polar_scan.subset(row_start=2, row_stop=6)
    assert crop.shape == (4, 12)
    assert crop.param("r0") == pytest.approx(1500.0)

    record = polar_scan.to_provenance()
    assert record["params"] == {"r0": 1000.0}
    assert GridSpec.from_provenance(record) == polar_scan


def test_a_required_geometry_parameter_is_refused_at_construction(polar_scan):
    """Not at the first gradient. A grid with no metric has no metric now, not later."""
    with pytest.raises(GridError, match="r0"):
        GridSpec(kind="polar_scan", shape=(6, 12), dy=250.0, dx=1.0)


def test_the_registry_orders_recognisers_and_ends_at_pixel(polar_scan):
    order = [g.name for g in recognisers()]
    assert order[-1] == "pixel"
    assert order == ["latlon", "cartesian", "polar_scan", "pixel"]


# ===================================================================== 3. the else branch

def test_gradient_asks_the_geometry_which_way_is_north(polar_scan):
    """A plane's rows point +y; a descending-latitude grid's rows point south."""
    data = torch.arange(72, dtype=torch.float64).reshape(6, 12)

    descending = GridSpec.latlon((6, 12), 40.0, -1.0, 170.0, 1.0)
    out = gradient(PhysicalField(data, grid=descending))
    assert torch.allclose(out["d_dnorth"], -out["d_drow"])
    assert "north_sign=-1" in out["axis_convention"]

    ascending = GridSpec.latlon((6, 12), 30.0, 1.0, 170.0, 1.0)
    assert "north_sign=+1" in gradient(PhysicalField(data, grid=ascending))["axis_convention"]

    # A geometry that never declared a north gets the stated plane default, not a branch.
    out = gradient(PhysicalField(data, grid=polar_scan))
    assert torch.allclose(out["d_dnorth"], out["d_drow"])


def test_gradient_uses_the_per_row_metric_of_a_geometry_it_has_never_heard_of(polar_scan):
    """`gradient` was already capability-shaped: it divides by `dx_metres()` per row.

    Asserted anyway, because it is the half of the operator layer that generalised for free
    and the contrast with `laplacian` is the finding.
    """
    data = torch.zeros(6, 12, dtype=torch.float64)
    data[:, :] = torch.arange(12, dtype=torch.float64)
    out = gradient(PhysicalField(data, grid=polar_scan))
    # d/dcol of a ramp is 1 per column, divided by the arc length at that range.
    interior = out["d_dcol"][:, 1:-1]
    expected = 1.0 / polar_scan.dx_metres().unsqueeze(1).expand(6, 10)
    assert torch.allclose(interior, expected)
    assert out["units"] == "value per m"


def test_the_cartesian_laplacian_refuses_a_geometry_whose_metric_varies(polar_scan):
    """The defect the closed enum was hiding.

    Old code: ``if kind == "latlon": spherical else: cartesian_5point``. `polar_scan` is not
    ``latlon``, so it would have taken the else branch, been given one representative spacing
    for a metric that varies by a factor of 2.25 across the grid, and returned a finite array
    labelled ``value per m^2``. No exception, no warning, no NaN.
    """
    field = PhysicalField(torch.randn(6, 12, dtype=torch.float64), grid=polar_scan)
    with pytest.raises(GridError, match="uniform_metric=False"):
        laplacian(field)


def test_the_spherical_laplacian_is_selected_by_capability_not_by_name():
    spec = GridSpec.latlon((16, 32), 40.0, -1.0, 0.0, 1.0)
    out = laplacian(PhysicalField(torch.randn(16, 32, dtype=torch.float64), grid=spec))
    assert out["operator"] == "laplace_beltrami_sphere"

    flat = GridSpec.cartesian((16, 32), 1000.0)
    out = laplacian(PhysicalField(torch.randn(16, 32, dtype=torch.float64), grid=flat))
    assert out["operator"] == "cartesian_5point"


def test_latitudes_refuses_by_capability_rather_than_by_kind(polar_scan):
    with pytest.raises(GridError, match="has_latitude"):
        polar_scan.latitudes()
    with pytest.raises(GridError, match="has_latitude"):
        GridSpec.cartesian((4, 4), 100.0).longitudes()


def test_wavenumber_labels_follow_the_declared_length_unit(polar_scan):
    assert GridSpec.pixel((8, 8)).wavelength_units("rad_per_m") == "rad pixel^-1"
    assert polar_scan.wavelength_units("rad_per_m") == "rad m^-1"
    with pytest.raises(GridError, match="cycles_per_km"):
        GridSpec.pixel((8, 8)).wavelength_units("cycles_per_km")
