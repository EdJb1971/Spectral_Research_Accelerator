"""T4F.5: a pattern and a rule put back on the map, and what a projection may not claim.

1.  **The acceptance is a measurement, not an assertion.** The planted vortex is at a known cell
    in every frame, and the projection is graded against it across the whole record rather than
    at a chosen frame. The result has a boundary in it: a level-4 footprint holds the planted
    cell for nine frames and loses it for fifteen, because the flank offset is about `1.1` times
    the structure's width while the footprint reaches only half the filter support. Level 5
    holds it throughout. That boundary is the finding, and it is pinned here.

2.  **A single cell would have been wrong every time.** Not once in the record does a member's
    own peak cell coincide with the planted cell. Projecting a coefficient maximum onto one
    pixel would have looked precise and been wrong by six to twelve cells.

3.  **An intersection is not a location either.** Two bands of one level that resolve different
    axes intersect over the planted cell in 11 of 11 occurrences; two bands of one orientation
    at two levels intersect *beside* it in 11 of 11, because both flanks point the same way.
    Both counts are pinned, because the second is what stops the first being read as a rule.

4.  **Each grid says only what it can.** Degrees from a `latlon` grid, metres from the crop's
    origin on a `cartesian` one, cells and nothing else on a `pixel` one; a level with no
    declared axis is a number; a frame is a frame unless the record supplied carries dates.

5.  **The instances come from the code that counted them.** `anchor_verdicts` decides an
    anchor's eligibility once, `count_sequence` tallies it and the projection lists it, and
    every enumerated total is reconciled against the figures the rule published before anything
    is shown.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, ConstellationPattern, PatternCatalogue, SignatureMetric, SignaturePoint,
    calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_constellation import extract_constellations
from src.analysis_engine.spectral_events import (
    EventSeries, ObservationGrid, PatternEvent, events_from_catalogue,
)
from src.analysis_engine.spectral_feature import detect_features
from src.analysis_engine.spectral_invariance import sign_constellations
from src.analysis_engine.spectral_precursors import (
    LagFamily, RULE_NOT_DISTINGUISHED, RULE_PRECURSOR, precursor_report,
)
from src.analysis_engine.spectral_projection import (
    INSTANCE_SCHEMA, INSTANCE_SUPPORTING, INSTANCE_TRUNCATED, INSTANCE_UNFOLLOWED,
    INSTANCE_UNOBSERVED, LEVEL_DECLARED, LEVEL_NONE, LEVEL_UNDECLARED_AXIS, PLACE_CELLS_ONLY,
    PLACE_GEOGRAPHIC, PLACE_OFFSET, PROJECTION_CLAIM_BOUNDARY, PROJECTION_SCHEMA,
    REDUNDANCY_NOTE, TIME_CALENDAR, TIME_FRAME, common_extent, describe_projection,
    evidence_index,
    geography_of, project_occurrence, project_pattern, project_rule, scale_supports,
)
from src.analysis_engine.spectral_sequences import TransitionWindow
from src.analysis_engine.spectral_tracking import track_spectral_features
from src.benchmarks.core import get_benchmark
from src.core.claim_ladder import OUTSIDE_THE_LADDER
from src.core.errors import InvalidParameterError
from src.core.level_axis import PRESSURE_HPA
from src.physical_core.field import PhysicalField
from src.physical_core.grid import GridSpec
from src.physical_core.sequence import FieldSequence
from src.transform_engine.coefficient_field import CoefficientField, decompose_sequence

WAVELET = "haar"
LEVELS = 5
SCALES = [4, 5]
THRESHOLD_SIGMA = 4.0
METRIC = SignatureMetric(AttributeWeights(geometry=1.0, bearings=1.0, strengths=1.0,
                                          scales=1.0))

#: The haar filter's own parent-grid support at the two tracked levels, and half of it, which
#: is how far a footprint reaches. Written here so a test reads the number rather than the code.
SUPPORT = {"4": 16, "5": 32}

_CACHE: dict = {}


# ---------------------------------------------------------------------------------------------
# the record: one planted vortex, decomposed, tracked and enumerated once
# ---------------------------------------------------------------------------------------------


def _record():
    """The T4D.2 acceptance sequence, carried as far as T4E.1 and reused.

    Half a second of work, so it is done once. Everything below that needs a real coefficient,
    a real position and a known answer to grade against reads this.
    """
    if "record" not in _CACHE:
        bench = get_benchmark("advected_vortex_sequence")
        built = bench.make()
        truth = bench.known_answer()
        sequence = FieldSequence(built.fields, np.arange(len(built.fields), dtype=float))
        field = decompose_sequence(
            sequence, "swt", {"levels": LEVELS, "wavelet": WAVELET}).select(scales=SCALES)
        tracked = track_spectral_features(
            field, domain="synthetic", dataset="advected_vortex_sequence",
            variable="amplitude", time_units="frames", threshold_sigma=THRESHOLD_SIGMA)
        constellations = extract_constellations(tracked, cardinalities=(2,))
        _CACHE["record"] = (sequence, field, tracked, constellations, truth)
    return _CACHE["record"]


@pytest.fixture(scope="module")
def record():
    return _record()


@pytest.fixture(scope="module")
def geography(record):
    sequence, field = record[0], record[1]
    return geography_of(field, time_units="frames", values=sequence)


@pytest.fixture(scope="module")
def bare(record):
    """The same map with no physical record attached, so the refusals can be read."""
    return geography_of(record[1], time_units="frames")


def _of_bands(bands, constellations):
    return tuple(item for item in constellations if item.bands == bands)


def _planted(truth, time):
    return truth["positions_rowcol"][int(time)]


def _catalogue(constellations, signatures=None):
    """One pattern per band pair, assembled rather than agglomerated.

    T4E.3's clustering has its own suite and costs eleven seconds on this record; what this one
    needs is a catalogue whose members are real signatures of real constellations. One test
    below does run `cluster_signatures`, so the index is held to a genuinely clustered
    catalogue as well as to this one.
    """
    if signatures is None:
        signatures = sign_constellations(constellations, scale_invariant=False).signatures
    by_key = {tuple(item.key()): item for item in constellations}
    groups: dict = {}
    for signature in signatures:
        groups.setdefault(by_key[tuple(signature.key)].bands, []).append(signature)
    replicates = sorted(groups.values(), key=len)[-1][:3]
    tolerance = calibrate_signature_tolerance(replicates, metric=METRIC)
    patterns = []
    for index, bands in enumerate(sorted(groups)):
        members = tuple(sorted(groups[bands], key=lambda item: (item.time, item.key)))
        points = [SignaturePoint.from_signature(member) for member in members]
        patterns.append(ConstellationPattern(
            pattern_id=index + 1, members=members, centroid=points[0],
            tolerance_radius=tolerance.value,
            observed_radius=max(METRIC.distance(point, points[0]) for point in points)))
    return PatternCatalogue(patterns=tuple(patterns), metric=METRIC, tolerance=tolerance,
                            n_signatures=len(signatures))


@pytest.fixture(scope="module")
def catalogue(record):
    return _catalogue(record[3])


@pytest.fixture(scope="module")
def index(catalogue, record, geography):
    return evidence_index(catalogue, record[3], geography)


def _regrid(field, grid):
    """The same coefficients under a different geometry, so a grid's answer can be read."""
    return CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times, grid=grid,
        config=dict(field.config), level=field.level, level_axis=field.level_axis)


# ---------------------------------------------------------------------------------------------
# section 1: the map, and what each grid is entitled to say
# ---------------------------------------------------------------------------------------------


def test_a_cartesian_grid_gives_a_distance_and_refuses_to_call_it_a_position(geography):
    place = geography.place(10.0, 20.0)
    assert place.basis == PLACE_OFFSET
    assert place.coordinates["northing"] == pytest.approx(10.0 * geography.grid.dy)
    assert place.coordinates["easting"] == pytest.approx(20.0 * geography.grid.dx)
    assert "not a position on the Earth" in place.refusal
    assert "latitude" not in place.coordinates


def test_a_latlon_grid_gives_degrees_in_its_own_convention(record):
    grid = GridSpec.latlon((128, 128), lat0=70.0, dlat=-0.25, lon0=-30.0, dlon=0.25)
    geo = geography_of(_regrid(record[1], grid), time_units="frames")
    place = geo.place(8.0, 12.0)
    assert place.basis == PLACE_GEOGRAPHIC
    assert place.coordinates["latitude"] == pytest.approx(70.0 - 0.25 * 8.0)
    assert place.coordinates["longitude"] == pytest.approx(-30.0 + 0.25 * 12.0)
    assert place.refusal is None
    assert "row 0 at latitude 70" in place.units


def test_a_pixel_grid_gives_cells_and_names_the_kind_it_refused_for(record):
    geo = geography_of(_regrid(record[1], GridSpec.pixel((128, 128))), time_units="frames")
    place = geo.place(3.0, 4.0)
    assert place.basis == PLACE_CELLS_ONLY
    assert place.coordinates == {}
    assert "pixel grid declares no length and no latitude" in place.refusal
    assert "invent a geography" in place.refusal


def test_only_a_grid_whose_columns_close_declares_a_seam(record):
    closed = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=360.0 / 128.0)
    regional = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=0.25)
    assert geography_of(_regrid(record[1], closed)).wraps is True
    assert geography_of(_regrid(record[1], regional)).wraps is False


def test_a_longitude_past_the_seam_comes_back_inside_the_grids_own_range(record):
    closed = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=360.0 / 128.0)
    geo = geography_of(_regrid(record[1], closed))
    inside = geo.place(4.0, 4.0)
    beyond = geo.place(4.0, 4.0 + 128.0)
    assert inside.coordinates["longitude"] == pytest.approx(beyond.coordinates["longitude"])
    assert 0.0 <= beyond.coordinates["longitude"] < 360.0
    # And the cell it names comes back round with it rather than being clipped to the edge.
    assert beyond.col_cell == inside.col_cell == 4


def test_a_sub_pixel_position_names_the_cell_it_is_nearest_to(geography):
    assert geography.place(41.6, 12.4).row_cell == 42
    assert geography.place(41.6, 12.4).col_cell == 12
    assert geography.place(41.4, 12.6).row_cell == 41
    assert geography.place(41.4, 12.6).col_cell == 13


def test_a_decomposition_with_no_level_says_so_rather_than_guessing_one(geography):
    vertical = geography.vertical()
    assert vertical.status == LEVEL_NONE
    assert vertical.value is None and vertical.units is None
    assert "declares no vertical coordinate" in vertical.note


def test_a_level_without_its_axis_is_a_number_and_not_hectopascals(record):
    field = record[1]
    bare_level = CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times, grid=field.grid,
        config=dict(field.config), level=500.0)
    vertical = geography_of(bare_level).vertical()
    assert vertical.status == LEVEL_UNDECLARED_AXIS
    assert vertical.value == 500.0
    assert vertical.axis is None and vertical.units is None
    assert "hectopascals" in vertical.note


def test_a_declared_axis_carries_its_own_units(record):
    field = record[1]
    declared = CoefficientField(
        field.data, wavelet_family=field.wavelet_family, scales=field.scales,
        orientations=field.orientations, times=field.times, grid=field.grid,
        config=dict(field.config), level=500.0, level_axis=PRESSURE_HPA)
    vertical = geography_of(declared).vertical()
    assert vertical.status == LEVEL_DECLARED
    assert vertical.axis == PRESSURE_HPA
    assert vertical.units == "hPa"


def test_a_frame_is_a_frame_until_the_record_supplied_carries_dates(record, geography):
    moment = geography.when(6.0)
    assert moment.basis == TIME_FRAME
    assert moment.frame_index == 6 and moment.timestamp is None
    assert "not a date" in moment.refusal

    stamps = np.array(["2021-06-01T00", "2021-06-01T06", "2021-06-01T12"], dtype="datetime64[h]")
    dated = FieldSequence(record[0].fields[:3], stamps)
    field = decompose_sequence(
        FieldSequence(record[0].fields[:3], stamps), "swt",
        {"levels": LEVELS, "wavelet": WAVELET}).select(scales=SCALES)
    geo = geography_of(field, time_units="hours", values=dated)
    dated_moment = geo.when(float(field.times[1]))
    assert dated_moment.basis == TIME_CALENDAR
    assert dated_moment.frame_index == 1
    assert dated_moment.timestamp.startswith("2021-06-01T06")


def test_a_frame_this_decomposition_never_had_is_refused(geography):
    with pytest.raises(InvalidParameterError, match="different record"):
        geography.when(999.0)


def test_the_footprints_extent_is_the_filters_support_and_not_the_octave_label(record):
    supports = scale_supports(record[1])
    assert {key: item["support_cells"] for key, item in supports.items()} == SUPPORT
    assert supports["4"]["radius_cells"] == 8.0
    # The dyadic octave the scale ratios use is 2 ** (level - 1): 8 cells at level 4, half the
    # support. Two different numbers about two different things, and the receipt says so.
    assert supports["4"]["support_cells"] != 2 ** (4 - 1)
    assert "haar" in supports["4"]["basis"]


# ---------------------------------------------------------------------------------------------
# section 2: a footprint is a region, and its edges are the record's business
# ---------------------------------------------------------------------------------------------


def test_a_footprint_covers_the_filters_support_around_the_maximum(record, geography):
    occurrence = project_occurrence(_of_bands(("L4/LH", "L4/HL"), record[3])[0],
                                    geography=geography)
    for footprint in occurrence.footprints:
        assert footprint.support_cells == SUPPORT[footprint.scale_label]
        assert footprint.radius_cells == SUPPORT[footprint.scale_label] / 2.0
        side_rows = footprint.rows[1] - footprint.rows[0] + 1
        side_cols = footprint.columns[0][1] - footprint.columns[0][0] + 1
        assert side_rows in (SUPPORT[footprint.scale_label], SUPPORT[footprint.scale_label] + 1)
        assert side_cols in (SUPPORT[footprint.scale_label], SUPPORT[footprint.scale_label] + 1)
        assert footprint.n_cells == side_rows * side_cols
        assert len(footprint.cells()) == footprint.n_cells
        assert footprint.contains(footprint.centre.row, footprint.centre.col)
        assert not footprint.contains(footprint.rows[0] - 1, footprint.centre.col)
        # No cell in it is further from the maximum than the filter reaches, which is what
        # makes the footprint a statement about the support rather than a box around it.
        for row, col in footprint.cells():
            assert abs(row - footprint.centre.row) <= footprint.radius_cells
            assert abs(col - footprint.centre.col) <= footprint.radius_cells


def test_the_octave_travels_beside_the_support_rather_than_instead_of_it(record, geography):
    occurrence = project_occurrence(_of_bands(("L5/LH", "L5/HL"), record[3])[0],
                                    geography=geography)
    footprint = occurrence.footprints[0]
    assert footprint.octave_cells == float(2 ** (5 - 1))
    assert footprint.support_cells == 32
    described = footprint.describe()
    assert "NOT the dyadic octave" in described["extent_basis"]
    assert "%g cells" % footprint.octave_cells in described["extent_basis"]


def test_a_footprint_at_a_closing_grids_seam_is_two_segments_and_not_one_wide_one(record):
    closed = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=360.0 / 128.0)
    geo = geography_of(_regrid(record[1], closed))
    rows, columns, wraps, clipped = geo.extent(40.0, 2.0, 8.0)
    assert wraps is True and clipped is False
    assert len(columns) == 2
    assert sum(high - low + 1 for low, high in columns) == 17
    assert (0, 10) in columns and (122, 127) in columns

    # A filter reaching further than the grid is wide covers every column once and still
    # crosses the seam.
    whole_rows, whole_cols, whole_wraps, whole_clipped = geo.extent(40.0, 2.0, 80.0)
    assert whole_cols == ((0, 127),) and whole_wraps is True
    # The rows have an edge and the columns do not, so only the rows are recorded as clipped.
    assert whole_rows == (0, 120) and whole_clipped is True

    wrapped = replace(
        project_occurrence(record[3][0], geography=geo).footprints[0],
        rows=rows, columns=columns, wraps_seam=True)
    assert wrapped.contains(40, 125) and wrapped.contains(40, 3)
    assert not wrapped.contains(40, 60)
    assert wrapped.n_cells == len(wrapped.cells()) == (rows[1] - rows[0] + 1) * 17


def test_the_same_position_on_a_grid_that_does_not_close_is_clipped_and_says_so(geography):
    rows, columns, wraps, clipped = geography.extent(40.0, 2.0, 8.0)
    assert wraps is False and clipped is True
    assert columns == ((0, 10),)
    # And the rows are cut by the same rule as the columns, which is a separate branch.
    rows, columns, wraps, clipped = geography.extent(2.0, 40.0, 8.0)
    assert clipped is True and rows == (0, 10)
    assert columns == ((32, 48),)
    inside, _columns, _wraps, untouched = geography.extent(64.0, 64.0, 8.0)
    assert untouched is False and inside == (56, 72)


def test_an_extent_narrower_than_a_cell_does_not_pretend_to_cross_the_seam(record):
    closed = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=360.0 / 128.0)
    geo = geography_of(_regrid(record[1], closed))
    rows, columns, wraps, clipped = geo.extent(40.0, 2.0, 0.4)
    assert columns == ((2, 2),)
    assert wraps is False


def test_every_detection_sits_far_enough_inside_the_crop_that_R13_already_held(record,
                                                                              geography):
    """The margin was cut at detection, so a footprint's centre is never in it."""
    margins = {"4": 8, "5": 16}
    for constellation in record[3][:20]:
        for footprint in project_occurrence(constellation, geography=geography).footprints:
            margin = margins[footprint.scale_label]
            assert margin <= footprint.centre.row <= 128 - margin
            assert margin <= footprint.centre.col <= 128 - margin


def test_two_footprints_that_do_not_overlap_share_no_cells(record, geography):
    """The intersection is a real box and an empty one is a real answer.

    Every occurrence on this record is of one advecting vortex, so its members always overlap.
    Two footprints seventeen frames apart on that same trajectory do not, and the arithmetic
    has to say so rather than returning the box that spans them.
    """
    early = project_occurrence(_of_bands(("L4/LH", "L4/HL"), record[3])[0],
                               geography=geography).footprints[0]
    late = project_occurrence(_of_bands(("L5/LH", "L5/HL"), record[3])[-1],
                              geography=geography).footprints[0]
    assert late.moment.time - early.moment.time > 16
    rows, columns = common_extent([early, late])
    assert rows is None and columns == ()
    assert common_extent([early, early]) == (early.rows, early.columns)

    # And columns that do overlap are still reported empty when the rows do not, because a
    # shared band of longitude between two footprints that share no latitude is not a region
    # either of them covers.
    moved = replace(early, rows=(early.rows[0] + 60, early.rows[1] + 60))
    assert moved.columns == early.columns
    assert common_extent([early, moved]) == (None, ())


# ---------------------------------------------------------------------------------------------
# section 3: the acceptance -- the cells that were planted
# ---------------------------------------------------------------------------------------------


def _offsets(record, geography):
    """Every member's offset from the planted centre, with its band and the vortex's width."""
    truth = record[4]
    rows = []
    for constellation in record[3]:
        planted = _planted(truth, constellation.time)
        sigma = truth["sigma_cells"][int(constellation.time)]
        occurrence = project_occurrence(constellation, geography=geography)
        for footprint in occurrence.footprints:
            rows.append({
                "time": int(constellation.time), "band": footprint.band, "sigma": sigma,
                "row_offset": footprint.centre.row - planted[0],
                "col_offset": footprint.centre.col - planted[1],
                "contains": footprint.contains(*planted),
                "radius": footprint.radius_cells,
                "peak_cell": (footprint.centre.row_cell, footprint.centre.col_cell),
                "planted_cell": (int(planted[0]), int(planted[1])),
            })
    return rows


def test_a_detail_band_locates_the_axis_it_resolves_and_flanks_the_one_it_does_not(record,
                                                                                  geography):
    for item in _offsets(record, geography):
        across, along = ((item["row_offset"], item["col_offset"]) if item["band"].endswith("LH")
                         else (item["col_offset"], item["row_offset"]))
        assert abs(across) < 0.75, item          # the resolved axis: sub-pixel
        assert along > 5.5, item                 # the high-passed axis: a flank, cells away


def test_the_flank_offset_is_about_one_structure_width(record, geography):
    ratios: dict = {}
    for item in _offsets(record, geography):
        offset = max(abs(item["row_offset"]), abs(item["col_offset"]))
        ratios.setdefault(item["band"][:2], []).append(offset / item["sigma"])
    for level, values in sorted(ratios.items()):
        assert 0.9 < float(np.mean(values)) < 1.45, (level, float(np.mean(values)))
    assert float(np.mean(ratios["L4"])) == pytest.approx(1.080, abs=0.005)
    assert float(np.mean(ratios["L5"])) == pytest.approx(1.261, abs=0.005)


def test_projecting_a_maximum_onto_one_cell_would_have_been_wrong_every_single_time(record,
                                                                                   geography):
    rows = _offsets(record, geography)
    assert rows
    assert not any(item["peak_cell"] == item["planted_cell"] for item in rows)
    assert min(max(abs(item["row_offset"]), abs(item["col_offset"])) for item in rows) > 5.0


def test_the_footprint_holds_the_planted_cell_exactly_while_the_filter_can_reach_it(record,
                                                                                   geography):
    """The acceptance, and the boundary in it: a level far finer than the structure loses it."""
    held: dict = {}
    for item in _offsets(record, geography):
        offset = max(abs(item["row_offset"]), abs(item["col_offset"]))
        # Containment is the filter's reach against the flank distance, and the test says so
        # in geometry rather than taking the code's word for it. Within one cell of the reach
        # the answer turns on where the planted half-cell rounds, so only the two clear sides
        # are asserted -- and the record spends five frames in that band.
        if offset <= item["radius"] - 1.0:
            assert item["contains"], item
        if offset >= item["radius"] + 1.0:
            assert not item["contains"], item
        held.setdefault(item["band"][:2], []).append((item["time"], item["contains"]))

    level4 = sorted({time for time, ok in held["L4"] if ok})
    lost4 = sorted({time for time, ok in held["L4"] if not ok})
    level5 = sorted({time for time, ok in held["L5"] if ok})
    lost5 = sorted({time for time, ok in held["L5"] if not ok})
    # Level 4 reaches the planted cell while the vortex is narrower than its own filter, out to
    # frame 9, and never once after -- the vortex passes 8 cells of width at frame 10.
    assert max(level4) == 9 and min(lost4) == 7
    assert set(range(0, 7)) <= set(level4)
    assert set(range(10, 24)) == set(lost4) - {7, 8, 9}
    assert not set(range(10, 24)) & set(level4)
    # Level 5 detects nothing before frame 9 and loses the planted cell in none of the fifteen
    # frames after it, because its reach is sixteen cells and the flank never exceeds it.
    assert lost5 == []
    assert min(level5) == 9 and max(level5) == 23


def test_two_bands_of_one_level_intersect_over_the_planted_cell(record, geography):
    truth = record[4]
    pairs = _of_bands(("L5/LH", "L5/HL"), record[3])
    assert len(pairs) == 11
    for constellation in pairs:
        planted = _planted(truth, constellation.time)
        occurrence = project_occurrence(constellation, geography=geography)
        cell = (int(planted[0]), int(planted[1]))
        assert cell in occurrence.common_cells, constellation.time
        # An intersection is inside every member, never a box drawn around them.
        for row, col in occurrence.common_cells:
            assert all(item.contains(row, col) for item in occurrence.footprints)
        assert len(set(occurrence.orientations)) == 2
        assert "common_caution" not in occurrence.describe()


def test_two_bands_of_one_orientation_intersect_beside_it_and_the_receipt_warns(record,
                                                                               geography):
    truth = record[4]
    pairs = _of_bands(("L4/LH", "L5/LH"), record[3])
    assert len(pairs) == 11
    for constellation in pairs:
        planted = _planted(truth, constellation.time)
        occurrence = project_occurrence(constellation, geography=geography)
        assert occurrence.common_cells                      # not empty
        assert (int(planted[0]), int(planted[1])) not in occurrence.common_cells
        assert set(occurrence.orientations) == {"LH"}
        assert "flanks point the same way" in occurrence.describe()["common_caution"]


def test_the_intersection_basis_states_both_counts_it_was_measured_from(record, geography):
    described = project_occurrence(record[3][0], geography=geography).describe()
    assert "11 of 11" in described["common_basis"]
    assert "not a location of anything" in described["common_basis"]
    assert described["claim_boundary"] == PROJECTION_CLAIM_BOUNDARY


# ---------------------------------------------------------------------------------------------
# section 4: the field's values are supplied, never derived
# ---------------------------------------------------------------------------------------------


def test_the_values_under_a_footprint_are_the_records_own(record, geography):
    sequence = record[0]
    occurrence = project_occurrence(record[3][0], geography=geography)
    footprint = occurrence.footprints[0]
    values = sequence.at(footprint.moment.frame_index).data.numpy()
    assert footprint.values.at_centre == pytest.approx(
        float(values[footprint.centre.row_cell, footprint.centre.col_cell]))
    block = values[footprint.rows[0]:footprint.rows[1] + 1,
                   footprint.columns[0][0]:footprint.columns[0][1] + 1]
    assert footprint.values.n_cells == block.size == footprint.n_cells
    assert footprint.values.mean == pytest.approx(float(block.mean()))
    assert footprint.values.maximum == pytest.approx(float(block.max()))
    assert footprint.values.quantity == "field value"


def test_with_no_record_supplied_the_values_are_absent_and_not_zero(record, bare):
    footprint = project_occurrence(record[3][0], geography=bare).footprints[0]
    assert footprint.values is None
    assert "not zero" in footprint.refusals["values"]
    assert "R11" in footprint.refusals["anomalies"]


def test_an_anomaly_is_read_from_an_anomaly_record_and_never_made_from_a_raw_one(record):
    sequence = record[0]
    baseline = sequence.mean_field()
    departures = FieldSequence(
        [PhysicalField(item.data - baseline.data, grid=item.grid, units=item.units)
         for item in sequence.fields],
        np.arange(len(sequence.fields), dtype=float))
    geo = geography_of(record[1], time_units="frames", values=sequence, anomalies=departures)
    footprint = project_occurrence(record[3][0], geography=geo).footprints[0]
    assert footprint.anomalies is not None
    assert footprint.anomalies.quantity == "anomaly"
    assert footprint.anomalies.at_centre != pytest.approx(footprint.values.at_centre)
    assert footprint.anomalies.at_centre == pytest.approx(
        footprint.values.at_centre
        - float(baseline.data[footprint.centre.row_cell, footprint.centre.col_cell]))
    assert "anomalies" not in footprint.refusals


def test_an_anomaly_record_may_be_supplied_without_a_value_record(record):
    sequence = record[0]
    geo = geography_of(record[1], time_units="frames", anomalies=sequence)
    footprint = project_occurrence(record[3][0], geography=geo).footprints[0]
    assert footprint.values is None and footprint.anomalies is not None
    assert "values" in footprint.refusals and "anomalies" not in footprint.refusals


def test_a_record_of_the_wrong_length_shape_or_clock_is_refused(record):
    sequence, field = record[0], record[1]
    with pytest.raises(InvalidParameterError, match="one frame per decomposed frame"):
        geography_of(field, values=sequence[:10])
    smaller = FieldSequence(
        [PhysicalField(item.data[:64, :64], grid=GridSpec.cartesian((64, 64), 31000.0, 31000.0),
                       units=item.units) for item in sequence.fields],
        np.arange(len(sequence.fields), dtype=float))
    with pytest.raises(InvalidParameterError, match="parent grid these coefficients live on"):
        geography_of(field, values=smaller)
    shifted = FieldSequence(sequence.fields, np.arange(len(sequence.fields), dtype=float) + 100.0)
    with pytest.raises(InvalidParameterError, match="the same frames the decomposition ran on"):
        geography_of(field, values=shifted)
    with pytest.raises(InvalidParameterError, match="a FieldSequence"):
        geography_of(field, values=[1, 2, 3])


def test_a_wrapped_footprint_is_sampled_from_both_of_its_segments(record):
    closed = GridSpec.latlon((128, 128), lat0=60.0, dlat=-0.5, lon0=0.0, dlon=360.0 / 128.0)
    wrapped = FieldSequence(
        [PhysicalField(item.data, grid=closed, units=item.units) for item in record[0].fields],
        np.arange(len(record[0].fields), dtype=float))
    geo = geography_of(_regrid(record[1], closed), values=wrapped)
    rows, columns, wraps, _clipped = geo.extent(40.0, 2.0, 8.0)
    sample = geo.sample(wrapped, "field value", 0, rows, columns, geo.place(40.0, 2.0))
    assert wraps is True and len(columns) == 2
    expected = (rows[1] - rows[0] + 1) * sum(high - low + 1 for low, high in columns)
    assert sample.n_cells == expected


# ---------------------------------------------------------------------------------------------
# section 5: per-scale contribution, and what a share of a redundant bank is not
# ---------------------------------------------------------------------------------------------


def _two_scale_pattern(catalogue, constellations):
    """A pattern whose members straddle two levels, so a share of one is not a share of all."""
    by_key = {tuple(item.key()): item for item in constellations}
    for pattern in catalogue:
        levels = {str(node.band).split("/")[0]
                  for member in pattern.members
                  for node in by_key[tuple(member.key)].nodes}
        if len(levels) == 2:
            return pattern
    raise AssertionError("expected a pattern spanning two levels")


def test_the_scale_shares_are_the_squared_magnitudes_of_this_patterns_own_members(
        index, catalogue, record):
    pattern = _two_scale_pattern(catalogue, record[3])
    projection = project_pattern(pattern.pattern_id, index)
    by_key = {tuple(item.key()): item for item in record[3]}
    expected: dict = {}
    for member in pattern.members:
        for node in by_key[tuple(member.key)].nodes:
            level = str(node.band).split("/")[0]
            expected[level] = expected.get(level, 0.0) + float(node.strength) ** 2
    total = sum(expected.values())
    assert len(projection.scale_shares) == 2
    for share in projection.scale_shares:
        assert share["energy"] == pytest.approx(expected[share["name"]])
        assert share["share_of_energy"] == pytest.approx(expected[share["name"]] / total)
        # The largest share is strictly below one, so dividing by the largest rather than by
        # the total would be visible here rather than hidden by a single-scale pattern.
        assert 0.0 < share["share_of_energy"] < 1.0
    assert sum(item["share_of_energy"] for item in projection.scale_shares) == pytest.approx(1.0)
    assert sum(item["share_of_nodes"] for item in projection.scale_shares) == pytest.approx(1.0)
    assert projection.n_members == len(projection.occurrences)


def test_the_share_says_it_does_not_partition_the_fields_variance(index, catalogue):
    described = project_pattern(catalogue.patterns[0].pattern_id, index).describe()
    assert REDUNDANCY_NOTE in described["share_basis"]
    assert "undecimated and therefore redundant" in described["share_basis"]
    assert described["schema"] == PROJECTION_SCHEMA


def test_a_band_share_is_finer_than_a_scale_share(index, record, catalogue):
    pattern = [item for item in catalogue
               if len({member.bands for member in item.members}) == 1
               and len(set(item.members[0].bands)) == 2][0]
    projection = project_pattern(pattern.pattern_id, index)
    assert len(projection.band_shares) >= len(projection.scale_shares)
    assert all("/" in item["name"] for item in projection.band_shares)
    assert all("/" not in item["name"] for item in projection.scale_shares)


def test_the_shares_are_ordered_by_energy(index, catalogue):
    projection = project_pattern(
        max(catalogue, key=lambda item: item.support).pattern_id, index)
    energies = [item["energy"] for item in projection.band_shares]
    assert energies == sorted(energies, reverse=True)


def test_the_occurrences_of_a_pattern_come_back_in_time_order(record, geography, catalogue):
    """Ordered by the projection, not by the order the catalogue happened to hold them in."""
    biggest = max(catalogue, key=lambda item: item.support)
    reversed_members = type(biggest)(
        pattern_id=biggest.pattern_id, members=tuple(reversed(biggest.members)),
        centroid=biggest.centroid, tolerance_radius=biggest.tolerance_radius,
        observed_radius=biggest.observed_radius)
    shuffled = type(catalogue)(
        patterns=(reversed_members,) + tuple(item for item in catalogue
                                             if item is not biggest),
        metric=catalogue.metric, tolerance=catalogue.tolerance,
        n_signatures=catalogue.n_signatures)
    held = [float(item.time) for item in reversed_members.members]
    assert held != sorted(held), "the fixture must disagree with time order to test anything"
    projection = project_pattern(
        biggest.pattern_id, evidence_index(shuffled, record[3], geography))
    times = [item.moment.time for item in projection]
    assert times == sorted(held)
    assert len(projection) == projection.n_members


# ---------------------------------------------------------------------------------------------
# section 6: the index binds a catalogue to the constellations behind it
# ---------------------------------------------------------------------------------------------


def test_a_catalogue_whose_members_are_not_in_this_extraction_is_refused(catalogue, record,
                                                                        geography):
    partial = extract_constellations(record[2], cardinalities=(3,))
    with pytest.raises(InvalidParameterError, match="have no constellation in this set"):
        evidence_index(catalogue, partial, geography)


def test_the_refusal_counts_the_members_it_would_have_skipped(catalogue, record, geography):
    trimmed = type(record[3])(
        constellations=record[3].constellations[:5],
        frames_searched=record[3].frames_searched, cardinalities=record[3].cardinalities)
    with pytest.raises(InvalidParameterError) as caught:
        evidence_index(catalogue, trimmed, geography)
    message = str(caught.value)
    assert "understating the evidence" in message
    assert "First missing key" in message


def test_the_index_binds_a_genuinely_clustered_catalogue_too(record, geography):
    """Not only the assembled fixture: T4E.3's own agglomeration, on a slice of the record."""
    subset = extract_constellations(record[2], cardinalities=(2,))
    signatures = sign_constellations(subset, scale_invariant=False).signatures[:12]
    tolerance = calibrate_signature_tolerance(signatures[:3], metric=METRIC)
    clustered = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
    bound = evidence_index(clustered, subset, geography)
    projection = project_pattern(clustered.patterns[0].pattern_id, bound)
    assert len(projection) == clustered.patterns[0].support
    assert all(item.footprints for item in projection)


def test_a_pattern_the_catalogue_does_not_contain_is_refused(index):
    with pytest.raises(InvalidParameterError, match="a pattern this catalogue contains"):
        project_pattern(9999, index)


def test_occurrences_at_a_frame_the_pattern_missed_is_empty_rather_than_wrong(index,
                                                                             catalogue):
    pattern = catalogue.patterns[0]
    time = float(pattern.members[0].time)
    assert index.occurrences_at(pattern.pattern_id, time)
    absent = index.occurrences_at(pattern.pattern_id, time + 1000.0)
    assert absent == ()


def test_project_pattern_needs_an_index(catalogue):
    with pytest.raises(InvalidParameterError, match="an EvidenceIndex"):
        project_pattern(catalogue.patterns[0].pattern_id, None)


# ---------------------------------------------------------------------------------------------
# section 7: the instances a rule was counted on
# ---------------------------------------------------------------------------------------------

FRAMES = 120
#: Twelve occurrences of the consequent, and three of them earn their place: `64` sits four
#: frames after an antecedent and so falls *outside* the declared window, `119` is inside the
#: window of the anchor at `118` whose window the record's end truncates, and the record's last
#: two anchors have nothing measurable after them at all.
B_TIMES = [8, 15, 31, 36, 41, 50, 55, 64, 68, 77, 91, 99, 108, 119]
A_TIMES = [time - 2 for time in B_TIMES[:7]] + [66, 60, 118, 119]
C_TIMES = list(range(0, FRAMES, 7))
WINDOW = TransitionWindow(1.0, 3.0, "frames")
SURROGATES = 113
SEED = 20260906


def _series(frames=None):
    grid = ObservationGrid(
        frames=tuple(float(item) for item in (frames if frames is not None
                                              else range(FRAMES))),
        time_units="frames", cadence=1.0)
    searched = set(grid.frames)
    events = []
    for pattern_id, times in ((1, A_TIMES), (2, B_TIMES), (3, C_TIMES)):
        for time in times:
            if float(time) not in searched:
                continue
            events.append(PatternEvent(
                pattern_id=pattern_id, time=float(time),
                identity="p%d@%d" % (pattern_id, time), cardinality=2,
                mode="scale_specific"))
    events.sort(key=lambda item: (item.time, item.pattern_id))
    return EventSeries(events=tuple(events), grid=grid, mode="scale_specific",
                       units_confirmed_by_members=len(events),
                       units_uncarried_by_members=0)


@pytest.fixture(scope="module")
def planted():
    series = _series()
    report = precursor_report(series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES,
                              seed=SEED, pairs=[(1, 2), (3, 2)])
    return series, report


def test_the_planted_rule_is_a_signature_and_the_decoy_is_not(planted):
    _series_, report = planted
    signal = report.rule(1, 2, WINDOW)
    decoy = report.rule(3, 2, WINDOW)
    assert signal.status == RULE_PRECURSOR
    assert decoy.status == RULE_NOT_DISTINGUISHED
    assert signal.support == 8 and signal.eligible_antecedents == 9
    assert signal.antecedent_occurrences == 11 and signal.ineligible_truncated == 2
    assert signal.ineligible_unobserved == 0


def test_every_anchor_is_listed_under_its_own_verdict(planted):
    series, report = planted
    projection = project_rule(report.rule(1, 2, WINDOW), series)
    verdicts = [item.verdict for item in projection.instances]
    assert len(projection.instances) == 11
    assert verdicts.count(INSTANCE_SUPPORTING) == 8
    assert verdicts.count(INSTANCE_UNFOLLOWED) == 1
    assert verdicts.count(INSTANCE_TRUNCATED) == 2
    assert [item.anchor for item in projection.supporting] == sorted(
        [float(time - 2) for time in B_TIMES[:7]] + [66.0])
    unfollowed = [item for item in projection.instances
                  if item.verdict == INSTANCE_UNFOLLOWED][0]
    assert unfollowed.anchor == 60.0 and unfollowed.completions == ()


def test_a_truncated_anchor_is_truncated_even_when_something_did_follow_it(planted):
    """The anchor at 118 is followed at 119 and is still not support.

    Its window reaches to 121 and the record stops at 119, so nobody watched the rest of it.
    Counting it would price the tail of every record as a success whenever the consequent
    happened to be common, which is the asymmetry T4F.2's denominator was built to refuse.
    """
    series, report = planted
    projection = project_rule(report.rule(1, 2, WINDOW), series)
    truncated = [item for item in projection.instances
                 if item.verdict == INSTANCE_TRUNCATED]
    assert [item.anchor for item in truncated] == [118.0, 119.0]
    assert truncated[0].completions == (119.0,) and truncated[0].lags == (1.0,)
    assert truncated[1].completions == ()
    assert not any(item.supporting for item in truncated)
    assert report.rule(1, 2, WINDOW).support == 8


def test_an_anchor_whose_window_spans_unobserved_time_is_kept_apart_from_a_truncated_one():
    """Two frames go unsearched, and the anchor whose window needed them is neither.

    On the full record the anchor at 60 is eligible and simply not followed -- a real failure
    of the rule. With frames 62 and 63 unsearched it becomes undecidable instead, and the two
    are different findings: one lowers the confidence, the other leaves the denominator.
    """
    series = _series([item for item in range(FRAMES) if item not in (62, 63)])
    report = precursor_report(series, lags=LagFamily((WINDOW,)), n_surrogates=SURROGATES,
                              seed=SEED, pairs=[(1, 2), (3, 2)])
    rule = report.rule(1, 2, WINDOW)
    projection = project_rule(rule, series)
    verdicts = {item.anchor: item.verdict for item in projection.instances}
    assert verdicts[60.0] == INSTANCE_UNOBSERVED
    assert verdicts[118.0] == INSTANCE_TRUNCATED
    counted = [item.verdict for item in projection.instances]
    assert counted.count(INSTANCE_UNOBSERVED) == rule.ineligible_unobserved == 1
    assert counted.count(INSTANCE_TRUNCATED) == rule.ineligible_truncated == 2
    assert rule.eligible_antecedents == 8 and rule.support == 8
    assert rule.confidence == 1.0


def test_every_enumerated_total_is_reconciled_against_the_rules_own_figures(planted):
    series, report = planted
    for pair in ((1, 2), (3, 2)):
        rule = report.rule(pair[0], pair[1], WINDOW)
        projection = project_rule(rule, series)
        published = projection.reconciliation["published"]
        assert published == projection.reconciliation["enumerated"]
        assert published["support"] == rule.support
        assert published["eligible_antecedents"] == rule.eligible_antecedents
        assert published["ineligible_unobserved"] == rule.ineligible_unobserved
        assert projection.reconciliation["agree"] is True
        assert "anchor_verdicts" in projection.reconciliation["decided_by"]


def test_a_series_that_is_not_the_one_the_rule_was_measured_on_is_refused(planted):
    series, report = planted
    events = list(series.events) + [PatternEvent(
        pattern_id=1, time=70.0, identity="p1@70", cardinality=2, mode="scale_specific")]
    events.sort(key=lambda item: (item.time, item.pattern_id))
    other = EventSeries(events=tuple(events), grid=series.grid, mode=series.mode,
                        units_confirmed_by_members=len(events), units_uncarried_by_members=0)
    with pytest.raises(InvalidParameterError, match="not the ones the correction was paid on"):
        project_rule(report.rule(1, 2, WINDOW), other)


def test_a_rule_the_null_explained_still_gets_its_instances(planted):
    series, report = planted
    rule = report.rule(3, 2, WINDOW)
    projection = project_rule(rule, series)
    assert projection.status == RULE_NOT_DISTINGUISHED
    assert len(projection.supporting) == rule.support > 0
    assert len(projection.instances) == rule.antecedent_occurrences


def test_the_figures_are_echoed_and_not_recomputed(planted):
    series, report = planted
    rule = report.rule(1, 2, WINDOW)
    projection = project_rule(rule, series)
    assert projection.figures["q_value"] == rule.q_value
    assert projection.figures["p_value"] == rule.p_value
    assert projection.figures["lift"] == rule.lift
    assert projection.figures["confidence"] == rule.confidence
    described = projection.describe()
    assert described["schema"] == INSTANCE_SCHEMA
    assert "is a view of a test that has already been corrected" in described["figures_basis"]
    assert described["n_supporting"] == rule.support


def test_a_completion_outside_the_declared_window_is_not_listed(planted):
    """The anchor at 60 is followed at 64, four frames later, and the window reaches three."""
    series, report = planted
    projection = project_rule(report.rule(1, 2, WINDOW), series)
    unfollowed = [item for item in projection.instances if item.anchor == 60.0][0]
    assert 64.0 in {float(time) for time in B_TIMES}
    assert unfollowed.completions == () and unfollowed.verdict == INSTANCE_UNFOLLOWED
    for instance in projection.instances:
        for lag in instance.lags:
            assert WINDOW.minimum_lag <= lag <= WINDOW.maximum_lag
        for completion in instance.completions:
            assert completion in {float(time) for time in B_TIMES}


def test_a_selected_lag_row_is_refused_by_name(planted):
    series, report = planted
    with pytest.raises(InvalidParameterError, match="a T4F.3 PrecursorRule"):
        project_rule(report.selected_rule(1, 2), series)


def test_project_rule_needs_the_event_series_and_an_index_or_nothing(planted):
    series, report = planted
    rule = report.rule(1, 2, WINDOW)
    with pytest.raises(InvalidParameterError, match="EventSeries"):
        project_rule(rule, None)
    with pytest.raises(InvalidParameterError, match="an EvidenceIndex, or nothing at all"):
        project_rule(rule, series, "not an index")


def test_with_an_index_both_ends_of_an_instance_are_on_the_map(record, geography, catalogue,
                                                               index):
    """The rule is over the record's own patterns, so its instances carry real footprints."""
    series = events_from_catalogue(catalogue, grid=ObservationGrid(
        frames=tuple(float(item) for item in range(24)), time_units="frames", cadence=1.0))
    ids = sorted(series.pattern_ids)
    report = precursor_report(series, lags=LagFamily((TransitionWindow(1.0, 1.0, "frames"),)),
                              n_surrogates=19, seed=SEED, pairs=[(ids[0], ids[1])])
    rule = report.rules[0]
    projection = project_rule(rule, series, index)
    assert projection.instances
    for instance in projection.instances:
        for occurrence in instance.antecedents:
            assert occurrence.pattern_id == rule.antecedent
            assert occurrence.moment.time == instance.anchor
            assert occurrence.footprints
        for occurrence in instance.consequents:
            assert occurrence.pattern_id == rule.consequent
            assert occurrence.moment.time in instance.completions
    assert any(instance.consequents for instance in projection.instances)


# ---------------------------------------------------------------------------------------------
# section 8: what is refused, and what is never claimed
# ---------------------------------------------------------------------------------------------


def test_a_detection_whose_positions_were_never_aligned_is_refused(record):
    """D88: an unaligned index is at the filter's anchor, not at the place it responded to."""
    field = record[1].select(scales=[4])
    detection = detect_features(field, threshold_sigma=THRESHOLD_SIGMA, align=False)
    tracked = track_spectral_features(
        field, domain="synthetic", dataset="advected_vortex_sequence", variable="amplitude",
        time_units="frames", detection=detection)
    unaligned = extract_constellations(tracked, cardinalities=(2,))
    geo = geography_of(field, time_units="frames")
    with pytest.raises(InvalidParameterError, match="aligned to the parent grid"):
        project_occurrence(unaligned[0], geography=geo)


def test_a_member_measured_at_a_scale_this_decomposition_lacks_is_refused(record):
    """A constellation from a wider bank cannot be drawn on a narrower bank's map."""
    narrow = geography_of(record[1].select(scales=[4]), time_units="frames")
    coarse = _of_bands(("L5/LH", "L5/HL"), record[3])[0]
    with pytest.raises(InvalidParameterError, match="a different decomposition"):
        project_occurrence(coarse, geography=narrow)
    fine = _of_bands(("L4/LH", "L4/HL"), record[3])[0]
    assert project_occurrence(fine, geography=narrow).footprints


def test_a_constellation_that_dropped_its_features_cannot_be_projected(record, geography):
    stripped = type(record[3][0])(
        graph=record[3][0].graph, nodes=record[3][0].nodes, relations=record[3][0].relations,
        time=record[3][0].time, time_units=record[3][0].time_units)
    with pytest.raises(InvalidParameterError, match="carrying the features"):
        project_occurrence(stripped, geography=geography)


def test_a_signature_is_not_a_constellation_and_says_why(geography, catalogue):
    with pytest.raises(InvalidParameterError, match="a signature does not carry them"):
        project_occurrence(catalogue.patterns[0].members[0], geography=geography)


def test_a_bare_array_is_not_a_decomposition(record):
    with pytest.raises(InvalidParameterError, match="a CoefficientField"):
        geography_of(torch.zeros(3, 3))


def test_the_index_refuses_anything_that_is_not_a_catalogue_a_set_and_a_map(record, catalogue,
                                                                           geography):
    with pytest.raises(InvalidParameterError, match="PatternCatalogue"):
        evidence_index(None, record[3], geography)
    with pytest.raises(InvalidParameterError, match="carries no position by design"):
        evidence_index(catalogue, None, geography)
    with pytest.raises(InvalidParameterError, match="one map"):
        evidence_index(catalogue, record[3], None)


def test_no_verdict_or_status_this_module_emits_names_a_term_the_ladder_excludes():
    for name in (INSTANCE_SUPPORTING, INSTANCE_UNFOLLOWED, INSTANCE_TRUNCATED,
                 INSTANCE_UNOBSERVED, PLACE_GEOGRAPHIC, PLACE_OFFSET, PLACE_CELLS_ONLY,
                 LEVEL_DECLARED, LEVEL_UNDECLARED_AXIS, LEVEL_NONE, TIME_CALENDAR, TIME_FRAME):
        words = set(name.lower().split("_"))
        assert not words & {term.lower() for term in OUTSIDE_THE_LADDER}


def test_the_claim_boundary_refuses_causation_and_says_where_a_footprint_is_not(record,
                                                                               geography):
    lowered = PROJECTION_CLAIM_BOUNDARY.lower()
    for forbidden in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert forbidden in lowered
    assert "it is where a coefficient was, not where a structure is" in lowered
    assert "peaks at a flank" in lowered
    described = project_occurrence(record[3][0], geography=geography).describe()
    assert described["claim_boundary"] == PROJECTION_CLAIM_BOUNDARY


def test_the_rule_receipt_carries_the_inference_boundary_as_well_as_its_own(planted):
    series, report = planted
    described = project_rule(report.rule(1, 2, WINDOW), series).describe()
    assert described["claim_boundary"] == PROJECTION_CLAIM_BOUNDARY
    assert "precursor signature" in described["inference_claim_boundary"].lower()
    assert described["window"]["minimum_lag"] == WINDOW.minimum_lag


def test_the_map_receipt_says_what_it_could_and_could_not_name(geography, bare):
    described = geography.describe()
    assert described["schema"] == PROJECTION_SCHEMA
    assert described["grid_kind"] == "cartesian"
    assert described["coordinate_basis"] == PLACE_OFFSET
    assert described["columns_close_on_a_circle"] is False
    assert described["values_supplied"] is True
    assert described["supports"]["4"]["support_cells"] == 16
    assert bare.describe()["values_supplied"] is False
    assert bare.describe()["anomalies_supplied"] is False


def test_describe_projection_refuses_something_it_did_not_produce(index, record, geography):
    assert describe_projection(geography)["schema"] == PROJECTION_SCHEMA
    assert describe_projection(index)["n_patterns"] > 0
    assert describe_projection(
        project_occurrence(record[3][0], geography=geography))["cardinality"] == 2
    with pytest.raises(InvalidParameterError, match="a projection this module produced"):
        describe_projection({"schema": PROJECTION_SCHEMA})


def test_a_scale_invariant_catalogue_projects_exactly_as_a_scale_specific_one(record,
                                                                              geography):
    """Unlike T4F.4's ordering, a projection reads the record's positions and not the signature.

    T4E.2's toggle is what a top-down query has to refuse: under scale invariance every
    pattern's scale statistic is exactly 1.0 and there is nothing to order. It costs a
    projection nothing at all, and the way to show that is to sign the same triples both ways,
    cluster and project each, and compare the footprints rather than to assert it.
    """
    triples = extract_constellations(record[2], cardinalities=(3,))
    both = {}
    for invariant in (False, True):
        signatures = sign_constellations(triples, scale_invariant=invariant).signatures[:8]
        assert len(signatures) == 8
        tolerance = calibrate_signature_tolerance(signatures[:3], metric=METRIC)
        clustered = cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)
        bound = evidence_index(clustered, triples, geography)
        both[invariant] = {
            tuple(occurrence.key): occurrence
            for pattern in clustered
            for occurrence in project_pattern(pattern.pattern_id, bound)}
    assert set(both[False]) == set(both[True])
    for key, occurrence in both[False].items():
        other = both[True][key]
        assert [item.rows for item in occurrence.footprints] ==                [item.rows for item in other.footprints]
        assert [item.columns for item in occurrence.footprints] ==                [item.columns for item in other.footprints]
        assert occurrence.common_cells == other.common_cells
