"""TG2.3: frame-to-frame association, in the order the slice has to be believed.

1.  **The acceptance criterion, against an answer written before the tracker.**
    `advected_vortex_sequence` records one track, one birth, no deaths, an exact trajectory
    and an exact doubling time. `4D.tracking` has been `NOT_YET_RUNNABLE` since T3.5.17.
2.  **The gate is derived, not chosen.** The one free number a tracker usually carries is
    how far a feature may move between frames, and it decides how many objects exist. Here
    it is a coincidence radius computed from the same alpha the extraction was calibrated at
    and the frame's own density, and these tests measure it moving when the frame does.
3.  **Association is a registry.** Two associators, disagreeing on a constructed case where
    the disagreement is measurable, plus a third registered from this module.
4.  **The declaration changes the arithmetic.** A seam crossing is one track on a torus and
    two on a plane, and an orientation gate reads the convention rather than the number.
5.  **Silence is a fact.** A frame the extractor found nothing in ends a track, and a clock
    shorter than the run cannot report a death at the end of one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmarks.core import Outcome, get_benchmark
from src.core.domain import AxisSpec
from src.core.errors import InvalidParameterError, UnknownNameError
from src.core.extraction import ExtractionField, calibrate, extract
from src.core.feature import (
    FeatureLocation, FeatureSet, Orientation, Quantity, SemanticComparisonError,
    SpectralFeature,
)
from src.core.registry import restore, snapshot
from src.core.tracking import (
    ASSOCIATORS, MotionBounds, SearchVolume, Track, coincidence_radius, greedy_nearest,
    hungarian, track, track_extractions,
)

GRID_N = 128
ROW = AxisSpec("row", "space", units="cells", ordinal=0)
COL = AxisSpec("col", "space", units="cells", ordinal=1)
WRAP_ROW = AxisSpec("row", "space", units="cells", periodic=True, ordinal=0)
WRAP_COL = AxisSpec("col", "space", units="cells", periodic=True, ordinal=1)
FLAT = SearchVolume({"row": float(GRID_N), "col": float(GRID_N)}, units="cells")
N_SURROGATES = 99
SEED = 4321


def _feature(time, row, col, *, scale=5.0, axes=(ROW, COL), orientation=None,
             magnitude=1.0, time_units="frames", dataset="toy"):
    """A record placed by hand, so that the correct association is known by construction."""
    return SpectralFeature(
        domain="synthetic", dataset=dataset, variable="amplitude",
        magnitude=Quantity(magnitude, None),
        location=FeatureLocation({"row": float(row), "col": float(col)}, axes),
        time=float(time), representation="identity", time_units=time_units,
        spatial_scale=None if scale is None else Quantity(scale, "cells"),
        orientation=orientation)


def _run(features, *, times=None, volume=FLAT, **kwargs):
    ordered = list(features)
    if times is None:
        times = sorted({f.time for f in ordered})
    return track(FeatureSet(ordered), volume=volume, times=times, **kwargs)


def _vortex(**overrides):
    """Every frame of the benchmark, extracted under one calibration - TG2.2's output."""
    bench = get_benchmark("advected_vortex_sequence")
    sequence = bench.make(**overrides)
    frames = [np.asarray(f.data.numpy(), dtype=np.float64) for f in sequence.fields]
    calibration = calibrate(frames[0], alpha=0.05, n_surrogates=N_SURROGATES, seed=SEED)
    results = [
        extract(ExtractionField(
            values=values, axes=(ROW, COL), domain="synthetic",
            dataset="advected_vortex_sequence", variable="amplitude", units=None,
            time=float(t), time_units="frames", representation="identity"),
            calibration=calibration)
        for t, values in enumerate(frames)]
    return results, bench.truth(**overrides)


# ================================================== 1. the acceptance criterion


def test_the_gate_that_has_been_unrunnable_since_t3517_now_passes():
    """`4D.tracking` was a `NotImplementedError` holding a recorded answer. It is a check."""
    bench = get_benchmark("advected_vortex_sequence")
    results = {r.stage: r for r in bench.run()}
    assert results["4D.tracking"].outcome is Outcome.PASS
    assert results["4D.tracking"].measured["track_count"] == 1
    assert results["4D.tracking"].measured["births"] == 1
    assert results["4D.tracking"].measured["deaths"] == 0


def test_one_object_over_twenty_four_frames_is_one_track():
    results, truth = _vortex()
    tracking = track_extractions(results)
    assert len(tracking) == truth["track_count"] == 1
    assert tracking.births == truth["births"] == 1
    assert tracking.deaths == truth["deaths"] == 0
    assert len(tracking[0]) == len(results)
    assert tracking.empty_frames == ()


def test_every_position_on_the_track_is_within_a_cell_of_the_recorded_trajectory():
    results, truth = _vortex()
    only = track_extractions(results)[0]
    for item in only:
        row, col = truth["positions_rowcol"][int(item.time)]
        assert math.hypot(item.location.coords["row"] - row,
                          item.location.coords["col"] - col) < 1.0


def test_the_measured_velocity_is_the_velocity_the_benchmark_recorded():
    """The tracker was told the fields, the axes and an alpha. Not the velocity."""
    results, truth = _vortex()
    velocity = track_extractions(results)[0].velocity()
    known_row, known_col = truth["velocity_cells_per_step"]
    assert velocity["row"].value == pytest.approx(known_row, abs=0.05)
    assert velocity["col"].value == pytest.approx(known_col, abs=0.05)
    assert velocity["row"].units == "cells/frames"


def test_the_measured_growth_rate_recovers_the_recorded_doubling_time():
    results, truth = _vortex()
    only = track_extractions(results)[0]
    assert only.doubling_time().value == pytest.approx(
        truth["scale_doubling_steps"], rel=0.10)
    assert only.doubling_time().units == "frames"
    assert only.scale_velocity() > 0.0


def test_the_lifetime_is_the_span_of_the_evidence_and_carries_the_clocks_units():
    results, _ = _vortex()
    only = track_extractions(results)[0]
    assert only.lifetime().value == pytest.approx(float(len(results) - 1))
    assert only.lifetime().units == "frames"
    assert only.birth_time == 0.0 and only.last_time == float(len(results) - 1)


# ================================================== 2. the gate is derived, not chosen


def test_the_gate_is_the_radius_at_which_a_neighbour_stops_being_evidence():
    """One feature in a 128x128 frame at alpha = 0.05: 16.1 cells, from arithmetic."""
    radius = coincidence_radius(0.05, 1, float(GRID_N * GRID_N), 2)
    assert radius == pytest.approx(math.sqrt(0.05 * GRID_N * GRID_N / math.pi))
    expected_neighbours = 1 * math.pi * radius ** 2 / (GRID_N * GRID_N)
    assert expected_neighbours == pytest.approx(0.05)


def test_a_crowded_frame_gets_a_tighter_gate_than_an_empty_one():
    """The number moves with the frame, which is the whole difference from a constant."""
    sparse = coincidence_radius(0.05, 1, float(GRID_N * GRID_N), 2)
    crowded = coincidence_radius(0.05, 64, float(GRID_N * GRID_N), 2)
    assert crowded < sparse / 7.0
    assert crowded == pytest.approx(sparse / 8.0)      # radius falls as 1/sqrt(density)


def test_the_receipt_reports_the_gate_that_was_actually_applied():
    tracking = _run([_feature(0, 20, 20), _feature(1, 22, 24)])
    step = tracking.steps[1]
    assert step.coincidence_radius == pytest.approx(
        coincidence_radius(0.05, 1, float(GRID_N * GRID_N), 2))
    assert step.radius == step.coincidence_radius
    assert step.speed_radius is None
    assert step.active_gates == ("position",)


def test_a_jump_no_closer_than_coincidence_is_a_death_and_a_birth_not_a_link():
    """Refusing to link is a claim the receipt makes, not a silence."""
    tracking = _run([_feature(0, 20, 20), _feature(1, 20, 60)])
    assert len(tracking) == 2
    assert tracking.births == 2
    assert tracking.deaths == 1
    assert tracking.steps[1].linked == 0
    assert tracking.steps[1].refused_by_gate == 1


def test_nothing_is_declared_by_default_and_the_receipt_says_which_gates_were_active():
    tracking = _run([_feature(0, 20, 20), _feature(1, 22, 24)])
    assert tracking.describe()["bounds"]["declared"] == []
    assert tracking.describe()["gap_bridging"] == "none: a missed frame ends a track"


def test_a_declared_speed_is_a_rate_and_is_multiplied_by_the_elapsed_time():
    """An irregular clock must not silently re-tune the gate.

    The same 12-cell step is admissible over six frames and refused over one, under one
    declared bound of 3 cells per frame. A gate expressed per *frame* rather than per unit
    time would have answered the same in both cases and been wrong in one of them.
    """
    bounds = MotionBounds(max_speed=3.0)
    slow = _run([_feature(0, 20, 20), _feature(6, 20, 32)], bounds=bounds)
    fast = _run([_feature(0, 20, 20), _feature(1, 20, 32)], bounds=bounds)
    assert len(slow) == 1 and len(slow[0]) == 2
    assert len(fast) == 2
    assert slow.steps[1].speed_radius == pytest.approx(18.0)
    assert fast.steps[1].speed_radius == pytest.approx(3.0)


def test_a_declared_bound_can_only_tighten_the_coincidence_gate():
    """Declaring generous physics does not license a link chance already explains."""
    generous = MotionBounds(max_speed=1000.0)
    tracking = _run([_feature(0, 20, 20), _feature(1, 20, 60)], bounds=generous)
    assert tracking.steps[1].radius == pytest.approx(tracking.steps[1].coincidence_radius)
    assert len(tracking) == 2


def test_a_zero_bound_is_refused_because_it_is_a_gate_nothing_can_pass():
    for kwargs in ({"max_speed": 0.0}, {"max_doublings": 0.0},
                   {"max_turn_degrees": -1.0}):
        with pytest.raises(InvalidParameterError) as excinfo:
            MotionBounds(**kwargs)
        assert "not a loose bound" in str(excinfo.value)


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(InvalidParameterError):
        _run([_feature(0, 20, 20), _feature(1, 22, 24)], alpha=0.0)
    with pytest.raises(InvalidParameterError):
        _run([_feature(0, 20, 20), _feature(1, 22, 24)], alpha=1.0)


def test_the_searched_extent_is_required_and_not_inferred_from_the_features():
    """A volume guessed from the features found would tighten as the finding thinned."""
    with pytest.raises(InvalidParameterError) as excinfo:
        _run([_feature(0, 20, 20), _feature(1, 22, 24)],
             volume=SearchVolume({"row": 128.0}, units="cells"))
    assert "col" in str(excinfo.value)


def test_the_extent_and_the_axes_must_be_in_the_same_units():
    with pytest.raises(SemanticComparisonError) as excinfo:
        _run([_feature(0, 20, 20), _feature(1, 22, 24)],
             volume=SearchVolume({"row": 128.0, "col": 128.0}, units="m"))
    assert "different units" in str(excinfo.value)


# ================================================== 3. association is a registry


#: Rows are open tracks, columns are observations; the correct answer is the diagonal, by
#: construction, and the point is that the two registered associators do not agree on it.
CROSSING_FRAME_A = [(10.0, 10.0), (10.0, 14.0), (60.0, 60.0)]
CROSSING_FRAME_B = [(13.0, 10.0), (10.0, 11.0), (63.0, 60.0)]


def _crossing():
    return ([_feature(0, r, c) for r, c in CROSSING_FRAME_A]
            + [_feature(1, r, c) for r, c in CROSSING_FRAME_B])


def _pairing(tracking):
    """(first position, last position) for each track, as a comparable set."""
    return {(tuple(t[0].location.coords[n] for n in ("row", "col")),
             tuple(t[-1].location.coords[n] for n in ("row", "col")))
            for t in tracking if len(t) == 2}


def test_both_associators_are_registered_and_declare_whether_they_are_optimal():
    assert set(ASSOCIATORS.names()) >= {"greedy_nearest", "hungarian"}
    assert ASSOCIATORS.entry("hungarian").capabilities["optimal"] is True
    assert ASSOCIATORS.entry("greedy_nearest").capabilities["optimal"] is False
    assert [e.name for e in ASSOCIATORS.with_capability("optimal", True)] == ["hungarian"]


def test_greedy_takes_the_cheapest_pair_and_strands_the_right_one():
    """The reason there are two associators rather than one with a caveat.

    Three objects, one of which sits one cell from another object's true partner. Greedy
    takes that pair first because it is the cheapest single link in the frame, and the
    stranded track then has to take an observation nine times further away. The Hungarian
    assignment pays one cell more on the cheapest link and less overall.

    The correct answer here is correct *by construction* - these features were placed, not
    measured - and the measurable claim is the one asserted: the two associators return
    different pairings, and the greedy total is higher.
    """
    greedy = _run(_crossing(), associator="greedy_nearest")
    optimal = _run(_crossing(), associator="hungarian")
    truth = {((10.0, 10.0), (13.0, 10.0)), ((10.0, 14.0), (10.0, 11.0)),
             ((60.0, 60.0), (63.0, 60.0))}
    assert _pairing(optimal) == truth
    assert _pairing(greedy) != truth
    assert len(_pairing(greedy) & truth) == 1        # only the isolated third object


def test_the_greedy_pairing_costs_more_than_the_one_it_was_preferred_over():
    """Measured on the matrix itself, so the claim is arithmetic and not narrative."""
    cost = np.array([[9.0, 1.0, 1e3], [25.0, 9.0, 1e3], [1e3, 1e3, 9.0]])
    admissible = cost < 1e2
    greedy_total = sum(cost[i, j] for i, j in greedy_nearest(cost, admissible))
    optimal_total = sum(cost[i, j] for i, j in hungarian(cost, admissible))
    assert optimal_total == pytest.approx(27.0)
    assert greedy_total == pytest.approx(35.0)
    assert optimal_total < greedy_total


def test_a_third_associator_registered_from_the_test_module_drives_the_tracker():
    """No edit to `src`, and the framework's accounting still holds around it."""
    state = snapshot(ASSOCIATORS)
    try:
        @ASSOCIATORS.register("link_nothing", capabilities={"optimal": False})
        def link_nothing(cost, admissible):
            return []

        tracking = _run([_feature(0, 20, 20), _feature(1, 22, 24)],
                        associator="link_nothing")
        assert len(tracking) == 2
        assert tracking.births == 2 and tracking.deaths == 1
        assert tracking.describe()["associator"] == "link_nothing"
    finally:
        restore(ASSOCIATORS, state)


def test_an_associator_that_returns_a_pair_the_gate_refused_is_itself_refused():
    """Otherwise every refusal in the module is advisory."""
    state = snapshot(ASSOCIATORS)
    try:
        @ASSOCIATORS.register("link_everything")
        def link_everything(cost, admissible):
            return [(0, 0)]

        with pytest.raises(InvalidParameterError) as excinfo:
            _run([_feature(0, 20, 20), _feature(1, 20, 60)], associator="link_everything")
        assert "widen a gate" in str(excinfo.value)
    finally:
        restore(ASSOCIATORS, state)


def test_an_associator_that_claims_one_observation_for_two_tracks_is_refused():
    state = snapshot(ASSOCIATORS)
    try:
        @ASSOCIATORS.register("merge_everything")
        def merge_everything(cost, admissible):
            return [(0, 0), (1, 0)]

        with pytest.raises(InvalidParameterError) as excinfo:
            _run([_feature(0, 20, 20), _feature(0, 24, 20), _feature(1, 22, 20)],
                 associator="merge_everything")
        assert "merge" in str(excinfo.value)
    finally:
        restore(ASSOCIATORS, state)


def test_an_unregistered_associator_names_the_ones_that_exist():
    with pytest.raises(UnknownNameError) as excinfo:
        _run([_feature(0, 20, 20), _feature(1, 22, 24)], associator="hungarain")
    assert "hungarian" in str(excinfo.value)


# ================================================== 4. the declaration changes the arithmetic


def _seam_crossing(axes):
    return [_feature(0, 20, 126, axes=axes), _feature(1, 20, 2, axes=axes)]


def test_a_feature_crossing_the_seam_is_one_track_when_the_axis_says_it_wraps():
    tracking = _run(_seam_crossing((WRAP_ROW, WRAP_COL)))
    assert len(tracking) == 1
    assert len(tracking[0]) == 2
    assert tracking[0].velocity()["col"].value == pytest.approx(4.0)


def test_the_same_crossing_declared_flat_is_two_objects_and_not_one_moved_wrongly():
    """124 cells is well past the coincidence radius, so the honest answer is two."""
    tracking = _run(_seam_crossing((ROW, COL)))
    assert len(tracking) == 2
    assert tracking.deaths == 1


def test_a_track_on_a_periodic_axis_refuses_a_displacement_without_the_period():
    only = _run(_seam_crossing((WRAP_ROW, WRAP_COL)))[0]
    stripped = Track(track_id=only.track_id, observations=only.observations, periods={})
    with pytest.raises(InvalidParameterError) as excinfo:
        stripped.displacement()
    assert "wrong by an axis" in str(excinfo.value)


def test_a_displacement_is_accumulated_link_by_link_and_not_taken_end_to_end():
    """A full lap and standing still have the same endpoints on a torus."""
    lap = [_feature(t, 20, (20 + 8 * t) % GRID_N, axes=(WRAP_ROW, WRAP_COL))
           for t in range(17)]
    only = _run(lap)[0]
    assert len(only) == 17
    assert only.displacement()["col"] == pytest.approx(128.0)
    assert only[0].location.coords["col"] == only[-1].location.coords["col"]


def test_a_scale_gate_refuses_a_link_the_declared_growth_rate_cannot_support():
    close_pair = [_feature(0, 20, 20, scale=5.0), _feature(1, 21, 21, scale=20.0)]
    ungated = _run(close_pair)
    gated = _run(close_pair, bounds=MotionBounds(max_doublings=0.5))
    assert len(ungated) == 1, "position alone cannot tell these apart"
    assert len(gated) == 2
    assert gated.steps[1].active_gates == ("position", "scale")


def test_an_orientation_gate_reads_the_convention_and_not_the_number():
    """170 and 350 degrees: the same axis, or opposed directions. The gate must know.

    Under `axis_180` the two features are the same ridge and link; under `direction_360`
    they are pointing opposite ways and the same declared turn rate refuses. The numbers on
    the records are identical in both runs.
    """
    bounds = MotionBounds(max_turn_degrees=30.0)
    axial = [_feature(0, 20, 20, orientation=Orientation(170.0, "axis_180")),
             _feature(1, 21, 21, orientation=Orientation(350.0, "axis_180"))]
    directed = [_feature(0, 20, 20, orientation=Orientation(170.0, "direction_360")),
                _feature(1, 21, 21, orientation=Orientation(350.0, "direction_360"))]
    assert len(_run(axial, bounds=bounds)) == 1
    assert len(_run(directed, bounds=bounds)) == 2


def test_two_orientation_conventions_in_one_set_are_refused_rather_than_averaged():
    mixed = [_feature(0, 20, 20, orientation=Orientation(10.0, "axis_180")),
             _feature(1, 21, 21, orientation=Orientation(20.0, "direction_360"))]
    with pytest.raises(SemanticComparisonError):
        _run(mixed, bounds=MotionBounds(max_turn_degrees=30.0))


def test_gating_on_orientation_is_refused_when_the_extractor_does_not_report_one():
    """The load-bearing refusal. `local_maximum` declares `reports_orientation: False`.

    A gate that silently passes every pair because the quantity is missing appears in the
    receipt, refuses nothing, and is indistinguishable from a gate that was doing work. The
    features here are the real output of TG2.2's extractor on the real benchmark, so this is
    the state a caller would actually be in.
    """
    from src.core.extraction import EXTRACTORS

    assert EXTRACTORS.entry("local_maximum").capabilities["reports_orientation"] is False
    results, _ = _vortex()
    found = [f for r in results for f in r]
    assert all(f.orientation is None for f in found)
    with pytest.raises(InvalidParameterError) as excinfo:
        track_extractions(results, bounds=MotionBounds(max_turn_degrees=45.0))
    assert "reports_orientation=False" in str(excinfo.value)


def test_gating_on_scale_is_refused_when_a_feature_does_not_carry_one():
    with pytest.raises(InvalidParameterError) as excinfo:
        _run([_feature(0, 20, 20, scale=None), _feature(1, 21, 21)],
             bounds=MotionBounds(max_doublings=1.0))
    assert "spatial scale" in str(excinfo.value)


def test_a_velocity_needs_a_clock_whose_units_were_recorded():
    only = _run([_feature(0, 20, 20, time_units=None),
                 _feature(1, 22, 22, time_units=None)])[0]
    with pytest.raises(InvalidParameterError) as excinfo:
        only.lifetime()
    assert "unrecorded unit" in str(excinfo.value)


def test_a_single_sighting_has_no_velocity_and_zero_is_not_the_answer():
    lone = _run([_feature(0, 20, 20)], times=[0.0, 1.0])[0]
    assert len(lone) == 1
    for method in (lone.velocity, lone.speed, lone.scale_velocity):
        with pytest.raises(InvalidParameterError) as excinfo:
            method()
        assert "never made" in str(excinfo.value)


def test_a_shrinking_track_has_no_doubling_time():
    shrinking = [_feature(t, 20, 20 + t, scale=10.0 - t) for t in range(4)]
    only = _run(shrinking)[0]
    assert only.scale_velocity() < 0.0
    with pytest.raises(InvalidParameterError) as excinfo:
        only.doubling_time()
    assert "read as a magnitude" in str(excinfo.value)


def test_a_track_cannot_be_assembled_across_two_datasets():
    """Inherited from `FeatureSet` rather than restated here - which is the point."""
    with pytest.raises(SemanticComparisonError):
        Track(track_id=0, observations=FeatureSet(
            [_feature(0, 20, 20), _feature(1, 22, 22, dataset="somewhere_else")]))


def test_a_track_refuses_two_observations_of_one_object_at_one_instant():
    with pytest.raises(InvalidParameterError) as excinfo:
        Track(track_id=0, observations=FeatureSet(
            [_feature(0, 20, 20), _feature(0, 22, 22)]))
    assert "not a fast-moving object" in str(excinfo.value)


# ================================================== 5. silence is a fact about the search


def test_a_frame_the_extractor_found_nothing_in_ends_the_track():
    """Bridging that gap would assert what happened while the object was invisible."""
    seen = [_feature(0, 20, 20), _feature(2, 20, 24)]
    tracking = _run(seen, times=[0.0, 1.0, 2.0])
    assert len(tracking) == 2
    assert tracking.deaths == 1
    assert tracking.empty_frames == (1.0,)
    assert tracking.steps[1].observations == 0


def test_the_same_two_observations_on_a_two_frame_clock_are_one_track():
    """The difference is the clock, and nothing else - which is why it must be supplied."""
    seen = [_feature(0, 20, 20), _feature(2, 20, 24)]
    tracking = _run(seen, times=[0.0, 2.0])
    assert len(tracking) == 1
    assert tracking.empty_frames == ()


def test_a_clock_that_stops_where_the_evidence_stops_cannot_report_a_death():
    """The regression for the defect this slice's own benchmark run found.

    Reading the clock off the features found makes the last frame with a feature in it the
    end of the run, so a track that stopped six frames into a twenty-four frame sequence
    looked complete. The vortex advected out of the frame is the case: the extractor refuses
    the clipped blob by rule R13, and the recorded answer says one death.
    """
    seen = [_feature(t, 20, 20 + 2 * t) for t in range(6)]
    short = _run(seen, times=[float(t) for t in range(6)])
    full = _run(seen, times=[float(t) for t in range(24)])
    assert short.deaths == 0
    assert full.deaths == 1
    assert len(short) == len(full) == 1


def test_the_vortex_that_leaves_the_frame_is_a_death_the_recorded_answer_agrees_with():
    """D59's case, end to end: extractor, tracker and recorded answer on the same topology."""
    results, truth = _vortex(start=(100.0, 100.0))
    tracking = track_extractions(results)
    assert [r.found for r in results] == [1] * 6 + [0] * 18
    assert truth["measurable_frames"] == list(range(6))
    assert len(tracking) == truth["track_count"] == 1
    assert tracking.deaths == truth["deaths"] == 1
    assert len(tracking.empty_frames) == 18
    assert results[6].rejected["outside_valid_interior"] >= 1


def test_a_run_that_found_nothing_anywhere_is_not_a_tracking_result_with_zero_tracks():
    field = ExtractionField(
        values=np.zeros((8, 8)), axes=(ROW, COL), domain="synthetic", dataset="toy",
        variable="amplitude", units=None, time=0.0, time_units="frames",
        representation="identity")

    class _Empty:
        def __init__(self, f):
            self.field = f

        def __iter__(self):
            return iter(())

    assert track_extractions([_Empty(field)]) is None


def test_a_feature_at_a_time_the_run_does_not_list_is_refused():
    with pytest.raises(InvalidParameterError) as excinfo:
        _run([_feature(0, 20, 20), _feature(5, 22, 22)], times=[0.0, 1.0, 2.0])
    assert "another run" in str(excinfo.value)


def test_an_unordered_clock_is_refused_rather_than_sorted():
    with pytest.raises(InvalidParameterError) as excinfo:
        _run([_feature(0, 20, 20), _feature(1, 22, 22)], times=[1.0, 0.0])
    assert "strictly increasing" in str(excinfo.value)


def test_the_receipt_carries_the_gate_the_bounds_and_every_step():
    results, _ = _vortex()
    record = track_extractions(results).describe()
    assert record["associator"] == "hungarian"
    assert record["alpha"] == 0.05
    assert record["frames"] == len(results)
    assert record["empty_frames"] == []
    assert len(record["steps"]) == len(results)
    assert record["tracks"][0]["velocity"]["col"]["units"] == "cells/frames"
    assert record["volume"]["extent"] == {"row": float(GRID_N), "col": float(GRID_N)}


def test_the_same_input_tracks_identically_twice():
    """Standard E4: nothing here depends on dictionary or set iteration order."""
    first = _run(_crossing(), associator="hungarian")
    second = _run(_crossing(), associator="hungarian")
    assert first.describe() == second.describe()
