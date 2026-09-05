"""T4F.1: the timed event substrate, the clock it will not assume, and the order it will not invent."""

from __future__ import annotations

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_events import (
    COVERAGE_INCOMPLETE, COVERAGE_MEASURED, COVERAGE_UNDECLARED, EVENT_SCHEMA,
    ObservationGrid, events_from_catalogue,
)
from src.analysis_engine.spectral_invariance import AxisAdmission, ConstellationSignature
from src.core.errors import InvalidParameterError


METRIC = SignatureMetric(AttributeWeights(
    geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))

#: identity -> (geometry factor, frame). Five occurrences of one configuration and two of
#: another, with both present at frame 4 so simultaneity has something to be tested on.
PLANTED = {1: (1.00, 0.0), 2: (1.01, 2.0), 3: (1.02, 4.0), 4: (1.03, 8.0), 5: (1.04, 9.0),
           6: (2.00, 4.0), 7: (2.02, 7.0)}

FRAMES = tuple(float(index) for index in range(12))


def _signature(factor, identity, time, *, units="frames", mode="scale_specific"):
    return ConstellationSignature(
        key=(float(identity), 0, 1, 2), time=float(time), cardinality=3,
        mode=mode, order=(0, 1, 2),
        geometry=tuple(factor * value for value in (1.0, 1.3, 1.6)),
        geometry_relation="distance", bearings=(10.0, 35.0, 70.0),
        strengths=(0.75, 1.0, 4.0 / 3.0), scales=(8.0, 8.0, 16.0),
        scale_units="cells", axis=AxisAdmission(True, 3.0, 1.0421, 25.0),
        track_ids=(0, 1, 2), bands=("L1", "L1", "L2"), time_units=units)


def _catalogue(order=None, **overrides):
    calibration = [_signature(1.0, 100, 0.0, **overrides),
                   _signature(1.05, 101, 1.0, **overrides)]
    tolerance = calibrate_signature_tolerance(calibration, metric=METRIC)
    signatures = [_signature(factor, identity, time, **overrides)
                  for identity, (factor, time) in sorted(PLANTED.items())]
    if order is not None:
        signatures = [signatures[index] for index in order]
    return cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)


def _grid(frames=FRAMES, **kwargs):
    kwargs.setdefault("time_units", "frames")
    kwargs.setdefault("cadence", 1.0)
    return ObservationGrid(frames=tuple(frames), **kwargs)


def _series(**kwargs):
    return events_from_catalogue(_catalogue(), grid=_grid(**kwargs))


def _by_support(catalogue, support):
    matches = [pattern for pattern in catalogue if pattern.support == support]
    assert len(matches) == 1
    return matches[0]


# --- the grid: what was looked at, in what unit, on what cadence -----------------------------

def test_a_grid_without_a_named_time_unit_is_refused_because_a_span_would_have_none():
    for invalid in (None, "", "   ", 3):
        with pytest.raises(InvalidParameterError, match="non-empty unit name"):
            ObservationGrid(frames=FRAMES, time_units=invalid)


def test_an_empty_grid_is_an_unrun_pass_rather_than_a_pass_that_observed_nothing():
    with pytest.raises(InvalidParameterError, match="at least one searched frame"):
        ObservationGrid(frames=(), time_units="frames")


def test_repeated_or_reordered_frames_are_refused_rather_than_sorted():
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        ObservationGrid(frames=(0.0, 1.0, 1.0), time_units="frames")
    with pytest.raises(InvalidParameterError, match="strictly increasing"):
        ObservationGrid(frames=(0.0, 2.0, 1.0), time_units="frames")
    with pytest.raises(InvalidParameterError, match="finite frame times"):
        ObservationGrid(frames=(0.0, float("nan")), time_units="frames")


def test_a_cadence_is_positive_and_finite_or_absent_and_the_frames_must_lie_on_it():
    with pytest.raises(InvalidParameterError, match="positive finite sampling interval"):
        ObservationGrid(frames=FRAMES, time_units="frames", cadence=0.0)
    with pytest.raises(InvalidParameterError, match="declared cadence"):
        ObservationGrid(frames=(0.0, 1.0, 2.5), time_units="frames", cadence=1.0)
    assert ObservationGrid(frames=(0.0, 2.0, 6.0), time_units="frames", cadence=2.0).cadence == 2.0


def test_coverage_is_measured_against_the_cadence_and_refused_without_one():
    complete = _grid()
    assert complete.coverage_between(2.0, 4.0) == (COVERAGE_MEASURED, 1, 0)

    gapped = _grid(frames=tuple(value for value in FRAMES if value != 3.0))
    status, observed, missing = gapped.coverage_between(2.0, 4.0)
    assert (status, observed, missing) == (COVERAGE_INCOMPLETE, 0, 1)

    undeclared = _grid(cadence=None)
    assert undeclared.coverage_between(2.0, 4.0) == (COVERAGE_UNDECLARED, 1, None)
    assert undeclared.describe()["coverage_decidable"] is False
    assert "never read" in undeclared.describe()["coverage_basis"]


# --- reading a catalogue as events ------------------------------------------------------------

def test_the_frames_that_were_searched_cannot_be_recovered_from_what_was_found():
    with pytest.raises(InvalidParameterError, match="explicit ObservationGrid"):
        events_from_catalogue(_catalogue(), grid=FRAMES)
    with pytest.raises(InvalidParameterError, match="complete T4E.3 PatternCatalogue"):
        events_from_catalogue([], grid=_grid())


def test_an_empty_catalogue_is_an_unrun_clustering_not_an_absence_of_events():
    empty = _catalogue()
    with pytest.raises(InvalidParameterError, match="at least one pattern"):
        events_from_catalogue(
            type(empty)(patterns=(), metric=empty.metric, tolerance=empty.tolerance,
                        n_signatures=0),
            grid=_grid())


def test_an_occurrence_at_a_frame_nothing_searched_is_refused_by_name():
    with pytest.raises(InvalidParameterError, match="sighting where nothing looked"):
        events_from_catalogue(_catalogue(), grid=_grid(frames=(0.0, 2.0, 4.0)))


def test_two_scale_modes_in_one_series_would_sequence_two_different_questions():
    specific = _catalogue()
    invariant = _catalogue(mode="scale_invariant")
    mixed = type(specific)(
        patterns=specific.patterns[:1] + invariant.patterns[1:],
        metric=specific.metric, tolerance=specific.tolerance, n_signatures=len(PLANTED))
    with pytest.raises(InvalidParameterError, match="one scale mode"):
        events_from_catalogue(mixed, grid=_grid())


def test_one_identity_appearing_twice_would_manufacture_a_transition():
    catalogue = _catalogue()
    duplicated = type(catalogue)(
        patterns=catalogue.patterns + catalogue.patterns[:1],
        metric=catalogue.metric, tolerance=catalogue.tolerance,
        n_signatures=catalogue.n_signatures)
    with pytest.raises(InvalidParameterError, match="manufacture a transition"):
        events_from_catalogue(duplicated, grid=_grid())


def test_a_member_measured_in_another_unit_than_the_grid_declares_is_refused():
    catalogue = _catalogue(units="hours")
    with pytest.raises(InvalidParameterError, match="unit the grid declares"):
        events_from_catalogue(catalogue, grid=_grid())


def test_members_carrying_no_unit_are_counted_rather_than_assumed_to_agree():
    series = events_from_catalogue(_catalogue(units=None), grid=_grid())
    assert series.units_uncarried_by_members == len(PLANTED)
    assert series.units_confirmed_by_members == 0
    assert _series().units_confirmed_by_members == len(PLANTED)


# --- what the substrate then says -------------------------------------------------------------

def test_the_event_order_is_the_records_and_not_the_input_lists():
    reference = [(event.time, event.identity) for event in _series()]
    reversed_input = events_from_catalogue(
        _catalogue(order=list(reversed(range(len(PLANTED))))), grid=_grid())
    assert [(event.time, event.identity) for event in reversed_input] == reference
    assert [event.time for event in reversed_input] == [0.0, 2.0, 4.0, 4.0, 7.0, 8.0, 9.0]


def test_events_sharing_a_frame_are_reported_as_simultaneous_and_carry_no_order():
    series = _series()
    co_occurring = series.co_occurrences()
    assert list(co_occurring) == [4.0]
    assert len(co_occurring[4.0]) == 2
    assert len(series.at(4.0)) == 2
    assert "artefact of sorting" in series.describe()["co_occurrence_basis"]


def test_each_pattern_reports_its_own_spans_in_the_declared_unit():
    series = _series()
    catalogue = _catalogue()
    frequent = _by_support(catalogue, 5).pattern_id
    spans = series.spans(frequent)
    assert [span.duration for span in spans] == [2.0, 2.0, 4.0, 1.0]
    assert {span.time_units for span in spans} == {"frames"}
    assert [span.coverage for span in spans] == [COVERAGE_MEASURED] * 4
    assert [span.frames_observed_between for span in spans] == [1, 1, 3, 0]
    assert series.spans(_by_support(catalogue, 2).pattern_id)[0].duration == 3.0


def test_a_span_crossing_an_unsearched_frame_says_so_and_counts_what_was_missed():
    series = _series(frames=tuple(value for value in FRAMES if value not in (3.0, 6.0)))
    spans = series.spans(_by_support(_catalogue(), 5).pattern_id)
    by_pair = {(span.earlier, span.later): span for span in spans}
    assert by_pair[(2.0, 4.0)].coverage == COVERAGE_INCOMPLETE
    assert by_pair[(2.0, 4.0)].frames_unobserved_between == 1
    assert by_pair[(4.0, 8.0)].coverage == COVERAGE_INCOMPLETE
    assert by_pair[(4.0, 8.0)].frames_unobserved_between == 1
    assert by_pair[(0.0, 2.0)].coverage == COVERAGE_MEASURED


def test_without_a_cadence_every_span_declares_its_coverage_unknown_rather_than_complete():
    spans = _series(cadence=None).spans(_by_support(_catalogue(), 5).pattern_id)
    assert {span.coverage for span in spans} == {COVERAGE_UNDECLARED}
    assert {span.frames_unobserved_between for span in spans} == {None}


def test_the_receipt_publishes_the_schema_the_grid_and_the_boundary_it_stops_at():
    body = _series().describe()
    assert body["schema"] == EVENT_SCHEMA
    assert body["n_events"] == len(PLANTED)
    assert body["n_patterns"] == 2
    assert body["mode"] == "scale_specific"
    assert body["grid"]["n_frames_searched"] == len(FRAMES)
    assert body["grid"]["time_units"] == "frames"
    assert body["frames_with_co_occurrence"] == 1
    assert len(body["spans"]) == len(PLANTED) - body["n_patterns"]
    for word in ("recurrence", "period", "precursor", "discovery"):
        assert word in body["claim_boundary"]
