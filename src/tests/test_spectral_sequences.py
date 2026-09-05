"""T4F.2: counted sequences, the denominator they are counted over, and a gap that is no period."""

from __future__ import annotations

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_events import (
    COVERAGE_UNDECLARED, ObservationGrid, WINDOW_INCOMPLETE, WINDOW_MEASURED, WINDOW_TRUNCATED,
    events_from_catalogue,
)
from src.analysis_engine.spectral_invariance import AxisAdmission, ConstellationSignature
from src.analysis_engine.spectral_mining import MiningBudget, MiningBudgetExceededError
from src.analysis_engine.spectral_sequences import (
    CONFIDENCE_MEASURED, CONFIDENCE_NO_ELIGIBLE_ANTECEDENT, CONFIDENCE_UNDECIDABLE,
    RECURRENCE_NO_REPEAT, RECURRENCE_REPEATS, RECURRENCE_SCHEMA, RECURRENCE_TOO_FEW_SPANS,
    RECURRENCE_UNMEASURED, SEQUENCE_BELOW_MINIMUM, SEQUENCE_SCHEMA, SEQUENCE_SUPPORTED,
    TransitionWindow, mine_frequent_sequences, recurrence_report,
)
from src.core.errors import InvalidParameterError


METRIC = SignatureMetric(AttributeWeights(
    geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))

#: Three well-separated configurations. The jitter is additive so it stays inside the calibrated
#: tolerance at every scale rather than growing with the configuration it is applied to.
FACTORS = {"A": 1.0, "B": 2.0, "C": 3.0}

BUDGET = MiningBudget(max_candidates=10_000, max_seconds=60.0)

#: The planted chain: A, then B two frames later, then C two after that, four times over.
CHAIN = [(letter, anchor + offset)
         for anchor in (0.0, 6.0, 12.0, 18.0)
         for letter, offset in (("A", 0.0), ("B", 2.0), ("C", 4.0))]


def _signature(factor, identity, time, *, units="frames", mode="scale_specific"):
    return ConstellationSignature(
        key=(float(identity), 0, 1, 2), time=float(time), cardinality=3,
        mode=mode, order=(0, 1, 2),
        geometry=tuple(factor * value for value in (1.0, 1.3, 1.6)),
        geometry_relation="distance", bearings=(10.0, 35.0, 70.0),
        strengths=(0.75, 1.0, 4.0 / 3.0), scales=(8.0, 8.0, 16.0),
        scale_units="cells", axis=AxisAdmission(True, 3.0, 1.0421, 25.0),
        track_ids=(0, 1, 2), bands=("L1", "L1", "L2"), time_units=units)


def _catalogue(plan, **overrides):
    calibration = [_signature(1.0, 900, 0.0, **overrides),
                   _signature(1.05, 901, 1.0, **overrides)]
    tolerance = calibrate_signature_tolerance(calibration, metric=METRIC)
    signatures = [_signature(FACTORS[letter] + 0.01 * (index % 4), 1000 + index, time,
                             **overrides)
                  for index, (letter, time) in enumerate(plan)]
    return cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)


def _grid(frames=tuple(float(index) for index in range(26)), **kwargs):
    kwargs.setdefault("time_units", "frames")
    kwargs.setdefault("cadence", 1.0)
    return ObservationGrid(frames=tuple(frames), **kwargs)


def _series(plan=None, grid=None, **overrides):
    return events_from_catalogue(_catalogue(CHAIN if plan is None else plan, **overrides),
                                 grid=_grid() if grid is None else grid)


def _ids(series, plan=None):
    """Map planted letter to the pattern id clustering gave it, via a frame only it occupies."""
    plan = CHAIN if plan is None else plan
    mapping = {}
    for letter, time in plan:
        if letter in mapping:
            continue
        events = series.at(time)
        assert len(events) == 1, "the lookup frame must belong to one letter alone"
        mapping[letter] = events[0].pattern_id
    return mapping


def _window(minimum=1.0, maximum=3.0, units="frames"):
    return TransitionWindow(minimum_lag=minimum, maximum_lag=maximum, time_units=units)


def _mine(series, window=None, minimum_support=1, maximum_length=3, budget=BUDGET, **kwargs):
    return mine_frequent_sequences(
        series, window=_window() if window is None else window,
        minimum_support=minimum_support, maximum_length=maximum_length, budget=budget, **kwargs)


def _find(result, sequence):
    matches = [item for item in result.sequences if item.sequence == tuple(sequence)]
    assert len(matches) == 1, "each candidate is counted once"
    return matches[0]


# --- the window a transition has to be declared in -------------------------------------------

def test_a_zero_minimum_lag_would_make_two_events_on_one_frame_a_succession():
    with pytest.raises(InvalidParameterError, match="strictly positive lag"):
        TransitionWindow(minimum_lag=0.0, maximum_lag=3.0, time_units="frames")
    with pytest.raises(InvalidParameterError, match="strictly positive lag"):
        TransitionWindow(minimum_lag=-1.0, maximum_lag=3.0, time_units="frames")


def test_a_window_states_finite_ordered_lags_in_a_named_unit():
    with pytest.raises(InvalidParameterError, match="at or above the minimum"):
        TransitionWindow(minimum_lag=3.0, maximum_lag=1.0, time_units="frames")
    with pytest.raises(InvalidParameterError, match="a finite lag"):
        TransitionWindow(minimum_lag=1.0, maximum_lag=float("inf"), time_units="frames")
    for invalid in (None, "", "  ", 7):
        with pytest.raises(InvalidParameterError, match="non-empty unit name"):
            TransitionWindow(minimum_lag=1.0, maximum_lag=3.0, time_units=invalid)


def test_a_lag_in_one_unit_against_frames_in_another_is_refused_rather_than_converted():
    with pytest.raises(InvalidParameterError, match="the unit the grid declares"):
        _mine(_series(), window=_window(units="hours"))


def test_a_window_no_frame_can_fall_inside_is_refused_rather_than_counted_as_zero():
    with pytest.raises(InvalidParameterError, match="at least one lag on the grid's own cadence"):
        _mine(_series(), window=_window(minimum=1.2, maximum=1.8))
    # The same window is admissible on a grid whose cadence can reach into it.
    fine = _grid(frames=tuple(0.5 * index for index in range(52)), cadence=0.5)
    result = _mine(_series(grid=fine), window=_window(minimum=1.2, maximum=1.8))
    assert result.window.describe()["minimum_lag"] == 1.2


# --- what the grid can say about a proposed window -------------------------------------------

def test_a_window_is_measured_holed_or_truncated_and_the_three_are_kept_apart():
    grid = _grid(frames=tuple(float(index) for index in range(24) if index != 13))
    assert grid.observation_of_window(2.0, 5.0) == (WINDOW_MEASURED, 4, 0)
    assert grid.observation_of_window(12.0, 15.0) == (WINDOW_INCOMPLETE, 3, 1)
    status, _observed, missing = grid.observation_of_window(21.0, 26.0)
    assert (status, missing) == (WINDOW_TRUNCATED, None)


def test_without_a_cadence_a_window_cannot_be_called_searched_at_all():
    grid = _grid(cadence=None)
    assert grid.observation_of_window(2.0, 5.0)[0] == COVERAGE_UNDECLARED


# --- counting the chain ----------------------------------------------------------------------

def test_the_planted_chain_is_counted_with_its_support_and_its_confidence():
    series = _series()
    ids = _ids(series)
    result = _mine(series, minimum_support=4)

    pair = _find(result, (ids["A"], ids["B"]))
    assert (pair.support, pair.eligible_antecedents, pair.antecedent_occurrences) == (4, 4, 4)
    assert (pair.confidence, pair.confidence_status) == (1.0, CONFIDENCE_MEASURED)
    assert pair.status == SEQUENCE_SUPPORTED

    triple = _find(result, (ids["A"], ids["B"], ids["C"]))
    assert triple.support == 4 and triple.confidence == 1.0
    assert triple.status == SEQUENCE_SUPPORTED

    # The reversed chain is present as a candidate and is counted at zero rather than omitted.
    reversed_pair = _find(result, (ids["C"], ids["B"]))
    assert reversed_pair.support == 0
    assert reversed_pair.status == SEQUENCE_BELOW_MINIMUM


def test_a_longer_chain_needs_a_longer_window_to_have_been_searched():
    grid = _grid(frames=tuple(float(index) for index in range(24)))
    series = _series(grid=grid)
    ids = _ids(series)
    result = _mine(series, minimum_support=4)

    pair = _find(result, (ids["A"], ids["B"]))
    triple = _find(result, (ids["A"], ids["B"], ids["C"]))

    assert pair.eligible_antecedents == 4, "one step from the last A still fits in the record"
    assert triple.eligible_antecedents == 3, (
        "two steps from the last A could have finished past frame 23, so that A was never given "
        "the whole window a three-link chain needs")
    assert triple.ineligible_truncated == 1
    assert triple.observed_completions_including_ineligible == 4
    assert triple.support == 3


def test_a_sequence_one_occurrence_short_of_the_minimum_is_not_admitted():
    series = _series(grid=_grid(frames=tuple(float(index) for index in range(24))))
    ids = _ids(series)
    triple = (ids["A"], ids["B"], ids["C"])

    assert _find(_mine(series, minimum_support=4), triple).status == SEQUENCE_BELOW_MINIMUM
    at_threshold = _find(_mine(series, minimum_support=3), triple)
    assert (at_threshold.support, at_threshold.status) == (3, SEQUENCE_SUPPORTED), (
        "the declared minimum is inclusive, and one below it is out")


def test_two_events_on_one_frame_are_never_counted_as_a_step():
    series = _series(plan=[("A", 3.0), ("B", 3.0), ("A", 7.0), ("B", 7.0)])
    first, second = (event.pattern_id for event in series.at(3.0))
    result = _mine(series, maximum_length=2)

    assert _find(result, (first, second)).support == 0
    assert _find(result, (second, first)).support == 0
    assert series.co_occurrences() == {3.0: (first, second), 7.0: (first, second)}, (
        "both frames are simultaneity, and neither direction may be read out of it")


def test_a_pattern_following_itself_is_a_chain_and_is_counted_as_one():
    plan = [("A", time) for time in (0.0, 2.0, 4.0, 6.0)]
    series = _series(plan=plan)
    ids = _ids(series, [("A", 0.0)])
    self_pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["A"]))

    assert (self_pair.support, self_pair.eligible_antecedents) == (3, 4)
    assert self_pair.confidence == 0.75, (
        "the fourth A's window was searched and held nothing, so unlike a censored antecedent "
        "it belongs in the denominator as a miss")


# --- the denominator, which is the part that can be quietly wrong -----------------------------

def test_an_antecedent_the_record_ended_before_is_censored_rather_than_counted_unfollowed():
    plan = [("A", 0.0), ("B", 2.0), ("A", 6.0), ("B", 8.0), ("A", 12.0), ("B", 14.0),
            ("A", 18.0)]
    grid = _grid(frames=tuple(float(index) for index in range(20)))
    series = _series(plan=plan, grid=grid)
    ids = _ids(series, plan)
    pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["B"]))

    assert pair.antecedent_occurrences == 4
    assert (pair.eligible_antecedents, pair.support) == (3, 3)
    assert pair.ineligible_truncated == 1 and pair.ineligible_unobserved == 0
    assert pair.confidence == 1.0, (
        "the fourth A was never given a window to be followed in; dividing by it would price "
        "the end of the record as a failure to follow")


def test_an_antecedent_whose_window_was_not_read_is_excluded_and_counted_apart():
    plan = [("A", 0.0), ("B", 2.0), ("A", 6.0), ("B", 8.0), ("A", 12.0)]
    grid = _grid(frames=tuple(float(index) for index in range(20) if index != 13))
    series = _series(plan=plan, grid=grid)
    ids = _ids(series, plan)
    pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["B"]))

    assert (pair.eligible_antecedents, pair.support, pair.confidence) == (2, 2, 1.0)
    assert pair.ineligible_unobserved == 1 and pair.ineligible_truncated == 0


def test_a_completion_at_an_inadmissible_antecedent_is_published_rather_than_dropped():
    plan = [("A", 0.0), ("B", 2.0), ("A", 6.0), ("B", 8.0), ("A", 12.0), ("B", 14.0)]
    grid = _grid(frames=tuple(float(index) for index in range(20) if index != 13))
    series = _series(plan=plan, grid=grid)
    ids = _ids(series, plan)
    pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["B"]))

    assert pair.observed_completions_including_ineligible == 3
    assert pair.support == 2, "the third completion was seen and is not admissible into a ratio"
    assert pair.confidence == 1.0


def test_without_a_cadence_the_count_stands_and_the_ratio_is_refused():
    series = _series(grid=_grid(cadence=None))
    ids = _ids(series)
    pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["B"]))

    assert pair.observed_completions_including_ineligible == 4
    assert pair.eligible_antecedents == 0
    assert pair.confidence is None
    assert pair.confidence_status == CONFIDENCE_UNDECIDABLE


def test_a_pattern_with_no_admissible_antecedent_reports_that_rather_than_a_zero_ratio():
    plan = [("A", 25.0), ("B", 2.0), ("B", 8.0)]
    series = _series(plan=plan)
    ids = _ids(series, plan)
    pair = _find(_mine(series, maximum_length=2), (ids["A"], ids["B"]))
    assert pair.confidence is None, (
        "the record's last frame is its only A, so no A here was ever given a window; a zero "
        "ratio would report that as a pattern that fails to be followed")
    assert pair.confidence_status == CONFIDENCE_NO_ELIGIBLE_ANTECEDENT
    assert pair.ineligible_truncated == 1


def test_no_sequence_claims_more_support_than_the_occurrences_it_was_counted_over():
    result = _mine(_series())
    for item in result.sequences:
        assert item.support <= item.eligible_antecedents <= item.antecedent_occurrences
        assert item.support <= item.observed_completions_including_ineligible
        if item.confidence is not None:
            assert 0.0 <= item.confidence <= 1.0


# --- the pruning rule, which this module can prove -------------------------------------------

def test_support_cannot_rise_when_a_sequence_is_extended():
    result = _mine(_series(), minimum_support=1, maximum_length=4)
    by_sequence = {item.sequence: item for item in result.sequences}
    checked = 0
    for sequence, item in by_sequence.items():
        prefix = sequence[:-1]
        if len(prefix) >= 2:
            assert item.support <= by_sequence[prefix].support
            assert item.eligible_antecedents <= by_sequence[prefix].eligible_antecedents
            checked += 1
    assert checked > 0, "the sweep must actually have reached an extension to check"


def test_only_extensions_of_supported_sequences_are_examined():
    series = _series()
    patterns = len(series.pattern_ids)
    result = _mine(series, minimum_support=4, maximum_length=3)
    supported_pairs = len(result.of_length(2))

    assert supported_pairs < patterns ** 2, "the fixture must prune something to test pruning"
    assert result.candidates_examined == patterns ** 2 + supported_pairs * patterns
    assert result.candidates_pruned_by_antimonotonicity == (
        (patterns ** 2 - supported_pairs) * patterns)
    exhaustive = sum(patterns ** length for length in (2, 3))
    assert result.candidates_examined + result.candidates_pruned_by_antimonotonicity == (
        exhaustive), (
        "what was counted plus what the rule saved must be the sweep nobody had to run")


# --- refusing the sweep whole -----------------------------------------------------------------

def test_a_sequence_length_below_two_is_an_occurrence_count_and_not_a_transition():
    for invalid in (1, 0, -3, True, 2.5):
        with pytest.raises(InvalidParameterError, match="length of at least two"):
            _mine(_series(), maximum_length=invalid)


def test_a_minimum_support_below_one_or_fractional_is_refused():
    for invalid in (0, -1, 1.5, True):
        with pytest.raises(InvalidParameterError, match="positive integer occurrence count"):
            _mine(_series(), minimum_support=invalid)


def test_the_series_and_the_budget_are_both_explicit_objects():
    with pytest.raises(InvalidParameterError, match="a T4F.1 EventSeries"):
        _mine("not a series")
    with pytest.raises(InvalidParameterError, match="an explicit MiningBudget"):
        _mine(_series(), budget=None)
    with pytest.raises(InvalidParameterError, match="an explicit TransitionWindow"):
        mine_frequent_sequences(_series(), window=None, minimum_support=1, maximum_length=2,
                                budget=BUDGET)


def test_either_budget_overrun_refuses_the_sweep_without_returning_a_prefix():
    with pytest.raises(MiningBudgetExceededError) as candidate_error:
        _mine(_series(), budget=MiningBudget(max_candidates=4, max_seconds=60.0))
    assert candidate_error.value.context["partial_result"] is False

    ticks = iter([0.0] + [99.0] * 200)
    with pytest.raises(MiningBudgetExceededError) as clock_error:
        _mine(_series(), budget=MiningBudget(max_candidates=10_000, max_seconds=1.0),
              clock=lambda: next(ticks))
    assert clock_error.value.context["resource"] == "wall-clock"


def test_the_sequence_receipt_publishes_its_window_its_eligibility_rule_and_its_boundary():
    receipt = _mine(_series()).describe()
    assert receipt["schema"] == SEQUENCE_SCHEMA
    assert receipt["window"] == {"minimum_lag": 1.0, "maximum_lag": 3.0, "time_units": "frames"}
    assert "was searched" in receipt["eligibility_basis"]
    assert "truncated rather than unfollowed" in receipt["eligibility_basis"]
    for forbidden in ("base rate", "lift", "p-value", "precursor", "cause"):
        assert forbidden in receipt["claim_boundary"], (
            "the boundary must name what support is not, and %r is one of them" % forbidden)
    assert receipt["grid"]["n_frames_searched"] == 26


# --- recurrence: a repeated gap, and the several ways it is not a period -----------------------

def test_a_gap_that_repeats_is_reported_with_its_count_and_its_share():
    series = _series()
    ids = _ids(series)
    interval = recurrence_report(series, minimum_spans=2).for_pattern(ids["A"])

    assert interval.status == RECURRENCE_REPEATS
    assert (interval.spans_total, interval.spans_measured) == (3, 3)
    assert (interval.modal_interval, interval.modal_count) == (6.0, 3)
    assert interval.concentration == 1.0
    assert interval.distinct_intervals == 1, (
        "three gaps that agree are one interval seen three times, not three intervals")
    assert (interval.resolution, interval.time_units) == (1.0, "frames")
    assert (interval.shortest, interval.longest) == (6.0, 6.0)


def test_a_gap_containing_an_unread_instant_bounds_an_interval_and_is_not_tallied():
    plan = [("A", 0.0), ("A", 6.0), ("A", 12.0), ("A", 21.0)]
    grid = _grid(frames=tuple(float(index) for index in range(24) if index not in (16, 17)))
    series = _series(plan=plan, grid=grid)
    ids = _ids(series, [("A", 0.0)])
    interval = recurrence_report(series, minimum_spans=2).for_pattern(ids["A"])

    assert (interval.spans_total, interval.spans_measured) == (3, 2)
    assert interval.spans_bounded_above_only == 1
    assert (interval.modal_interval, interval.modal_count) == (6.0, 2)
    assert interval.concentration == 1.0, (
        "the share is of the gaps this record could measure; dividing by the bounded one too "
        "would report the excluded gap as a gap that disagreed")
    assert interval.longest == 6.0, (
        "the nine-frame gap is an upper bound on an interval and must not be its longest")


def test_a_series_whose_every_gap_is_unmeasured_reports_no_measured_span():
    plan = [("A", 0.0), ("A", 12.0), ("A", 21.0)]
    grid = _grid(frames=tuple(float(index) for index in range(24)
                              if index not in (5, 6, 16, 17)))
    series = _series(plan=plan, grid=grid)
    ids = _ids(series, [("A", 0.0)])
    interval = recurrence_report(series, minimum_spans=2).for_pattern(ids["A"])

    assert interval.status == RECURRENCE_UNMEASURED
    assert interval.spans_bounded_above_only == 2
    assert interval.modal_interval is None and interval.concentration is None


def test_without_a_cadence_no_gap_is_measured_and_the_report_says_which_kind_of_ignorance():
    series = _series(grid=_grid(cadence=None))
    ids = _ids(series)
    report = recurrence_report(series, minimum_spans=2)
    interval = report.for_pattern(ids["A"])

    assert interval.status == RECURRENCE_UNMEASURED
    assert (interval.spans_undecidable, interval.spans_bounded_above_only) == (3, 0)
    assert interval.resolution is None
    assert report.admissible_lattice_values is None


def test_gaps_that_never_agree_are_reported_as_no_repeated_interval():
    plan = [("A", 0.0), ("A", 1.0), ("A", 3.0), ("A", 6.0), ("A", 10.0)]
    series = _series(plan=plan)
    ids = _ids(series, [("A", 0.0)])
    interval = recurrence_report(series, minimum_spans=2).for_pattern(ids["A"])

    assert interval.status == RECURRENCE_NO_REPEAT
    assert interval.distinct_intervals == 4 and interval.spans_measured == 4
    assert interval.modal_interval is None


def test_fewer_measured_gaps_than_declared_report_too_few_rather_than_a_modal_interval():
    plan = [("A", 0.0), ("A", 6.0), ("A", 12.0)]
    series = _series(plan=plan)
    ids = _ids(series, [("A", 0.0)])
    interval = recurrence_report(series, minimum_spans=3).for_pattern(ids["A"])

    assert interval.status == RECURRENCE_TOO_FEW_SPANS
    assert interval.spans_measured == 2
    assert interval.modal_interval is None, (
        "two agreeing gaps below a declared minimum are a subthreshold tally, not a recurrence")


def test_a_single_gap_cannot_be_a_recurrence_and_the_minimum_says_so():
    for invalid in (1, 0, -2, True, 2.5):
        with pytest.raises(InvalidParameterError, match="at least two measured spans"):
            recurrence_report(_series(), minimum_spans=invalid)
    with pytest.raises(InvalidParameterError, match="a T4F.1 EventSeries"):
        recurrence_report("not a series", minimum_spans=2)


def test_the_recurrence_receipt_publishes_the_denominator_of_its_own_coincidence():
    report = recurrence_report(_series(), minimum_spans=2)
    receipt = report.describe()

    assert receipt["schema"] == RECURRENCE_SCHEMA
    assert receipt["admissible_lattice_values"] == 25
    assert "denominator of any concentration" in receipt["coincidence_basis"]
    assert "bounds an interval from above" in receipt["exclusion_basis"]
    for forbidden in ("period", "frequency", "oscillation"):
        assert forbidden in receipt["claim_boundary"]
    assert len(receipt["intervals"]) == len(report.intervals) == 3


def test_a_pattern_absent_from_the_series_is_refused_rather_than_reported_empty():
    with pytest.raises(InvalidParameterError, match="a pattern present in this series"):
        recurrence_report(_series(), minimum_spans=2).for_pattern(9999)
