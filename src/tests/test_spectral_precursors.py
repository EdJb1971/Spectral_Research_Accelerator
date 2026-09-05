"""T4F.3: the first null this phase draws, and the denominator the null is drawn against."""

from __future__ import annotations

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_events import ObservationGrid, events_from_catalogue
from src.analysis_engine.spectral_invariance import AxisAdmission, ConstellationSignature
from src.analysis_engine.spectral_mining import MiningBudget, MiningBudgetExceededError
from src.analysis_engine.spectral_precursors import (
    INTERVAL_METHOD, NULL_CAUTION, NULL_CIRCULAR_SHIFT, NULL_DESTROYS, NULL_PRESERVES,
    NULL_UNIFORM_RELOCATION, PRECURSOR_CLAIM_BOUNDARY, PRECURSOR_SCHEMA, RULE_BASE_RATE_ZERO,
    RULE_NOT_DISTINGUISHED, RULE_NO_ELIGIBLE_ANTECEDENT, RULE_PRECURSOR, LagFamily,
    _available_shifts, _overlapping_windows, _rotate, _shift_floor, precursor_report,
    window_base_rate,
)
from src.analysis_engine.spectral_sequences import (
    TransitionWindow, count_sequence, times_by_pattern,
)
from src.core.claim_ladder import OUTSIDE_THE_LADDER
from src.core.errors import InvalidParameterError
from src.core.translation import AssociationFigures


METRIC = SignatureMetric(AttributeWeights(
    geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))

FACTORS = {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0}

WINDOW = TransitionWindow(1.0, 3.0, "frames")
LAGS = LagFamily((WINDOW,))

#: A precursor planted plainly: A every ten frames, B two frames behind each one, and three
#: further B occurrences in a stretch A never enters so the consequent is not merely A relabelled.
PLANTED = ([("A", float(time)) for time in range(0, 100, 10)]
           + [("B", float(time + 2)) for time in range(0, 100, 10)]
           + [("B", 120.0), ("B", 125.0), ("B", 130.0)])

#: The same lift, arranged so that the choice of null decides the answer: B falls in four dense
#: blocks and A is one clump of four sitting just before the first of them.
BLOCKED = ([("B", float(time))
            for start in (20, 60, 100, 140) for time in range(start, start + 5)]
           + [("A", float(time)) for time in (17, 18, 19, 20)])


def _signature(factor, identity, time, *, units="frames", cardinality=3):
    return ConstellationSignature(
        key=(float(identity), 0, 1, 2), time=float(time), cardinality=cardinality,
        mode="scale_specific", order=(0, 1, 2),
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


def _series(plan, *, frames=150, cadence=1.0, **overrides):
    grid = ObservationGrid(frames=tuple(float(index) for index in range(frames)),
                           time_units="frames", cadence=cadence)
    return events_from_catalogue(_catalogue(plan, **overrides), grid=grid)


def _ids(series, plan):
    mapping = {}
    for letter, time in plan:
        if letter in mapping:
            continue
        events = series.at(time)
        if len(events) == 1:
            mapping[letter] = events[0].pattern_id
    return mapping


@pytest.fixture(scope="module")
def planted():
    series = _series(PLANTED)
    return series, _ids(series, PLANTED)


@pytest.fixture(scope="module")
def planted_report(planted):
    series, _ = planted
    return precursor_report(series, lags=LAGS, n_surrogates=99, seed=20260905)


@pytest.fixture(scope="module")
def blocked():
    series = _series(BLOCKED)
    identities = _ids(series, BLOCKED)
    return series, [(identities["A"], identities["B"])]


# ------------------------------------------------------------------------------ the base rate


def test_the_base_rate_is_a_window_probability_and_not_a_frame_one(planted):
    """A confidence over a three-frame window compared against a per-frame rate is a lift that
    grows with the width of the window and with nothing else."""
    series, identity = planted
    narrow, _positions, _hits = window_base_rate(
        series, identity["B"], TransitionWindow(1.0, 1.0, "frames"))
    wide, _positions, _hits = window_base_rate(
        series, identity["B"], TransitionWindow(1.0, 5.0, "frames"))
    assert wide > narrow, "a wider window can only contain the consequent more often"
    hand = sum(1 for frame in series.grid.frames
               if any(abs(float(frame) + 1.0 - event.time) < 1e-9
                      for event in series.for_pattern(identity["B"])))
    assert narrow == pytest.approx(hand / float(len(series.grid.frames) - 1))


def test_the_base_rate_does_not_depend_on_which_antecedent_is_tested(planted):
    """Two rules sharing a consequent must be divided by the same number, or the lifts they
    report are not comparable with each other."""
    series, identity = planted
    plan = PLANTED + [("C", float(time)) for time in (5.0, 45.0, 85.0)]
    with_c = _series(plan)
    letters = _ids(with_c, plan)
    report = precursor_report(with_c, lags=LAGS, n_surrogates=99, seed=11,
                              pairs=[(letters["A"], letters["B"]),
                                     (letters["C"], letters["B"])])
    first = report.rule(letters["A"], letters["B"], WINDOW)
    second = report.rule(letters["C"], letters["B"], WINDOW)
    assert first.base_rate == second.base_rate
    assert first.base_rate_positions == second.base_rate_positions


def test_a_position_whose_window_was_not_wholly_observed_is_not_a_base_rate_position(planted):
    """The base rate is estimated over positions that were actually searched, by the same rule
    that decides an antecedent's eligibility."""
    series, identity = planted
    _rate, positions, _hits = window_base_rate(series, identity["B"], WINDOW)
    assert positions == len(series.grid.frames) - 3, (
        "the last three frames cannot carry a wholly observed window three frames wide")


def test_the_base_rate_travels_with_its_own_sample_size(planted_report, planted):
    _series_, identity = planted
    rule = planted_report.rule(identity["A"], identity["B"], WINDOW)
    assert rule.base_rate_positions > 0 and rule.base_rate_hits > 0
    assert rule.base_rate == pytest.approx(
        rule.base_rate_hits / float(rule.base_rate_positions))


def test_a_consequent_never_seen_in_any_window_has_no_lift_rather_than_an_infinite_one():
    # The consequent sits on the first searched frame and nowhere else, so no window that was
    # wholly observed can contain it: every window opens strictly after its anchor.
    plan = [("B", 0.0), ("A", 10.0), ("A", 20.0), ("A", 30.0)]
    series = _series(plan, frames=40)
    identity = _ids(series, plan)
    report = precursor_report(series, lags=LagFamily((TransitionWindow(1.0, 2.0, "frames"),)),
                              n_surrogates=19, seed=3,
                              pairs=[(identity["A"], identity["B"])])
    rule = report.rules[0]
    assert rule.base_rate == 0.0
    assert rule.base_rate_hits == 0 and rule.base_rate_positions > 0
    assert rule.status == RULE_BASE_RATE_ZERO
    assert rule.lift is None and rule.figures() is None
    assert rule.p_value is None, (
        "there is nothing for a null to be compared against when the consequent was never "
        "available to be found")


# ------------------------------------------------------------------------ the lift, and its CI


def test_lift_is_confidence_over_the_base_rate_and_nothing_else(planted_report):
    for rule in planted_report:
        if rule.lift is None:
            continue
        assert rule.lift == pytest.approx(rule.confidence / rule.base_rate)


def test_the_interval_brackets_the_estimate_and_narrows_as_the_denominator_grows(planted):
    series, identity = planted
    report = precursor_report(series, lags=LAGS, n_surrogates=99, seed=5)
    rule = report.rule(identity["A"], identity["B"], WINDOW)
    low, high = rule.lift_interval
    assert low <= rule.lift <= high + 1e-12
    thin = _series(PLANTED[:2] + PLANTED[10:12], frames=150)
    letters = _ids(thin, PLANTED[:2] + PLANTED[10:12])
    sparse = precursor_report(thin, lags=LAGS, n_surrogates=99, seed=5,
                              pairs=[(letters["A"], letters["B"])]).rules[0]
    assert (high - low) < (sparse.lift_interval[1] - sparse.lift_interval[0]), (
        "ten anchors must give a tighter interval than two")


def test_the_interval_is_the_score_interval_and_not_the_normal_approximation(planted):
    """A Wald interval is symmetric about the estimate and collapses to zero width where every
    trial succeeded, which is exactly where a precursor rule lands. The Wilson bounds are the
    two proportions the observation sits `z` standard errors away from, and that defining
    property is what is asserted rather than a number copied out of the implementation."""
    from statistics import NormalDist
    series, identity = planted
    rule = precursor_report(series, lags=LAGS, n_surrogates=99, seed=5,
                            pairs=[(identity["A"], identity["B"])]).rules[0]
    low, high = rule.lift_interval
    rate = rule.base_rate
    estimate = rule.confidence
    trials = rule.eligible_antecedents
    z = NormalDist().inv_cdf(0.975)
    for bound in (low * rate, high * rate):
        if bound in (0.0, 1.0):
            continue
        error = (bound * (1.0 - bound) / trials) ** 0.5
        assert abs(estimate - bound) == pytest.approx(z * error, rel=1e-9)
    assert low * rate < estimate, (
        "every trial succeeded, and an interval that gives that no width is claiming the "
        "record settled the question exactly")


def test_the_interval_declares_its_method_and_counts_the_trials_that_were_not_independent():
    """The Wilson interval assumes independent trials. Overlapping windows are not independent,
    and the number of them is measured rather than merely disclaimed."""
    series = _series(BLOCKED)
    identity = _ids(series, BLOCKED)
    report = precursor_report(series, lags=LAGS, n_surrogates=99, seed=7,
                              pairs=[(identity["A"], identity["B"])])
    rule = report.rules[0]
    assert rule.overlapping_eligible_windows == 3, (
        "four anchors one frame apart under a three-frame window overlap three times")
    spaced = [0.0, 2.0, 4.0, 20.0]
    assert _overlapping_windows(spaced, WINDOW) == 2, (
        "anchors exactly the window's width apart share its boundary instant, and a trial "
        "that shares even one instant with its neighbour is not a fresh one")
    assert _overlapping_windows([0.0, 3.0, 20.0], WINDOW) == 0
    assert report.describe()["rules"][0]["lift_interval_method"] == INTERVAL_METHOD
    assert "not independent" in report.describe()["rules"][0]["lift_interval_assumption"]


def test_a_rule_with_six_figures_builds_the_programmes_own_r9_carrier(planted_report, planted):
    _series_, identity = planted
    figures = planted_report.rule(identity["A"], identity["B"], WINDOW).figures()
    assert isinstance(figures, AssociationFigures)
    assert figures.lift == pytest.approx(figures.confidence / figures.base_rate)


def test_a_rule_missing_one_of_the_six_refuses_to_produce_the_other_five(planted_report,
                                                                        planted):
    _series_, identity = planted
    reverse = planted_report.rule(identity["B"], identity["A"], WINDOW)
    assert reverse.support == 0 and reverse.confidence == 0.0
    assert reverse.figures() is None, (
        "a confidence of zero has no lift interval R9 will accept, and five figures are not a "
        "weaker finding than six")


# ------------------------------------------------------------------------------------ the null


def test_the_planted_precursor_clears_the_null_and_its_reverse_does_not(planted_report,
                                                                       planted):
    _series_, identity = planted
    forward = planted_report.rule(identity["A"], identity["B"], WINDOW)
    reverse = planted_report.rule(identity["B"], identity["A"], WINDOW)
    assert forward.status == RULE_PRECURSOR
    assert forward.confidence == 1.0 and forward.lift > 3.0
    assert forward.q_value <= planted_report.alpha
    assert reverse.status == RULE_NOT_DISTINGUISHED
    assert reverse.p_value == 1.0


def test_the_choice_of_null_decides_the_finding_on_one_unchanged_record(blocked):
    """The demonstration this module exists to make. Identical data, identical lift; the
    anti-conservative null calls it a precursor and the null that preserves the antecedent's own
    bursting does not."""
    series, pair = blocked
    circular = precursor_report(series, lags=LAGS, null=NULL_CIRCULAR_SHIFT,
                                n_surrogates=99, seed=7, pairs=pair).rules[0]
    uniform = precursor_report(series, lags=LAGS, null=NULL_UNIFORM_RELOCATION,
                               n_surrogates=99, seed=7, pairs=pair).rules[0]
    assert circular.confidence == uniform.confidence == 1.0
    assert circular.lift == pytest.approx(uniform.lift)
    assert uniform.status == RULE_PRECURSOR
    assert circular.status == RULE_NOT_DISTINGUISHED
    assert circular.p_value > uniform.p_value * 5, (
        "scattering a clumped antecedent makes the null far easier to beat, and the record "
        "supplies no evidence for the difference")


def test_the_shift_null_preserves_the_antecedents_count_and_every_gap_but_one():
    times = [17.0, 18.0, 19.0, 20.0]
    rotated = _rotate(times, 40.0, 0.0, 150.0)
    assert len(rotated) == len(times)
    assert [b - a for a, b in zip(rotated, rotated[1:])] == [1.0, 1.0, 1.0]
    wrapped = _rotate(times, 132.0, 0.0, 150.0)
    assert len(wrapped) == len(times)
    assert wrapped == [0.0, 1.0, 2.0, 149.0], (
        "the one gap the wrap point falls in is the one gap this null does not preserve")


def test_a_shift_smaller_than_the_widest_lag_is_never_drawn():
    """A rotation of one frame against a three-frame window is not a null draw, it is the
    observation slightly blurred."""
    grid = ObservationGrid(frames=tuple(float(index) for index in range(150)),
                           time_units="frames", cadence=1.0)
    floor = _shift_floor(LagFamily((WINDOW, TransitionWindow(1.0, 5.0, "frames"))), grid)
    assert floor == 6.0, "one cadence step beyond the widest declared lag"
    shifts = _available_shifts(grid, floor)
    assert min(shifts) == 6.0
    assert max(shifts) == 144.0, "a rotation near the full extent is a small one the other way"
    assert all(shift >= floor and 150.0 - shift >= floor for shift in shifts)


def test_an_ensemble_larger_than_the_records_distinct_rotations_is_refused(blocked):
    series, pair = blocked
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=LAGS, n_surrogates=400, seed=1, pairs=pair)
    assert "distinct rotations" in str(caught.value)
    assert "price one null draw as several" in str(caught.value)


def test_the_receipt_says_which_null_was_drawn_and_what_it_leaves_alone(planted_report):
    published = planted_report.describe()["null"]
    assert published["method"] == NULL_CIRCULAR_SHIFT
    assert published["preserves"] == list(NULL_PRESERVES[NULL_CIRCULAR_SHIFT])
    assert published["destroys"] == list(NULL_DESTROYS[NULL_CIRCULAR_SHIFT])
    assert published["caution"] is None
    assert published["seed"] == 20260905 and published["n_surrogates"] == 99


def test_the_easy_null_carries_the_warning_that_it_is_easy(blocked):
    series, pair = blocked
    report = precursor_report(series, lags=LAGS, null=NULL_UNIFORM_RELOCATION,
                              n_surrogates=99, seed=7, pairs=pair)
    caution = report.describe()["null"]["caution"]
    assert caution is not None and "anti-conservative" in caution.lower()
    assert "should not be used to support a claim" in caution
    assert NULL_CAUTION[NULL_CIRCULAR_SHIFT] is None


def test_the_same_seed_redraws_the_same_ensemble(blocked):
    series, pair = blocked
    first = precursor_report(series, lags=LAGS, n_surrogates=99, seed=99, pairs=pair).rules[0]
    again = precursor_report(series, lags=LAGS, n_surrogates=99, seed=99, pairs=pair).rules[0]
    assert first.p_value == again.p_value
    assert first.null_mean_confidence == again.null_mean_confidence


# ------------------------------------------------------------- the lag family, and the choice


def test_a_lag_family_must_be_declared_as_one(planted):
    series, _identity = planted
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=(WINDOW,), n_surrogates=99, seed=1)
    assert "size the correction never sees" in str(caught.value)


def test_an_empty_lag_family_is_refused():
    with pytest.raises(InvalidParameterError) as caught:
        LagFamily(())
    assert "it is not a test" in str(caught.value)


def test_a_window_declared_twice_is_refused():
    with pytest.raises(InvalidParameterError) as caught:
        LagFamily((WINDOW, TransitionWindow(1.0, 3.0, "frames")))
    assert "pays for a test nobody ran" in str(caught.value)


def test_two_units_in_one_family_are_refused():
    with pytest.raises(InvalidParameterError) as caught:
        LagFamily((WINDOW, TransitionWindow(1.0, 3.0, "hours")))
    assert "a conversion nobody performed" in str(caught.value)


def test_a_family_in_a_unit_the_grid_does_not_use_is_refused(planted):
    series, _identity = planted
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=LagFamily((TransitionWindow(1.0, 3.0, "hours"),)),
                         n_surrogates=99, seed=1)
    assert "refused rather than converted" in str(caught.value)


def test_overlapping_lag_windows_are_reported_rather_than_refused():
    family = LagFamily((TransitionWindow(1.0, 4.0, "frames"),
                        TransitionWindow(3.0, 6.0, "frames")))
    assert family.overlaps is True
    assert family.describe()["windows_overlap"] is True
    assert LagFamily((TransitionWindow(1.0, 2.0, "frames"),
                      TransitionWindow(3.0, 4.0, "frames"))).overlaps is False


def test_the_chosen_lag_is_tested_against_the_null_of_the_maximum_over_the_family(blocked):
    """A lag chosen by the data is a search. Testing the winner against a single-lag null would
    price a search of several lags as one test."""
    series, pair = blocked
    family = LagFamily((TransitionWindow(1.0, 3.0, "frames"),
                        TransitionWindow(4.0, 6.0, "frames")))
    report = precursor_report(series, lags=family, n_surrogates=99, seed=7, pairs=pair)
    chosen = report.selected_rule(*pair[0])
    best = max((rule for rule in report if rule.confidence is not None),
               key=lambda rule: rule.confidence)
    assert chosen.window == best.window and chosen.confidence == best.confidence
    assert chosen.p_value >= best.p_value, (
        "the maximum of two lags cannot be rarer under the null than the same value at one "
        "lag chosen in advance")
    assert chosen.n_lags_searched == 2
    assert "distribution of the maximum" in chosen.describe()["selection_note"]


def test_one_draw_is_shared_across_the_lag_family_so_the_maximum_has_a_null(blocked):
    """The null of the maximum is the maximum of one shifted record, not the maximum of several
    independently shifted ones. The observable consequence is that the ensemble a lag is tested
    against cannot depend on where that lag sits in the declared family."""
    series, pair = blocked
    narrow = TransitionWindow(1.0, 3.0, "frames")
    wide = TransitionWindow(4.0, 6.0, "frames")
    first = precursor_report(series, lags=LagFamily((narrow, wide)), n_surrogates=99, seed=7,
                             pairs=pair)
    second = precursor_report(series, lags=LagFamily((wide, narrow)), n_surrogates=99, seed=7,
                              pairs=pair)
    assert first.rule(pair[0][0], pair[0][1], narrow).null_confidences ==         second.rule(pair[0][0], pair[0][1], narrow).null_confidences, (
            "a fresh draw per lag would give this window a different ensemble depending on "
            "whether it was declared first or second")
    chosen = first.selected_rule(*pair[0])
    per_lag = [first.rule(pair[0][0], pair[0][1], window).null_confidences
               for window in (narrow, wide)]
    assert chosen.null_maximum == tuple(max(values) for values in zip(*per_lag)), (
        "the null of the maximum is the elementwise maximum of the shared draws")


def test_the_ensemble_a_p_value_came_from_is_published_and_not_only_summarised(planted_report,
                                                                              planted):
    _series_, identity = planted
    rule = planted_report.rule(identity["A"], identity["B"], WINDOW)
    assert len(rule.null_confidences) == 99
    assert all(0.0 <= value <= 1.0 for value in rule.null_confidences)
    exceeding = sum(1 for value in rule.null_confidences if value >= rule.confidence)
    assert rule.n_surrogates_at_least_as_extreme == exceeding
    assert rule.p_value == pytest.approx((1.0 + exceeding) / 100.0)
    quantiles = rule.null_quantiles()
    assert quantiles["minimum"] <= quantiles["median"] <= quantiles["maximum"]
    assert rule.null_mean_confidence == pytest.approx(
        sum(rule.null_confidences) / len(rule.null_confidences))
    assert rule.describe()["null_quantiles"] == quantiles


def test_each_family_is_corrected_against_its_own_size(blocked):
    series, pair = blocked
    family = LagFamily((TransitionWindow(1.0, 3.0, "frames"),
                        TransitionWindow(4.0, 6.0, "frames")))
    report = precursor_report(series, lags=family, n_surrogates=99, seed=7, pairs=pair)
    assert report.per_lag_correction["n_tests"] == 2, "one pair at two lags"
    assert report.selected_lag_correction["n_tests"] == 1, "one pair, one selected lag"
    assert report.per_lag_correction["method"] == report.correction


# --------------------------------------------------------------- multiplicity, and its ceiling


def test_a_design_that_could_never_reject_is_refused_before_anything_is_counted(planted):
    """An under-powered study reports an absence indistinguishable from a real one. The refusal
    is a fact about the declared design, so no absence is ever produced to be misread."""
    series, _identity = planted
    plan = PLANTED + [("C", 5.0), ("C", 45.0), ("C", 85.0)]
    wider = _series(plan)
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(wider, lags=LAGS, n_surrogates=19, seed=1)
    message = str(caught.value)
    assert "cannot produce a significant result" in message
    assert "not a result" in message


def test_the_family_size_counts_every_declared_test_and_not_the_reported_ones(planted_report):
    assert planted_report.per_lag_correction["n_tests"] == len(planted_report.rules)
    assert planted_report.power["n_tests"] == len(planted_report.rules)
    assert planted_report.counts_performed == len(planted_report.rules) * (1 + 99)


def test_the_family_is_sized_by_what_was_declared_and_not_by_what_returned_a_p_value():
    """Correcting only the tests that produced a number is correcting a family the study did
    not run: the members that returned nothing were still performed."""
    plan = PLANTED + [("C", 148.0), ("C", 149.0)]
    series = _series(plan)
    letters = _ids(series, plan)
    report = precursor_report(series, lags=LAGS, n_surrogates=99, seed=13,
                              pairs=[(letters["A"], letters["B"]),
                                     (letters["C"], letters["B"])])
    silent = report.rule(letters["C"], letters["B"], WINDOW)
    assert silent.status == RULE_NO_ELIGIBLE_ANTECEDENT and silent.p_value is None
    assert report.per_lag_correction["n_tests"] == 2
    assert report.per_lag_correction["n_p_values"] == 1
    assert any("absent tests" in warning
               for warning in report.per_lag_correction.get("warnings", [])), (
        "a family wider than the p-values in it is a fact the correction must declare")
    tested = report.rule(letters["A"], letters["B"], WINDOW)
    assert tested.q_value > tested.p_value, (
        "two declared tests cannot cost the same as one")


def test_a_rule_is_called_a_precursor_on_its_corrected_p_value_and_not_its_raw_one():
    """The correction exists to be able to withhold a rejection the raw p-value would have
    granted. Both members of this family clear alpha before correction and neither clears it
    after, which is the only arrangement that tells the two readings apart."""
    plan = ([("B", float(time)) for start in (20, 60, 100) for time in range(start, start + 5)]
            + [("A", float(time)) for time in (17, 18, 19, 20)])
    series = _series(plan, frames=280)
    identity = _ids(series, plan)
    family = LagFamily((TransitionWindow(1.0, 3.0, "frames"),
                        TransitionWindow(4.0, 6.0, "frames")))
    report = precursor_report(series, lags=family, n_surrogates=99, seed=7,
                              pairs=[(identity["A"], identity["B"])])
    for rule in report:
        assert rule.p_value <= report.alpha < rule.q_value, (
            "the fixture must keep the case that separates a raw reading from a corrected one")
        assert rule.status == RULE_NOT_DISTINGUISHED and not rule.rejected
        assert rule.rejected == (rule.q_value <= report.alpha)


def test_the_corrected_lift_is_against_the_null_and_the_plain_lift_is_against_the_base_rate(
        planted_report, planted):
    _series_, identity = planted
    rule = planted_report.rule(identity["A"], identity["B"], WINDOW)
    assert rule.surrogate_corrected_lift == pytest.approx(
        rule.confidence / rule.null_mean_confidence)
    assert rule.null_mean_confidence != rule.base_rate
    assert rule.surrogate_corrected_lift != rule.lift, (
        "two figures R9 requires separately must not be one figure printed twice")


def test_the_scattering_null_moves_every_occurrence_and_keeps_all_of_them():
    """A relocation drawn with replacement would quietly hand the surrogate fewer occurrences
    than the record has, and a null with less to find is a null that is easier to beat."""
    import numpy as np

    from src.analysis_engine.spectral_precursors import _relocate
    # Eight drawn from eight: with replacement a repeat is all but certain (a permutation is
    # one outcome in 2,097,152), so this is the shape that tells the two draws apart.
    frames = tuple(float(index) for index in range(8))
    for seed in range(6):
        drawn = _relocate(8, frames, np.random.default_rng(seed))
        assert len(drawn) == 8 and len(set(drawn)) == 8
        assert set(drawn) == set(frames)


def test_correction_can_only_move_a_p_value_upwards(planted_report):
    for rule in planted_report:
        if rule.p_value is None:
            continue
        assert rule.q_value >= rule.p_value - 1e-12


def test_every_member_of_the_family_is_reported_including_the_ones_that_found_nothing(
        planted_report, planted):
    _series_, identity = planted
    tested = {(rule.antecedent, rule.consequent) for rule in planted_report}
    assert tested == {(identity["A"], identity["B"]), (identity["B"], identity["A"])}
    assert any(rule.status == RULE_NOT_DISTINGUISHED for rule in planted_report)
    assert "no selection precedes this correction" in \
        planted_report.per_lag_correction["note"]


# -------------------------------------------------------------------------------- the refusals


def test_a_grid_without_a_cadence_is_refused_with_the_three_things_it_costs():
    grid = ObservationGrid(frames=tuple(float(index) for index in range(150)),
                           time_units="frames", cadence=None)
    series = events_from_catalogue(_catalogue(PLANTED), grid=grid)
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=LAGS, n_surrogates=99, seed=1)
    message = str(caught.value)
    assert "eligibility is undecidable" in message
    assert "no universe of positions" in message or "no lattice" in message
    assert "T4F.2 can still count" in message


def test_a_pattern_preceding_itself_is_sent_to_the_question_built_for_it(planted):
    series, identity = planted
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=LAGS, n_surrogates=99, seed=1,
                         pairs=[(identity["A"], identity["A"])])
    assert "recurrence_report" in str(caught.value)


@pytest.mark.parametrize("field,value,expected", [
    ("null", "shuffle_everything", "one of"),
    ("correction", "eyeball", "one of"),
    ("alpha", 1.5, "in (0, 1)"),
    ("interval_confidence", 0.0, "in (0, 1)"),
    ("n_surrogates", 0, "positive ensemble size"),
    ("seed", "20260905", "nobody can redraw"),
])
def test_every_argument_that_changes_what_a_p_value_means_is_checked(planted, field, value,
                                                                    expected):
    series, identity = planted
    arguments = dict(lags=LAGS, n_surrogates=99, seed=1,
                     pairs=[(identity["A"], identity["B"])])
    arguments[field] = value
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, **arguments)
    assert expected in str(caught.value)


def test_a_sweep_that_does_not_fit_its_budget_is_refused_whole_before_it_starts(planted):
    series, identity = planted
    with pytest.raises(MiningBudgetExceededError) as caught:
        precursor_report(series, lags=LAGS, n_surrogates=99, seed=1,
                         pairs=[(identity["A"], identity["B"])],
                         budget=MiningBudget(max_candidates=10, max_seconds=60.0))
    assert caught.value.context["partial_result"] is False
    assert caught.value.context["observed"] == 100


def test_a_pair_naming_a_pattern_the_series_does_not_contain_is_refused(planted):
    series, identity = planted
    with pytest.raises(InvalidParameterError) as caught:
        precursor_report(series, lags=LAGS, n_surrogates=99, seed=1,
                         pairs=[(identity["A"], 9999)])
    assert "a pattern this series contains" in str(caught.value)


def test_an_antecedent_whose_every_window_was_censored_has_no_denominator():
    plan = [("A", 148.0), ("A", 149.0), ("B", 10.0), ("B", 20.0), ("B", 30.0)]
    series = _series(plan, frames=150)
    identity = _ids(series, plan)
    report = precursor_report(series, lags=LAGS, n_surrogates=19, seed=1,
                              pairs=[(identity["A"], identity["B"])])
    rule = report.rules[0]
    assert rule.eligible_antecedents == 0
    assert rule.ineligible_truncated == 2
    assert rule.status == RULE_NO_ELIGIBLE_ANTECEDENT
    assert rule.confidence is None and rule.p_value is None and rule.q_value is None


# ------------------------------------------------------------------------- what is not claimed


def test_no_status_this_module_can_emit_names_a_thing_outside_the_ladder():
    statuses = (RULE_PRECURSOR, RULE_NOT_DISTINGUISHED, RULE_NO_ELIGIBLE_ANTECEDENT,
                RULE_BASE_RATE_ZERO)
    words = {word for status in statuses for word in status.lower().split("_")}
    assert words.isdisjoint(set(OUTSIDE_THE_LADDER))


def test_the_boundary_names_every_reading_a_rejected_rule_does_not_license(planted_report):
    boundary = planted_report.describe()["claim_boundary"]
    assert boundary == PRECURSOR_CLAIM_BOUNDARY
    for refused in ("cause", "driver", "mechanism", "trigger", "forecast", "intervention"):
        assert refused in boundary, "%s is refused by name, not by omission" % refused
    assert "still a correlation" in boundary
    assert "a statement about this record and not about the world" in boundary


def test_the_receipt_is_schema_stamped_and_carries_what_a_reader_would_have_to_ask_for(
        planted_report):
    published = planted_report.describe()
    assert published["schema"] == PRECURSOR_SCHEMA
    assert published["test_unit"].startswith("one ordered")
    for key in ("lags", "null", "alpha", "correction", "power", "per_lag_correction",
                "selected_lag_correction", "counts_performed", "claim_boundary"):
        assert key in published
    rule = published["rules"][0]
    for figure in ("support", "confidence", "base_rate", "lift", "lift_interval",
                   "surrogate_corrected_lift"):
        assert figure in rule, "%s is one of R9's six and must travel with the others" % figure


def test_the_counting_is_t4f2s_and_not_a_second_implementation_of_it(planted, planted_report):
    """Two denominators would agree on the planted case and diverge at the record's edge, which
    is the one case the denominator was written for."""
    series, identity = planted
    counted = count_sequence((identity["A"], identity["B"]), series,
                             times_by_pattern(series), WINDOW, 1)
    rule = planted_report.rule(identity["A"], identity["B"], WINDOW)
    assert (rule.support, rule.eligible_antecedents, rule.antecedent_occurrences) == (
        counted.support, counted.eligible_antecedents, counted.antecedent_occurrences)
    assert rule.confidence == counted.confidence
    assert rule.ineligible_truncated == counted.ineligible_truncated
