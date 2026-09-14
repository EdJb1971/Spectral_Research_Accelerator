"""Can the change about to be made move the answer at all?

T4E.27 predicted improvement from a bar change that could not produce it. These tests pin the
arithmetic that would have said so before the run, and -- more importantly -- pin the module's
own limits, because a check that implied it validated a design would sell the same false comfort
it exists to remove.
"""
from __future__ import annotations

import json

import pytest

from src.analysis_engine.prediction_sensitivity import (
    check_prediction,
    verdict_travel,
)
from src.core.errors import InvalidParameterError


# ---------------- the case that happened


def test_the_t4e27_bar_change_could_not_move_a_single_verdict():
    """The swing set is empty on the real inputs, so 'improve' was impossible, not unlucky."""
    record = json.loads(open("measurements/t4e27_restated_acceptance.json",
                             encoding="utf-8").read())
    storms = record["per_storm"]
    travel = verdict_travel(
        [s["nearest_km"] for s in storms],
        [s["catalogue_radius_km"] for s in storms],
        [s["tolerance_km"] for s in storms],
        [s["storm"] for s in storms])

    assert travel.inert is True
    assert travel.swing == ()
    assert travel.admitted_before == travel.admitted_after == 2
    # The refused tolerance is out of the judged population, not counted as immovable.
    assert travel.not_judged == ("LINDA",)
    assert travel.judged == 17


def test_the_prediction_t4e27_declared_is_adjudicated_as_impossible():
    record = json.loads(open("measurements/t4e27_restated_acceptance.json",
                             encoding="utf-8").read())
    storms = record["per_storm"]
    travel = verdict_travel(
        [s["nearest_km"] for s in storms],
        [s["catalogue_radius_km"] for s in storms],
        [s["tolerance_km"] for s in storms],
        [s["storm"] for s in storms])

    result = check_prediction(travel, "improve")
    assert result["verdict"] == "IMPOSSIBLE"
    assert "settled by arithmetic before any data is read" in result["reason"]


# ---------------- the arithmetic


def test_a_value_between_the_two_bars_is_the_only_thing_that_can_move():
    travel = verdict_travel([5.0, 15.0, 25.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0],
                            ["below", "between", "above"])

    assert travel.gained == ("between",)
    assert travel.lost == ()
    assert dict(travel.outcomes)["below"] == "admitted_either_way"
    assert dict(travel.outcomes)["above"] == "rejected_either_way"


def test_a_tightening_bar_loses_rather_than_gains():
    travel = verdict_travel([15.0], [20.0], [10.0], ["x"])
    assert travel.lost == ("x",) and travel.gained == ()
    assert check_prediction(travel, "worsen")["verdict"] == "POSSIBLE"
    assert check_prediction(travel, "improve")["verdict"] == "IMPOSSIBLE"


def test_the_bar_is_inclusive_matching_how_a_tolerance_admits():
    """`PositionTolerance.admits` uses `<=`; a different convention here would disagree with
    the thing it is checking."""
    assert verdict_travel([10.0], [10.0], [10.0], ["x"]).admitted_before == 1
    assert verdict_travel([10.0], [9.99], [10.0], ["x"]).gained == ("x",)


def test_a_refused_bar_keeps_an_observation_out_of_the_swing_set():
    """A missing input is not movement, and counting it as immovable would be equally wrong."""
    travel = verdict_travel([50.0, 5.0], [None, 1.0], [None, 10.0], ["refused", "real"])

    assert travel.not_judged == ("refused",)
    assert travel.judged == 1
    assert travel.gained == ("real",)


def test_a_non_finite_value_is_not_judged_rather_than_compared():
    travel = verdict_travel([float("nan")], [1.0], [10.0], ["x"])
    assert travel.not_judged == ("x",)


# ---------------- predictions


def test_unchanged_on_an_inert_change_is_reported_as_trivially_true():
    """Predicting no change where nothing can change is not a risk that was taken."""
    travel = verdict_travel([100.0], [1.0], [2.0], ["x"])
    result = check_prediction(travel, "unchanged")

    assert result["verdict"] == "TRIVIALLY_TRUE"
    assert "carries no evidential weight" in result["reason"]


def test_unchanged_where_movement_is_possible_is_a_real_prediction():
    travel = verdict_travel([5.0], [1.0], [10.0], ["x"])
    assert check_prediction(travel, "unchanged")["verdict"] == "POSSIBLE"


def test_an_unknown_direction_is_refused_with_the_known_ones():
    travel = verdict_travel([5.0], [1.0], [10.0], ["x"])
    with pytest.raises(InvalidParameterError) as caught:
        check_prediction(travel, "sideways")
    assert "improve" in str(caught.value)


# ---------------- refusing to be misused


def test_mismatched_lengths_are_refused_rather_than_zipped_short():
    with pytest.raises(InvalidParameterError):
        verdict_travel([1.0, 2.0], [1.0], [2.0])


def test_mismatched_labels_are_refused():
    with pytest.raises(InvalidParameterError):
        verdict_travel([1.0, 2.0], [1.0, 1.0], [2.0, 2.0], ["only_one"])


def test_the_report_states_what_it_does_not_check():
    """A clean report here is not a sound design, and the module has to say so itself."""
    described = verdict_travel([5.0], [1.0], [10.0], ["x"]).describe()
    text = described["what_this_does_not_check"]

    assert "right quantity" in text
    assert "population" in text
    assert "not a sound design" in text


def test_an_inert_report_explains_why_that_disarms_a_prediction():
    described = verdict_travel([100.0], [1.0], [2.0], ["x"]).describe()
    assert "arithmetically impossible" in described["what_inert_means"]


def test_an_empty_population_is_inert_and_judged_nothing():
    travel = verdict_travel([], [], [])
    assert travel.inert is True and travel.judged == 0
