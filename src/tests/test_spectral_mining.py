"""T4E.4: minimum support, early pruning, and refusing resource budgets."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.analysis_engine.spectral_clustering import (
    AttributeWeights, SignatureMetric, calibrate_signature_tolerance, cluster_signatures,
)
from src.analysis_engine.spectral_invariance import AxisAdmission, ConstellationSignature
from src.analysis_engine.spectral_mining import (
    MINING_SCHEMA, SUPPORT_UNIT, MiningBudget, MiningBudgetExceededError,
    mine_supported_patterns,
)
from src.core.errors import InvalidParameterError


METRIC = SignatureMetric(AttributeWeights(
    geometry=1.0, bearings=0.0, strengths=0.0, scales=0.0))


def _signature(factor, identity):
    return ConstellationSignature(
        key=(float(identity), 0, 1, 2), time=float(identity), cardinality=3,
        mode="scale_specific", order=(0, 1, 2),
        geometry=tuple(factor * value for value in (1.0, 1.3, 1.6)),
        geometry_relation="distance", bearings=(10.0, 35.0, 70.0),
        strengths=(0.75, 1.0, 4.0 / 3.0), scales=(8.0, 8.0, 16.0),
        scale_units="cells", axis=AxisAdmission(True, 3.0, 1.0421, 25.0),
        track_ids=(0, 1, 2), bands=("L1", "L1", "L2"))


def _catalogue(order=None):
    calibration = [_signature(1.0, 100), _signature(1.05, 101)]
    tolerance = calibrate_signature_tolerance(calibration, metric=METRIC)
    signatures = [
        _signature(1.00, 1), _signature(1.01, 2), _signature(1.02, 3),
        _signature(1.03, 4), _signature(1.04, 5),
        _signature(2.00, 6), _signature(2.02, 7),
    ]
    if order is not None:
        signatures = [signatures[index] for index in order]
    return cluster_signatures(signatures, metric=METRIC, tolerance=tolerance)


class _Clock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def test_both_budget_limits_are_positive_finite_and_candidate_count_is_integral():
    with pytest.raises(InvalidParameterError, match="positive integer"):
        MiningBudget(max_candidates=0, max_seconds=1.0)
    with pytest.raises(InvalidParameterError, match="positive integer"):
        MiningBudget(max_candidates=1.5, max_seconds=1.0)
    with pytest.raises(InvalidParameterError, match="positive finite"):
        MiningBudget(max_candidates=3, max_seconds=float("inf"))


def test_minimum_support_is_a_positive_integer_not_a_boolean_or_fraction():
    for invalid in (0, True, 2.5):
        with pytest.raises(InvalidParameterError, match="positive integer"):
            mine_supported_patterns(
                _catalogue(), minimum_support=invalid,
                budget=MiningBudget(10, 1.0))


def test_the_planted_five_and_two_occurrence_patterns_meet_only_the_declared_minimum():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0))
    assert len(result) == 1
    assert result.supported[0].support == 5
    assert sorted(decision.support for decision in result.decisions) == [2, 5]


def test_support_is_inclusive_at_the_declared_boundary():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=5, budget=MiningBudget(10, 1.0))
    assert [pattern.support for pattern in result.supported] == [5]


def test_every_candidate_reports_support_even_when_it_is_pruned():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0))
    report = result.describe()
    assert report["candidate_count"] == 2
    assert {row["support"] for row in report["patterns"]} == {2, 5}
    assert {row["support_unit"] for row in report["patterns"]} == {SUPPORT_UNIT}
    assert {row["status"] for row in report["patterns"]} == {
        "SUPPORTED", "PRUNED_BELOW_MINIMUM"}


def test_decreasing_support_order_prunes_the_entire_remaining_tail_early():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0))
    assert result.candidates_examined == 1
    assert result.candidates_pruned_early == 1
    none = mine_supported_patterns(
        _catalogue(), minimum_support=6, budget=MiningBudget(10, 1.0))
    assert none.candidates_examined == 0
    assert none.candidates_pruned_early == 2


def test_candidate_budget_refuses_before_a_partial_sweep_can_start():
    with pytest.raises(MiningBudgetExceededError, match="candidate-count") as error:
        mine_supported_patterns(
            _catalogue(), minimum_support=3, budget=MiningBudget(1, 1.0))
    assert error.value.context["partial_result"] is False
    assert error.value.context["observed"] == 2


def test_wall_clock_budget_refuses_the_whole_sweep_mid_scan():
    clock = _Clock([0.0, 0.0, 2.0])
    with pytest.raises(MiningBudgetExceededError, match="wall-clock") as error:
        mine_supported_patterns(
            _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0), clock=clock)
    assert error.value.context["partial_result"] is False


def test_wall_clock_budget_includes_identity_validation_and_support_preflight():
    clock = _Clock([0.0, 1.1])
    with pytest.raises(MiningBudgetExceededError, match="wall-clock"):
        mine_supported_patterns(
            _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0), clock=clock)


def test_duplicate_constellation_identity_is_refused_not_counted_twice():
    one = _signature(1.0, 1)
    tolerance = calibrate_signature_tolerance(
        [_signature(1.0, 100), _signature(1.05, 101)], metric=METRIC)
    duplicated = cluster_signatures([one, replace(one)], metric=METRIC, tolerance=tolerance)
    with pytest.raises(InvalidParameterError, match="manufacture support"):
        mine_supported_patterns(
            duplicated, minimum_support=2, budget=MiningBudget(10, 1.0))


def test_a_minimum_above_every_pattern_returns_a_complete_empty_result():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=8, budget=MiningBudget(10, 1.0))
    assert len(result) == 0
    assert len(result.decisions) == 2
    assert all(decision.status == "PRUNED_BELOW_MINIMUM"
               for decision in result.decisions)


def test_input_permutation_cannot_change_support_or_pattern_decisions():
    forward = mine_supported_patterns(
        _catalogue(), minimum_support=3, budget=MiningBudget(10, 1.0))
    reverse = mine_supported_patterns(
        _catalogue(order=list(reversed(range(7)))), minimum_support=3,
        budget=MiningBudget(10, 1.0))
    left = [(item.pattern.pattern_id, item.support, item.status)
            for item in forward.decisions]
    right = [(item.pattern.pattern_id, item.support, item.status)
             for item in reverse.decisions]
    assert left == right


def test_receipt_names_the_algorithm_budgets_and_non_significance_boundary():
    result = mine_supported_patterns(
        _catalogue(), minimum_support=3, budget=MiningBudget(10, 2.0),
        clock=_Clock([0.0, 0.1, 0.2, 0.3, 0.4]))
    report = result.describe()
    assert report["schema"] == MINING_SCHEMA
    assert report["algorithm"] == "decreasing-support scan with early tail pruning"
    assert report["budget"] == {"max_candidates": 10, "max_seconds": 2.0}
    assert report["elapsed_seconds_unasserted"] == pytest.approx(0.4)
    assert "not recurrence significance" in report["claim_boundary"]


def test_wrong_catalogue_or_missing_explicit_budget_is_refused():
    with pytest.raises(InvalidParameterError, match="PatternCatalogue"):
        mine_supported_patterns(
            object(), minimum_support=3, budget=MiningBudget(10, 1.0))
    with pytest.raises(InvalidParameterError, match="explicit MiningBudget"):
        mine_supported_patterns(
            _catalogue(), minimum_support=3, budget=None)
