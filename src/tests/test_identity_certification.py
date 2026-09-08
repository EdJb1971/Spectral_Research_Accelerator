"""T4E.9: the T4E identity path measured against an answer known by construction.

These tests pin the certification's own machinery -- the construction labels, the partition
discipline, and the refusals that stop a mislabelled scene being scored as an error rate --
rather than re-asserting the measured figures, which live in `VERIFICATION.md`.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.analysis_engine.spectral_clustering import SignatureMetric
from src.benchmarks import fields as F
from src.benchmarks.core import Outcome, get_benchmark
from src.benchmarks.identity_certification import (
    CALIBRATION_SEEDS, CLAIM_BOUNDARY, EVALUATION_SEEDS, MAX_FALSE_ADMISSION, MAX_FALSE_SPLIT,
    NULL_SEEDS, WEIGHTS, UnlabelledScene, _cross_scene_distances, _motif_track_ids, _scene,
    _signed_scene, certify,
)


def test_the_benchmark_is_registered_and_names_the_path_it_certifies():
    bench = get_benchmark("t4e_identity_certified")
    truth = bench.known_answer()
    assert "T4E identity path" in truth["path_under_test"]
    assert truth["mode"] == "spatial_geometry"
    # Six features choose three.
    assert truth["configurations_per_scene"] == 20
    assert truth["motif_configurations_per_planted_scene"] == 1
    assert truth["expected_motif_configurations_per_null_scene"] == 0


def test_the_three_partitions_are_disjoint():
    """A radius calibrated on scenes it is later evaluated on measures nothing."""
    assert not set(CALIBRATION_SEEDS) & set(EVALUATION_SEEDS)
    assert not set(CALIBRATION_SEEDS) & set(NULL_SEEDS)
    assert not set(EVALUATION_SEEDS) & set(NULL_SEEDS)


def test_a_planted_scene_yields_exactly_one_motif_configuration():
    points, is_motif = _signed_scene(CALIBRATION_SEEDS[0], plant=True)
    assert len(points) == 20
    assert sum(is_motif) == 1


def test_a_null_scene_yields_no_motif_configuration():
    """The null keeps the feature count and the family, and drops only the repetition."""
    points, is_motif = _signed_scene(NULL_SEEDS[0], plant=False)
    assert len(points) == 20
    assert sum(is_motif) == 0


def test_construction_labels_come_from_the_generator_not_from_the_matcher():
    features, planted = _scene(CALIBRATION_SEEDS[0], plant=True)
    assert len(planted) == 3
    ids = _motif_track_ids(features, planted)
    assert len(ids) == 3
    # Each planted position is claimed by a feature that really is close to it.
    for position in planted:
        best = min(math.hypot(float(f.location.coords["row"]) - position[0],
                              float(f.location.coords["col"]) - position[1])
                   for f in features)
        assert best <= 15.0


def test_a_planted_position_with_no_feature_near_it_is_refused_by_name():
    features, _ = _scene(CALIBRATION_SEEDS[0], plant=True)
    with pytest.raises(UnlabelledScene) as raised:
        _motif_track_ids(features, [(2.0, 2.0)])
    assert "no extracted feature within" in str(raised.value)


def test_two_planted_positions_claiming_one_feature_are_refused_as_ambiguous():
    features, planted = _scene(CALIBRATION_SEEDS[0], plant=True)
    duplicated = [planted[0], (planted[0][0] + 0.5, planted[0][1] + 0.5)]
    with pytest.raises(UnlabelledScene) as raised:
        _motif_track_ids(features, duplicated)
    assert "ambiguous" in str(raised.value)


def test_only_cross_scene_pairs_are_formed():
    """Two configurations inside one scene share features and observe nothing twice."""
    partition = [_signed_scene(seed, plant=True) for seed in CALIBRATION_SEEDS[:3]]
    positive, negative = _cross_scene_distances(partition, SignatureMetric(WEIGHTS))
    # Three scenes: three unordered scene pairs, one motif pair each.
    assert len(positive) == 3
    assert len(negative) == 3 * (20 * 20 - 1)


def test_the_identity_definition_separates_the_motif_from_everything_else():
    """The question D101 asked. Separation is the definition's job; the radius is not.

    Asserted as a floor rather than as the measured value, so an improvement does not fail
    the test and a regression does.
    """
    result = certify()
    assert result["planted_evaluation"]["auc"] >= 0.99
    assert result["planted_evaluation"]["false_admission_rate"] <= MAX_FALSE_ADMISSION


def test_nothing_is_admitted_where_nothing_recurs():
    """The null partition's correct answer is that there is nothing to find."""
    result = certify()
    null = result["null_evaluation"]
    assert null["unrelated_pairs"] > 0
    assert null["false_admission_rate"] <= MAX_FALSE_ADMISSION
    # No positive population exists here, so its rate must be unmeasured, never zero.
    assert null["repeat_pairs"] == 0
    assert null["false_split_rate"] is None


def test_a_frozen_radius_that_fails_while_a_feasible_one_exists_is_reported_as_both():
    """The finding T4E.9 produced: the definition separates, the calibration does not transfer.

    Pinned as a relationship rather than as two numbers -- if the frozen radius ever meets its
    bounds the outcome becomes PASS and this test says so instead.
    """
    result = certify()
    feasibility = result["empirical_radius_feasibility"]
    if result["outcome"] == "PASS":
        assert result["planted_evaluation"]["false_split_rate"] <= MAX_FALSE_SPLIT
        return
    assert result["outcome"] == "FAIL"
    assert result["planted_evaluation"]["false_split_rate"] > MAX_FALSE_SPLIT
    assert feasibility["any_radius_meets_both_empirical_bounds"] is True
    assert feasibility["smallest_split_feasible_radius"] > result["radius"]


def test_every_result_states_what_it_does_not_license():
    result = certify()
    assert result["claim_boundary"] == CLAIM_BOUNDARY
    assert "not the atmosphere" in CLAIM_BOUNDARY
    assert "no mining radius is approved" in CLAIM_BOUNDARY


def test_an_empty_calibration_population_is_invalid_rather_than_permissive():
    """No radius from no pairs. An unmeasured population must not become a wide radius."""
    result = certify(calibration_seeds=[], evaluation_seeds=EVALUATION_SEEDS[:2],
                     null_seeds=NULL_SEEDS[:2])
    assert result["outcome"] == "INVALID"
    assert result["radius"] is None
    assert "unmeasured population is not a permissive one" in result["reason"]


def test_the_check_reports_the_measurement_rather_than_swallowing_it():
    bench = get_benchmark("t4e_identity_certified")
    field = bench.make()
    result = bench.checks[0](field, bench.known_answer())
    assert result.stage == "4E.identity_certified"
    assert result.outcome in (Outcome.PASS, Outcome.FAIL, Outcome.NOT_YET_RUNNABLE)
    assert result.measured is not None
    assert result.measured["claim_boundary"] == CLAIM_BOUNDARY
    if result.outcome is Outcome.FAIL:
        assert "declared bounds" in result.detail


# ------------------------------------------------------ T4E.10: does any radius transfer at all?

def test_blocks_from_one_generator_are_not_exchangeable():
    """The finding that redirects D97: the premise of a frozen radius fails before the estimator.

    Four blocks of six scenes, identical generator and identical parameters. A tolerance bound
    is distribution-free but not assumption-free -- it covers the population its sample came
    from. If blocks differ, calibration and evaluation are not one population and no bound
    calibrated on the first says anything about the second, at any support.
    """
    from src.benchmarks.identity_certification import measure_block_exchangeability

    report = measure_block_exchangeability()
    assert len(report["blocks"]) == 4
    for block in report["blocks"]:
        assert block["pairs"] == 15
    # Asserted as a floor, so a narrowing of the spread fails this test and says so.
    assert report["block_mean_ratio"] > 1.5, (
        "blocks were expected to differ; if they no longer do, the T4E.10 finding has moved")
    assert "same generator" in report["boundary"]


def test_the_tolerance_bound_refuses_the_support_the_certification_actually_has():
    """Six scenes give 15 cross-scene motif pairs, and 90/90 needs 22."""
    from src.benchmarks.identity_certification import certify_estimators

    result = certify_estimators()
    thin = result["supports"]["below_requirement"]
    assert thin["calibration_pairs"] == 15
    assert thin["required_support"] == 22
    assert thin["estimators"]["nonparametric_tolerance_bound"]["outcome"] == "REFUSED"


def test_no_estimator_transfers_even_where_the_support_is_sufficient():
    """Recorded as a relationship: if one ever holds, this test says so rather than passing on."""
    from src.benchmarks.identity_certification import certify_estimators

    result = certify_estimators()
    thick = result["supports"]["above_requirement"]
    assert thick["calibration_pairs"] >= thick["required_support"]
    outcomes = {name: item["outcome"] for name, item in thick["estimators"].items()}
    holding = [name for name, outcome in outcomes.items() if outcome == "HOLDS"]
    if holding:
        for name in holding:
            errors = thick["estimators"][name]["planted_evaluation"]
            assert errors["false_split_rate"] <= 0.10
        return
    assert set(outcomes.values()) == {"DOES_NOT_HOLD"}
    # The tolerance bound is still the best of them: a guarantee it cannot honour across
    # non-exchangeable blocks still beats a quantile that claims none.
    split = {name: thick["estimators"][name]["planted_evaluation"]["false_split_rate"]
             for name in outcomes}
    assert split["nonparametric_tolerance_bound"] < split["empirical_quantile"]


def test_a_refusal_is_recorded_as_correct_behaviour_not_as_a_failure():
    from src.benchmarks.identity_certification import certify_estimators

    result = certify_estimators()
    assert "correct behaviour, not a failure" in result["refusal_boundary"]
    assert result["claim_boundary"] == CLAIM_BOUNDARY
