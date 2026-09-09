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


# ----------------------------------------- T4E.11: the confirmatory blocks, reserved in code

def test_the_reserved_confirmatory_blocks_are_refused_by_development_code():
    """Intention is not a reservation. Looking once is the whole cost, so the refusal is
    raised before the scene is built rather than before a result is reported."""
    from src.benchmarks.identity_certification import ReservedSceneOpened, _partition

    with pytest.raises(ReservedSceneOpened) as raised:
        _partition([700, 701], plant=True)
    assert "reserved for T4E.11's confirmatory evaluation" in str(raised.value)
    assert "never been generated" in str(raised.value)


def test_the_reservation_covers_every_declared_confirmatory_block():
    from src.benchmarks.identity_certification import RESERVED_CONFIRMATORY_SEEDS

    declared = {seed for start in (700, 710, 720, 730) for seed in range(start, start + 6)}
    assert RESERVED_CONFIRMATORY_SEEDS == declared
    assert len(RESERVED_CONFIRMATORY_SEEDS) == 24


def test_no_development_partition_overlaps_a_reserved_block():
    from src.benchmarks.identity_certification import (
        ESTIMATOR_SUPPORTS, EXCHANGEABILITY_BLOCKS, RESERVED_CONFIRMATORY_SEEDS,
    )

    used = set(CALIBRATION_SEEDS) | set(EVALUATION_SEEDS) | set(NULL_SEEDS)
    for seeds in ESTIMATOR_SUPPORTS.values():
        used |= set(seeds)
    for seeds in EXCHANGEABILITY_BLOCKS:
        used |= set(seeds)
    assert not used & RESERVED_CONFIRMATORY_SEEDS


def test_a_confirmatory_run_may_reach_them_only_by_saying_so():
    """The escape exists and is explicit; a silent one would make the guard decorative."""
    from src.benchmarks.identity_certification import _refuse_reserved

    _refuse_reserved([700, 701], confirmatory=True)
    with pytest.raises(Exception):
        _refuse_reserved([700], confirmatory=False)


def test_the_declaration_exists_and_is_not_yet_adopted():
    """Code does not sign a scientific declaration for a person."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e11-normalised-distance-declaration.json"
    ).read_text(encoding="utf-8"))
    assert body["status"] == "declared_before_measurement"
    assert "NOT VALID until the maintainer has reviewed" in body["declared_by"]
    assert body["confirmatory"]["status"].startswith("RESERVED AND NOT YET GENERATED")
    assert body["development"]["status"].startswith("ALREADY INSPECTED")
    # The falsifying outcome is named before anything is measured.
    assert "motivate candidate C" in body["what_would_falsify_this_candidate"]


# ------------------------------------ T4E.11 candidate B: adopted, measured, and falsified

def test_the_normaliser_is_label_free_by_construction():
    """It may not consult which configurations are the motif; that is the whole point of it."""
    import inspect
    from src.benchmarks.identity_certification import partition_normaliser

    source = inspect.getsource(partition_normaliser)
    assert "is_motif" not in source and "motif" not in source.split('"""')[2]


def test_candidate_B_does_not_collapse_the_spread_it_was_declared_to_collapse():
    """The declared falsification, measured. Recorded as a relationship, not as two numbers.

    Candidate B was adopted on the reasoning that block distributions shift in scale while
    holding their shape, so dividing by each partition's own close-pair scale should make one
    radius mean the same thing everywhere. It does not.
    """
    from src.benchmarks.identity_certification import measure_normalised_blocks

    report = measure_normalised_blocks()
    assert report["evidence_class"] == "development"
    assert "not confirmation" in report["boundary"]
    # If normalising ever does collapse the spread, this test says so rather than passing on.
    if report["normalised_mean_ratio"] < 1.2:
        return
    assert report["normalised_mean_ratio"] >= report["raw_mean_ratio"] * 0.9, (
        "candidate B was expected to leave the spread substantially uncorrected")


def test_the_normaliser_does_not_track_the_quantity_it_was_meant_to_track():
    """Why B failed, which is narrower than the declaration anticipated and is recorded as such.

    The declaration reasoned that failure would show the *shape* was moving rather than the
    scale. That is not what happened. The same-configuration scale still moves 1.88x; this
    label-free normaliser simply does not see it, because a median over every configuration's
    nearest cross-scene neighbour is dominated by the unrelated majority.
    """
    from src.benchmarks.identity_certification import measure_normalised_blocks

    report = measure_normalised_blocks()
    normalisers = [block["normaliser"] for block in report["blocks"]]
    raw_means = [block["raw_mean"] for block in report["blocks"]]
    normaliser_spread = max(normalisers) / min(normalisers)
    assert normaliser_spread < 1.2, "the normaliser barely varies between blocks"
    assert report["raw_mean_ratio"] > 1.5, "while the quantity it should track varies a lot"
    # It is measuring a different regime: nearest-unrelated, not same-configuration.
    for normaliser, raw in zip(normalisers, raw_means):
        assert normaliser / raw > 5.0


# --------------------------------- T4E.11 candidate C: adopted, measured, falsified precisely

def test_candidate_C_has_no_threshold_to_transfer():
    """The property it was adopted for: no radius, no normaliser, nothing to carry."""
    import inspect
    from src.benchmarks.identity_certification import mutual_nearest_matches

    source = inspect.getsource(mutual_nearest_matches)
    body = source.split('"""')[2]
    for magnitude in ("radius", "threshold", "normalis", "<=", ">="):
        assert magnitude not in body, (
            "candidate C must compare orderings, not magnitudes: found %r" % magnitude)


def test_candidate_C_recovers_every_motif_pair_in_every_block():
    """The half that works, and the property both a frozen and a normalised radius lacked.

    Asserted as a bound rather than as the measured zero, so an improvement cannot fail it.
    """
    from src.benchmarks.identity_certification import measure_criterion_c

    report = measure_criterion_c()
    assert report["evidence_class"] == "development"
    for block in report["planted_blocks"]:
        assert block["false_split_rate"] <= MAX_FALSE_SPLIT
    low, high = report["false_split_range"]
    assert high - low <= 0.05, "the recall half must be stable across blocks, not merely good"


def test_candidate_C_cannot_decline_to_match_and_that_is_what_falsifies_it():
    """The declared falsification, measured: a rule that always answers has no rejection.

    The null block keeps the feature count and the family and drops only the repetition, so
    every match returned there is a false positive by construction.
    """
    from src.benchmarks.identity_certification import measure_criterion_c

    report = measure_criterion_c()
    null = report["null_blocks"][0]
    assert null["motif_pairs"] == 0
    if null["matches_returned"] == 0:
        return  # a later criterion may decline; this test would then say so.
    assert null["false_admission_rate"] == 1.0, (
        "with nothing recurring, every match is false by construction")
    for block in report["planted_blocks"]:
        assert block["false_admission_rate"] > MAX_FALSE_ADMISSION


def test_the_failure_is_precision_and_not_the_shape_of_the_criterion():
    """Why the declaration's own escalation does not follow from this result.

    The declaration said a failure would mean the question is unanswerable by a criterion of
    this shape and the next move is the signature. The measurement says something narrower: the
    ordering transfers perfectly and it is the absence of a rejection rule that fails.
    """
    from src.benchmarks.identity_certification import measure_criterion_c

    report = measure_criterion_c()
    assert max(report["false_split_range"]) == 0.0
    assert min(report["false_admission_range"]) > 0.5


# ------------------------------------ T4E.11 candidate D: adopted, measured, and falsified
#
# The declaration was written, committed and hashed before the development blocks were evaluated,
# and adopted separately in `t4e11-ratio-margin-adoption.json`. The toy-scene tests below check
# the criterion's mechanics; the block tests after them record what the measurement said.


class _AbsoluteMetric:
    """A one-dimensional stand-in, so the mechanics are tested without the signature path.

    `scale` exists to test the property candidate D was declared for: the criterion is a ratio
    of two distances from the same scene pair, so multiplying every distance by a constant must
    leave its output identical.
    """

    def __init__(self, scale=1.0):
        self.scale = scale

    def distance(self, a, b):
        return abs(a - b) * self.scale


def _toy(points):
    return (list(points), [False] * len(points))


def test_tau_is_pinned_at_the_value_its_declaration_names():
    """It is 0.8 because Lowe (2004) says so, not because these blocks were consulted."""
    import json
    from pathlib import Path
    from src.benchmarks.identity_certification import MARGIN_TAU

    body = json.loads(Path(
        "data/identity_calibration/t4e11-ratio-margin-declaration.json"
    ).read_text(encoding="utf-8"))
    assert MARGIN_TAU == body["criterion"]["tau"] == 0.8
    assert "Lowe (2004)" in body["criterion"]["where_tau_comes_from_and_why_it_is_not_fitted"]


def test_the_declaration_for_candidate_D_is_not_yet_adopted():
    """Code does not sign a scientific declaration for a person."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e11-ratio-margin-declaration.json"
    ).read_text(encoding="utf-8"))
    assert body["status"] == "declared_before_measurement"
    assert "NOT VALID until the maintainer has reviewed" in body["declared_by"]
    assert body["confirmatory"]["status"].startswith("STILL RESERVED AND NOT YET GENERATED")
    assert body["development"]["status"].startswith("ALREADY INSPECTED")
    # The failure it would license is narrowed on purpose, after two declarations over-reached.
    assert "would NOT license" in body["what_a_failure_would_and_would_not_license"]


def test_the_margin_is_a_ratio_and_survives_a_change_of_scale():
    """The declared property: rescale every distance and the criterion does not notice.

    This is what an absolute radius cannot do (T4E.10) and what candidate B's estimated
    normaliser failed to achieve. Here it holds by construction rather than by estimate.
    """
    from src.benchmarks.identity_certification import ratio_margin_matches

    left, right = _toy([0.0, 10.0, 25.0]), _toy([0.2, 10.4, 25.9])
    unscaled = ratio_margin_matches(left, right, _AbsoluteMetric(1.0))
    for scale in (0.001, 3.7, 1000.0):
        assert ratio_margin_matches(left, right, _AbsoluteMetric(scale)) == unscaled
    assert unscaled, "the toy scenes should match at all, or the test proves nothing"


def test_candidate_D_can_only_remove_what_candidate_C_returned():
    """Stated in the declaration before measurement, so neither outcome can surprise afterwards.

    D is C plus a filter, so its false split can only rise from C's 0.0000 and its match count
    can only fall. The whole question the declaration asks is whether that trade is close to
    free -- which these toy scenes do not answer and are not meant to.
    """
    from src.benchmarks.identity_certification import (
        mutual_nearest_matches, ratio_margin_matches)

    metric = _AbsoluteMetric()
    for left_points, right_points in (
            ([0.0, 10.0], [0.1, 10.5]),
            ([0.0, 1.0], [0.5, 0.6]),
            ([0.0, 1.0, 2.0, 3.0], [0.05, 1.9, 2.95, 7.0])):
        left, right = _toy(left_points), _toy(right_points)
        loose = set(mutual_nearest_matches(left, right, metric))
        tight = set(ratio_margin_matches(left, right, metric))
        assert tight <= loose


def test_the_margin_declines_where_candidate_C_could_not():
    """The deficiency candidate D was declared to address, on a case built to show it.

    Both pairs are mutual nearest neighbours, so candidate C returns both. One of them is barely
    nearer than its runner-up, and the margin drops it.
    """
    from src.benchmarks.identity_certification import (
        mutual_nearest_matches, ratio_margin_matches)

    metric = _AbsoluteMetric()
    left, right = _toy([0.0, 1.0]), _toy([0.5, 0.6])
    assert len(mutual_nearest_matches(left, right, metric)) == 2
    assert ratio_margin_matches(left, right, metric) == [(1, 1)]


def test_an_undefined_margin_is_refused_by_name_rather_than_decided():
    """A scene of one configuration has no second nearest neighbour, so the test cannot be run.

    Admitting the pair would make the rule strongest where it has the least evidence; counting
    it as a non-match would charge a false split to a test that was never applied. The
    declaration requires a named refusal, and this is it.
    """
    import pytest
    from src.benchmarks.identity_certification import UndefinedMargin, ratio_margin_matches

    with pytest.raises(UndefinedMargin):
        ratio_margin_matches(_toy([0.0]), _toy([0.1, 5.0]), _AbsoluteMetric())
    with pytest.raises(UndefinedMargin):
        ratio_margin_matches(_toy([0.1, 5.0]), _toy([0.0]), _AbsoluteMetric())
    # An empty scene is a different case and keeps candidate C's behaviour.
    assert ratio_margin_matches(_toy([]), _toy([0.1, 5.0]), _AbsoluteMetric()) == []


def test_the_ratio_diagnostic_is_labelled_as_one_wherever_it_is_reported():
    """It may inform a candidate E; it may not be read as an operating point for this one."""
    import inspect
    from src.benchmarks.identity_certification import margin_ratios, measure_criterion_d

    assert "DIAGNOSTIC and not a criterion" in inspect.getdoc(margin_ratios)
    source = inspect.getsource(measure_criterion_d)
    assert "is_a_diagnostic_not_a_criterion" in source


def _candidate_d_report():
    """One measurement, reused. Each call rebuilds five partitions through the full T4E path."""
    import functools
    from src.benchmarks.identity_certification import measure_criterion_d

    if not hasattr(_candidate_d_report, "_cached"):
        _candidate_d_report._cached = measure_criterion_d()
    return _candidate_d_report._cached


def test_the_declaration_for_candidate_D_was_adopted_before_it_was_measured():
    """The adoption record names the hash the declaration had when it was adopted."""
    import hashlib
    import json
    from pathlib import Path

    declaration = Path("data/identity_calibration/t4e11-ratio-margin-declaration.json")
    adoption = json.loads(Path(
        "data/identity_calibration/t4e11-ratio-margin-adoption.json"
    ).read_text(encoding="utf-8"))
    assert adoption["adopts_sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()
    assert "not a prediction that D will work" in adoption["what_adoption_does_not_mean"]
    assert "stays at 0.8" in adoption["tau_is_not_adjustable_by_this_adoption"]


def test_the_margin_costs_nothing_in_recall_which_was_not_guaranteed():
    """The trade the declaration asked about, in the direction that went well.

    D can only remove pairs C returned, so the false split could only rise from C's 0.0000. It
    did not rise at all: every construction-labelled motif pair survives the margin in every
    block. Asserted as a bound, so an improvement cannot fail it.
    """
    report = _candidate_d_report()
    assert report["evidence_class"] == "development"
    for block in report["planted_blocks"]:
        assert block["false_split_rate"] <= MAX_FALSE_SPLIT
    low, high = report["false_split_range"]
    assert high - low <= 0.05, "recall must be stable across blocks, not merely good on average"


def test_the_structural_prediction_made_before_measurement_holds():
    """Declared in advance so that neither outcome could be presented afterwards as a surprise.

    D is C plus a filter, so on every block it must return no more matches than C did.
    """
    report = _candidate_d_report()
    for block in report["planted_blocks"] + report["null_blocks"]:
        assert block["matches_returned"] <= block["matches_returned_by_candidate_C"]
        assert block["scene_pairs_refused_for_undefined_margin"] == 0


def test_candidate_D_fails_the_rejection_condition_it_was_declared_against():
    """The declared falsification, measured. Recorded, not tuned away.

    Acceptance required the margin to discard at least ninety per cent of the matches candidate
    C returned where nothing recurs. It discards roughly half. The null partition keeps the
    feature count and the family and drops only the repetition, so every match it returns there
    is false by construction.
    """
    report = _candidate_d_report()
    null = report["null_blocks"][0]
    assert null["motif_pairs"] == 0
    assert null["false_admission_rate"] == 1.0
    assert null["retention_of_candidate_C"] > MAX_FALSE_ADMISSION, (
        "the declaration accepted at most 0.10 retention; this is the number that falsifies it")


def test_the_diagnostic_says_the_distributions_overlap_rather_than_that_tau_is_misplaced():
    """Why no operating point may be read off this, however tempting the ranges look.

    The declaration anticipated exactly this question -- whether a failure means the motif and
    non-motif ratio distributions overlap, or merely sit either side of a badly placed tau. They
    overlap: the largest motif ratio exceeds the smallest ratio the null block produces. The
    overlap is narrow, which is the tempting part, and a tau chosen to exploit it would be
    fitted to blocks that have already been inspected four times over. That is a candidate E,
    needing its own declaration and an evaluation on data these blocks did not select.
    """
    report = _candidate_d_report()
    motif_high = max(block["margin_ratio_diagnostic"]["motif_ratio_range"][1]
                     for block in report["planted_blocks"])
    null_low = report["null_blocks"][0]["margin_ratio_diagnostic"]["other_ratio_range"][0]
    assert motif_high > null_low, (
        "if this ever ceases to hold, the distributions have separated and the finding changes")
    # Every motif ratio sits far below the pinned tau, which is what makes 0.8 the wrong value
    # here -- but being the wrong value is not the same as there being a right one.
    assert motif_high < report["tau"]


# ------------------------------------ The diagnostic on the ground truth, after four failures
#
# Not a criterion and not preregistered: it adopts nothing, chooses no threshold and reports no
# operating point. It asks what the pairs called false admissions actually are, which no
# declaration had asked.


def _decomposition():
    """One run, reused. Each call rebuilds five partitions through the full T4E path."""
    from src.benchmarks.identity_certification import decompose_matched_pairs

    if not hasattr(_decomposition, "_cached"):
        _decomposition._cached = decompose_matched_pairs()
    return _decomposition._cached


def test_the_diagnostic_reports_no_operating_point_and_says_so():
    """Four criteria have been falsified; a diagnostic is not the place to smuggle in a fifth."""
    report = _decomposition()
    assert "Nothing is adopted" in report["is_a_diagnostic_not_a_criterion"]
    assert report["evidence_class"] == "development"


def test_the_labelling_is_not_what_the_false_admissions_are():
    """The hypothesis this diagnostic was built to test, and it does not survive.

    Nine of every twenty configurations in a scene share two motif features, which recur by
    construction, so the suspicion was that the ground truth was scoring real partial recurrence
    as error. It is not: most false admissions hold no motif vertex in common at all.
    """
    totals = _decomposition()["planted_totals"]["candidate_D"]
    non_motif = sum(count for key, count in totals.items() if key != "motif")
    same_vertices = totals.get("shared_2", 0) + totals.get("shared_1", 0)
    assert same_vertices < non_motif / 2, (
        "if this ever fails, the ground truth is scoring recurrence as error and the finding "
        "changes")


def test_the_signature_never_confuses_the_motif_with_a_partial_copy_of_itself():
    """The hardest discrimination in the scene, and the signature wins it outright.

    A `crossed_2` pair is the full motif against a configuration holding two of its three
    features -- one shared motif edge at fixed length and bearing, two members with identical
    strengths and scales, one member different. Not one such pair is matched by either
    criterion, out of more than a thousand available.
    """
    report = _decomposition()
    assert report["planted_totals"]["available"]["crossed_2"] > 1000
    for criterion in ("candidate_C", "candidate_D"):
        assert report["planted_totals"][criterion].get("crossed_2", 0) == 0


def test_every_motif_pair_is_recovered_in_the_decomposition_too():
    """The same 60 out of 60, arrived at by a second route, which is worth having."""
    report = _decomposition()
    assert report["planted_totals"]["available"]["motif"] == 60
    for criterion in ("candidate_C", "candidate_D"):
        assert report["planted_totals"][criterion]["motif"] == 60


def test_the_background_coincidence_rate_does_not_notice_whether_a_motif_is_present():
    """Why the null block's retention was never a null-specific artefact.

    Unrelated configurations -- no motif vertex in common -- match at essentially the same rate
    in the planted blocks as in the null. The same background operates everywhere, and it is
    that background, not the null's construction, which sets the false-admission floor.
    """
    report = _decomposition()
    rates = []
    for key in ("planted_totals", "null_totals"):
        totals = report[key]
        rates.append(totals["candidate_D"].get("unrelated", 0)
                     / totals["available"]["unrelated"])
    assert abs(rates[0] - rates[1]) < 0.005, (
        "planted %.5f against null %.5f" % tuple(rates))


def test_the_binding_constraint_is_specificity_and_the_shortfall_is_a_number():
    """What revisiting the signature has to achieve, stated as a factor rather than a wish.

    Each scene pair offers 400 candidate pairs and holds one true positive, so a per-pair false
    rate near one per cent yields four false admissions for every true one. Reaching the
    declared bound of 0.10 with 15 true positives per block requires the rate to fall by more
    than an order of magnitude. No threshold placed on this distance can supply that, which is
    why the next move is the signature and not a fifth criterion.
    """
    report = _decomposition()
    totals = report["planted_totals"]
    available = sum(count for key, count in totals["available"].items() if key != "motif")
    matched = sum(count for key, count in totals["candidate_D"].items() if key != "motif")
    rate = matched / available
    true_positives = totals["available"]["motif"]
    # False admissions permitted at the declared bound, pooled over the same blocks.
    permitted = MAX_FALSE_ADMISSION * true_positives / (1.0 - MAX_FALSE_ADMISSION)
    assert rate > 0.005, "the measured background rate, recorded so a change would be visible"
    assert matched / permitted > 10.0, (
        "the shortfall is more than an order of magnitude: %d false admissions against %.1f "
        "permitted" % (matched, permitted))


# ------------------------------------ T4E.12: what scene richness does to the identity problem
#
# Diagnostics, not criteria. Candidate D appears as a probe of the signature and is already
# falsified; nothing here adopts it or anything else, and no operating point is reported.


def _richness():
    """One sweep, reused. Each call rebuilds three partitions at three feature counts."""
    from src.benchmarks.identity_certification import measure_richness_scaling

    if not hasattr(_richness, "_cached"):
        _richness._cached = measure_richness_scaling()
    return _richness._cached


def test_the_required_pair_rate_is_arithmetic_and_approves_nothing():
    """It says what a bound demands, never what a signature delivers."""
    from src.benchmarks.identity_certification import required_pair_rate

    row = required_pair_rate(6)
    assert row["configurations_per_scene"] == 20
    assert row["candidate_pairs_per_scene_pair"] == 400
    assert "approves nothing" in row["boundary"]


def test_an_admission_bound_tightens_faster_than_the_scene_grows():
    """Why a bound met on one record need not hold on a richer one.

    Configurations grow as C(f,3) and candidate pairs as its square, while the number of true
    correspondences does not grow at all. So the per-pair rate a fixed admission bound permits
    collapses as scenes get richer -- roughly as the sixth power of the feature count -- and the
    admission FRACTION is therefore a property of the signature and the scene together, never of
    the signature alone. That is T4E.10's transfer problem in a new place.
    """
    from src.benchmarks.identity_certification import required_pair_rate

    rates = [required_pair_rate(f)["required_pair_rate"] for f in (6, 12, 20)]
    assert rates[0] > rates[1] > rates[2]
    assert rates[0] / rates[2] > 1000.0, (
        "six features to twenty tightens the requirement by more than three orders of magnitude")


def test_the_sweep_reports_no_operating_point_and_says_candidate_D_is_a_probe():
    """Four criteria are falsified; a diagnostic is not the place to revive one."""
    report = _richness()
    assert "already falsified" in report["is_a_diagnostic_not_a_criterion"]
    assert report["evidence_class"] == "development"


def test_recall_survives_richness_and_that_is_the_positive_finding():
    """The motif is still the mutual nearest neighbour among two hundred and twenty rivals.

    The information needed to identify the configuration is present in the signature at every
    richness measured. What is missing is the ability to decline everything else.
    """
    report = _richness()
    assert [row["features"] for row in report["rows"]] == [6, 9, 12]
    assert report["rows"][-1]["configurations_per_scene"] >= 220
    for row in report["rows"]:
        assert row["false_split_rate"] <= MAX_FALSE_SPLIT


def test_the_rule_matches_a_constant_fraction_of_configurations_however_rich_the_scene():
    """The measured law, and the reason the shortfall grows instead of closing.

    A mutual-nearest-neighbour matching returns at most one pair per configuration, so its
    output grows LINEARLY with the configuration count while the true correspondences stay at
    one per scene pair. Roughly a quarter of configurations are matched at every richness
    measured, so the admission fraction climbs towards one as scenes get richer.
    """
    report = _richness()
    fractions = []
    for row in report["rows"]:
        scene_pairs = row["candidate_pairs"] / row["configurations_per_scene"] ** 2
        fractions.append((row["matches_returned"] / scene_pairs)
                         / row["configurations_per_scene"])
    assert max(fractions) - min(fractions) < 0.10, (
        "the matched fraction is near-constant in richness: %s" % fractions)
    admissions = [row["false_admission_rate"] for row in report["rows"]]
    assert admissions == sorted(admissions), "the admission fraction must climb with richness"
    assert admissions[-1] > 0.95


def test_the_shortfall_grows_with_richness_rather_than_closing():
    """What T4E.12 has to fix, stated as a direction rather than a wish.

    The per-pair false rate does fall as scenes get richer -- a quarter of a growing population
    is a shrinking fraction of its square -- but it falls as 1/m where the bound demands 1/m^2.
    So the gap widens. No rule whose match count scales with the configuration count can close
    it, whatever threshold is placed on the distances.
    """
    report = _richness()
    rates = [row["false_pair_rate"] for row in report["rows"]]
    shortfalls = [row["shortfall_factor"] for row in report["rows"]]
    assert rates == sorted(rates, reverse=True), "the measured rate does fall with richness"
    assert shortfalls == sorted(shortfalls), "and the shortfall nonetheless widens"
    assert shortfalls[-1] > 10.0 * 1.0 and shortfalls[-1] > shortfalls[0] * 5.0


# ------------------------------------ T4E.12 candidate 1: adopted, measured, falsified
#
# Declared and committed at its hash before any development block was evaluated, adopted
# separately in `t4e12-multiplicity-adoption.json`, and falsified by the measurement that
# followed. The toy-scene tests below check mechanics; the block tests after them record what
# the measurement said, including that it did not test the signature.


def _spread_scene(count, *, seed, spacing=1.0):
    """A one-dimensional toy scene large enough for a tail fit to have support."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return (sorted(float(x) for x in rng.uniform(0.0, count * spacing, size=count)),
            [False] * count)


def test_the_declaration_for_T4E12_candidate_1_is_not_yet_adopted():
    """Code does not sign a scientific declaration for a person."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e12-multiplicity-declaration.json"
    ).read_text(encoding="utf-8"))
    assert body["status"] == "declared_before_measurement"
    assert "NOT VALID until the maintainer has reviewed" in body["declared_by"]
    assert body["confirmatory"]["status"].startswith("STILL RESERVED AND NOT YET GENERATED")
    assert body["development"]["status"].startswith("ALREADY INSPECTED")
    assert "would NOT license" in body["what_a_failure_would_and_would_not_license"]


def test_the_declaration_says_why_the_obvious_candidate_was_not_declared():
    """Reweighting cannot satisfy condition 2, and the reason is derivable, not measured."""
    import json
    from pathlib import Path
    from src.benchmarks.identity_certification import ADMISSION_ALPHA

    body = json.loads(Path(
        "data/identity_calibration/t4e12-multiplicity-declaration.json"
    ).read_text(encoding="utf-8"))
    assert ADMISSION_ALPHA == body["criterion"]["alpha"] == 0.05
    assert "CANNOT satisfy acceptance condition 2" in \
        body["why_this_candidate_and_why_not_the_obvious_one"]


def test_reweighting_could_not_have_satisfied_condition_2_and_here_is_why():
    """The derivation the declaration rests on, recorded as a test rather than as a claim.

    A mutual nearest-neighbour matching returns at most one pair per configuration, so its
    output is bounded by the smaller scene however the metric is weighted. The matched fraction
    is therefore pinned no matter what the weights do, which is exactly acceptance condition 2's
    demand and exactly what reweighting cannot move.
    """
    from src.benchmarks.identity_certification import mutual_nearest_matches

    for scale in (0.5, 1.0, 7.0):
        left, right = _spread_scene(30, seed=1), _spread_scene(24, seed=2)
        matches = mutual_nearest_matches(left, right, _AbsoluteMetric(scale))
        assert len(matches) <= min(len(left[0]), len(right[0]))
        assert len({i for i, _ in matches}) == len(matches), "one pair per configuration"


def test_the_admission_threshold_tightens_as_the_square_of_the_population():
    """The property the candidate was declared for, and the shape the acceptance demands.

    The bound is alpha / m^2, so it falls by four orders of magnitude between the sparsest and
    richest scenes measured -- by construction rather than by calibration.
    """
    from src.benchmarks.identity_certification import ADMISSION_ALPHA

    bounds = [ADMISSION_ALPHA / (m * m) for m in (20, 84, 220)]
    assert bounds[0] > bounds[1] > bounds[2]
    assert bounds[0] / bounds[2] > 100.0


def test_a_tail_that_cannot_be_fitted_is_refused_by_name():
    """Admitting on a failed fit would make the rule most permissive where its model is weakest."""
    import pytest
    from src.benchmarks.identity_certification import TailModelRefused, fit_lower_tail

    with pytest.raises(TailModelRefused):
        fit_lower_tail([0.1, 0.2, 0.3])
    with pytest.raises(TailModelRefused):
        fit_lower_tail([float("nan")] * 500)


def test_the_tail_probability_falls_as_the_distance_falls():
    """Monotone, and equal to the exceedance rate at the threshold itself."""
    import numpy as np
    from src.benchmarks.identity_certification import fit_lower_tail, tail_probability

    sample = np.random.default_rng(0).gamma(2.0, 0.1, size=40000)
    model = fit_lower_tail(sample)
    threshold = model["threshold"]
    probabilities = [tail_probability(threshold * f, model) for f in (1.0, 0.5, 0.1, 0.01)]
    assert probabilities == sorted(probabilities, reverse=True)
    assert abs(probabilities[0] - model["exceedance_rate"]) < 1e-9
    assert tail_probability(threshold * 2.0, model) == 1.0


def test_the_decision_does_not_notice_a_rescaling_of_the_metric():
    """Scale-free, as candidates C and D were: the null is the scene pair's own distances.

    T4E.10 ruled out an absolute radius and candidate B failed at an estimated normaliser. This
    criterion has neither -- rescale every distance and the admitted set is unchanged.
    """
    from src.benchmarks.identity_certification import multiplicity_aware_matches

    left, right = _spread_scene(90, seed=11), _spread_scene(90, seed=12)
    baseline, model = multiplicity_aware_matches(left, right, _AbsoluteMetric(1.0))
    assert model["candidate_pairs"] == 8100
    assert model["admission_bound"] == model["alpha"] / 8100
    for scale in (0.002, 55.0):
        rescaled, other = multiplicity_aware_matches(left, right, _AbsoluteMetric(scale))
        assert rescaled == baseline
        assert other["candidate_pairs"] == model["candidate_pairs"]


def test_the_criterion_admits_no_more_than_candidate_C_proposed():
    """It decides among candidate C's pairs; it cannot invent one."""
    from src.benchmarks.identity_certification import (
        multiplicity_aware_matches, mutual_nearest_matches)

    left, right = _spread_scene(90, seed=21), _spread_scene(90, seed=22)
    proposed = set(mutual_nearest_matches(left, right, _AbsoluteMetric()))
    admitted, _ = multiplicity_aware_matches(left, right, _AbsoluteMetric())
    assert set(admitted) <= proposed


def test_an_empty_scene_returns_no_model_rather_than_a_fitted_one():
    """A refusal is an outcome, and an absent scene is not a thin tail."""
    from src.benchmarks.identity_certification import multiplicity_aware_matches

    pairs, model = multiplicity_aware_matches(([], []), _spread_scene(90, seed=31),
                                              _AbsoluteMetric())
    assert pairs == [] and model is None


def _multiplicity_report():
    """One measurement at a single richness and block, reused. The full sweep is far larger."""
    from src.benchmarks.identity_certification import measure_multiplicity_criterion

    if not hasattr(_multiplicity_report, "_cached"):
        _multiplicity_report._cached = measure_multiplicity_criterion(
            richness_levels=(6, 9), blocks=(tuple(range(100, 104)),),
            null_blocks=(tuple(range(500, 504)),))
    return _multiplicity_report._cached


def test_the_declaration_was_adopted_before_it_was_measured():
    """The adoption record names the hash the declaration had when it was adopted."""
    import hashlib
    import json
    from pathlib import Path

    declaration = Path("data/identity_calibration/t4e12-multiplicity-declaration.json")
    adoption = json.loads(Path(
        "data/identity_calibration/t4e12-multiplicity-adoption.json"
    ).read_text(encoding="utf-8"))
    assert adoption["adopts_sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()
    assert "not a prediction" in adoption["what_adoption_does_not_mean"]
    assert "alpha stays at 0.05" in adoption["what_is_not_adjustable_by_this_adoption"]


def test_the_tail_model_has_no_support_at_six_features_and_says_so():
    """Derivable before adoption and not derived: 50 exceedances at the 1st percentile needs
    at least 5,000 distances, so at least 71 configurations, so at least nine features.

    Six features offer 400 distances and four exceedances. Every scene pair is refused by name,
    which is the declared behaviour -- but it means the candidate could never have been
    evaluated at the sparsest of its own three declared richness levels.
    """
    report = _multiplicity_report()
    sparse = [row for row in report["planted_blocks"] if row["richness"] == 6]
    assert sparse, "richness 6 must be measured, even though it can only refuse"
    for row in sparse:
        assert row["scene_pairs_refused"] == row["scene_pairs"]
        assert row["admitted"] == 0


def test_candidate_1_admits_nothing_which_is_what_falsifies_it():
    """The declared falsification, measured. Recorded, not tuned away.

    Acceptance condition 1 allows a false split of 0.10. Where the model could be fitted at
    all, nothing was admitted and the split is 1.0.
    """
    report = _multiplicity_report()
    fitted = [row for row in report["planted_blocks"]
              if row["richness"] == 9 and row["scene_pairs_refused"] == 0]
    assert fitted
    for row in fitted:
        assert row["pairs_proposed_by_candidate_C"] > 0, "candidate C proposed pairs to judge"
        assert row["admitted"] == 0
        assert row["false_split_rate"] > MAX_FALSE_SPLIT


def test_the_null_says_the_tail_model_is_conservative_so_the_signature_was_not_tested():
    """The most important line of the measurement, and it was declared in advance.

    If the tail model were correct, admissions per scene pair would equal alpha by construction
    whatever the signature is like. Nominal 0.05, measured 0.0000: the model under-admits
    relative to its own nominal rate, so the failure is not a verdict on the signature.
    """
    report = _multiplicity_report()
    nulls = [row for row in report["null_blocks"] if row["scene_pairs_refused"] == 0]
    assert nulls, "at least one null partition must have been evaluable"
    for row in nulls:
        assert row["admitted_per_scene_pair"] is not None
        assert row["admitted_per_scene_pair"] < report["alpha"] / 10.0, (
            "far below alpha, which the declaration's diagnostic did not name")
    assert "not a verdict on the signature" not in report["acceptance_boundary"]
    assert "tail model" in report["the_null_tests_the_tail_model_not_the_signature"]


def test_a_configuration_the_metric_cannot_compare_is_refused_rather_than_dropped_silently():
    """Found while measuring, named rather than worked around.

    At twelve features a few configurations are so nearly isotropic that their principal axis
    is refused, so they carry no bearing block and the metric will not compare them with
    anything that does. They are excluded and COUNTED, and a scene whose motif fell in the
    minority family would refuse outright rather than be measured around.
    """
    import inspect
    from src.benchmarks.identity_certification import comparable_subset

    report = _multiplicity_report()
    for row in report["planted_blocks"] + report["null_blocks"]:
        assert row["configurations_refused_as_incomparable"] == 0, (
            "none are expected at six or nine features; the count exists to be visible")
    assert "REFUSAL and not a rejection" in inspect.getdoc(comparable_subset)


# ------------------------------------ T4E.12 candidate 2: adopted, measured, and it passes
#
# The first candidate in this sequence to meet its acceptance conditions. Adopted for a
# DEVELOPMENT experiment only, with the reserved confirmatory blocks deliberately withheld
# because k was chosen after a feasibility probe on these same blocks. The tests below record
# what was measured AND which half of it is informative.


def _graph(*edges):
    """A partner map built from explicit mutual nearest-neighbour edges."""
    import collections

    partners = collections.defaultdict(dict)
    for left, right in edges:
        partners[left][right[0]] = right[1]
        partners[right][left[0]] = left[1]
    return partners


def test_the_declaration_for_candidate_2_is_not_yet_adopted():
    """Code does not sign a scientific declaration for a person."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e12-consistency-declaration.json"
    ).read_text(encoding="utf-8"))
    assert body["status"] == "declared_before_measurement"
    assert "NOT VALID until the maintainer has reviewed" in body["declared_by"]
    assert body["confirmatory"]["status"].startswith("STILL RESERVED AND NOT YET GENERATED")
    assert "would NOT license" in body["what_a_failure_would_and_would_not_license"]


def test_the_declaration_carries_the_feasibility_check_candidate_1_lacked():
    """The lesson from candidate 1, written into the next declaration rather than only recorded.

    It must show the criterion CAN admit something in principle, disclose that a probe was run
    before the declaration, and be honest that the probe informed the choice of k.
    """
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e12-consistency-declaration.json"
    ).read_text(encoding="utf-8"))
    feasibility = body["feasibility_checked_before_adoption"]
    assert "cannot be inert" in feasibility["derivable_without_the_probe"]
    assert "probe informed the choice of k" in feasibility["the_honest_disclosure_about_k"]
    assert body["limitations_declared_now_rather_than_discovered_later"]
    # Success is a live possibility here, so what a success would NOT license is declared too.
    assert "would NOT approve a mining radius" in \
        body["what_a_success_would_and_would_not_license"]


def test_a_fully_agreeing_set_is_found_whole():
    """Three configurations that all name each other are one correspondence, not three pairs."""
    from src.benchmarks.identity_certification import consistent_group

    partners = _graph(((0, 0), (1, 0)), ((0, 0), (2, 0)), ((1, 0), (2, 0)))
    assert consistent_group((0, 0), partners) == ((0, 0), (1, 0), (2, 0))


def test_a_set_that_does_not_close_is_cut_back_to_what_agrees():
    """A names B and C, but B and C do not name each other, so the three are not a group."""
    from src.benchmarks.identity_certification import consistent_group

    partners = _graph(((0, 0), (1, 0)), ((0, 0), (2, 0)))
    assert len(consistent_group((0, 0), partners)) == 2


def test_the_largest_agreeing_set_is_found_exactly_and_not_greedily():
    """Why the criterion enumerates rather than walks.

    Here node (0,0) has partners in scenes 1, 2 and 3, but only two of the three agree with
    each other. A greedy walk that accepted partners in the order it met them could return a
    pair; the exact rule returns the triple that closes.
    """
    from src.benchmarks.identity_certification import consistent_group

    partners = _graph(((0, 0), (1, 0)), ((0, 0), (2, 0)), ((0, 0), (3, 0)),
                      ((2, 0), (3, 0)))
    group = consistent_group((0, 0), partners)
    assert len(group) == 3
    assert {scene for scene, _ in group} == {0, 2, 3}


def test_one_configuration_per_scene_or_it_is_not_a_correspondence():
    """A group may not hold two configurations from the same scene, however well they agree."""
    from src.benchmarks.identity_certification import consistent_group

    partners = _graph(((0, 0), (1, 0)), ((0, 0), (1, 1)))
    group = consistent_group((0, 0), partners)
    assert len({scene for scene, _ in group}) == len(group)


def test_the_criterion_admits_only_pairs_inside_groups_that_span_enough_scenes():
    """The rule itself, on a partition where one group closes and another does not."""
    from src.benchmarks.identity_certification import consistency_admitted_pairs

    class _Cycle:
        """Distances that make configuration 0 of every scene mutually nearest throughout."""

        def distance(self, a, b):
            return 0.0 if (a == 0 and b == 0) else 1.0 + abs(a - b)

    partition = [([0, 1, 2], [False, False, False]) for _ in range(4)]
    admitted = consistency_admitted_pairs(partition, _Cycle(), k=4)
    assert admitted, "the agreeing groups span the whole partition and must be admitted"
    # Under this metric every configuration is its counterpart's mutual nearest neighbour, so
    # all three form partition-spanning groups. What the rule must never do is pair a
    # configuration with a different index, which would not be a consistent correspondence.
    assert all(i == j for _, i, _, j in admitted)
    assert len(admitted) == 18, "three groups across four scenes give six pairs each"
    # Demanding more scenes than the partition holds admits nothing rather than erroring.
    assert consistency_admitted_pairs(partition, _Cycle(), k=5) == []


def test_a_group_of_fewer_than_two_scenes_is_refused_as_meaningless():
    """A correspondence needs two scenes; asking for fewer is a parameter error, not a result."""
    import pytest
    from src.benchmarks.identity_certification import consistency_admitted_pairs

    with pytest.raises(ValueError):
        consistency_admitted_pairs([([0], [False])] * 3, _AbsoluteMetric(), k=1)


def _consistency_report():
    """One measurement at two richness levels, reused. The declared sweep is far larger."""
    from src.benchmarks.identity_certification import measure_consistency_criterion

    if not hasattr(_consistency_report, "_cached"):
        _consistency_report._cached = measure_consistency_criterion(
            richness_levels=(6, 9), blocks=(tuple(range(100, 106)),),
            null_blocks=(tuple(range(500, 506)),))
    return _consistency_report._cached


def test_the_adoption_withholds_the_confirmatory_blocks_and_says_why():
    """The maintainer's decision, recorded in the maintainer's words and stricter than asked.

    The declaration would have allowed the reserved blocks to be opened by a later amendment.
    The maintainer declined to spend them on this candidate at all, because k was influenced by
    the feasibility probe, so they stay clean for a criterion whose structural choices were
    fixed blind.
    """
    import hashlib
    import json
    from pathlib import Path

    declaration = Path("data/identity_calibration/t4e12-consistency-declaration.json")
    adoption = json.loads(Path(
        "data/identity_calibration/t4e12-consistency-adoption.json"
    ).read_text(encoding="utf-8"))
    assert adoption["adopts_sha256"] == hashlib.sha256(declaration.read_bytes()).hexdigest()
    assert adoption["adopted_as"] == "DEVELOPMENT_EXPERIMENT_ONLY"
    assert "genuinely clean" in adoption["the_confirmatory_blocks_are_explicitly_withheld"]
    assert "not yet a defensible real-world recurrence rule" in \
        adoption["the_maintainer_s_own_framing_of_this_candidate"]


def test_the_rejection_half_is_the_finding_and_the_null_admits_nothing():
    """What five previous candidates could not do, and the informative half of this result.

    Candidate C returned 116 matches on a null partition and candidate D returned 62. This
    criterion returns none, at either richness, out of the hundreds of pairs candidate C still
    proposes there.
    """
    report = _consistency_report()
    for row in report["null_blocks"]:
        assert row["pairs_proposed_by_candidate_C"] > 100, (
            "candidate C must still be proposing pairs, or nothing is being rejected")
        assert row["admitted"] == 0


def test_the_recall_half_was_very_nearly_guaranteed_and_is_not_the_finding():
    """Recorded so the result is not over-read, including by its own author later.

    The scenes are built with the motif in every scene and this criterion admits configurations
    present in every scene. Candidate C had already recovered every motif pair in every block,
    so the motif's match graph is complete and a partition-spanning group exists by
    construction. Condition 1 could hardly have failed; it is checked, not celebrated.
    """
    report = _consistency_report()
    for row in report["planted_blocks"]:
        assert row["false_split_rate"] <= MAX_FALSE_SPLIT
        assert row["motif_pairs_admitted"] == row["motif_pairs"]


def test_no_false_admission_survives_at_any_richness_measured():
    """The admission bound met, which no earlier candidate managed at any richness."""
    report = _consistency_report()
    for row in report["planted_blocks"]:
        assert row["false_admission_rate"] == 0.0
        assert row["admitted"] == row["motif_pairs"], (
            "exactly the motif's complete clique is admitted and nothing else")
        assert row["shortfall_factor"] == 0.0


def test_the_matched_fraction_falls_as_one_over_the_configuration_count():
    """Acceptance condition 2, and the shape it asked for.

    Only one group spans the partition, so the admitted count is fixed at C(S,2) however rich
    the scene, and the matched fraction is therefore exactly 1/m. No matching-based candidate
    could deliver this: a matching returns at most one pair per configuration, so its matched
    fraction is pinned.
    """
    report = _consistency_report()
    fractions = {}
    for row in report["planted_blocks"]:
        fractions[row["richness"]] = row["matched_fraction"]
        assert abs(row["matched_fraction"]
                   - 1.0 / row["configurations_per_scene"]) < 1e-9
    ordered = [fractions[key] for key in sorted(fractions)]
    assert ordered == sorted(ordered, reverse=True), "it must fall as richness grows"


def test_the_result_is_development_evidence_and_says_so():
    """No confirmatory evidence for this candidate exists, and none will under this adoption."""
    report = _consistency_report()
    assert report["evidence_class"] == "development"
    assert "reserved confirmatory blocks are untouched" in report["boundary"]


# ------------------------------------ The reserved evidence, split rather than spent
#
# Partitions 700-715 were opened once, for a confirmatory evaluation of T4E.12 candidate 2, and
# are inspected data permanently. Partitions 720-735 have never been generated and are refused
# in code even under confirmatory=True, because candidate 2's k was chosen after a probe and
# they are kept for a criterion whose structural choices are fixed blind.


def test_the_still_reserved_partitions_are_refused_even_under_confirmatory():
    """Enforced in code rather than intended, which is this programme's own standard.

    Intention has already been shown insufficient here -- it is why the forecast period is
    refused in code too. A flag that opens every reserved block at once would make the split a
    note in a document rather than a property of the system.
    """
    import pytest
    from src.benchmarks.identity_certification import (
        OPENED_CONFIRMATORY_SEEDS, RESERVED_CONFIRMATORY_SEEDS,
        ReservedSceneOpened, STILL_RESERVED_CONFIRMATORY_SEEDS, _refuse_reserved)

    assert sorted(OPENED_CONFIRMATORY_SEEDS) == list(range(700, 706)) + list(range(710, 716))
    assert sorted(STILL_RESERVED_CONFIRMATORY_SEEDS) == \
        list(range(720, 726)) + list(range(730, 736))
    assert OPENED_CONFIRMATORY_SEEDS | STILL_RESERVED_CONFIRMATORY_SEEDS == \
        RESERVED_CONFIRMATORY_SEEDS
    assert not (OPENED_CONFIRMATORY_SEEDS & STILL_RESERVED_CONFIRMATORY_SEEDS)
    for seed in sorted(STILL_RESERVED_CONFIRMATORY_SEEDS):
        for confirmatory in (False, True):
            with pytest.raises(ReservedSceneOpened):
                _refuse_reserved([seed], confirmatory=confirmatory)


def test_the_opened_partitions_still_refuse_development_code():
    """Spent for one confirmatory run does not mean available to anything that asks."""
    import pytest
    from src.benchmarks.identity_certification import (
        OPENED_CONFIRMATORY_SEEDS, ReservedSceneOpened, _refuse_reserved)

    for seed in sorted(OPENED_CONFIRMATORY_SEEDS):
        with pytest.raises(ReservedSceneOpened):
            _refuse_reserved([seed], confirmatory=False)
    _refuse_reserved(sorted(OPENED_CONFIRMATORY_SEEDS), confirmatory=True)


def test_the_amendment_records_what_a_confirmatory_pass_cannot_establish():
    """The ceiling survives the result, because it was fixed before the result existed."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e12-confirmatory-amendment.json"
    ).read_text(encoding="utf-8"))
    assert body["outcome"]["false_split_every_partition_every_richness"] == 0.0
    assert body["outcome"]["false_admission_every_partition_every_richness"] == 0.0
    cannot = " ".join(body["outcome"]["what_this_still_does_not_establish"])
    assert "cannot retrospectively make the choice of k blind" in cannot
    assert "REMAINS DEVELOPMENT ONLY" in cannot
    assert "not yet a defensible real-world recurrence rule" in \
        body["outcome"]["the_maintainer_s_framing_still_governs_the_write_up"]


def test_the_confirmatory_result_reproduces_on_the_partitions_that_were_spent():
    """Measured on 700-715 at the cheapest declared richness, and it holds.

    The full three-richness sweep is recorded in
    `measurements/t4e12_candidate_2_confirmatory.json`; this re-runs the sparsest level so the
    claim in the documents is checked by the suite rather than only asserted.
    """
    from src.benchmarks.identity_certification import measure_consistency_criterion

    report = measure_consistency_criterion(
        richness_levels=(6,), blocks=(tuple(range(700, 706)), tuple(range(710, 716))),
        null_blocks=(), confirmatory=True)
    assert report["evidence_class"] == "confirmatory"
    assert "720-735 remain reserved" in report["boundary"]
    assert "null evidence remains development only" in report["boundary"]
    for row in report["planted_blocks"]:
        assert row["false_split_rate"] == 0.0
        assert row["false_admission_rate"] == 0.0
        assert row["admitted"] == row["motif_pairs"] == 15


# --------------------------------- T4E.13 candidate 3: partial recurrence, declared blind
#
# The criterion is fixed in code here and MEASURED NOWHERE until the maintainer adopts its
# declaration. These tests therefore exercise the mechanics on toy metrics only. A test that
# ran the criterion on blocks 100-405 would BE the measurement, and would destroy the one
# property this candidate has that candidate 2 could not have.


def test_the_declaration_for_candidate_3_is_not_yet_adopted():
    """Code does not sign a scientific declaration for a person."""
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e13-partial-recurrence-declaration.json"
    ).read_text(encoding="utf-8"))
    assert body["status"] == "declared_before_measurement"
    assert "NOT VALID until the maintainer has reviewed" in body["declared_by"]
    assert body["confirmatory"]["status"].startswith("STILL RESERVED AND NOT YET GENERATED")
    assert body["development"]["status"].startswith("ALREADY INSPECTED")
    assert "would NOT license" in body["what_a_failure_would_and_would_not_license"]


def test_the_declaration_derives_its_recall_result_instead_of_reporting_it_later():
    """Candidate 1's lesson, applied before adoption rather than after the fact.

    Monotonicity makes condition 1 a matter of arithmetic. A declaration that let that pass as
    evidence would be claiming a finding it already knew it would get.
    """
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e13-partial-recurrence-declaration.json"
    ).read_text(encoding="utf-8"))
    derived = body["what_is_derivable_before_measuring_and_must_not_be_reported_as_a_finding"]
    assert "SUPERSET" in derived["monotonicity"]
    assert "0.0000" in derived["the_consequence_for_recall"]
    assert "admitted nothing anywhere" in derived["the_criterion_cannot_be_inert"]
    assert "PASSED BY ARITHMETIC" in body["acceptance"]["condition_1_recall"]
    assert "carries no evidential weight" in body["acceptance"]["condition_1_recall"]


def test_the_declaration_discloses_what_was_known_before_k_was_fixed():
    """Blindness is a claim about timing, and a claim about timing has to state its own limits.

    k = S - 1 sits at the coincidental ceiling candidate 2 already exposed. Saying so is what
    separates a severe test from a lucky one, and it is stated before the numbers exist.
    """
    import json
    from pathlib import Path

    body = json.loads(Path(
        "data/identity_calibration/t4e13-partial-recurrence-declaration.json"
    ).read_text(encoding="utf-8"))
    claim = body["the_blindness_claim_stated_exactly"]
    known = " ".join(claim["what_is_already_known_and_is_disclosed_rather_than_hidden"])
    assert "no coincidental group reaches 6" in known
    assert "reached 5" in known
    assert "most likely to fail" in claim["why_that_disclosure_does_not_void_the_claim"]
    assert "new candidate needing a new declaration" in claim["what_would_void_the_claim"]
    assert "cannot claim" in body["the_k_profile"]["what_it_is_explicitly_not_for"] or         "will NOT be able to make" in body["the_k_profile"]["what_it_is_explicitly_not_for"]


def test_k_is_a_function_of_the_partition_size_and_not_a_constant():
    """A literal 5 would silently change meaning when S changed, and the rates would not compare."""
    import pytest
    from src.benchmarks.identity_certification import ABSENCES_TOLERATED, consistency_k_for

    assert ABSENCES_TOLERATED == 1
    assert consistency_k_for(6) == 5
    assert consistency_k_for(9) == 8
    assert consistency_k_for(3) == 2
    assert [consistency_k_for(6, absences=a) for a in (0, 1, 2, 3)] == [6, 5, 4, 3]
    with pytest.raises(ValueError):
        consistency_k_for(6, absences=5)          # k = 1 is not a correspondence
    with pytest.raises(ValueError):
        consistency_k_for(6, absences=-1)


def test_partitions_of_different_sizes_are_refused_rather_than_pooled():
    """k tracks S, so unequal blocks would be measured under different criteria and averaged."""
    import pytest
    from src.benchmarks.identity_certification import _partition_size

    assert _partition_size([[1, 2, 3], [4, 5, 6]]) == 3
    with pytest.raises(ValueError):
        _partition_size([[1, 2, 3], [4, 5]])


def test_lowering_k_can_only_add_admissions_never_remove_them():
    """The monotonicity the declaration derives, checked on toys rather than assumed.

    This is what makes candidate 3's recall arithmetic rather than evidence, so it is worth a
    test of its own: if it failed, the declaration's central derivation would be wrong.
    """
    from src.benchmarks.identity_certification import consistency_admitted_pairs

    class _Ladder:
        """Configuration 0 agrees across all four scenes; configuration 1 across the first three."""

        def distance(self, a, b):
            return 0.0 if a == b else 1.0 + abs(a - b)

    partition = [([0, 1, 2], [False] * 3), ([0, 1, 2], [False] * 3),
                 ([0, 1, 2], [False] * 3), ([0, 2, 3], [False] * 3)]
    previous = None
    for k in (4, 3, 2):
        admitted = set(consistency_admitted_pairs(partition, _Ladder(), k=k))
        if previous is not None:
            assert previous <= admitted, "lowering k removed an admission at k=%d" % (k,)
        previous = admitted


def test_the_criterion_is_measured_nowhere_before_its_declaration_is_adopted():
    """The blindness claim is a property of the repository, not of a sentence in a file.

    Candidate 3's whole value is that k was fixed before any measurement at any k below S. A
    committed measurement, a recorded result or a test that ran one would spend that value
    silently, so its absence is asserted rather than intended.
    """
    from pathlib import Path

    assert not list(Path("measurements").glob("t4e13*")), "a candidate 3 measurement exists"
    assert not list(Path("data/identity_calibration").glob("t4e13*adoption*")),         "an adoption record exists, so this test is the one that should have been deleted"
    suite = Path("src/tests/test_identity_certification.py").read_text(encoding="utf-8")
    for call in ("measure_partial_recurrence" + "_criterion(", "measure_k" + "_profile("):
        assert call not in suite, "%s is called in the suite, which would be the measurement" % call


def test_the_k_profile_is_declared_to_adjudicate_nothing():
    """R20 forbids the horse race, and a sweep is the shape a horse race arrives in."""
    from src.benchmarks.identity_certification import PROFILE_ABSENCES, measure_k_profile

    assert PROFILE_ABSENCES == (0, 1, 2, 3)
    assert "adjudicates nothing" in measure_k_profile.__doc__ or         "Characterisation, not adjudication" in measure_k_profile.__doc__
    assert "own declaration" in measure_k_profile.__doc__
