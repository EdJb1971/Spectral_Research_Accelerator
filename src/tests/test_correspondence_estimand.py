"""TG17.15 slice 1: the question declared before it is answered, and the trap that decides it."""

from __future__ import annotations

import pytest

from src.benchmarks.shape_fixtures import minimum_resolvable_family
from src.core.correspondence_estimand import (
    ALPHA, CORRECTION, ESTIMAND_SCHEMA, ESTIMANDS, JOINT_STRUCTURE, PER_CORRESPONDENCE,
    estimand_report, joint_reassignment_resolution, minimum_pool_size,
    minimum_pool_size_for_detected_fraction, require_declared_estimand, sparsest_detectable_count,
)
from src.core.errors import InvalidParameterError
from src.core.structural_nulls import MAX_REASSIGNABLE_PAIRINGS
from src.statistics.multiple_comparisons import adjust


# --- the measurement that chooses between the two estimands ---------------------------------

def test_joint_reassignment_resolves_by_partner_count_and_not_by_reassignment_count():
    """The single most likely error in any reimplementation, pinned as a number.

    The valid reassignments are the derangement numbers and grow factorially. The partners one
    member can actually receive are `k - 1`. A member's statistic depends only on its partner, so
    the reassignment count is a count of spellings; using it as the reference-set size counts
    duplicates as independent evidence.
    """
    derangements = {4: 9, 5: 44, 6: 265, 7: 1854, 8: 14833}
    for k, expected in derangements.items():
        measured = joint_reassignment_resolution(k)
        assert measured["valid_reassignments"] == expected
        assert measured["distinct_partners_for_one_member"] == k - 1
        assert measured["p_floor_partner_based"] == pytest.approx(1.0 / k)


def test_miscounting_the_reference_set_is_anticonservative_by_orders_of_magnitude():
    """At the largest enumerable inventory the two readings differ by more than 1000x."""
    measured = joint_reassignment_resolution(MAX_REASSIGNABLE_PAIRINGS)
    honest = measured["p_floor_partner_based"]
    wrong = measured["p_floor_if_reassignments_miscounted"]
    assert honest == pytest.approx(0.125)
    assert wrong == pytest.approx(1.0 / 14834)
    assert honest / wrong > 1000.0
    # The direction matters more than the size: the mistake makes a null look easier to reject.
    assert wrong < honest


def test_resolution_is_only_measured_where_the_null_would_actually_run():
    with pytest.raises(InvalidParameterError, match="at most %d" % MAX_REASSIGNABLE_PAIRINGS):
        joint_reassignment_resolution(MAX_REASSIGNABLE_PAIRINGS + 1)
    for invalid in (1, 0, True, 2.5):
        with pytest.raises(InvalidParameterError):
            joint_reassignment_resolution(invalid)


# --- why the gap is structural rather than computational -------------------------------------

def test_the_declared_null_cannot_reach_its_own_resolvable_size():
    """TG17.11's refusal, restated as the relation between two measured numbers."""
    assert minimum_resolvable_family() > MAX_REASSIGNABLE_PAIRINGS


def test_at_its_minimum_size_the_joint_family_has_no_margin_at_all():
    """The measurement that says reaching 105 would not have produced a usable instrument.

    One member a single step off the floor and *nothing* rejects -- not 104 of 105. Graceful
    degradation begins only above the minimum, so the smallest resolvable family is a knife-edge
    demanding that every member be a perfect planted match simultaneously.
    """
    k = minimum_resolvable_family()
    labels = [str(index) for index in range(k)]
    at_floor = adjust([1.0 / k] * k, method=CORRECTION, alpha=ALPHA, n_tests=k, labels=labels)
    assert all(at_floor["rejected"])

    one_off = adjust([2.0 / k] + [1.0 / k] * (k - 1), method=CORRECTION, alpha=ALPHA,
                     n_tests=k, labels=labels)
    assert sum(one_off["rejected"]) == 0

    bigger = k + 1
    labels = [str(index) for index in range(bigger)]
    relaxed = adjust([2.0 / bigger] + [1.0 / bigger] * (bigger - 1), method=CORRECTION,
                     alpha=ALPHA, n_tests=bigger, labels=labels)
    assert sum(relaxed["rejected"]) == bigger - 1


# --- what the chosen estimand buys ------------------------------------------------------------

def test_the_pool_size_is_derived_against_the_real_correction_not_written_down():
    for m, expected in ((1, 19), (5, 45), (6, 48), (10, 58)):
        size = minimum_pool_size(m)
        assert size == expected
        outcome = adjust([1.0 / (size + 1)] * m, method=CORRECTION, alpha=ALPHA, n_tests=m,
                         labels=[str(i) for i in range(m)])
        assert all(outcome["rejected"])
        # One smaller must fail, or the number is not the minimum it claims to be.
        short = adjust([1.0 / size] * m, method=CORRECTION, alpha=ALPHA, n_tests=m,
                       labels=[str(i) for i in range(m)])
        assert not all(short["rejected"])


def test_margin_can_be_bought_from_the_pool_at_no_multiplicity_cost():
    """The property the joint estimand cannot have, which is the whole reason to change.

    Under `joint_structure` the only way to lower the floor is to grow the family, which grows
    the correction burden with it. Under `per_correspondence` the pool grows alone.
    """
    m = 6
    tight = minimum_pool_size(m)
    generous = 2 * tight
    labels = [str(index) for index in range(m)]
    off_floor_tight = adjust([2.0 / (tight + 1)] + [1.0 / (tight + 1)] * (m - 1),
                             method=CORRECTION, alpha=ALPHA, n_tests=m, labels=labels)
    off_floor_generous = adjust([2.0 / (generous + 1)] + [1.0 / (generous + 1)] * (m - 1),
                                method=CORRECTION, alpha=ALPHA, n_tests=m, labels=labels)
    assert sum(off_floor_tight["rejected"]) == 0
    assert sum(off_floor_generous["rejected"]) == m - 1
    # The number of tests, and therefore the correction burden, did not change between them.


def test_a_bigger_pool_never_costs_more_tests():
    assert minimum_pool_size(1) < minimum_pool_size(20)
    for m in (1, 3, 5, 6, 10, 20):
        assert minimum_pool_size(m) < minimum_resolvable_family()


# --- the declaration itself -------------------------------------------------------------------

def test_a_correspondence_test_with_no_declared_estimand_is_refused_rather_than_defaulted():
    for missing in (None, "", "   "):
        with pytest.raises(InvalidParameterError, match="no default question"):
            require_declared_estimand(missing)
    with pytest.raises(InvalidParameterError, match="registered estimands"):
        require_declared_estimand("whatever_the_caller_meant")


def test_the_refused_estimand_is_registered_so_it_can_be_refused_by_name():
    """Leaving it out would make it a question nobody had answered."""
    assert JOINT_STRUCTURE.name in set(ESTIMANDS.names())
    assert JOINT_STRUCTURE.admissible is False
    assert JOINT_STRUCTURE.resolution_coupled_to_family is True
    with pytest.raises(InvalidParameterError, match="refused by name"):
        require_declared_estimand("joint_structure")


def test_the_chosen_estimand_resolves_independently_of_the_family_size():
    chosen = require_declared_estimand("per_correspondence")
    assert chosen is PER_CORRESPONDENCE
    assert chosen.admissible is True
    assert chosen.resolution_coupled_to_family is False


def test_the_correction_choice_is_recorded_with_its_reason_because_it_is_load_bearing():
    """BY rather than BH: family members share a pool and are dependent."""
    body = estimand_report()
    assert body["correction"] == CORRECTION == "benjamini_yekutieli"
    assert "arbitrary dependence" in body["correction_basis"]
    assert "Benjamini-Hochberg is not" in body["correction_basis"]


def test_the_report_carries_the_measurement_the_decision_rests_on():
    body = estimand_report()
    assert body["schema"] == ESTIMAND_SCHEMA
    assert body["chosen"] == "per_correspondence"
    assert body["largest_enumerable_inventory"] == MAX_REASSIGNABLE_PAIRINGS
    measured = {row["pairings"]: row for row in body["joint_reassignment_resolution"]}
    assert measured[8]["valid_reassignments"] == 14833
    assert measured[8]["distinct_partners_for_one_member"] == 7
    assert set(body["pool_sizes_required"]) == {"1", "3", "5", "6", "10", "20"}
    for word in ("calibration", "partner pool", "result"):
        assert word in body["claim_boundary"]


def test_this_slice_claims_no_calibration_and_no_pool():
    """The boundary matters: declaring the question is not answering it."""
    body = estimand_report()
    assert "It is not a calibration" in body["claim_boundary"]
    assert not any(key.startswith("power") or key.startswith("false_positive") for key in body)


# --- the all-genuine assumption, named and measured ---------------------------------------------

def test_minimum_pool_size_is_the_detected_fraction_function_at_fraction_one():
    for m in (3, 6, 10):
        assert minimum_pool_size_for_detected_fraction(m, 1.0) == minimum_pool_size(m)


def test_a_sparser_family_needs_a_larger_pool_and_the_gap_is_not_marginal():
    """`minimum_pool_size` sizes for the most favourable world there is; this measures the rest."""
    sizes = [minimum_pool_size_for_detected_fraction(6, f) for f in (1.0, 5 / 6, 0.5, 1 / 6)]
    assert sizes == sorted(sizes)
    assert sizes[0] == 48 and sizes[-1] >= 6 * sizes[0] / 2
    assert minimum_pool_size_for_detected_fraction(6, 0.5) > 2 * minimum_pool_size(6) - 10


def test_a_fraction_rounding_to_no_planted_member_is_refused_rather_than_answered():
    with pytest.raises(InvalidParameterError) as excinfo:
        minimum_pool_size_for_detected_fraction(3, 0.01)
    assert "has no answer at any pool size" in str(excinfo.value)


@pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5])
def test_an_impossible_detected_fraction_is_refused(fraction):
    with pytest.raises(InvalidParameterError):
        minimum_pool_size_for_detected_fraction(6, fraction)


def test_sparsest_detectable_count_measures_from_the_floors_the_pools_actually_have():
    """Six pools of 58 clear `minimum_pool_size(6)` and still need five genuine members."""
    assert sparsest_detectable_count([1 / 59.0] * 6, tested_correspondences=6) == 5
    # A larger pool lowers every floor and buys back the sparsity the family can detect.
    assert sparsest_detectable_count([1 / 301.0] * 6, tested_correspondences=6) == 1


def test_pools_too_small_to_ever_reject_return_none_rather_than_a_number():
    assert sparsest_detectable_count([1 / 3.0] * 6, tested_correspondences=6) is None
    assert sparsest_detectable_count([], tested_correspondences=6) is None


def test_the_estimand_report_publishes_the_sparse_case_beside_the_favourable_one():
    body = estimand_report()
    sparse = body["pool_sizes_required_when_only_some_correspond"]
    assert sparse["m=6"]["1"] == body["pool_sizes_required"]["6"]
    assert sparse["m=6"]["0.5"] > sparse["m=6"]["1"]
    assert "most favourable world" in sparse["basis"]
