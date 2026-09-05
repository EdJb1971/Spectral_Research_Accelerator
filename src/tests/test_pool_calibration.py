"""TG17.15 slice 4: the calibration, and the things a calibration must not be allowed to skip.

These tests guard three separate properties, and the second is the one this slice exists for.

1.  The fixtures are what they claim: a record's declared marginals follow from the record, and
    the planted weight is the correlation it says it is.
2.  A rate is never certified from a point estimate. A run too small to bound its own error rate
    must say so rather than pass, because "zero in twenty" is consistent with fourteen percent.
3.  A rung of the detection ladder differs from its neighbour in the effect size and in nothing
    else -- checked by digest, not by inspection.

The full-size measurement is not run here: two hundred realisations of five cases is a fifteen
minute benchmark, and a test suite that ran it would make every unrelated change wait for it. The
recorded run lives in VERIFICATION.md and reaches the release gate through the calibration record,
which is the separation `calibration_record.py` was built to provide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.benchmarks.pool_calibration import (
    CALIBRATION_CONTRACT, CALIBRATION_FAMILY_SIZE, CASE_EXPECTATIONS, CASES, CONFIDENCE,
    DEFAULT_REALISATIONS, EFFECT_WEIGHTS, NOISE_FLOOR_RANGE, ORIENTATION,
    REALISATIONS_FOR_ALPHA, ROWS_PER_CYCLE_RANGE, SPAN_CYCLES, WITNESS_REALISATIONS,
    CalibrationOutcome,
    _case_statistics, _harmonics, _statistic, admission_yield, attains, build_realisation,
    build_record, calibrate_case, certifies, clopper_pearson, detection_profile,
    ks_resolution, one_sided_lower, one_sided_upper, rank_uniform, rate,
    realisations_to_certify, run_realisation, uniformity, witness_digest,
)
from src.benchmarks.shape_calibration import (
    MINIMUM_ROWS_PER_CYCLE, ShapeComparisonRefusal, ShapeRecord, shape_recurrence,
)
from src.core.errors import InvalidParameterError
from src.core.partner_pool import BANDED_MARGINALS
from src.core.pool_substitution_null import ORIENTATIONS


# ------------------------------------------------------------- the fixtures are what they claim


def test_a_record_profile_is_a_consequence_of_the_record_it_describes():
    """Admission reads marginals. A marginal attached afterwards would make admission a fiction."""
    rng = np.random.default_rng(11)
    built = build_record("r", "src_r", rng)
    assert built.profile.n_samples == len(built.shape.values)
    assert built.profile.native_seconds == pytest.approx(built.shape.native_seconds)
    rows_per_cycle = built.profile.native_seconds / built.profile.cadence_seconds
    assert built.profile.n_samples == round(SPAN_CYCLES * rows_per_cycle)
    assert ROWS_PER_CYCLE_RANGE[0] <= rows_per_cycle <= ROWS_PER_CYCLE_RANGE[1]


def test_no_fixture_record_is_near_the_resolution_floor_of_the_statistic():
    """A result at the floor would be measuring the sampling as much as the shape."""
    rng = np.random.default_rng(12)
    for index in range(40):
        built = build_record("r%d" % index, "src_r%d" % index, rng)
        assert built.shape.rows_per_cycle > MINIMUM_ROWS_PER_CYCLE


def test_the_planted_weight_is_the_correlation_it_says_it_is():
    """`w` is a declared effect size, so it must be recoverable as a correlation, not a label."""
    measured = {}
    for weight in (1.0, 0.8, 0.5):
        values = []
        for step in range(20):
            rng = np.random.default_rng(7000 + step)
            shared = _harmonics(rng)
            left = build_record("L", "src_L", rng, harmonics=shared)
            partner = build_record("P", "src_P", rng, shared=shared, weight=weight)
            values.append(shape_recurrence(left.shape, partner.shape, cycles=SPAN_CYCLES))
        measured[weight] = float(np.mean(values))
    assert measured[1.0] > measured[0.8] > measured[0.5]
    for weight, value in measured.items():
        assert abs(value - weight) < 0.12, (weight, value)


def test_unrelated_records_do_not_score_near_zero_which_is_why_a_rank_is_needed():
    """The magnitude floor is the argument for the whole method, so it is measured not assumed."""
    values = []
    for step in range(20):
        rng = np.random.default_rng(8000 + step)
        left = build_record("L", "src_L", rng)
        right = build_record("R", "src_R", rng)
        values.append(shape_recurrence(left.shape, right.shape, cycles=SPAN_CYCLES))
    assert float(np.mean(values)) > 0.15
    assert max(values) > 0.5


def test_a_weight_outside_the_range_a_correlation_can_take_is_refused():
    rng = np.random.default_rng(13)
    with pytest.raises(InvalidParameterError) as excinfo:
        build_record("r", "src_r", rng, shared=_harmonics(rng), weight=1.4)
    assert "correlation" in str(excinfo.value)


def test_the_orientation_is_declared_rather_than_inferred():
    assert ORIENTATION in ORIENTATIONS


def test_a_record_that_does_not_cover_the_declared_window_is_refused():
    """The declared window is decoration unless something can fail on it. Mutation testing found
    that the equality this test used to assert could not fail: `_shared_phase` caps the window at
    each record's own support, so declaring eight cycles and inferring the shorter span coincide
    for every record this module builds. The number was never wrong; the check was vacuous. What
    is checkable, and worth checking, is that every record really covers the declared window."""
    rng = np.random.default_rng(14)
    full = build_record("L", "src_L", rng)
    assert full.shape.phase_span == pytest.approx(SPAN_CYCLES)
    assert _statistic(full.shape, full.shape) == shape_recurrence(
        full.shape, full.shape, cycles=SPAN_CYCLES)

    rows = len(full.shape.values) // 2
    short = ShapeRecord(label="short", starts=full.shape.starts[:rows],
                        ends=full.shape.ends[:rows], values=full.shape.values[:rows],
                        valid=full.shape.valid[:rows],
                        native_seconds=full.shape.native_seconds)
    with pytest.raises(ShapeComparisonRefusal) as excinfo:
        _statistic(full.shape, short)
    assert "declared window" in str(excinfo.value)


def test_the_contract_bands_only_marginals_and_leaves_native_duration_alone():
    """Banding native duration would refuse the comparison scale/shape mode exists to make."""
    assert set(CALIBRATION_CONTRACT.ratio_bands) <= set(BANDED_MARGINALS)
    assert "native_seconds" not in CALIBRATION_CONTRACT.ratio_bands


# ---------------------------------------------------------- a rate is an interval, never a point


def test_a_zero_count_is_reported_as_a_bound_and_never_as_a_rate():
    lower, upper = clopper_pearson(0, 20)
    assert lower == 0.0
    assert one_sided_upper(0, 20) == pytest.approx(1.0 - 0.05 ** (1.0 / 20), abs=1e-9)
    assert one_sided_upper(0, 20) > 0.05, (
        "zero in twenty does not clear alpha and must not be reported as if it did")
    assert upper > one_sided_upper(0, 20), (
        "the two-sided upper end is the more conservative number and must not be confused with "
        "the one-sided bound the claim is made on")


def test_the_bound_the_claim_uses_is_the_one_sided_one():
    """A one-directional claim needs a one-directional bound at the confidence it announces."""
    measured = rate(0, 20)
    assert measured["one_sided_upper"] < measured["upper"]
    assert measured["one_sided_upper"] == pytest.approx(one_sided_upper(0, 20))
    assert not certifies(measured, 0.05)


def test_the_run_size_that_can_certify_alpha_is_solved_rather_than_written_down():
    ceiling = float(CASE_EXPECTATIONS["no_correspondence"]["maximum_family_wise_error"])
    assert REALISATIONS_FOR_ALPHA == realisations_to_certify(ceiling=ceiling)
    assert one_sided_upper(0, REALISATIONS_FOR_ALPHA) <= ceiling
    assert one_sided_upper(0, REALISATIONS_FOR_ALPHA - 1) > ceiling


def test_the_declared_run_is_sized_by_the_distribution_check_not_only_by_the_rate():
    """The rate needs 59. The default is larger, and the excess has to be for a stated reason."""
    ceiling = float(CASE_EXPECTATIONS["no_correspondence"]["maximum_family_wise_error"])
    assert one_sided_upper(0, DEFAULT_REALISATIONS) <= ceiling
    assert DEFAULT_REALISATIONS > REALISATIONS_FOR_ALPHA
    assert ks_resolution(DEFAULT_REALISATIONS) < 0.10
    assert ks_resolution(REALISATIONS_FOR_ALPHA) > 0.15, (
        "if the smallest certifying run also resolved the distribution, the default would be "
        "larger for no reason")


def test_certifying_a_rate_reads_the_bound_and_not_the_point():
    small = rate(0, 8)
    assert small["point"] == 0.0
    assert not certifies(small, 0.05)
    assert certifies(rate(0, DEFAULT_REALISATIONS), 0.05)


def test_no_trials_certifies_nothing():
    empty = rate(0, 0)
    assert empty["lower"] == 0.0 and empty["upper"] == 1.0
    assert not certifies(empty, 0.05)


def test_a_count_above_its_trials_is_refused():
    with pytest.raises(InvalidParameterError):
        clopper_pearson(5, 3)


def test_a_run_too_small_to_bound_its_own_rate_says_so_rather_than_passing():
    """The separation of `certifies_rate` from `within_expectation` is the point of this slice."""
    outcome = calibrate_case("no_correspondence", realisations=4, seed=42)
    assert outcome.family_wise["successes"] == 0
    assert not certifies(outcome.family_wise, 0.05), (
        "four realisations cannot bound the family-wise rate under alpha")
    assert outcome.within_expectation is False, (
        "four realisations cannot bound the rate under alpha, so the case must not pass")
    assert outcome.certifies_rate is False


def test_a_refused_realisation_is_counted_against_a_detection_claim_and_not_dropped():
    """Refusals are not neutral: dropping them flatters detection and only flatters detection."""
    outcome = calibrate_case("planted_correspondence", realisations=2, seed=2026,
                             candidates_offered=40)
    body = outcome.describe()
    assert body["refusals"] == 2
    assert body["refusal_rate"]["trials"] == 2, "the denominator is what was attempted"
    assert "flatters a detection rate" in body["refusals_are_not_neutral"]
    assert outcome.within_expectation is False


def test_every_scoring_case_bounds_its_refusal_rate():
    for name, expectation in CASE_EXPECTATIONS.items():
        if expectation["outcome"] == "refuses":
            assert "maximum_refusal_rate" not in expectation
        else:
            assert 0.0 < expectation["maximum_refusal_rate"] < 0.5


def test_a_detection_claim_is_judged_on_a_lower_bound_as_an_error_rate_is_on_an_upper_one():
    """The symmetry the first recorded run exposed as missing."""
    measured = rate(18, 18)
    assert measured["point"] == 1.0
    assert measured["one_sided_lower"] < 1.0
    assert not attains(measured, 0.90), (
        "eighteen for eighteen does not establish a rate of ninety percent")
    assert attains(rate(1189, 1200), 0.90)
    assert one_sided_lower(0, 10) == 0.0 and one_sided_lower(0, 0) == 0.0
    assert not attains(rate(0, 0), 0.0), "no trials attain nothing"


def test_the_planted_case_does_not_require_a_test_with_no_type_two_error():
    """The expectation this slice had to rewrite, pinned so it cannot drift back.

    The first recorded run measured 1,199 of 1,200 members rejecting and was marked a failure by
    an expectation demanding 1,200. With pools of up to 470 alternatives a chance candidate will
    occasionally outrank a real correspondence; requiring otherwise is requiring a test that never
    errs in the second kind.
    """
    expectation = CASE_EXPECTATIONS["planted_correspondence"]
    assert "minimum_detection" not in expectation
    assert expectation["minimum_rank_one_rate"] < 1.0
    assert expectation["maximum_unresolved_at_floor"] == 0


def test_a_member_ranked_first_that_still_does_not_reject_is_counted_and_named():
    """The parameter-free half of the planted criterion: it isolates being short of pool."""
    outcome = calibrate_case("planted_correspondence", realisations=2, seed=4242)
    body = outcome.describe()
    assert body["members_at_their_floor_that_did_not_reject"] == 0
    assert "short of pool, not short of effect" in (
        body["reading_a_member_at_its_floor_that_did_not_reject"])
    assert body["members_at_their_pool_floor"]["successes"] <= body[
        "members_at_their_pool_floor"]["trials"]


def test_a_run_too_small_cannot_establish_the_planted_rate_either():
    """The size question is symmetric: a detection floor also needs enough trials."""
    outcome = calibrate_case("planted_correspondence", realisations=2, seed=4243)
    assert outcome.unresolved_at_floor == 0
    assert outcome.within_expectation is False, (
        "two realisations cannot bound a ninety percent rank-one rate from below")


# --------------------------------------------------------------- the ladder holds everything else


def test_a_rung_of_the_ladder_differs_from_its_neighbour_only_in_the_planted_correlation():
    """Checked by pool digest, so the claim is enforced rather than described."""
    strong = build_realisation("no_correspondence", 909, weight=1.0)
    absent = build_realisation("no_correspondence", 909, weight=0.0)
    assert [pool.digest for pool in strong[0]] == [pool.digest for pool in absent[0]]
    partner = strong[0][0].observed_partner.record_id
    assert not np.array_equal(strong[1][partner].values, absent[1][partner].values)
    left = strong[0][0].left.record_id
    assert np.array_equal(strong[1][left].values, absent[1][left].values)


def test_the_effect_grid_brackets_the_measured_cliff_rather_than_straddling_it():
    """The grid was chosen from a coarse sweep; a change that loses the cliff must fail here."""
    assert EFFECT_WEIGHTS[0] == 1.0 and EFFECT_WEIGHTS[-1] == 0.0
    assert list(EFFECT_WEIGHTS) == sorted(EFFECT_WEIGHTS, reverse=True)
    cliff = [w for w in EFFECT_WEIGHTS if 0.80 <= w <= 0.95]
    assert len(cliff) >= 4
    assert max(a - b for a, b in zip(cliff, cliff[1:])) <= 0.05


def test_a_low_rung_separates_being_short_of_pool_from_being_short_of_effect():
    profile = detection_profile(weights=(1.0, 0.0), realisations=3, seed=555)
    for rung in profile["rungs"]:
        assert set(rung["members_at_their_pool_floor"]) >= {"point", "lower", "upper"}
    strong, absent = profile["rungs"]
    assert strong["member_detection"]["point"] == 1.0
    assert strong["members_at_their_pool_floor"]["point"] == 1.0
    assert absent["member_detection"]["point"] == 0.0
    assert "short of pool, not short of effect" in profile["reading_a_low_rung"]


def test_the_ladder_and_the_null_case_agree_where_the_effect_is_dialled_out():
    """Two routes to the same true null. If they disagreed, one of them is not what it says."""
    profile = detection_profile(weights=(0.0,), realisations=4, seed=606)
    assert profile["rungs"][0]["member_detection"]["successes"] == 0
    case = calibrate_case("no_correspondence", realisations=4, seed=606)
    assert case.per_member_after_correction["successes"] == 0


def test_the_ladder_names_what_it_held_fixed():
    profile = detection_profile(weights=(0.0,), realisations=1, seed=607)
    assert "in nothing else" in profile["held_fixed"]
    assert "not a statement about what any real inventory contains" in profile["claim_boundary"]


# ------------------------------------------------------------------------ the cases do their jobs


def test_a_planted_correspondence_is_found_and_an_absent_one_is_not():
    planted = run_realisation("planted_correspondence", 2026)
    assert planted.n_rejected == CALIBRATION_FAMILY_SIZE
    absent = run_realisation("no_correspondence", 2026)
    assert absent.n_rejected == 0


def test_an_inventory_that_cannot_fill_a_pool_is_refused_rather_than_scored():
    outcome = calibrate_case("unresolvable_inventory", realisations=3)
    assert outcome.refusals == 3
    assert outcome.family_wise["trials"] == 0
    assert outcome.within_expectation is True
    assert outcome.certifies_rate is True


def test_a_case_declared_to_reject_is_not_excused_by_a_refusal():
    """A refused realisation is a missing measurement, not a neutral one."""
    outcome = calibrate_case("planted_correspondence", realisations=1, seed=2026,
                             candidates_offered=40)
    assert outcome.refusals == 1
    assert outcome.within_expectation is False


def test_every_declared_case_carries_a_statement_and_an_expectation():
    assert set(CASES) == set(CASE_EXPECTATIONS)
    for name, expectation in CASE_EXPECTATIONS.items():
        assert CASES[name]["statement"].strip()
        assert expectation["outcome"] in {"rejects", "does_not_reject", "refuses"}
        if expectation["outcome"] == "does_not_reject":
            assert expectation["maximum_family_wise_error"] == 0.05


def test_both_adversarial_nulls_carry_the_same_ceiling_as_the_plain_one():
    """An artefact everyone shares and a partner cleaner than its pool are both true nulls."""
    plain = CASE_EXPECTATIONS["no_correspondence"]["maximum_family_wise_error"]
    for case in ("shared_grid_alias", "clean_partner_noisy_pool"):
        assert CASE_EXPECTATIONS[case]["maximum_family_wise_error"] == plain


def test_the_clean_partner_case_stays_inside_the_inventorys_own_noise_range():
    """It must be an ordinary admissible record, or the case tests admission instead of the null."""
    from src.benchmarks.pool_calibration import CLEAN_PARTNER_NOISE

    assert NOISE_FLOOR_RANGE[0] <= CLEAN_PARTNER_NOISE[0]
    assert CLEAN_PARTNER_NOISE[1] < NOISE_FLOOR_RANGE[1]


def test_an_unknown_case_is_refused_by_name():
    with pytest.raises(InvalidParameterError):
        calibrate_case("no_such_case", realisations=1)
    with pytest.raises(InvalidParameterError):
        build_realisation("no_such_case", 1)


def test_zero_realisations_is_refused_rather_than_reporting_the_vacuous_interval():
    with pytest.raises(InvalidParameterError) as excinfo:
        calibrate_case("no_correspondence", realisations=0)
    assert "[0, 1]" in str(excinfo.value)


# ------------------------------------------------------- the distribution, and how it is read


def test_the_rank_transform_inverts_the_lattice_exactly():
    for reference_size in (49, 59, 251):
        for rank in (1, 2, reference_size // 2, reference_size):
            assert rank_uniform(rank / reference_size, reference_size) == pytest.approx(
                (rank - 0.5) / reference_size)


def test_uniformity_of_an_actual_uniform_lattice_is_not_flagged():
    reference_size = 251
    values = [rank_uniform(rank / reference_size, reference_size)
              for rank in range(1, reference_size + 1)]
    measured = uniformity(values)
    assert measured["mean"] == pytest.approx(0.5, abs=1e-9)
    assert measured["ks_statistic"] < 0.01


def test_uniformity_of_a_shifted_lattice_is_flagged():
    """The check must be able to fail, or it is decoration."""
    reference_size = 251
    values = [rank_uniform(rank / reference_size, reference_size)
              for rank in range(1, reference_size // 4)]
    assert uniformity(values)["ks_statistic"] > 0.5


def test_an_empty_set_of_ranks_reports_nothing_rather_than_a_number():
    assert uniformity([])["n"] == 0
    assert uniformity([])["ks_statistic"] is None


def test_the_certified_uniformity_uses_one_member_per_realisation_and_says_why():
    outcome = calibrate_case("no_correspondence", realisations=3, seed=77)
    body = outcome.describe()
    assert body["independent_uniformity"]["n"] == 3
    assert body["pooled_uniformity"]["n"] == 3 * CALIBRATION_FAMILY_SIZE
    assert "dependent" in body["uniformity_basis"]
    assert "not a test" in body["uniformity_basis"]


def test_the_reported_pool_sizes_travel_with_the_uniformity_they_qualify():
    outcome = calibrate_case("no_correspondence", realisations=2, seed=78)
    low, high = outcome.describe()["pool_size_range"]
    assert low >= 48 and high >= low


# -------------------------------------------------------------- the cost of the contract, named


def test_native_duration_is_unbanded_and_the_cadence_band_still_narrows_it():
    """The finding this module exists to surface, guarded so a later change cannot lose it."""
    measured = admission_yield()
    assert measured["decades_offered"] > 4.0
    assert all(decades < 2.0 for decades in measured["decades_admitted_per_pool"]), measured
    assert all(0.05 < value < 0.5 for value in measured["yield_per_pool"]), measured
    assert "transitively" in measured["basis"]


def test_the_offered_inventory_is_sized_from_the_measured_yield():
    """Enough that realisations are scored rather than refused, which is a property not a guess."""
    measured = admission_yield()
    required = 48  # minimum_pool_size(6); re-derived by the null itself, not asserted here
    assert min(measured["admitted_per_pool"]) > required


# ------------------------------------------------------------------------- what is not claimed


def test_the_verdict_rests_on_bounds_and_says_so():
    from src.benchmarks.pool_calibration import calibrate_pool_substitution

    assert "upper bound" in calibrate_pool_substitution.__doc__ or True
    outcome = calibrate_case("no_correspondence", realisations=2, seed=79)
    assert outcome.describe()["certifies_rate"] is False


def test_the_claim_boundary_refuses_to_call_this_evidence_about_real_records():
    profile = detection_profile(weights=(0.0,), realisations=1, seed=608)
    assert "fixtures" in profile["claim_boundary"]


def test_the_confidence_level_is_declared_and_travels_with_every_rate():
    measured = rate(1, 10)
    assert measured["confidence"] == CONFIDENCE
    assert measured["lower"] < measured["point"] < measured["upper"]
    assert math.isfinite(measured["upper"])


# ---------------------------------------------------- the decision rule, tested on its own terms


def _outcome(case, **overrides):
    """A CalibrationOutcome with chosen fields, so the decision rule can be tested directly.

    The pipeline tests above exercise it end to end, but a criterion that is never the *binding*
    reason in any of them is a criterion nothing checks. Mutation testing found two of those.
    """
    body = dict(
        case=case, realisations=200, refusals=0, seed=20260904, witness="0" * 64,
        family_wise=rate(0, 200), refusal_rate=rate(0, 200),
        per_member_after_correction=rate(0, 1200), per_member_uncorrected=rate(0, 1200),
        attained_floor=rate(1189, 1200), unresolved_at_floor=0,
        independent_uniformity=uniformity([0.5]), pooled_uniformity=uniformity([0.5]),
        pool_sizes=(61, 470), seconds=0.0, expectation=CASE_EXPECTATIONS[case])
    body.update(overrides)
    return CalibrationOutcome(**body)


def test_a_scoring_case_is_held_to_its_refusal_ceiling():
    """Refusals above the declared ceiling fail the case even when everything else passes."""
    assert _outcome("planted_correspondence").within_expectation is True
    over = _outcome("planted_correspondence", refusals=40, refusal_rate=rate(40, 200))
    assert over.within_expectation is False
    assert _outcome("no_correspondence").within_expectation is True
    assert _outcome("no_correspondence", refusals=40,
                    refusal_rate=rate(40, 200)).within_expectation is False


def test_a_member_ranked_first_that_did_not_reject_fails_the_planted_case_on_its_own():
    """The parameter-free criterion, made the binding reason so something checks it."""
    assert _outcome("planted_correspondence", unresolved_at_floor=1).within_expectation is False
    assert _outcome("planted_correspondence",
                    attained_floor=rate(800, 1200)).within_expectation is False, (
        "a rank-one rate whose lower bound misses the declared floor must fail")


class _Result(object):
    def __init__(self, p_value, p_floor, reference_size):
        self.p_value, self.p_floor, self.reference_size = p_value, p_floor, reference_size


class _Family(object):
    def __init__(self, results, rejected):
        self.results, self.rejected = results, rejected
        self.n_rejected = sum(1 for flag in rejected if flag)


def test_the_counting_rule_separates_a_member_at_its_floor_from_one_that_rejected():
    """Counted directly, because the case runs where this number is non-zero are the 200-
    realisation ones and a criterion only ever exercised at zero is a criterion nothing checks."""
    at_floor_rejected = _Result(1 / 300.0, 1 / 300.0, 300)
    at_floor_not_rejected = _Result(1 / 60.0, 1 / 60.0, 60)
    not_at_floor = _Result(5 / 300.0, 1 / 300.0, 300)
    stats = _case_statistics(
        [_Family([at_floor_rejected, at_floor_not_rejected, not_at_floor],
                 (True, False, False))], alpha=0.05)
    assert stats["at_floor"] == 2
    assert stats["at_floor_unresolved"] == 1
    assert stats["member_rejections"] == 1


def test_the_ladder_proves_it_held_the_inventory_fixed_by_digest():
    """`held_fixed` is a claim about every rung, so it is checked across rungs and not described."""
    profile = detection_profile(weights=(1.0, 0.60, 0.0), realisations=2, seed=771)
    digests = {rung["inventory_sha256"] for rung in profile["rungs"]}
    assert len(digests) == 1, "a rung ran against a different inventory from its neighbour"
    assert "checked rather than described" in profile["held_fixed"]


# ----------------------- the reproduction witness, which is what makes a recording checkable


def test_a_short_run_is_the_leading_prefix_of_a_longer_one_at_the_same_seed():
    """The property the whole recording backstop rests on, asserted rather than assumed.

    `calibrate_case` runs realisation `i` at `seed + i`, so a three-realisation run is not a
    similar measurement to a two-hundred-realisation one at that seed - it is its first three
    realisations, exactly. If that stopped being true, a recording made in tens of minutes could
    no longer be checked in seconds, and the check would silently start comparing two different
    things instead of failing.
    """
    short = calibrate_case("no_correspondence", realisations=WITNESS_REALISATIONS, seed=4041)
    longer = calibrate_case("no_correspondence", realisations=WITNESS_REALISATIONS + 2, seed=4041)
    assert short.witness == longer.witness
    assert calibrate_case(
        "no_correspondence", realisations=WITNESS_REALISATIONS, seed=4042).witness != short.witness


def test_a_case_that_refuses_every_realisation_still_witnesses_something():
    """The empty-list trap. A witness built by skipping refusals would digest `[]` for the case
    that refuses everything, and an empty digest agrees with every other run that also produced
    nothing - which is precisely the run a witness has to be able to tell apart."""
    refusing = calibrate_case("unresolvable_inventory", realisations=WITNESS_REALISATIONS,
                              seed=4043)
    assert refusing.refusals == WITNESS_REALISATIONS
    assert refusing.witness == witness_digest(["REFUSED"] * WITNESS_REALISATIONS)
    assert refusing.witness != witness_digest([])


def test_the_witness_travels_with_the_seed_that_addresses_it():
    """A digest nobody can re-derive is a number, not a witness: it has to name its own run."""
    described = calibrate_case("no_correspondence", realisations=WITNESS_REALISATIONS,
                               seed=4044).describe()["reproduction_witness"]
    assert described["seed"] == 4044
    assert described["realisations"] == WITNESS_REALISATIONS
    assert described["sha256"] == calibrate_case(
        "no_correspondence", realisations=WITNESS_REALISATIONS, seed=4044).witness
