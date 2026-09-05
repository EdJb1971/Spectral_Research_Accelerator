"""TG17.15 slice 3: the exact pool-substitution null, and the family size that cannot be rechosen."""

from __future__ import annotations

import dataclasses

import pytest

from src.core.correspondence_estimand import (
    ALPHA, CORRECTION, minimum_pool_size, minimum_pool_size_for_detected_fraction,
)
from src.core.errors import InvalidParameterError
from src.core.partner_pool import AdmissionContract, PartnerPool, RecordProfile, build_partner_pool
from src.core.pool_substitution_null import (
    FAMILY_SCHEMA, NULL_SCHEMA, ORIENTATIONS, CorrespondenceFamily, SubstitutionResult,
    correspondence_family, exact_pool_substitution, monte_carlo_pool_substitution,
    require_declared_orientation,
)
from src.core.structural_nulls import NullRefusal


CONTRACT = AdmissionContract(ratio_bands={
    "n_samples": 2.0, "effective_sample_size": 2.0, "noise_floor": 3.0, "cadence_seconds": 4.0})

M = 6  # the declared family size used throughout; minimum_pool_size(6) is what the pools must clear


def profile(index, **changes) -> RecordProfile:
    values = dict(record_id="r%03d" % index, provenance_key="src%03d" % index,
                  n_samples=1000, effective_sample_size=400.0,
                  native_seconds=900.0 * (1.0 + index % 37),
                  cadence_seconds=60.0, coverage_fraction=0.95, noise_floor=0.1)
    values.update(changes)
    return RecordProfile(**values)


def build(left_id="LEFT", partner_id="PARTNER", *, first=2, count=58, m=M):
    left = profile(0, record_id=left_id, provenance_key="src" + left_id)
    partner = profile(1, record_id=partner_id, provenance_key="src" + partner_id)
    candidates = [profile(i) for i in range(first, first + count)]
    return build_partner_pool(left=left, observed_partner=partner, candidates=candidates,
                              contract=CONTRACT, estimand="per_correspondence",
                              tested_correspondences=m)


def payloads(pool, *, observed=10.0, alternative=0.0):
    """Records keyed by id. The statistic below reads them, so the ranking is fully controlled."""
    body = {pool.left.record_id: 0.0, pool.observed_partner.record_id: observed}
    for index, item in enumerate(pool.admitted):
        body[item.record_id] = alternative(index) if callable(alternative) else alternative
    return body


def similarity(left, right):
    """A statistic of two payloads and nothing else. Deliberately blind to which is observed."""
    return float(right) - float(left)


# --- orientation is declared, never inferred ------------------------------------------------

def test_orientation_has_no_default_and_both_tails_are_named():
    assert set(ORIENTATIONS) == {"larger_is_more_similar", "smaller_is_more_similar"}
    for missing in (None, "", "   "):
        with pytest.raises(InvalidParameterError) as excinfo:
            require_declared_orientation(missing)
        assert "opposite tails" in str(excinfo.value)
    with pytest.raises(InvalidParameterError):
        require_declared_orientation("bigger_is_better")
    assert require_declared_orientation(" larger_is_more_similar ") == "larger_is_more_similar"


def test_the_two_orientations_invert_the_result_on_identical_numbers():
    """The reason orientation cannot be defaulted: the same values give opposite answers."""
    pool = build()
    records = payloads(pool, observed=10.0, alternative=lambda i: float(i))
    high = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                   orientation="larger_is_more_similar")
    low = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                  orientation="smaller_is_more_similar")
    assert high.p_value != low.p_value
    # Every alternative is either strictly above, strictly below, or tied with the observation,
    # so the two tails and the ties partition the reference set exactly.
    assert high.n_strictly_more_extreme + low.n_strictly_more_extreme + high.n_tied \
        == len(pool.admitted)


# --- the exact arithmetic -------------------------------------------------------------------

def test_p_value_is_the_exact_rank_with_the_observation_in_its_own_reference_set():
    pool = build()
    n = len(pool.admitted)
    # Ten alternatives strictly beat the observation; the rest fall below it.
    records = payloads(pool, observed=10.0,
                       alternative=lambda i: 20.0 if i < 10 else 1.0)
    result = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.n_strictly_more_extreme == 10
    assert result.n_tied == 0
    assert result.reference_size == n + 1
    assert result.p_value == pytest.approx(11.0 / (n + 1))
    assert result.p_floor == pytest.approx(1.0 / (n + 1))


def test_an_unbeaten_observation_attains_the_floor_and_never_reports_zero():
    pool = build()
    records = payloads(pool, observed=10.0, alternative=0.0)
    result = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.n_strictly_more_extreme == 0 and result.n_tied == 0
    assert result.p_value == result.p_floor > 0.0
    assert result.describe()["attained_floor"] is True


def test_ties_count_toward_the_numerator_because_indistinguishable_is_not_beaten():
    pool = build()
    n = len(pool.admitted)
    records = payloads(pool, observed=10.0, alternative=lambda i: 10.0 if i < 4 else 0.0)
    result = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.n_strictly_more_extreme == 0 and result.n_tied == 4
    assert result.p_value == pytest.approx(5.0 / (n + 1))


def test_a_statistic_that_separates_nothing_is_reported_as_degenerate_not_as_a_pass():
    """`p = 1.0` from an undiscriminating statistic is a foregone conclusion, not a safeguard."""
    pool = build()
    records = payloads(pool, observed=3.0, alternative=3.0)
    result = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.p_value == 1.0
    assert result.degenerate is True
    assert result.describe()["degenerate"] is True


def test_alternatives_are_reported_in_pool_order_with_the_ids_that_produced_them():
    pool = build()
    records = payloads(pool, observed=1.0, alternative=lambda i: float(i))
    result = exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.alternative_ids == tuple(item.record_id for item in pool.admitted)
    assert result.alternative_statistics == tuple(float(i) for i in range(len(pool.admitted)))


# --- one statistic path -----------------------------------------------------------------------

def test_the_observed_statistic_cannot_be_supplied_and_travels_the_same_path():
    """An observed value computed elsewhere may carry a different normalisation or code version."""
    parameters = set(dataclasses.asdict(build().left))  # touch the pool API without asserting it
    assert parameters  # sanity
    pool = build()
    seen = []

    def recording(left, right):
        seen.append((left, right))
        return similarity(left, right)

    result = exact_pool_substitution(pool=pool, records=payloads(pool), statistic=recording,
                                     orientation="larger_is_more_similar",
                                     verify_determinism=False)
    assert len(seen) == len(pool.admitted) + 1
    assert all(call[0] == seen[0][0] for call in seen)  # the same left payload every time
    assert result.reference_size == len(seen)


def test_a_precomputed_observed_statistic_is_not_an_accepted_argument():
    pool = build()
    with pytest.raises(TypeError):
        exact_pool_substitution(pool=pool, records=payloads(pool), statistic=similarity,
                                orientation="larger_is_more_similar", observed_statistic=9.9)


def test_a_non_deterministic_statistic_is_refused_rather_than_ranked():
    pool = build()
    counter = {"n": 0}

    def drifting(left, right):
        counter["n"] += 1
        return float(right) + counter["n"] * 1e-9

    with pytest.raises(NullRefusal) as excinfo:
        exact_pool_substitution(pool=pool, records=payloads(pool), statistic=drifting,
                                orientation="larger_is_more_similar")
    assert "not exact" in str(excinfo.value)


# --- refusals that keep the denominator honest -------------------------------------------------

def test_a_missing_record_payload_is_refused_rather_than_skipped():
    """Skipping shrinks an exact denominator without changing the digest it is reported against."""
    pool = build()
    records = payloads(pool)
    del records[pool.admitted[3].record_id]
    with pytest.raises(NullRefusal) as excinfo:
        exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                orientation="larger_is_more_similar")
    assert pool.admitted[3].record_id in str(excinfo.value)
    assert "anticonservative" in str(excinfo.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_alternative_is_refused_rather_than_dropped(bad):
    pool = build()
    records = payloads(pool)
    records[pool.admitted[0].record_id] = bad
    with pytest.raises(NullRefusal) as excinfo:
        exact_pool_substitution(pool=pool, records=records, statistic=similarity,
                                orientation="larger_is_more_similar")
    assert "resolution the pool does not have" in str(excinfo.value)


def test_a_statistic_that_is_not_real_valued_is_refused():
    pool = build()
    with pytest.raises(NullRefusal):
        exact_pool_substitution(pool=pool, records=payloads(pool),
                                statistic=lambda a, b: object(),
                                orientation="larger_is_more_similar")


def test_an_empty_pool_is_refused_because_p_would_be_one_by_arithmetic():
    pool = dataclasses.replace(build(), admitted=())
    with pytest.raises(NullRefusal) as excinfo:
        exact_pool_substitution(pool=pool, records={"LEFT": 0.0, "PARTNER": 1.0},
                                statistic=similarity, orientation="larger_is_more_similar")
    assert "foregone conclusion" in str(excinfo.value)


def test_the_observed_partner_appearing_in_its_own_pool_is_refused():
    """`build_partner_pool` excludes it; `PartnerPool` is public, so the null checks again."""
    pool = build()
    pool = dataclasses.replace(pool, admitted=pool.admitted + (pool.observed_partner,))
    with pytest.raises(NullRefusal) as excinfo:
        exact_pool_substitution(pool=pool, records=payloads(pool), statistic=similarity,
                                orientation="larger_is_more_similar")
    assert "count it twice" in str(excinfo.value)


def test_a_non_callable_statistic_is_refused_by_name():
    pool = build()
    with pytest.raises(InvalidParameterError) as excinfo:
        exact_pool_substitution(pool=pool, records=payloads(pool), statistic=3.5,
                                orientation="larger_is_more_similar")
    assert "same code path" in str(excinfo.value)


# --- sampling is refused by name ---------------------------------------------------------------

def test_monte_carlo_pool_substitution_is_registered_and_refused_with_its_measurement():
    with pytest.raises(NullRefusal) as excinfo:
        monte_carlo_pool_substitution()
    message = str(excinfo.value)
    assert "sealed finite pool" in message
    assert "chosen by" in message
    assert "1/31" in monte_carlo_pool_substitution.__doc__


# --- the family, at the size it was sealed at ---------------------------------------------------

def family_pools(m=M, count=58):
    return [build(left_id="L%d" % i, partner_id="P%d" % i, first=2 + 100 * i, count=count, m=m)
            for i in range(m)]


def family_records(pools, observed=10.0, alternative=0.0):
    body = {}
    for pool in pools:
        body.update(payloads(pool, observed=observed, alternative=alternative))
    return body


def test_a_family_corrects_once_at_the_declared_size():
    pools = family_pools()
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    assert isinstance(family, CorrespondenceFamily)
    assert family.tested_correspondences == M == len(family.results)
    assert family.correction == CORRECTION and family.alpha == ALPHA
    assert family.n_rejected == M  # every member sits at its own floor
    assert family.describe()["schema"] == FAMILY_SCHEMA
    assert family.describe()["members"][0]["schema"] == NULL_SCHEMA


def test_a_family_narrowed_after_the_pools_were_sealed_is_refused():
    """The 'run 500, correct the surviving 20' failure, made inexpressible rather than warned about.

    The family size was fixed inside every pool digest before a statistic existed, so presenting
    fewer correspondences than were declared contradicts a number that predates the p-values.
    """
    pools = family_pools()
    with pytest.raises(NullRefusal) as excinfo:
        correspondence_family(pools=pools[:3], records=family_records(pools),
                              statistic=similarity, orientation="larger_is_more_similar")
    assert "Narrowing" in str(excinfo.value) and "selection" in str(excinfo.value)


def test_a_family_enlarged_beyond_its_declared_size_is_refused_too():
    pools = family_pools()
    extra = build(left_id="LX", partner_id="PX", first=9000, count=58, m=M)
    with pytest.raises(NullRefusal) as excinfo:
        correspondence_family(pools=pools + [extra],
                              records=family_records(pools + [extra]),
                              statistic=similarity, orientation="larger_is_more_similar")
    assert "Enlarging" in str(excinfo.value)


def test_pools_declaring_different_family_sizes_are_not_one_family():
    pools = family_pools()
    pools[2] = build(left_id="L2", partner_id="P2", first=202, count=58, m=M + 1)
    with pytest.raises(NullRefusal) as excinfo:
        correspondence_family(pools=pools, records=family_records(pools), statistic=similarity,
                              orientation="larger_is_more_similar")
    assert "different multiplicity burdens" in str(excinfo.value)


def test_the_same_correspondence_presented_twice_is_refused():
    pools = family_pools()
    pools[4] = pools[1]
    with pytest.raises(NullRefusal) as excinfo:
        correspondence_family(pools=pools, records=family_records(pools), statistic=similarity,
                              orientation="larger_is_more_similar")
    assert "one hypothesis twice" in str(excinfo.value)


def test_an_empty_family_is_refused():
    with pytest.raises(InvalidParameterError):
        correspondence_family(pools=[], records={}, statistic=similarity,
                              orientation="larger_is_more_similar")


def test_the_correction_is_benjamini_yekutieli_and_costs_something():
    """BY rather than BH is load-bearing: family members share candidate inventories."""
    pools = family_pools()
    records = family_records(pools)
    family = correspondence_family(pools=pools, records=records, statistic=similarity,
                                   orientation="larger_is_more_similar")
    assert family.correction == "benjamini_yekutieli"
    assert all(adj >= p for adj, p in zip(family.adjusted,
                                          [item.p_value for item in family.results]))
    under_bh = correspondence_family(pools=pools, records=records, statistic=similarity,
                                     orientation="larger_is_more_similar",
                                     correction="benjamini_hochberg")
    assert min(family.adjusted) > min(under_bh.adjusted)


# --- resolution is measured again, not assumed --------------------------------------------------

def test_resolution_is_verified_against_the_real_correction_not_inherited_from_the_builder():
    pools = family_pools()
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    resolution = family.resolution()
    assert resolution["every_member_can_reject_at_its_own_floor"] is True
    assert resolution["unresolvable_members"] == []
    assert resolution["required_pool_size"] == minimum_pool_size(M)
    assert resolution["worst_p_floor"] == max(item.p_floor for item in family.results)


def test_a_directly_constructed_underpowered_pool_is_caught_by_the_family_resolution_check():
    """`PartnerPool` is a public dataclass, so the builder's refusal is not the only line."""
    pools = family_pools()
    starved = pools[0]
    pools[0] = dataclasses.replace(starved, admitted=starved.admitted[:3])
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    resolution = family.resolution()
    assert resolution["every_member_can_reject_at_its_own_floor"] is False
    assert [row["left"] for row in resolution["unresolvable_members"]] == [starved.left.record_id]
    assert resolution["unresolvable_members"][0]["required_pool_size"] == minimum_pool_size(M)


# --- what the receipt does and does not claim ---------------------------------------------------

def test_the_result_is_bound_to_the_pool_and_contract_digests_it_was_computed_against():
    pool = build()
    result = exact_pool_substitution(pool=pool, records=payloads(pool), statistic=similarity,
                                     orientation="larger_is_more_similar")
    assert result.pool_sha256 == pool.digest
    assert result.contract_sha256 == pool.contract.digest
    moved = dataclasses.replace(pool, admitted=pool.admitted[:-1])
    assert moved.digest != result.pool_sha256


def test_the_family_digest_moves_when_any_reported_number_moves():
    pools = family_pools()
    first = correspondence_family(pools=pools, records=family_records(pools),
                                  statistic=similarity, orientation="larger_is_more_similar")
    second = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    assert first.digest == second.digest
    shifted = correspondence_family(
        pools=pools,
        records=family_records(pools, observed=10.0, alternative=lambda i: 99.0 if i < 5 else 0.0),
        statistic=similarity, orientation="larger_is_more_similar")
    assert shifted.digest != first.digest


def test_the_claim_boundary_refuses_to_call_this_a_calibration():
    pools = family_pools()
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    boundary = family.describe()["claim_boundary"]
    assert "necessary and not a sufficient" in boundary
    assert "NOT a calibration" in boundary
    assert "no false-positive rate has been measured" in boundary
    assert "exchangeable" in boundary


def test_the_receipt_states_that_nothing_was_sampled():
    pool = build()
    body = exact_pool_substitution(pool=pool, records=payloads(pool), statistic=similarity,
                                   orientation="larger_is_more_similar").describe()
    assert body["sampling"].startswith("none")
    assert body["inference"] == "exact_pool_substitution"
    assert body["orientation_meaning"] == ORIENTATIONS["larger_is_more_similar"]


def test_a_substitution_result_carries_no_quantity_the_pool_was_admitted_on():
    """Slice 2's structural defence read forward: admission saw marginals, inference sees records.

    The two must not meet. A `SubstitutionResult` records statistics and identifiers, never a
    marginal, so no downstream step can re-admit a pool using a number the inference produced.
    """
    fields = {f.name for f in dataclasses.fields(SubstitutionResult)}
    for marginal in ("n_samples", "effective_sample_size", "native_seconds", "cadence_seconds",
                     "coverage_fraction", "noise_floor"):
        assert marginal not in fields, marginal


# --- the resolution check that reports the world it is actually powered for ---------------------

def test_a_family_clearing_the_required_pool_size_still_names_the_sparsity_it_needs():
    """The defect this test exists for: `minimum_pool_size` sizes for the all-genuine world.

    Six pools of 58 each clear `minimum_pool_size(6) = 48`, and every member can reject when all
    six sit at their floors. But members that do not correspond consume the Benjamini-Yekutieli
    step-up ranks the genuine ones need, so this family can reject nothing unless five of its six
    correspondences are real. Reporting only the all-genuine case would show a green light for a
    family that cannot produce a finding.
    """
    pools = family_pools()
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    resolution = family.resolution()
    assert resolution["every_member_can_reject_at_its_own_floor"] is True
    assert resolution["sparsest_detectable_count"] == 5
    assert resolution["sparsest_detectable_fraction"] == pytest.approx(5.0 / 6.0)
    assert "at least 5 of its 6" in family.describe()["powered_for"]


def test_three_genuine_correspondences_of_six_do_not_reject_and_the_receipt_predicted_it():
    """The measured consequence: three members at the exact floor, and zero rejections."""
    pools = family_pools()
    records = {}
    for index, pool in enumerate(pools):
        records.update(payloads(pool, observed=10.0 if index < 3 else -10.0, alternative=0.0))
    family = correspondence_family(pools=pools, records=records, statistic=similarity,
                                   orientation="larger_is_more_similar")
    at_floor = [item for item in family.results if item.p_value == item.p_floor]
    assert len(at_floor) == 3
    assert family.n_rejected == 0
    assert family.resolution()["sparsest_detectable_count"] == 5 > 3


def test_the_pool_size_that_would_be_needed_for_half_the_family_is_reported():
    pools = family_pools()
    family = correspondence_family(pools=pools, records=family_records(pools),
                                   statistic=similarity, orientation="larger_is_more_similar")
    needed = family.resolution()["pool_size_for_half_the_family"]
    assert needed == minimum_pool_size_for_detected_fraction(M, 0.5)
    assert needed > minimum_pool_size(M)  # sizing for the all-genuine world understates it


def test_a_family_that_can_never_reject_says_so_rather_than_reporting_a_number():
    starved = [dataclasses.replace(pool, admitted=pool.admitted[:2]) for pool in family_pools()]
    records = family_records(starved)
    family = correspondence_family(pools=starved, records=records, statistic=similarity,
                                   orientation="larger_is_more_similar")
    resolution = family.resolution()
    assert resolution["sparsest_detectable_count"] is None
    assert "cannot reject at any effect size" in resolution["sparsity_basis"]
    assert "cannot produce a rejection at any effect size" in family.describe()["powered_for"]
