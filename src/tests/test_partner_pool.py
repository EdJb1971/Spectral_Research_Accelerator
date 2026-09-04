"""TG17.15 slice 2: exchangeability made checkable, and circularity made inexpressible."""

from __future__ import annotations

import dataclasses

import pytest

from src.core.correspondence_estimand import minimum_pool_size
from src.core.errors import InvalidParameterError
from src.core.partner_pool import (
    BANDED_MARGINALS, POOL_SCHEMA, UNBANDED_MARGINALS, AdmissionContract, PartnerPool,
    PoolAdmissionRefusal, RecordProfile, build_partner_pool,
)


CONTRACT = AdmissionContract(ratio_bands={
    "n_samples": 2.0, "effective_sample_size": 2.0, "noise_floor": 3.0, "cadence_seconds": 4.0})


def profile(index, **changes) -> RecordProfile:
    values = dict(record_id="r%03d" % index, provenance_key="src%03d" % index,
                  n_samples=1000, effective_sample_size=400.0,
                  # Native duration spans two orders of magnitude across the pool on purpose.
                  native_seconds=900.0 * (1.0 + index % 37),
                  cadence_seconds=60.0, coverage_fraction=0.95, noise_floor=0.1)
    values.update(changes)
    return RecordProfile(**values)


LEFT = profile(0, record_id="LEFT", provenance_key="srcLEFT")
PARTNER = profile(1, record_id="PARTNER", provenance_key="srcPARTNER")


def pool(candidates=None, *, contract=CONTRACT, m=6, left=LEFT, partner=PARTNER):
    if candidates is None:
        candidates = [profile(i) for i in range(2, 60)]
    return build_partner_pool(left=left, observed_partner=partner, candidates=candidates,
                              contract=contract, estimand="per_correspondence",
                              tested_correspondences=m)


# --- the structural defence: circularity is not expressible --------------------------------

def test_a_record_profile_carries_no_quantity_computed_from_two_records():
    """The defence is structural, not disciplinary.

    Admission sees `RecordProfile` and never a record, and every field of a profile is computable
    from one record alone. If a joint quantity -- a similarity, a distance, a correlation -- could
    live on a profile, an admission rule could key on the statistic under test and no amount of
    care downstream would prevent it.
    """
    fields = {f.name for f in dataclasses.fields(RecordProfile)}
    assert fields == {"record_id", "provenance_key"} | set(BANDED_MARGINALS) \
        | set(UNBANDED_MARGINALS)
    for forbidden in ("similarity", "distance", "correlation", "score", "affinity", "statistic",
                      "match", "against", "versus", "pair"):
        assert not any(forbidden in name for name in fields), forbidden


def test_every_banded_marginal_is_actually_readable_from_a_profile():
    item = profile(5)
    for name in BANDED_MARGINALS + UNBANDED_MARGINALS:
        assert isinstance(item.marginal(name), float)
    with pytest.raises(InvalidParameterError, match="one of"):
        item.marginal("similarity_to_left")


def test_native_duration_cannot_be_banded_because_it_is_what_the_mode_compares():
    """Getting this backwards would quietly convert scale/shape mode into something else."""
    with pytest.raises(InvalidParameterError, match="deliberately"):
        AdmissionContract(ratio_bands={"native_seconds": 2.0})
    admitted = pool()
    spread = admitted.spread()["native_seconds"]
    assert spread["pool_max"] / spread["pool_min"] > 10.0


# --- the observation must clear the bands its own alternatives clear -------------------------

def test_an_observed_partner_that_fails_its_own_contract_is_refused():
    """Otherwise the reference set is not exchangeable with the thing it is a reference for."""
    thin = profile(1, record_id="PARTNER", provenance_key="srcPARTNER", coverage_fraction=0.2)
    with pytest.raises(PoolAdmissionRefusal, match="observed partner"):
        pool(partner=thin)


def test_a_partner_sharing_the_left_members_provenance_is_refused_as_leakage():
    leaked = profile(1, record_id="PARTNER", provenance_key="srcLEFT")
    with pytest.raises(PoolAdmissionRefusal, match="shares provenance"):
        pool(partner=leaked)


def test_a_record_cannot_be_its_own_partner():
    with pytest.raises(InvalidParameterError, match="maximal similarity by construction"):
        pool(partner=LEFT)


# --- nothing is ever dropped silently ---------------------------------------------------------

def test_every_refused_candidate_is_carried_with_the_band_it_failed_and_by_how_much():
    """A pool that quietly shrinks is a pool whose resolution nobody can audit."""
    candidates = [profile(i) for i in range(2, 60)]
    candidates.append(profile(900, record_id="TOO_SHORT", n_samples=50,
                              effective_sample_size=20.0))
    candidates.append(profile(901, record_id="LEAKED", provenance_key="srcLEFT"))
    candidates.append(profile(902, record_id="UNCOVERED", coverage_fraction=0.05))
    built = pool(candidates)

    assert len(built.admitted) + len(built.refused) == len(candidates)
    refused = {row["record_id"]: row["reasons"] for row in built.refused}
    assert set(refused) == {"TOO_SHORT", "LEAKED", "UNCOVERED"}
    assert any(r["check"] == "effective_sample_size" for r in refused["TOO_SHORT"])
    assert any(r["check"] == "provenance_independence" for r in refused["LEAKED"])
    assert any(r["check"] == "minimum_coverage" for r in refused["UNCOVERED"])
    # The statement must carry the measured numbers, not just the verdict.
    statement = [r for r in refused["TOO_SHORT"] if r["check"] == "effective_sample_size"][0]
    assert "20" in statement["statement"] and "400" in statement["statement"]


def test_the_left_member_and_the_observed_partner_are_refused_from_their_own_pool():
    candidates = [profile(i) for i in range(2, 60)] + [LEFT, PARTNER]
    built = pool(candidates)
    refused = {row["record_id"] for row in built.refused}
    assert refused == {"LEFT", "PARTNER"}
    assert all(item.record_id not in {"LEFT", "PARTNER"} for item in built.admitted)


def test_one_record_offered_twice_is_refused_rather_than_counted_twice():
    candidates = [profile(i) for i in range(2, 60)]
    candidates.append(profile(5))
    with pytest.raises(InvalidParameterError, match="one entry per record"):
        pool(candidates)


# --- the pool must be able to resolve ----------------------------------------------------------

def test_a_pool_too_small_to_reject_is_refused_rather_than_returned():
    candidates = [profile(i) for i in range(2, 20)]
    with pytest.raises(PoolAdmissionRefusal, match="cannot resolve"):
        pool(candidates, m=6)


def test_the_required_size_comes_from_the_declared_number_of_tested_correspondences():
    candidates = [profile(i) for i in range(2, 30)]
    built = pool(candidates, m=1)
    assert built.describe()["minimum_pool_size"] == minimum_pool_size(1) == 19
    with pytest.raises(PoolAdmissionRefusal, match="at least 48"):
        pool(candidates, m=6)


def test_the_resolution_floor_counts_the_observation_alongside_its_alternatives():
    built = pool()
    assert built.resolution_floor == pytest.approx(1.0 / (len(built.admitted) + 1))


# --- declarations, seals and what the pool refuses to claim -------------------------------------

def test_the_estimand_must_be_declared_and_the_inadmissible_one_is_refused():
    with pytest.raises(InvalidParameterError, match="no default question"):
        build_partner_pool(left=LEFT, observed_partner=PARTNER,
                           candidates=[profile(i) for i in range(2, 60)],
                           contract=CONTRACT, estimand=None, tested_correspondences=6)
    with pytest.raises(InvalidParameterError, match="refused by name"):
        build_partner_pool(left=LEFT, observed_partner=PARTNER,
                           candidates=[profile(i) for i in range(2, 60)],
                           contract=CONTRACT, estimand="joint_structure",
                           tested_correspondences=6)


def test_a_contract_with_no_declared_band_is_refused():
    with pytest.raises(InvalidParameterError, match="at least one declared band"):
        AdmissionContract(ratio_bands={})


def test_a_band_below_one_is_refused_because_it_could_admit_nothing():
    with pytest.raises(InvalidParameterError, match="at least 1.0"):
        AdmissionContract(ratio_bands={"n_samples": 0.5})


def test_an_effective_sample_size_above_the_nominal_count_is_refused():
    with pytest.raises(InvalidParameterError, match="autocorrelation had added evidence"):
        profile(3, n_samples=100, effective_sample_size=200.0)


def test_the_pool_is_sealed_and_its_digest_moves_when_a_member_does():
    first = pool()
    assert first.digest == pool().digest
    changed = [profile(i) for i in range(2, 60)]
    changed[0] = profile(2, noise_floor=0.11)
    assert pool(changed).digest != first.digest
    assert first.contract.digest == CONTRACT.digest


def test_the_spread_shows_where_the_observed_partner_sits_inside_its_own_pool():
    """Reported, not checked: the partner is inside every band by construction.

    A partner at the edge of its pool on some marginal is exactly where an undeclared property is
    most likely to be doing the work, and a reader who cannot see it cannot judge it.
    """
    built = pool()
    spread = built.spread()
    assert set(spread) == set(BANDED_MARGINALS) | set(UNBANDED_MARGINALS)
    for name, row in spread.items():
        assert row["pool_min"] <= row["pool_median"] <= row["pool_max"]
        assert 0.0 <= row["observed_partner_percentile"] <= 100.0


def test_the_pool_says_admission_is_necessary_and_not_sufficient():
    """A pool that reads as a proof of exchangeability is worse than no pool."""
    body = pool().describe()
    assert body["schema"] == POOL_SCHEMA
    assert "NECESSARY condition" in body["claim_boundary"]
    assert "not a sufficient one" in body["claim_boundary"]
    assert "cannot know about it" in body["claim_boundary"]
    assert "never on any quantity computed between" in body["exchangeability_basis"]


def test_the_receipt_carries_every_refusal_and_the_contract_that_produced_them():
    candidates = [profile(i) for i in range(2, 60)]
    candidates.append(profile(903, record_id="NOISY", noise_floor=5.0))
    body = pool(candidates).describe()
    assert body["n_refused"] == 1
    assert body["refused"][0]["record_id"] == "NOISY"
    assert body["contract"]["contract_sha256"] == CONTRACT.digest
    assert body["contract"]["unbanded"] == list(UNBANDED_MARGINALS)
    assert body["estimand"] == "per_correspondence"
    assert isinstance(body["pool_sha256"], str) and len(body["pool_sha256"]) == 64
