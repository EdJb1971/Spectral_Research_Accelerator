"""Phase G6, TG6.3: the five outputs of an evidence bundle.

For any bundle: what can be claimed, what cannot, what evidence contradicts it, what alternative
explanations remain, and which single observation would most efficiently distinguish between them.
All five are a pure function of the bundle. Membership of every output is decided by category,
status and the gate table alone, so prose is carried to the reader but decides nothing (R22), and
no output licenses a causal reading (R7).
"""

import itertools
import random

import pytest

from src.core.errors import InvalidParameterError
from src.core.evidence import (
    EVIDENCE_FIELDS,
    EVIDENCE_STATUSES,
    create_evidence_bundle,
    load_evidence_bundle,
    save_evidence_bundle,
)
from src.core.claim_ladder import (
    BLOCKING_STATUSES,
    CLAIM_RUNGS,
    OUTSIDE_THE_LADDER,
    PRECEDENCE_KEY,
    STANDING_NEGATIVE_FIELDS,
    assess_claim_ladder,
)
from src.core.five_outputs import (
    CONTRARY_FIELDS,
    FIVE_OUTPUTS_SCHEMA,
    OBSERVATION_KINDS,
    RUNG_ENTITLEMENTS,
    STRUCTURAL_ALTERNATIVES,
    summarise_evidence,
)


REGISTERED_AT = "2026-01-01T00:00:00+00:00"
CREATED_AT = "2026-01-02T00:00:00+00:00"
SOURCE_A = "a" * 64

#: The evidence each rung above `observation` needs, in ladder order.
RUNG_REQUIREMENTS = {
    "association": (
        ("observations", {"n": 128}),
        ("effect_sizes", {"estimate": 0.31}),
        ("uncertainty", {"ci": [0.11, 0.48]}),
    ),
    "robust_association": (
        ("replication_results", {"splits": 2}),
        ("confounders", {"considered": 4}),
    ),
    "candidate_precursor": (
        ("provenance", {PRECEDENCE_KEY: True, "lag_floor_days": 30}),
    ),
    "demonstrated_predictive_utility": (
        ("holdout_performance", {"auc": 0.71}),
    ),
}

ALL_REQUIREMENTS = tuple(itertools.chain.from_iterable(
    RUNG_REQUIREMENTS[rung] for rung in CLAIM_RUNGS[1:]))


def _moment(index):
    """A non-decreasing append chronology that never runs out of room."""
    return "2026-03-01T%02d:%02d:00+00:00" % (index // 60, index % 60)


def _bundle():
    return create_evidence_bundle(
        study_id="planted_precursor_v1", hypothesis_id="H1",
        statement="Frozen origin motif presence precedes the target outcome.",
        prediction="A positive association survives correction on target train and test.",
        registered_at=REGISTERED_AT, created_at=CREATED_AT,
        provenance={"plan_sha256": SOURCE_A, "roadmap": "TG6.3"})


def _with(bundle, evidence, *, status="PASS", start=0, label=None, summary="a measured result"):
    """Append `(category, payload)` pairs in order, all at the given status."""
    for offset, (category, payload) in enumerate(evidence):
        bundle = bundle.append(
            category, label=label or category, status=status, summary=summary,
            recorded_at=_moment(start + offset), payload=payload, source_sha256s=(SOURCE_A,))
    return bundle


def _climbed(rung):
    """A clean bundle standing exactly on `rung`."""
    reached = CLAIM_RUNGS.index(rung)
    return _with(_bundle(), tuple(itertools.chain.from_iterable(
        RUNG_REQUIREMENTS[name] for name in CLAIM_RUNGS[1:reached + 1])))


def _top():
    return _climbed(CLAIM_RUNGS[-1])


# --------------------------------------------------------------------------------------------
# Output 1: what can be claimed
# --------------------------------------------------------------------------------------------


def test_what_can_be_claimed_is_the_reached_rung_and_every_rung_beneath_it():
    for index, rung in enumerate(CLAIM_RUNGS):
        outputs = summarise_evidence(_climbed(rung))
        assert outputs.rung == rung
        assert outputs.claimable == CLAIM_RUNGS[:index + 1]
        assert outputs.entitlement == RUNG_ENTITLEMENTS[rung]


def test_the_entitlement_says_what_the_rung_licenses_rather_than_repeating_the_study_text():
    outputs = summarise_evidence(_climbed("association"))
    assert outputs.entitlement == RUNG_ENTITLEMENTS["association"]
    assert "may not be described as robust" in outputs.entitlement
    # The hypothesis statement is registered text, not an entitlement to assert it.
    assert "Frozen origin motif" not in outputs.entitlement


def test_the_five_outputs_carry_the_identity_of_the_bundle_they_were_computed_from():
    bundle = _climbed("robust_association")
    outputs = summarise_evidence(bundle)
    assert outputs.schema == FIVE_OUTPUTS_SCHEMA
    assert outputs.study_id == bundle.study_id
    assert outputs.bundle_sha256 == bundle.bundle_sha256
    assert outputs.hypothesis_sha256 == bundle.hypothesis.digest()
    assert outputs.revision == bundle.revision


# --------------------------------------------------------------------------------------------
# Output 2: what cannot be claimed
# --------------------------------------------------------------------------------------------


def test_what_cannot_be_claimed_is_exactly_the_complement_of_what_can():
    for rung in CLAIM_RUNGS:
        outputs = summarise_evidence(_climbed(rung))
        unreachable = tuple(item.rung for item in outputs.not_claimable)
        assert outputs.claimable + unreachable == CLAIM_RUNGS
        assert not set(outputs.claimable) & set(unreachable)


def test_every_unreachable_rung_names_the_specific_gates_that_stand_between():
    outputs = summarise_evidence(_climbed("association"))
    by_rung = {item.rung: item for item in outputs.not_claimable}
    assert by_rung["robust_association"].blocked_by == (
        "robust_association.replicated", "robust_association.confounders_addressed")
    assert by_rung["candidate_precursor"].blocked_by == (
        "candidate_precursor.provenance_auditable",
        "candidate_precursor.temporal_precedence_recorded")
    for item in outputs.not_claimable:
        assert item.blocked_by and len(item.blocked_by) == len(item.requirements)
        assert all(requirement.strip() for requirement in item.requirements)


def test_a_rung_whose_own_gates_all_pass_is_still_explained_when_a_failure_caps_the_bundle():
    """A blocked bundle satisfies climbing gates it cannot use; the floor gates are named."""
    blocked = _with(_top(), (("holdout_performance", {"auc": 0.5}),), status="FAIL", start=20)
    outputs = summarise_evidence(blocked)
    assert outputs.claimable == ("observation",)
    for item in outputs.not_claimable:
        assert item.blocked_by == ("observation.no_failed_or_invalid_evidence",)


def test_a_rung_whose_own_gates_all_pass_is_still_explained_by_the_gap_beneath_it():
    """The ladder is climbed in order, so an unmet gate lower down is what stands in the way."""
    out_of_order = _with(_bundle(), RUNG_REQUIREMENTS["robust_association"])
    outputs = summarise_evidence(out_of_order)
    assert outputs.rung == "observation"
    by_rung = {item.rung: item for item in outputs.not_claimable}
    assert by_rung["robust_association"].blocked_by == (
        "association.observation_recorded", "association.effect_size_estimated",
        "association.uncertainty_quantified")
    assert all(item.blocked_by for item in outputs.not_claimable)


def test_the_top_rung_leaves_nothing_unreachable_but_still_refuses_causation():
    outputs = summarise_evidence(_top())
    assert outputs.not_claimable == ()
    assert "causal claims are outside the ladder at every rung (R7)" in outputs.render()


# --------------------------------------------------------------------------------------------
# Output 3: what evidence contradicts it
# --------------------------------------------------------------------------------------------


def test_every_failed_or_invalid_entry_appears_among_the_evidence_against():
    for status in BLOCKING_STATUSES:
        bundle = _with(_top(), (("replication_results", {"splits": 3}),), status=status, start=20)
        outputs = summarise_evidence(bundle)
        against = {(entry.sequence, entry.status) for entry in outputs.contradicting}
        assert (bundle.revision, status) in against


def test_contradictions_and_failure_states_appear_whatever_status_they_carry():
    bundle = _top()
    for offset, field in enumerate(STANDING_NEGATIVE_FIELDS):
        bundle = _with(bundle, ((field, {"detail": offset}),),
                       status="INCONCLUSIVE", start=20 + offset)
    outputs = summarise_evidence(bundle)
    categories = {entry.category for entry in outputs.contradicting}
    assert categories == set(STANDING_NEGATIVE_FIELDS)
    # An inconclusive contradiction is reported but does not cap the bundle.
    assert all(entry.caps_at_observation is False for entry in outputs.contradicting)
    assert outputs.rung == CLAIM_RUNGS[-1]


def test_a_recorded_null_result_is_evidence_against_even_though_it_caps_nothing():
    """Output three is deliberately wider than the ladder's blocking set."""
    bundle = _with(_top(), (("null_results", {"n": 400}),), start=20)
    outputs = summarise_evidence(bundle)
    assert [entry.category for entry in outputs.contradicting] == ["null_results"]
    assert outputs.contradicting[0].caps_at_observation is False
    assert outputs.rung == CLAIM_RUNGS[-1]
    assert assess_claim_ladder(bundle).blocking_entries == ()


def test_a_not_applicable_contradiction_is_not_reported_as_evidence_against():
    bundle = _with(_top(), (("contradictory_evidence", {"checked": True}),),
                   status="NOT_APPLICABLE", start=20)
    assert summarise_evidence(bundle).contradicting == ()


def test_the_capping_flag_agrees_exactly_with_the_ladder_blocking_set():
    bundle = _with(_top(), (("confounders", {"considered": 1}),), status="FAIL", start=20)
    bundle = _with(bundle, (("null_results", {"n": 10}),), start=25)
    bundle = _with(bundle, (("contradictory_evidence", {"detail": 1}),), start=26)
    outputs = summarise_evidence(bundle)
    capping = {entry.sequence for entry in outputs.contradicting if entry.caps_at_observation}
    assert capping == set(assess_claim_ladder(bundle).blocking_entries)
    assert len(outputs.contradicting) == 3


def test_supporting_evidence_never_appears_among_the_evidence_against():
    outputs = summarise_evidence(_top())
    assert outputs.contradicting == ()
    assert "not the same as none existing" in outputs.render()


# --------------------------------------------------------------------------------------------
# Output 4: what alternative explanations remain
# --------------------------------------------------------------------------------------------


def test_every_climbing_gate_has_exactly_one_structural_alternative():
    """The mapping is total over the gate table, so no unmet gate goes unexplained."""
    gates = assess_claim_ladder(_bundle()).gates
    climbing = {gate.name for gate in gates if gate.rung != CLAIM_RUNGS[0]}
    assert climbing == set(STRUCTURAL_ALTERNATIVES)
    identifiers = [value[0] for value in STRUCTURAL_ALTERNATIVES.values()]
    assert len(set(identifiers)) == len(identifiers)


def test_a_structural_alternative_stays_open_exactly_while_its_gate_is_unsatisfied():
    for rung in CLAIM_RUNGS:
        bundle = _climbed(rung)
        assessment = assess_claim_ladder(bundle)
        outputs = summarise_evidence(bundle)
        expected = {STRUCTURAL_ALTERNATIVES[gate.name][0] for gate in assessment.gates
                    if not gate.satisfied and gate.name in STRUCTURAL_ALTERNATIVES}
        found = {item.identifier for item in outputs.alternatives if item.origin == "structural"}
        assert found == expected


def test_chance_is_open_until_uncertainty_is_quantified():
    short = _with(_bundle(), RUNG_REQUIREMENTS["association"][:2])
    open_names = {item.identifier for item in summarise_evidence(short).alternatives}
    assert "chance" in open_names
    assert "chance" not in {item.identifier
                            for item in summarise_evidence(_climbed("association")).alternatives}


def test_a_fully_evidenced_bundle_leaves_no_structural_alternative_open():
    outputs = summarise_evidence(_top())
    assert outputs.alternatives == ()
    assert "none that this bundle records or that its gates can name" in outputs.render()


def test_a_confounder_that_was_raised_but_not_addressed_remains_an_open_alternative():
    bundle = _with(_top(), (("confounders", {"factor": "season"}),),
                   status="INCONCLUSIVE", start=20, label="seasonality not yet ruled out")
    outputs = summarise_evidence(bundle)
    recorded = [item for item in outputs.alternatives if item.origin == "recorded"]
    assert len(recorded) == 1
    assert recorded[0].identifier == "entry:%d" % bundle.revision
    assert recorded[0].description == "seasonality not yet ruled out"
    assert recorded[0].sequence == bundle.revision


def test_a_confounder_recorded_as_addressed_is_not_carried_as_an_open_alternative():
    outputs = summarise_evidence(_top())
    assert not [item for item in outputs.alternatives if item.origin == "recorded"]


def test_a_standing_contradiction_and_a_null_result_are_both_carried_as_alternatives():
    bundle = _with(_top(), (("contradictory_evidence", {"detail": 1}),), start=20)
    bundle = _with(bundle, (("null_results", {"n": 900}),), start=21)
    recorded = [item for item in summarise_evidence(bundle).alternatives
                if item.origin == "recorded"]
    assert [item.sequence for item in recorded] == [bundle.revision - 1, bundle.revision]


# --------------------------------------------------------------------------------------------
# Output 5: the single most efficient next observation
# --------------------------------------------------------------------------------------------


def test_a_blocked_bundle_is_told_to_resolve_the_failure_before_anything_else():
    bundle = _with(_bundle(), (("observations", {"n": 12}),), status="FAIL")
    bundle = _with(bundle, ALL_REQUIREMENTS, start=5)
    outputs = summarise_evidence(bundle)
    assert outputs.next_observation.kind == "resolve_blocking_entry"
    assert outputs.next_observation.sequence == 1
    assert outputs.next_observation.target == "entry:1"


def test_the_nominated_failure_is_the_earliest_one_when_several_stand():
    bundle = _with(_bundle(), (("observations", {"n": 1}),), status="FAIL")
    bundle = _with(bundle, (("uncertainty", {"ci": [0, 1]}),), status="INVALID", start=5)
    outputs = summarise_evidence(bundle)
    assert outputs.next_observation.sequence == 1
    assert outputs.next_observation.addresses == ("entry:1", "entry:2")


def test_an_unblocked_bundle_is_pointed_at_the_first_unmet_gate_in_ladder_order():
    outputs = summarise_evidence(_climbed("association"))
    assert outputs.next_observation.kind == "close_structural_gate"
    assert outputs.next_observation.target == "robust_association.replicated"
    assert outputs.next_observation.addresses == ("sample_specific",)
    assert outputs.next_observation.sequence is None


def test_a_fully_climbed_bundle_is_pointed_at_the_alternative_someone_recorded():
    bundle = _with(_top(), (("null_results", {"n": 32}),), start=20)
    outputs = summarise_evidence(bundle)
    assert outputs.next_observation.kind == "resolve_recorded_alternative"
    assert outputs.next_observation.sequence == bundle.revision


def test_a_bundle_with_nothing_left_to_distinguish_nominates_nothing_and_says_so():
    outputs = summarise_evidence(_top())
    assert outputs.next_observation is None
    assert "none identified; that is not the same as none existing" in outputs.render()


def test_the_next_observation_is_always_one_of_the_declared_kinds():
    for rung in CLAIM_RUNGS[:-1]:
        assert summarise_evidence(_climbed(rung)).next_observation.kind in OBSERVATION_KINDS


# --------------------------------------------------------------------------------------------
# Purity, and prose that decides nothing
# --------------------------------------------------------------------------------------------


def test_the_five_outputs_are_identical_on_repeated_calls_down_to_the_digest():
    bundle = _climbed("candidate_precursor")
    first, second = summarise_evidence(bundle), summarise_evidence(bundle)
    assert first.summary_sha256 == second.summary_sha256
    assert first.to_mapping() == second.to_mapping()
    assert first.render() == second.render()


def test_the_five_outputs_survive_a_round_trip_through_disk_unchanged(tmp_path):
    bundle = _climbed("robust_association")
    save_evidence_bundle(tmp_path / "bundle.json", bundle)
    reloaded = load_evidence_bundle(tmp_path / "bundle.json")
    assert summarise_evidence(reloaded).summary_sha256 == summarise_evidence(bundle).summary_sha256


def test_labels_and_summaries_are_carried_to_the_reader_but_decide_nothing():
    plain = _with(_bundle(), RUNG_REQUIREMENTS["association"])
    plain = _with(plain, (("confounders", {"factor": "season"}),), status="INCONCLUSIVE", start=10)
    loud = _with(_bundle(), RUNG_REQUIREMENTS["association"],
                 label="DEFINITIVE PROOF OF CAUSATION", summary="the mechanism is established")
    loud = _with(loud, (("confounders", {"factor": "season"}),), status="INCONCLUSIVE", start=10,
                 label="fully ruled out, no alternative remains")
    quiet, shouted = summarise_evidence(plain), summarise_evidence(loud)
    assert quiet.rung == shouted.rung
    assert quiet.claimable == shouted.claimable
    assert [item.blocked_by for item in quiet.not_claimable] == \
           [item.blocked_by for item in shouted.not_claimable]
    assert [item.identifier for item in quiet.alternatives] == \
           [item.identifier for item in shouted.alternatives]
    assert quiet.next_observation.target == shouted.next_observation.target
    # Only the display text differs.
    assert quiet.alternatives[-1].description != shouted.alternatives[-1].description


def test_an_unread_payload_key_cannot_move_any_of_the_five_outputs():
    plain = _with(_bundle(), (("observations", {"n": 4}),))
    laden = _with(_bundle(), (("observations", {"n": 4, "interpretation": "causal",
                                                "confidence": 0.99}),))
    assert summarise_evidence(plain).to_mapping()["not_claimable"] == \
           summarise_evidence(laden).to_mapping()["not_claimable"]
    assert [item.identifier for item in summarise_evidence(plain).alternatives] == \
           [item.identifier for item in summarise_evidence(laden).alternatives]


def test_the_digest_changes_when_the_bundle_underneath_it_changes():
    before = summarise_evidence(_climbed("association"))
    after = summarise_evidence(_climbed("robust_association"))
    assert before.summary_sha256 != after.summary_sha256


def test_something_other_than_a_bundle_is_refused():
    for value in ({}, None, "bundle", 7):
        with pytest.raises(InvalidParameterError):
            summarise_evidence(value)


# --------------------------------------------------------------------------------------------
# Causal claims remain outside every output (R7)
# --------------------------------------------------------------------------------------------


def test_asking_whether_a_causal_claim_is_permitted_is_refused_at_every_rung():
    for rung in CLAIM_RUNGS:
        outputs = summarise_evidence(_climbed(rung))
        for claim in OUTSIDE_THE_LADDER:
            with pytest.raises(InvalidParameterError) as refusal:
                outputs.permits(claim)
            assert "R7" in str(refusal.value)


def test_the_rendered_report_never_offers_a_causal_reading():
    for rung in CLAIM_RUNGS:
        rendered = summarise_evidence(_climbed(rung)).render()
        assert "causal claims are outside the ladder at every rung (R7)" in rendered
    assert "never about mechanism" in RUNG_ENTITLEMENTS[CLAIM_RUNGS[-1]]


def test_permits_still_answers_ordinary_rung_questions():
    outputs = summarise_evidence(_climbed("robust_association"))
    assert outputs.permits("association") is True
    assert outputs.permits("robust_association") is True
    assert outputs.permits("candidate_precursor") is False


# --------------------------------------------------------------------------------------------
# The invariants over randomised whole bundles
# --------------------------------------------------------------------------------------------


def _swept_bundle(rng):
    """A bundle drawn to land anywhere on the ladder, blocked or clear, contradicted or not."""
    bundle = _bundle()
    offset = 0
    for category, payload in ALL_REQUIREMENTS:
        if rng.random() < 0.75:
            bundle = bundle.append(
                category, label=category, status="PASS", summary="swept",
                recorded_at=_moment(offset), payload=payload, source_sha256s=(SOURCE_A,))
            offset += 1
    for _ in range(rng.randint(0, 4)):
        field = rng.choice(EVIDENCE_FIELDS)
        status = rng.choices(EVIDENCE_STATUSES, weights=(6, 1, 1, 2, 1))[0]
        payload = {"n": rng.randint(1, 500)}
        if field == "provenance":
            payload[PRECEDENCE_KEY] = rng.choice([True, False, "true", 1])
        bundle = bundle.append(field, label=field, status=status, summary="swept",
                               recorded_at=_moment(offset), payload=payload,
                               source_sha256s=(SOURCE_A,))
        offset += 1
    return bundle


def test_the_five_outputs_hold_their_invariants_over_a_randomised_sweep():
    rng = random.Random(20260826)
    reached = set()
    for _ in range(600):
        bundle = _swept_bundle(rng)
        assessment = assess_claim_ladder(bundle)
        outputs = summarise_evidence(bundle)
        reached.add(outputs.rung)

        # One: the claimable rungs are exactly those at or below the ladder's verdict.
        assert outputs.claimable == CLAIM_RUNGS[:assessment.rung_index + 1]

        # Two: claimable and not-claimable partition the ladder, and nothing is unexplained.
        unreachable = tuple(item.rung for item in outputs.not_claimable)
        assert outputs.claimable + unreachable == CLAIM_RUNGS
        assert all(item.blocked_by for item in outputs.not_claimable)

        # Three: the evidence against is exactly the adverse and contrary entries, in order.
        against = tuple(entry.sequence for entry in outputs.contradicting)
        expected = tuple(entry.sequence for entry in bundle.entries
                         if entry.status in BLOCKING_STATUSES
                         or (entry.category in CONTRARY_FIELDS
                             and entry.status != "NOT_APPLICABLE"))
        assert against == expected
        assert set(assessment.blocking_entries) <= set(against)

        # Four: open structural alternatives mirror the unsatisfied climbing gates exactly.
        structural = {item.identifier for item in outputs.alternatives
                      if item.origin == "structural"}
        assert structural == {STRUCTURAL_ALTERNATIVES[gate.name][0] for gate in assessment.gates
                              if not gate.satisfied and gate.name in STRUCTURAL_ALTERNATIVES}

    # A sweep that only ever produced one rung would assert almost nothing.
    assert reached == set(CLAIM_RUNGS)


def test_the_next_observation_follows_its_stated_precedence_over_a_randomised_sweep():
    rng = random.Random(90210)
    seen = {kind: 0 for kind in OBSERVATION_KINDS}
    nominated_nothing = 0
    for _ in range(600):
        bundle = _swept_bundle(rng)
        assessment = assess_claim_ladder(bundle)
        outputs = summarise_evidence(bundle)
        unmet = [gate.name for gate in assessment.gates
                 if not gate.satisfied and gate.name in STRUCTURAL_ALTERNATIVES]
        recorded = [item for item in outputs.alternatives if item.origin == "recorded"]

        if assessment.blocking_entries:
            assert outputs.next_observation.kind == "resolve_blocking_entry"
            assert outputs.next_observation.sequence == min(assessment.blocking_entries)
        elif unmet:
            assert outputs.next_observation.kind == "close_structural_gate"
            assert outputs.next_observation.target == unmet[0]
        elif recorded:
            assert outputs.next_observation.kind == "resolve_recorded_alternative"
            assert outputs.next_observation.sequence == min(item.sequence for item in recorded)
        else:
            assert outputs.next_observation is None
            nominated_nothing += 1
            continue
        seen[outputs.next_observation.kind] += 1

    # Every branch of the precedence, including the empty one, must occur or it is untested.
    assert all(count > 0 for count in seen.values())
    assert nominated_nothing > 0
