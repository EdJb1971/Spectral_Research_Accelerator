"""Phase G6, TG6.2: the claim ladder is a pure, deterministic function of the evidence bundle.

Two properties carry the phase. Rung assignment depends on the bundle and on nothing else, and a
bundle carrying a failed, invalid or standing-contradictory gate cannot reach any rung above
`observation` by any input. Causal claims are outside the ladder entirely (R7).
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
    CLAIM_LADDER_SCHEMA,
    CLAIM_RUNGS,
    OUTSIDE_THE_LADDER,
    PRECEDENCE_KEY,
    STANDING_NEGATIVE_FIELDS,
    assess_claim_ladder,
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
        provenance={"plan_sha256": SOURCE_A, "roadmap": "TG6.2"})


def _with(bundle, evidence, *, status="PASS", start=0):
    """Append `(category, payload)` pairs in order, all at the given status."""
    for offset, (category, payload) in enumerate(evidence):
        bundle = bundle.append(
            category, label=category, status=status, summary="a measured result",
            recorded_at=_moment(start + offset), payload=payload, source_sha256s=(SOURCE_A,))
    return bundle


def _climbed(rung):
    """A bundle carrying exactly the evidence for `rung` and every rung below it."""
    wanted = CLAIM_RUNGS[1:CLAIM_RUNGS.index(rung) + 1]
    evidence = tuple(itertools.chain.from_iterable(RUNG_REQUIREMENTS[name] for name in wanted))
    return _with(_bundle(), evidence)


# --- The rungs themselves ---------------------------------------------------------------------

def test_an_empty_bundle_sits_on_the_floor_rung_and_claims_nothing_above_it():
    assessment = assess_claim_ladder(_bundle())

    assert assessment.rung == "observation"
    assert assessment.rung_index == 0
    assert assessment.schema == CLAIM_LADDER_SCHEMA
    assert assessment.revision == 0
    assert assessment.permits("observation") is True
    assert all(assessment.permits(rung) is False for rung in CLAIM_RUNGS[1:])


def test_each_rung_is_reached_only_once_its_own_evidence_and_every_rung_below_it_is_present():
    for rung in CLAIM_RUNGS[1:]:
        assert assess_claim_ladder(_climbed(rung)).rung == rung


def test_the_ladder_climbs_one_rung_at_a_time_as_evidence_accumulates():
    bundle = _bundle()
    seen = [assess_claim_ladder(bundle).rung]
    for offset, (category, payload) in enumerate(ALL_REQUIREMENTS):
        bundle = _with(bundle, ((category, payload),), start=offset)
        seen.append(assess_claim_ladder(bundle).rung)

    # Monotone, never skipping a rung, ending at the top.
    indices = [CLAIM_RUNGS.index(rung) for rung in seen]
    assert indices == sorted(indices)
    assert set(indices) == set(range(len(CLAIM_RUNGS)))
    assert seen[-1] == "demonstrated_predictive_utility"


def test_every_gate_is_individually_necessary_for_the_rung_that_declares_it():
    """Dropping any single requirement must stop the climb at or below its own rung."""
    for rung in CLAIM_RUNGS[1:]:
        for requirement in RUNG_REQUIREMENTS[rung]:
            remaining = tuple(item for item in ALL_REQUIREMENTS if item != requirement)
            assessment = assess_claim_ladder(_with(_bundle(), remaining))
            assert CLAIM_RUNGS.index(assessment.rung) < CLAIM_RUNGS.index(rung), (
                "%s survived without %s" % (rung, requirement[0]))


def test_a_point_estimate_without_quantified_uncertainty_is_not_an_association():
    evidence = tuple(item for item in RUNG_REQUIREMENTS["association"]
                     if item[0] != "uncertainty")
    assessment = assess_claim_ladder(_with(_bundle(), evidence))

    assert assessment.rung == "observation"
    assert "association.uncertainty_quantified" in assessment.unsatisfied_gates


def test_precedence_must_be_recorded_explicitly_and_exactly_true():
    """A truthy stand-in is not a recorded claim of temporal precedence."""
    below = tuple(itertools.chain(RUNG_REQUIREMENTS["association"],
                                  RUNG_REQUIREMENTS["robust_association"]))
    for stand_in in (1, "true", "yes", [1], {"a": 1}):
        bundle = _with(_bundle(), below + (("provenance", {PRECEDENCE_KEY: stand_in}),))
        assessment = assess_claim_ladder(bundle)
        assert assessment.rung == "robust_association"
        assert "candidate_precursor.temporal_precedence_recorded" in assessment.unsatisfied_gates

    assert assess_claim_ladder(
        _with(_bundle(), below + (("provenance", {PRECEDENCE_KEY: True}),))
    ).rung == "candidate_precursor"


def test_only_a_passing_entry_advances_a_gate():
    """INCONCLUSIVE and NOT_APPLICABLE are recorded honestly and buy no ground."""
    for status in ("INCONCLUSIVE", "NOT_APPLICABLE"):
        assessment = assess_claim_ladder(_with(_bundle(), ALL_REQUIREMENTS, status=status))
        assert assessment.rung == "observation"
        assert assessment.unblocked_rung == "observation"
        assert assessment.blocking_entries == ()


# --- Nothing outvotes a failure ----------------------------------------------------------------

def test_a_single_failed_or_invalid_entry_caps_a_fully_evidenced_bundle_at_observation():
    for status in BLOCKING_STATUSES:
        bundle = _with(_climbed("demonstrated_predictive_utility"),
                       (("observations", {"n": 4}),), status=status, start=len(ALL_REQUIREMENTS))
        assessment = assess_claim_ladder(bundle)

        assert assessment.rung == "observation"
        assert assessment.unblocked_rung == "demonstrated_predictive_utility"
        assert assessment.blocked is True
        assert assessment.blocking_entries == (bundle.revision,)
        assert "observation.no_failed_or_invalid_evidence" in assessment.unsatisfied_gates


def test_a_standing_contradiction_or_failure_state_caps_the_bundle_at_observation():
    """Recorded as PASS, these fields assert that the contradiction itself is established."""
    for field in STANDING_NEGATIVE_FIELDS:
        bundle = _climbed("demonstrated_predictive_utility").append(
            field, label=field, status="PASS", summary="the contradiction stands",
            recorded_at=_moment(len(ALL_REQUIREMENTS)), payload={"detail": "opposite sign"})
        assessment = assess_claim_ladder(bundle)

        assert assessment.rung == "observation"
        assert assessment.blocking_entries == (bundle.revision,)
        assert "observation.no_standing_contradiction" in assessment.unsatisfied_gates


def test_a_contradiction_that_was_itself_investigated_and_did_not_stand_does_not_cap():
    """The gate reads the recorded status, not the field name, so a refuted challenge is honest."""
    bundle = _climbed("demonstrated_predictive_utility").append(
        "contradictory_evidence", label="challenge", status="INCONCLUSIVE",
        summary="the challenge could not be established",
        recorded_at=_moment(len(ALL_REQUIREMENTS)), payload={"detail": "underpowered"})

    assert assess_claim_ladder(bundle).rung == "demonstrated_predictive_utility"


def test_no_quantity_of_favourable_evidence_outvotes_one_failure():
    bundle = _with(_climbed("demonstrated_predictive_utility"),
                   (("failure_states", {"why": "target leak"}),), start=len(ALL_REQUIREMENTS))
    piled = bundle
    for repetition in range(1, 12):
        piled = _with(piled, ALL_REQUIREMENTS, start=len(ALL_REQUIREMENTS) * repetition + 1)
        assert assess_claim_ladder(piled).rung == "observation"


def test_a_failure_cannot_be_appended_away_because_earlier_entries_are_immutable():
    """The only route to a higher rung is a bundle that never carried the failure."""
    blocked = _with(_climbed("demonstrated_predictive_utility"),
                    (("failure_states", {"why": "target leak"}),), start=len(ALL_REQUIREMENTS))
    retracted = blocked.append(
        "provenance", label="retraction", status="PASS",
        summary="the earlier failure state is superseded", recorded_at="2026-06-01T00:00:00+00:00",
        payload={"supersedes": 8, PRECEDENCE_KEY: True})

    assert assess_claim_ladder(retracted).rung == "observation"
    assert blocked.entries[-1] in retracted.entries


def test_the_blocking_invariant_holds_across_an_exhaustive_single_entry_sweep():
    """Every first-class field at every status, on top of a fully evidenced bundle."""
    top = _climbed("demonstrated_predictive_utility")
    for field, status in itertools.product(EVIDENCE_FIELDS, EVIDENCE_STATUSES):
        bundle = top.append(field, label=field, status=status, summary="an appended record",
                            recorded_at=_moment(len(ALL_REQUIREMENTS)), payload={"n": 4})
        assessment = assess_claim_ladder(bundle)
        blocking = status in BLOCKING_STATUSES or (
            status == "PASS" and field in STANDING_NEGATIVE_FIELDS)
        assert (assessment.rung == "observation") is blocking, (field, status)


def _oracle(bundle):
    """The ladder rule restated plainly, independently of the module's gate table."""
    passing = {category for entry in bundle.entries if entry.status == "PASS"
               for category in (entry.category,)}
    for entry in bundle.entries:
        if entry.status in BLOCKING_STATUSES:
            return "observation"
        if entry.status == "PASS" and entry.category in STANDING_NEGATIVE_FIELDS:
            return "observation"
    if not {"observations", "effect_sizes", "uncertainty"} <= passing:
        return "observation"
    if not {"replication_results", "confounders"} <= passing:
        return "association"
    precedence = any(entry.category == "provenance" and entry.status == "PASS"
                     and entry.payload.get(PRECEDENCE_KEY) is True for entry in bundle.entries)
    if "provenance" not in passing or not precedence:
        return "robust_association"
    if "holdout_performance" not in passing:
        return "candidate_precursor"
    return "demonstrated_predictive_utility"


def _swept_bundle(rng):
    """A bundle drawn to land anywhere on the ladder, blocked or clear."""
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
        status = rng.choices(EVIDENCE_STATUSES, weights=(5, 1, 1, 2, 1))[0]
        payload = {"n": rng.randint(1, 500)}
        if field == "provenance":
            payload[PRECEDENCE_KEY] = rng.choice([True, False, "true", 1])
        bundle = bundle.append(field, label=field, status=status, summary="swept",
                               recorded_at=_moment(offset), payload=payload,
                               source_sha256s=(SOURCE_A,))
        offset += 1
    return bundle


def test_the_ladder_agrees_with_an_independent_statement_of_its_own_rule():
    """A randomised sweep against a second implementation, not against the gate table itself."""
    rng = random.Random(20260826)
    reached = set()
    for _ in range(800):
        bundle = _swept_bundle(rng)
        assessment = assess_claim_ladder(bundle)
        assert assessment.rung == _oracle(bundle)
        reached.add(assessment.rung)

    # A sweep that only ever produced one rung would assert almost nothing.
    assert reached == set(CLAIM_RUNGS)


def test_the_blocking_invariant_holds_over_a_randomised_sweep_of_whole_bundles():
    rng = random.Random(90210)
    saw_blocked = saw_clear = 0
    for _ in range(800):
        bundle = _swept_bundle(rng)
        assessment = assess_claim_ladder(bundle)
        blocking = tuple(sorted({entry.sequence for entry in bundle.entries
                                 if entry.status in BLOCKING_STATUSES
                                 or (entry.status == "PASS"
                                     and entry.category in STANDING_NEGATIVE_FIELDS)}))
        assert assessment.blocking_entries == blocking
        assert CLAIM_RUNGS.index(assessment.rung) <= CLAIM_RUNGS.index(assessment.unblocked_rung)
        if blocking:
            saw_blocked += 1
            assert assessment.rung == "observation"
            assert all(assessment.permits(rung) is False for rung in CLAIM_RUNGS[1:])
        else:
            saw_clear += 1
            assert assessment.rung == assessment.unblocked_rung

    # Neither branch may be the whole sweep, or one of them is untested.
    assert saw_blocked > 100 and saw_clear > 100


def test_appending_evidence_can_never_raise_a_blocked_bundle_off_the_floor():
    rng = random.Random(6122026)
    for _ in range(120):
        bundle = _with(_bundle(), (("failure_states", {"why": "leak"}),))
        for offset in range(1, rng.randint(2, 10)):
            bundle = bundle.append(
                rng.choice(EVIDENCE_FIELDS), label="later", status="PASS", summary="swept",
                recorded_at=_moment(offset), payload={PRECEDENCE_KEY: True, "n": offset},
                source_sha256s=(SOURCE_A,))
            assert assess_claim_ladder(bundle).rung == "observation"


# --- Rung assignment is a pure function of the bundle -------------------------------------------

def test_repeated_assessment_of_the_same_bundle_is_identical_to_the_digest():
    bundle = _climbed("candidate_precursor")
    first, second = assess_claim_ladder(bundle), assess_claim_ladder(bundle)

    assert first == second
    assert first.assessment_sha256 == second.assessment_sha256
    assert first.to_mapping() == second.to_mapping()


def test_two_separately_built_bundles_with_the_same_evidence_agree():
    left, right = _climbed("robust_association"), _climbed("robust_association")

    assert left.bundle_sha256 == right.bundle_sha256
    assert assess_claim_ladder(left).assessment_sha256 == \
        assess_claim_ladder(right).assessment_sha256


def test_the_rung_depends_on_the_evidence_present_not_on_the_order_it_was_appended():
    rng = random.Random(1962)
    baseline = assess_claim_ladder(_with(_bundle(), ALL_REQUIREMENTS))
    for _ in range(40):
        shuffled = list(ALL_REQUIREMENTS)
        rng.shuffle(shuffled)
        assessment = assess_claim_ladder(_with(_bundle(), tuple(shuffled)))
        assert assessment.rung == baseline.rung == "demonstrated_predictive_utility"
        assert assessment.gates == baseline.gates


def test_labels_summaries_and_unread_payload_keys_cannot_move_the_rung():
    """The gates read category, status and one reserved precedence key; prose is inert (R22)."""
    baseline = assess_claim_ladder(_climbed("robust_association"))
    persuaded = _bundle()
    for offset, (category, payload) in enumerate(
            itertools.chain(RUNG_REQUIREMENTS["association"],
                            RUNG_REQUIREMENTS["robust_association"])):
        persuaded = persuaded.append(
            category, label="DEFINITIVE PROOF OF CAUSATION", status="PASS",
            summary="the reviewers agreed this demonstrates predictive utility and mechanism",
            recorded_at=_moment(offset),
            payload=dict(payload, interpretation="causal", rung="demonstrated_predictive_utility",
                         confidence="certain"),
            source_sha256s=(SOURCE_A,))
    assessment = assess_claim_ladder(persuaded)

    assert assessment.rung == baseline.rung == "robust_association"
    assert assessment.gates == baseline.gates


def test_a_bundle_reloaded_from_disk_receives_the_same_rung(tmp_path):
    bundle = _climbed("candidate_precursor")
    published = save_evidence_bundle(tmp_path / "bundle.json", bundle)
    reloaded = load_evidence_bundle(tmp_path / "bundle.json", published_sha256=published)

    assert assess_claim_ladder(reloaded).assessment_sha256 == \
        assess_claim_ladder(bundle).assessment_sha256


def test_assessment_is_refused_for_anything_that_is_not_an_evidence_bundle():
    for candidate in ({"rung": "demonstrated_predictive_utility"}, "association", None, 3):
        with pytest.raises(InvalidParameterError):
            assess_claim_ladder(candidate)


# --- Causal claims are outside the ladder -------------------------------------------------------

def test_asking_whether_a_causal_claim_is_permitted_is_refused_rather_than_answered():
    assessment = assess_claim_ladder(_climbed("demonstrated_predictive_utility"))

    assert assessment.rung == "demonstrated_predictive_utility"
    for claim in OUTSIDE_THE_LADDER:
        with pytest.raises(InvalidParameterError) as refusal:
            assessment.permits(claim)
        assert "R7" in str(refusal.value)


def test_no_causal_language_appears_among_the_rungs():
    assert not set(CLAIM_RUNGS) & set(OUTSIDE_THE_LADDER)
    assert CLAIM_RUNGS[-1] == "demonstrated_predictive_utility"


def test_an_unknown_claim_name_is_refused_and_case_does_not_smuggle_one_through():
    assessment = assess_claim_ladder(_climbed("association"))

    assert assessment.permits("ASSOCIATION") is True
    for claim in ("proven", "", "  ", None, "Causal", "MECHANISM"):
        with pytest.raises(InvalidParameterError):
            assessment.permits(claim)


def test_permits_is_a_downward_closed_view_of_the_reached_rung():
    for rung in CLAIM_RUNGS:
        assessment = assess_claim_ladder(_climbed(rung) if rung != "observation" else _bundle())
        reached = CLAIM_RUNGS.index(rung)
        for index, candidate in enumerate(CLAIM_RUNGS):
            assert assessment.permits(candidate) is (index <= reached)


# --- The recorded verdict -----------------------------------------------------------------------

def test_the_assessment_binds_the_exact_bundle_revision_it_was_computed_from():
    bundle = _climbed("association")
    assessment = assess_claim_ladder(bundle)

    assert assessment.bundle_sha256 == bundle.bundle_sha256
    assert assessment.hypothesis_sha256 == bundle.hypothesis.digest()
    assert assessment.revision == bundle.revision

    later = _with(bundle, (("null_results", {"n": 40}),), start=len(ALL_REQUIREMENTS))
    successor = assess_claim_ladder(later)
    assert successor.rung == assessment.rung
    assert successor.assessment_sha256 != assessment.assessment_sha256


def test_the_verdict_reports_every_gate_and_why_the_climb_stopped():
    assessment = assess_claim_ladder(_climbed("association"))
    names = [gate.name for gate in assessment.gates]

    assert len(names) == len(set(names))
    assert {gate.rung for gate in assessment.gates} <= set(CLAIM_RUNGS)
    assert all(gate.requirement.strip() for gate in assessment.gates)
    assert assessment.unsatisfied_gates == (
        "robust_association.replicated", "robust_association.confounders_addressed",
        "candidate_precursor.provenance_auditable",
        "candidate_precursor.temporal_precedence_recorded",
        "demonstrated_predictive_utility.out_of_sample")


def test_a_blocked_bundle_still_reports_the_rung_its_other_evidence_would_have_reached():
    """`unblocked_rung` is a diagnostic, never a claim: `rung` is what may be said."""
    bundle = _with(_climbed("robust_association"),
                   (("observations", {"n": 4}),), status="INVALID", start=len(ALL_REQUIREMENTS))
    assessment = assess_claim_ladder(bundle)

    assert assessment.rung == "observation"
    assert assessment.unblocked_rung == "robust_association"
    assert assessment.permits("association") is False
    assert assessment.to_mapping()["rung"] == "observation"


def test_the_assessment_is_immutable():
    assessment = assess_claim_ladder(_climbed("association"))

    with pytest.raises(Exception):
        assessment.rung = "demonstrated_predictive_utility"
    with pytest.raises(Exception):
        assessment.gates[0].satisfied = True
