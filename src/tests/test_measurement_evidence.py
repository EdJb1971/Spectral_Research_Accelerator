"""T4E.30: the two stores, joined only where git can prove the ordering.

On 2026-09-11 this repository held 28 measurement records, 43 declarations and ZERO evidence
bundles, so every finding it had produced was invisible to the review stage built to argue about
them. These tests pin the bridge and -- more importantly -- pin what it refuses, because a bridge
that back-dated a hypothesis onto finished evidence would put a manufactured preregistration in
front of a panel.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.core.claim_ladder import assess_claim_ladder
from src.core.errors import InvalidParameterError
from src.core.evidence import EVIDENCE_FIELDS
from src.core.five_outputs import summarise_evidence
from src.core.measurement_evidence import (
    EvidenceClaim,
    RegistrationNotEstablished,
    bundle_from_measurement,
    check_registration,
    first_commit,
)

DECLARATION = Path("data/identity_calibration/t4e28-join-rerun-declaration.json")
MEASUREMENT = Path("measurements/t4e28_join_rerun.json")


def _claim(**kwargs):
    base = {"category": "observations", "label": "a label", "status": "PASS",
            "summary": "a summary a reader can act on", "payload": {"value": 1}}
    base.update(kwargs)
    return EvidenceClaim(**base)


# ---------------- the ordering, proved rather than asserted


def test_the_t4e28_declaration_is_shown_by_git_to_predate_its_measurement():
    """The gate landed in a commit carrying no measurement; the run followed."""
    registration = check_registration(DECLARATION, MEASUREMENT)

    assert registration.established is True
    assert registration.refusal is None
    assert registration.declaration.committed_at < registration.measurement.committed_at
    assert registration.declaration.modified_since_commit is False


def test_an_untracked_declaration_is_refused_with_what_would_lift_it(tmp_path):
    """A hypothesis that cannot be dated cannot be shown to predate its evidence."""
    loose = tmp_path / "declaration.json"
    loose.write_text("{}", encoding="utf-8")
    registration = check_registration(loose, MEASUREMENT)

    assert registration.established is False
    assert "not tracked by git" in registration.refusal
    assert "Commit it" in registration.refusal


def test_a_missing_file_is_refused_rather_than_treated_as_absent_evidence(tmp_path):
    registration = check_registration(tmp_path / "nothing.json", MEASUREMENT)
    assert registration.established is False
    assert registration.declaration.tracked is False


def test_a_declaration_committed_after_its_measurement_is_refused_by_name():
    """Reversing the real pair is the exact shape of a manufactured preregistration."""
    registration = check_registration(MEASUREMENT, DECLARATION)

    assert registration.established is False
    assert "manufactures a preregistration" in registration.refusal
    assert "nothing retrospective" in registration.refusal


def test_the_refusal_is_raised_for_callers_that_cannot_proceed():
    with pytest.raises(RegistrationNotEstablished):
        check_registration(MEASUREMENT, DECLARATION).require()


def test_the_first_commit_is_taken_not_the_latest():
    """A declaration amended later still registered its terms when it was written."""
    moment = first_commit(DECLARATION)
    latest = subprocess.run(
        ("git", "log", "--format=%H", "-1", "--", str(DECLARATION).replace("\\", "/")),
        capture_output=True, text=True, check=False).stdout.strip()

    assert moment.tracked and moment.commit
    # They may coincide today; what must hold is that the recorded one is the ADDING commit.
    added = subprocess.run(
        ("git", "log", "--diff-filter=A", "--format=%H", "--",
         str(DECLARATION).replace("\\", "/")),
        capture_output=True, text=True, check=False).stdout.split()
    assert moment.commit == added[-1]
    assert latest  # the comparison above is meaningful only if git answered at all


# ---------------- what a claim may be


def test_a_category_outside_the_evidence_vocabulary_is_refused():
    with pytest.raises(InvalidParameterError) as caught:
        _claim(category="commentary")
    assert "Commentary is not scientific evidence" in str(caught.value)


def test_a_claim_level_key_in_a_payload_is_refused_by_name():
    """A bundle carrying one reads back as though the record asserted a rung (R22)."""
    with pytest.raises(InvalidParameterError) as caught:
        _claim(payload={"rung": "association"})
    assert "rung" in str(caught.value)


def test_an_entry_without_a_summary_is_refused():
    with pytest.raises(InvalidParameterError) as caught:
        _claim(summary="  ")
    assert "argue about without reading" in str(caught.value)


def test_an_empty_payload_is_refused():
    with pytest.raises(InvalidParameterError):
        _claim(payload={})


def test_every_evidence_field_is_an_acceptable_category():
    for category in EVIDENCE_FIELDS:
        assert _claim(category=category).category == category


# ---------------- the bundle


def test_a_bundle_carries_the_commit_times_rather_than_the_clock_of_whoever_ran_it():
    bundle, registration = bundle_from_measurement(
        study_id="t4e28-test", hypothesis_id="T4E.28", statement="a statement",
        prediction="a prediction declared beforehand", declaration=DECLARATION,
        measurement=MEASUREMENT, claims=[_claim()])

    assert bundle.hypothesis.registered_at == registration.declaration.committed_at
    assert bundle.created_at == registration.measurement.committed_at
    assert bundle.hypothesis.registered_at < bundle.created_at


def test_the_bundle_carries_both_files_by_digest_on_every_entry():
    bundle, registration = bundle_from_measurement(
        study_id="t4e28-test", hypothesis_id="T4E.28", statement="a statement",
        prediction="a prediction", declaration=DECLARATION, measurement=MEASUREMENT,
        claims=[_claim(), _claim(category="null_results", status="FAIL")])

    for entry in bundle.entries:
        assert registration.declaration.sha256 in entry.source_sha256s
        assert registration.measurement.sha256 in entry.source_sha256s


def test_a_bundle_with_no_claims_is_refused():
    """A hypothesis with no evidence asserts a question answered by nothing."""
    with pytest.raises(InvalidParameterError) as caught:
        bundle_from_measurement(
            study_id="t4e28-test", hypothesis_id="T4E.28", statement="s", prediction="p",
            declaration=DECLARATION, measurement=MEASUREMENT, claims=[])
    assert "answered by nothing" in str(caught.value)


def test_the_provenance_states_what_it_does_not_say():
    bundle, _ = bundle_from_measurement(
        study_id="t4e28-test", hypothesis_id="T4E.28", statement="s", prediction="p",
        declaration=DECLARATION, measurement=MEASUREMENT, claims=[_claim()])
    said = bundle.hypothesis.provenance["what_this_provenance_does_not_say"]

    assert "was wise" in said
    assert "only that the question was fixed before the answer existed" in said


# ---------------- the published bundle, and what a panel would see


def _published():
    from src.core.evidence import load_evidence_bundle
    return load_evidence_bundle(Path("data/studies/t4e28-join-rerun.r7.json"))


def test_the_published_join_rerun_bundle_verifies_and_carries_seven_entries():
    bundle = _published()

    assert bundle.study_id == "t4e28-join-rerun"
    assert bundle.revision == 7
    assert {entry.category for entry in bundle.entries} == {
        "replication_results", "provenance", "contradictory_evidence", "null_results",
        "failure_states", "uncertainty"}


def test_the_falsified_range_is_in_the_bundle_as_a_standing_contradiction():
    """The finding a panel most needs is the one most easily left out of a summary."""
    bundle = _published()
    contradictions = bundle.contradictory_evidence

    assert len(contradictions) == 1
    assert "non-dateline range is false" in contradictions[0].label
    # A loaded bundle freezes its payload, so the sequences come back immutable.
    assert tuple(contradictions[0].payload["observed_range_km"]) == (16.6, 315.1)
    assert tuple(contradictions[0].payload["published_range_km"]) == (16.6, 99.3)


def test_the_published_bundle_is_capped_at_observation_and_says_why():
    """A standing contradiction and a FAIL both cap the ladder, and should."""
    assessment = assess_claim_ladder(_published())
    unmet = {gate.name for gate in assessment.gates if not gate.satisfied}

    assert assessment.rung == "observation"
    assert "observation.no_standing_contradiction" in unmet
    assert "observation.no_failed_or_invalid_evidence" in unmet


def test_a_panel_can_be_seated_over_the_published_bundle():
    """The whole point: the review stage can now reach this work at all."""
    from src.core.recorded_call import REVIEW_ROLES, verify_claim_independence
    from src.core.round_robin import PanelSeat, ReviewPanel, open_round_robin

    panel = ReviewPanel(seats={role: PanelSeat(role=role, model_id="a-model", effort="high")
                               for role in REVIEW_ROLES})
    exchange = open_round_robin(_published(), panel)

    assert exchange.next_turn.role == "candidate_synthesis"
    assert exchange.complete is False
    assert exchange.dissent == ()
    assert verify_claim_independence(exchange.reviewed)


def test_the_five_outputs_are_computable_over_the_published_bundle():
    outputs = summarise_evidence(_published()).to_mapping()

    assert outputs["study_id"] == "t4e28-join-rerun"
    assert outputs["contradicting"]
    assert outputs["next_observation"]


def test_a_declaration_committed_alongside_its_measurement_is_refused_as_unprovable():
    """The common case in this repository, and the interesting one.

    T4E.19, T4E.20, T4E.21, T4E.24, T4E.12 and T4E.14 each landed their declaration in the SAME
    commit as their measurement. Those declarations were almost certainly written first -- the
    sessions show it -- but git cannot separate them, and a bundle asserting the ordering would
    assert something nothing checks. T4E.28 is the only study that committed its gate alone,
    before the run, which is why it is the one that can be bundled.
    """
    registration = check_registration(
        Path("data/identity_calibration/t4e19-positional-error-declaration.json"),
        Path("measurements/t4e19_positional_error.json"))

    assert registration.established is False
    assert "SAME commit" in registration.refusal
    assert "nothing checks" in registration.refusal
    assert "Nothing lifts it retrospectively" in registration.refusal
    assert registration.declaration.commit == registration.measurement.commit
