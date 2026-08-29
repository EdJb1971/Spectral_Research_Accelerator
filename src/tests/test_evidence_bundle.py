"""Phase G6, TG6.1: evidence accumulates as a hashed, append-only structure.

Contradictory evidence and failure states are ordinary first-class scientific fields here, and
free commentary has no route into the chain that the TG6.2 gates will read.
"""

import json
from dataclasses import replace

import pytest

from src.core.errors import InvalidParameterError
from src.core.evidence import (
    EVIDENCE_BUNDLE_SCHEMA,
    EVIDENCE_FIELDS,
    EvidenceBundle,
    EvidenceEntry,
    Hypothesis,
    create_evidence_bundle,
    load_evidence_bundle,
    save_evidence_bundle,
)


REGISTERED_AT = "2026-01-01T00:00:00+00:00"
CREATED_AT = "2026-01-02T00:00:00+00:00"
T1 = "2026-01-03T00:00:00+00:00"
T2 = "2026-01-04T00:00:00+00:00"
T3 = "2026-01-05T00:00:00+00:00"

SOURCE_A = "a" * 64
SOURCE_B = "b" * 64


def _bundle(**overrides):
    parameters = dict(
        study_id="planted_precursor_v1",
        hypothesis_id="H1",
        statement="Frozen origin motif presence precedes the target outcome.",
        prediction="A positive association survives correction on target train and test.",
        registered_at=REGISTERED_AT,
        created_at=CREATED_AT,
        provenance={"plan_sha256": SOURCE_A, "roadmap": "TG5.3"},
    )
    parameters.update(overrides)
    return create_evidence_bundle(**parameters)


def _appended(bundle, category, *, label="entry", status="PASS", summary="a measured result",
              recorded_at=T1, payload=None, source_sha256s=(SOURCE_A,)):
    return bundle.append(
        category, label=label, status=status, summary=summary, recorded_at=recorded_at,
        payload=payload if payload is not None else {"n": 128, "estimate": 0.31},
        source_sha256s=source_sha256s)


# --- The structure itself ---------------------------------------------------------------------

def test_revision_zero_is_a_complete_hashed_bundle_before_any_evidence():
    bundle = _bundle()

    assert bundle.schema == EVIDENCE_BUNDLE_SCHEMA
    assert bundle.revision == 0
    assert bundle.entries == ()
    assert len(bundle.bundle_sha256) == 64
    assert len(bundle.head_sha256) == 64
    for category in EVIDENCE_FIELDS:
        assert bundle.evidence(category) == ()


def test_appending_returns_a_new_snapshot_and_leaves_the_receiver_byte_for_byte_unchanged():
    bundle = _bundle()
    before_digest = bundle.bundle_sha256
    before_bytes = json.dumps(bundle.to_mapping(), sort_keys=True)

    extended = _appended(bundle, "observations")

    assert extended is not bundle
    assert extended.revision == 1
    assert bundle.revision == 0
    assert bundle.bundle_sha256 == before_digest
    assert json.dumps(bundle.to_mapping(), sort_keys=True) == before_bytes
    assert extended.bundle_sha256 != before_digest


def test_every_entry_binds_the_previous_digest_into_an_ordered_chain():
    bundle = _bundle()
    first = _appended(bundle, "observations", label="obs", recorded_at=T1)
    second = _appended(first, "effect_sizes", label="effect", recorded_at=T2)
    third = _appended(second, "uncertainty", label="ci", recorded_at=T3)

    entries = third.entries
    assert [entry.sequence for entry in entries] == [1, 2, 3]
    assert entries[0].previous_sha256 == bundle.head_sha256
    assert entries[1].previous_sha256 == entries[0].entry_sha256
    assert entries[2].previous_sha256 == entries[1].entry_sha256
    assert third.head_sha256 == entries[2].entry_sha256


def test_entries_are_immutable_and_the_bundle_exposes_read_only_payloads():
    bundle = _appended(_bundle(), "observations", payload={"nested": {"n": 4}})
    entry = bundle.entries[0]

    with pytest.raises(Exception):
        entry.label = "rewritten"
    with pytest.raises(TypeError):
        entry.payload["nested"] = {}
    with pytest.raises(TypeError):
        entry.payload["nested"]["n"] = 9


def test_all_scientific_fields_are_routed_and_queryable_by_category():
    bundle = _bundle()
    for index, category in enumerate(EVIDENCE_FIELDS, 1):
        bundle = _appended(bundle, category, label="entry-%d" % index, recorded_at=T1)

    assert bundle.revision == len(EVIDENCE_FIELDS)
    for category in EVIDENCE_FIELDS:
        found = bundle.evidence(category)
        assert len(found) == 1
        assert found[0].category == category


# --- Contradictory evidence and failure states are first class ---------------------------------

def test_contradictory_evidence_and_failure_states_are_first_class_fields_not_remarks():
    assert "contradictory_evidence" in EVIDENCE_FIELDS
    assert "failure_states" in EVIDENCE_FIELDS

    bundle = _appended(_bundle(), "observations", label="obs")
    bundle = _appended(bundle, "contradictory_evidence", label="opposite-sign holdout",
                       status="FAIL", summary="the holdout effect reverses sign",
                       recorded_at=T2, payload={"estimate": -0.22})
    bundle = _appended(bundle, "failure_states", label="degenerate null", status="INVALID",
                       summary="the surrogate null collapsed", recorded_at=T3,
                       payload={"unique_surrogates": 1})

    assert [entry.label for entry in bundle.contradictory_evidence] == ["opposite-sign holdout"]
    assert [entry.label for entry in bundle.failure_states] == ["degenerate null"]
    body = bundle.body()
    assert len(body["contradictory_evidence"]) == 1
    assert len(body["failure_states"]) == 1
    # They ride the same chain as supporting evidence and are covered by the bundle digest.
    assert bundle.entries[-1].entry_sha256 == bundle.head_sha256


def test_contradictory_evidence_cannot_be_removed_by_appending_more_evidence():
    bundle = _appended(_bundle(), "contradictory_evidence", label="contradiction", status="FAIL",
                       summary="the replication failed", recorded_at=T1,
                       payload={"replicated": False})
    extended = _appended(bundle, "observations", label="more", recorded_at=T2)

    assert len(extended.contradictory_evidence) == 1
    assert extended.contradictory_evidence[0].entry_sha256 == \
        bundle.contradictory_evidence[0].entry_sha256


# --- Commentary is refused ---------------------------------------------------------------------

def test_commentary_is_refused_as_a_category():
    bundle = _bundle()
    for category in ("commentary", "notes", "llm_review", "interpretation", "discussion"):
        with pytest.raises(InvalidParameterError):
            _appended(bundle, category)


def test_an_entry_needs_a_structured_payload_rather_than_prose_alone():
    bundle = _bundle()
    for payload in ({}, "the effect looked convincing", ["a", "b"]):
        with pytest.raises(InvalidParameterError):
            _appended(bundle, "observations", payload=payload)


def test_non_finite_and_unserialisable_payloads_are_refused():
    bundle = _bundle()
    for payload in ({"estimate": float("nan")}, {"estimate": float("inf")},
                    {"callback": len}, {"nested": {"p": float("-inf")}}):
        with pytest.raises(InvalidParameterError):
            _appended(bundle, "observations", payload=payload)


def test_labels_summaries_statuses_and_source_digests_are_validated():
    bundle = _bundle()
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", label="   ")
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", summary="")
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", status="PROBABLY")
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", source_sha256s=("not-a-digest",))
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", source_sha256s=(SOURCE_A, SOURCE_A))


def test_timestamps_must_carry_an_explicit_offset_and_never_run_backwards():
    bundle = _bundle()
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", recorded_at="2026-01-03T00:00:00")
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", recorded_at="not a time")
    with pytest.raises(InvalidParameterError):
        _appended(bundle, "observations", recorded_at="2026-01-01T12:00:00+00:00")

    first = _appended(bundle, "observations", recorded_at=T2)
    with pytest.raises(InvalidParameterError):
        _appended(first, "effect_sizes", recorded_at=T1)


def test_a_hypothesis_cannot_be_registered_after_the_bundle_it_anchors():
    with pytest.raises(InvalidParameterError):
        _bundle(registered_at=CREATED_AT, created_at=REGISTERED_AT)
    with pytest.raises(InvalidParameterError):
        _bundle(statement="  ")
    with pytest.raises(InvalidParameterError):
        _bundle(prediction="")


# --- Tamper evidence ---------------------------------------------------------------------------

def test_editing_an_entry_body_invalidates_its_own_digest():
    entry = _appended(_bundle(), "observations").entries[0]
    with pytest.raises(InvalidParameterError):
        replace(entry, summary="a friendlier summary")
    with pytest.raises(InvalidParameterError):
        replace(entry, status="FAIL")
    with pytest.raises(InvalidParameterError):
        replace(entry, payload={"estimate": 0.99})


def test_dropping_reordering_or_relinking_entries_breaks_the_chain():
    bundle = _appended(_appended(_bundle(), "observations", recorded_at=T1),
                       "effect_sizes", label="effect", recorded_at=T2)
    first, second = bundle.entries

    with pytest.raises(InvalidParameterError):  # dropping the first entry
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=(second,))
    with pytest.raises(InvalidParameterError):  # reordering
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=(second, first))
    with pytest.raises(InvalidParameterError):  # a duplicated entry under a stale link
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=(first, first))


def test_substituting_an_earlier_entry_breaks_the_link_the_later_entry_committed_to():
    """Sequence numbers alone cannot catch this; only the bound previous digest can."""
    bundle = _appended(_appended(_bundle(), "observations", label="obs", recorded_at=T1),
                       "effect_sizes", label="effect", recorded_at=T2)
    first, second = bundle.entries

    # A well-formed replacement for entry 1, correct sequence and chronology, different content.
    forged = EvidenceEntry(
        sequence=1, category="observations", label="obs", status="PASS",
        summary="a rewritten history", recorded_at=T1, payload={"n": 128, "estimate": 0.99},
        source_sha256s=(SOURCE_A,), previous_sha256=first.previous_sha256)

    assert forged.sequence == first.sequence
    assert forged.entry_sha256 != first.entry_sha256
    with pytest.raises(InvalidParameterError):
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=(forged, second))

    # An entry whose sequence is right but which was never linked to its predecessor.
    unlinked = EvidenceEntry(
        sequence=2, category="effect_sizes", label="effect", status="PASS",
        summary="a measured result", recorded_at=T2, payload={"n": 128, "estimate": 0.31},
        source_sha256s=(SOURCE_A,), previous_sha256=first.previous_sha256)
    with pytest.raises(InvalidParameterError):
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=(first, unlinked))


def test_a_declared_bundle_digest_must_match_the_recomputed_one():
    bundle = _appended(_bundle(), "observations")
    with pytest.raises(InvalidParameterError):
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=bundle.hypothesis, entries=bundle.entries,
                       bundle_sha256="f" * 64)
    with pytest.raises(InvalidParameterError):
        bundle.verify_published("f" * 64)
    bundle.verify_published(bundle.bundle_sha256)


def test_swapping_the_hypothesis_under_an_existing_chain_is_refused():
    bundle = _appended(_bundle(), "observations")
    other = Hypothesis(identifier="H2", statement="A different claim.",
                       prediction="A different prediction.", registered_at=REGISTERED_AT)

    with pytest.raises(InvalidParameterError):
        EvidenceBundle(study_id=bundle.study_id, created_at=bundle.created_at,
                       hypothesis=other, entries=bundle.entries)


# --- Canonical publication and reload ----------------------------------------------------------

def test_publication_round_trips_and_verifies_the_separately_published_digest(tmp_path):
    bundle = _appended(_appended(_bundle(), "observations", recorded_at=T1),
                       "contradictory_evidence", label="counter", status="FAIL",
                       summary="a contradiction", recorded_at=T2, payload={"estimate": -0.1},
                       source_sha256s=(SOURCE_B,))
    path = tmp_path / "bundle.json"

    published = save_evidence_bundle(path, bundle)
    assert published == bundle.bundle_sha256

    reloaded = load_evidence_bundle(path, published_sha256=published)
    assert reloaded.to_mapping() == bundle.to_mapping()
    assert reloaded.head_sha256 == bundle.head_sha256
    assert [entry.label for entry in reloaded.contradictory_evidence] == ["counter"]

    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(path, published_sha256="c" * 64)


def test_publication_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "bundle.json"
    bundle = _bundle()
    save_evidence_bundle(path, bundle)

    with pytest.raises(FileExistsError):
        save_evidence_bundle(path, _appended(bundle, "observations"))

    assert load_evidence_bundle(path).revision == 0


def test_a_tampered_published_file_is_refused_on_reload(tmp_path):
    bundle = _appended(_bundle(), "contradictory_evidence", label="counter", status="FAIL",
                       summary="a contradiction", recorded_at=T1, payload={"estimate": -0.1})
    path = tmp_path / "bundle.json"
    save_evidence_bundle(path, bundle)
    record = json.loads(path.read_text())

    dropped = dict(record, contradictory_evidence=[])
    (tmp_path / "dropped.json").write_text(json.dumps(dropped))
    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(tmp_path / "dropped.json")

    softened = json.loads(json.dumps(record))
    softened["contradictory_evidence"][0]["summary"] = "a minor caveat"
    (tmp_path / "softened.json").write_text(json.dumps(softened))
    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(tmp_path / "softened.json")

    unknown = dict(record, commentary=["the result looks strong"])
    (tmp_path / "unknown.json").write_text(json.dumps(unknown))
    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(tmp_path / "unknown.json")

    reskinned = dict(record, schema="evidence-bundle/v99")
    (tmp_path / "reskinned.json").write_text(json.dumps(reskinned))
    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(tmp_path / "reskinned.json")


def test_an_entry_filed_under_the_wrong_first_class_field_is_refused(tmp_path):
    bundle = _appended(_bundle(), "contradictory_evidence", label="counter", status="FAIL",
                       summary="a contradiction", recorded_at=T1, payload={"estimate": -0.1})
    path = tmp_path / "bundle.json"
    save_evidence_bundle(path, bundle)
    record = json.loads(path.read_text())

    relocated = json.loads(json.dumps(record))
    relocated["observations"] = relocated["contradictory_evidence"]
    relocated["contradictory_evidence"] = []
    (tmp_path / "relocated.json").write_text(json.dumps(relocated))
    with pytest.raises(InvalidParameterError):
        load_evidence_bundle(tmp_path / "relocated.json")


def test_reload_is_independent_of_the_process_that_built_the_bundle(tmp_path):
    bundle = _bundle()
    for index, category in enumerate(EVIDENCE_FIELDS, 1):
        bundle = _appended(bundle, category, label="entry-%d" % index,
                           payload={"index": index}, recorded_at=T1)
    path = tmp_path / "bundle.json"
    save_evidence_bundle(path, bundle)

    reloaded = load_evidence_bundle(path, published_sha256=bundle.bundle_sha256)
    assert reloaded.revision == len(EVIDENCE_FIELDS)
    assert [entry.sequence for entry in reloaded.entries] == list(
        range(1, len(EVIDENCE_FIELDS) + 1))
    assert [entry.category for entry in reloaded.entries] == list(EVIDENCE_FIELDS)
    assert reloaded.bundle_sha256 == bundle.bundle_sha256

    # Appending to the reloaded bundle continues the same chain.
    extended = _appended(reloaded, "observations", label="later", recorded_at=T2)
    assert extended.entries[-1].previous_sha256 == bundle.head_sha256


def test_the_bundle_digest_is_order_sensitive_not_merely_content_sensitive():
    base = _bundle()
    forward = _appended(_appended(base, "observations", label="a", recorded_at=T1),
                        "effect_sizes", label="b", recorded_at=T2)
    reverse = _appended(_appended(base, "effect_sizes", label="b", recorded_at=T1),
                        "observations", label="a", recorded_at=T2)

    assert forward.bundle_sha256 != reverse.bundle_sha256
    assert forward.head_sha256 != reverse.head_sha256
