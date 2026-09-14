"""G19's pure guards: whole records, receipts, chains and deletion independence."""
from __future__ import annotations

import pytest

from src.core.conversation import (append_turn, grounding_receipt, select_whole_records,
    verify_grounding, verify_transcript_independence, verify_turn_chain)
from src.core.errors import InvalidParameterError
from src.tests.test_review_cost import _recorded_review


def _record(study_id="one"):
    bundle = _recorded_review().bundle
    return {"study_id": study_id, "bundle_sha256": bundle.bundle_sha256,
            "bundle_revision": bundle.revision, "evidence_bundle": bundle.to_mapping()}


def test_corpus_selection_is_deterministic_and_never_returns_fragments():
    rows = [{"readable": True, "study_id": "primary", "hypothesis": "ocean heat", "rung": "TESTED"},
            {"readable": True, "study_id": "related", "hypothesis": "ocean salinity", "rung": "TESTED"},
            {"readable": True, "study_id": "other", "hypothesis": "stellar spectra", "rung": "TESTED"}]
    assert select_whole_records(rows, "salinity ocean", "primary", 3) == ["primary", "related"]


def test_grounding_receipt_is_exactly_derivable_from_named_whole_records():
    records = [_record()]
    receipt = grounding_receipt(records)
    assert verify_grounding(records, receipt) == receipt["scientific_context_sha256"]
    receipt["contains_dialogue"] = True
    with pytest.raises(InvalidParameterError, match="grounding receipt"):
        verify_grounding(records, receipt)


def test_typed_turns_form_an_append_only_chain_and_tampering_refuses():
    turns = []
    turns.append(append_turn(turns, role="researcher", text="why?"))
    turns.append(append_turn(turns, role="model", text="because"))
    assert verify_turn_chain(turns) == turns[-1]["turn_sha256"]
    turns[0]["text"] = "changed"
    with pytest.raises(InvalidParameterError, match="turn_sha256"):
        verify_turn_chain(turns)


def test_deleting_the_entire_transcript_leaves_every_touched_claim_unchanged():
    bundle = _recorded_review().bundle
    turns = [append_turn([], role="researcher", text="why?")]
    turns.append(append_turn(turns, role="model", text="interpretation only"))
    proof = verify_transcript_independence(turns, [bundle])
    assert proof["verified"] is True
    assert proof["claim_digests_with_conversation"] == proof["claim_digests_after_deletion"]
    assert proof["conversation_is_claim_input"] is False
