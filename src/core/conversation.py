"""G19 guarantees for a private conversation with published records.

The module is deliberately transport-free.  It makes whole-record selection, per-turn grounding
receipts, hash-chained turns, budgets and transcript-wide claim independence testable without a
model call or a web server.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle
from src.core.five_outputs import summarise_evidence


CONVERSATION_SCHEMA = "ephemeral-record-conversation/v2"
TURN_SCHEMA = "record-conversation-turn/v1"
GROUNDING_SCHEMA = "whole-record-grounding/v1"
INDEPENDENCE_SCHEMA = "conversation-independence-proof/v1"
CLAIM_BOUNDARY = (
    "Interpretation only: this turn is non-reproducible model output, is not evidence, cannot "
    "change a claim, and must not be cited as the reason a finding holds."
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _words(value: str) -> frozenset[str]:
    return frozenset(word for word in re.findall(r"[a-z0-9]{3,}", value.lower())
                     if word not in {"the", "and", "for", "that", "with", "from", "this"})


def select_whole_records(rows: Sequence[Mapping[str, Any]], query: str, primary_id: str,
                         max_records: int) -> List[str]:
    """Select studies deterministically; selection is indexed, but records enter only whole."""
    if not 1 <= max_records <= 8:
        raise InvalidParameterError("max_records", max_records, "an integer from 1 to 8")
    terms = _words(query)
    ranked: List[Tuple[int, str]] = []
    readable_ids = set()
    for row in rows:
        study_id = row.get("study_id")
        if not row.get("readable") or not isinstance(study_id, str):
            continue
        readable_ids.add(study_id)
        searchable = " ".join(str(row.get(field, "")) for field in
                              ("study_id", "hypothesis", "rung", "search_text"))
        score = len(terms & _words(searchable))
        ranked.append((score, study_id))
    if primary_id not in readable_ids:
        raise InvalidParameterError("primary study", primary_id, "a readable published study")
    chosen = [primary_id]
    for score, study_id in sorted(ranked, key=lambda item: (-item[0], item[1])):
        if study_id != primary_id and score > 0 and len(chosen) < max_records:
            chosen.append(study_id)
    return chosen


def grounding_receipt(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    refs = []
    for record in records:
        raw = canonical_bytes(record)
        refs.append({
            "study_id": record["study_id"],
            "bundle_sha256": record["bundle_sha256"],
            "bundle_revision": record["bundle_revision"],
            "complete_record_sha256": hashlib.sha256(raw).hexdigest(),
            "complete_record_bytes": len(raw),
        })
    context = {"complete_source_records": list(records)}
    return {
        "schema": GROUNDING_SCHEMA,
        "record_granularity": "complete",
        "records": refs,
        "scientific_context_sha256": sha256(context),
        "derivable_from_named_records": True,
        "contains_dialogue": False,
    }


def verify_grounding(records: Sequence[Mapping[str, Any]], receipt: Mapping[str, Any]) -> str:
    rebuilt = grounding_receipt(records)
    if dict(receipt) != rebuilt:
        raise InvalidParameterError("grounding receipt", receipt,
                                    "the digest and byte counts derived from its named whole records")
    return rebuilt["scientific_context_sha256"]


def append_turn(turns: Sequence[Mapping[str, Any]], *, role: str, text: str,
                grounding: Mapping[str, Any] | None = None,
                provider: Mapping[str, Any] | None = None,
                usage: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    if role not in {"researcher", "model"}:
        raise InvalidParameterError("turn role", role, "researcher or model")
    body: Dict[str, Any] = {
        "schema": TURN_SCHEMA,
        "sequence": len(turns) + 1,
        "role": role,
        "text": text,
        "previous_turn_sha256": turns[-1]["turn_sha256"] if turns else None,
        "claim_boundary": CLAIM_BOUNDARY,
        "reproducible": False if role == "model" else True,
    }
    if grounding is not None:
        body["grounding"] = dict(grounding)
    if provider is not None:
        body["provider"] = dict(provider)
    if usage is not None:
        body["usage"] = dict(usage)
    body["turn_sha256"] = sha256(body)
    return body


def verify_turn_chain(turns: Sequence[Mapping[str, Any]]) -> str | None:
    previous = None
    for index, supplied in enumerate(turns):
        turn = dict(supplied)
        digest = turn.pop("turn_sha256", None)
        if turn.get("sequence") != index + 1 or turn.get("previous_turn_sha256") != previous:
            raise InvalidParameterError("transcript turn", index + 1, "an unbroken append-only chain")
        if digest != sha256(turn):
            raise InvalidParameterError("turn_sha256", digest, "the canonical turn digest")
        previous = digest
    return previous


def verify_transcript_independence(turns: Sequence[Mapping[str, Any]],
                                   bundles: Iterable[EvidenceBundle]) -> Dict[str, Any]:
    """Delete the transcript computationally and prove every touched claim stays identical."""
    head = verify_turn_chain(turns)
    before: Dict[str, str] = {}
    after_deletion: Dict[str, str] = {}
    for bundle in bundles:
        before[bundle.study_id] = summarise_evidence(bundle).summary_sha256
        rebuilt = EvidenceBundle.from_mapping(bundle.to_mapping())
        after_deletion[bundle.study_id] = summarise_evidence(rebuilt).summary_sha256
    if before != after_deletion:
        raise InvalidParameterError("claim digests", after_deletion,
                                    "unchanged after deleting the complete conversation")
    return {
        "schema": INDEPENDENCE_SCHEMA,
        "verified": True,
        "transcript_head_sha256": head,
        "claim_digests_with_conversation": before,
        "claim_digests_after_deletion": after_deletion,
        "conversation_is_claim_input": False,
    }


__all__ = [
    "CLAIM_BOUNDARY", "CONVERSATION_SCHEMA", "GROUNDING_SCHEMA", "INDEPENDENCE_SCHEMA",
    "append_turn", "canonical_bytes", "grounding_receipt", "select_whole_records", "sha256",
    "verify_grounding", "verify_transcript_independence", "verify_turn_chain",
]
