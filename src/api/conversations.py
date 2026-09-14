"""Private, whole-record, source-grounded conversations about published findings (G19)."""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.findings import StudyStore, _declaration_payload, _figures_for, _unadmitted_reading
from src.api.reviews import DEFAULT_REVIEW_MODEL, KEY_VARIABLES, _key, study_review_payload
from src.core.conversation import (CLAIM_BOUNDARY, CONVERSATION_SCHEMA, append_turn,
    canonical_bytes, grounding_receipt, select_whole_records, sha256, verify_grounding,
    verify_transcript_independence)
from src.core.errors import DataSourceError, InvalidParameterError
from src.core.experiment_run import RunStore
from src.core.five_outputs import summarise_evidence
from src.core.review_cost import GEMINI_API_ROOT
from src.core.translation import glossary_for, translate

router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])
CONVERSATION_ROOT_ENV = "SPECTRAL_CONVERSATION_ROOT"
DEFAULT_CONVERSATION_ROOT = Path("data") / "conversations"
MAX_TURN_CHARS, MAX_RECORD_BYTES, MAX_CONTEXT_BYTES = 8_000, 750_000, 2_000_000
DEFAULT_MAX_CALLS, DEFAULT_MAX_TOTAL_TOKENS, MAX_OUTPUT_TOKENS = 12, 120_000, 4_096
PROVIDERS = {"google-gemini": {"provider_id": "google-gemini", "display_name": "Google Gemini",
    "supports_declared_json_schema": True,
    "sampling_policy": "provider default; no sampling overrides"}}


class AskFindingRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)
    glossary: str = Field(default="reanalysis", min_length=1, max_length=100)
    model_id: str = Field(default=DEFAULT_REVIEW_MODEL, min_length=1, max_length=100)
    provider_id: str = "google-gemini"
    expected_bundle_sha256: str = Field(..., min_length=64, max_length=64)
    conversation_id: Optional[str] = None
    max_records: int = Field(default=3, ge=1, le=8)
    max_calls: int = Field(default=DEFAULT_MAX_CALLS, ge=1, le=50)
    max_total_tokens: int = Field(default=DEFAULT_MAX_TOTAL_TOKENS, ge=8_192, le=1_000_000)
    i_authorise_paid_call: bool = False


class SaveFindingDiscussionRequest(BaseModel):
    conversation_id: str = Field(..., min_length=32, max_length=64)
    title: Optional[str] = Field(default=None, max_length=160)


class CorpusSelectionRequest(BaseModel):
    primary_study_id: str = Field(..., min_length=1, max_length=200)
    question: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)
    glossary: str = Field(default="reanalysis", min_length=1, max_length=100)
    max_records: int = Field(default=3, ge=1, le=8)


@dataclass
class _Session:
    conversation_id: str
    primary_study_id: str
    glossary: str
    provider_id: str
    model_id: str
    max_calls: int
    max_total_tokens: int
    turns: List[Dict[str, Any]] = field(default_factory=list)
    bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    calls_used: int = 0
    tokens_used: int = 0


_SESSIONS: Dict[str, _Session] = {}
_SESSION_LOCK = threading.RLock()


def _run_root(request: Request) -> Path:
    return Path(getattr(request.app.state, "experiment_run_dir", None)
                or os.getenv("EXPERIMENT_RUN_DIR", "data/experiment_runs"))


def _conversation_root() -> Path:
    return Path(os.getenv(CONVERSATION_ROOT_ENV) or DEFAULT_CONVERSATION_ROOT)


def _grounding(study_id: str, glossary_name: str, request: Request) -> Dict[str, Any]:
    try:
        bundle, bundle_path = StudyStore().load(study_id)
        glossary = glossary_for(glossary_name)
    except (KeyError, FileNotFoundError, InvalidParameterError):
        raise HTTPException(status_code=404, detail="No readable published study or glossary matches %r." % study_id)
    outputs = summarise_evidence(bundle)
    document = translate(outputs, glossary, figures=_figures_for(bundle))
    translated = dict(document.to_mapping(), rendered_text=document.render(),
        figures_text=None if document.figures is None else document.figures.render(),
        domain_limits=_declaration_payload(glossary_name),
        unadmitted_reading=_unadmitted_reading(outputs.rung, glossary_name))
    runs = [row for row in RunStore(_run_root(request)).list_runs() if row.get("study_id") == study_id]
    return {"study_id": study_id, "bundle_file": bundle_path.name,
        "bundle_sha256": bundle.bundle_sha256, "bundle_revision": bundle.revision,
        "evidence_bundle": bundle.to_mapping(),
        "derived_finding": dict(outputs.to_mapping(), summary_sha256=outputs.summary_sha256),
        "finding_in_selected_vocabulary": translated, "experiment_runs": runs,
        "formal_round_table": study_review_payload(study_id)}


def _whole_records(ids: List[str], glossary: str, request: Request) -> List[Dict[str, Any]]:
    records, total = [], 0
    for study_id in ids:
        record = _grounding(study_id, glossary, request)
        size = len(canonical_bytes(record))
        if size > MAX_RECORD_BYTES:
            raise HTTPException(status_code=413, detail={"refused_record": study_id,
                "complete_record_bytes": size, "maximum_bytes": MAX_RECORD_BYTES,
                "deterministic_view_required": "conversation-study-view/v1",
                "reason": "The record cannot enter whole. Build the named deterministic view; fragments are forbidden."})
        records.append(record); total += size
    if total > MAX_CONTEXT_BYTES:
        raise HTTPException(status_code=413, detail={"complete_context_bytes": total,
            "maximum_bytes": MAX_CONTEXT_BYTES,
            "deterministic_view_required": "conversation-corpus-view/v1",
            "reason": "The selected records cannot enter together whole. No chunks were substituted."})
    return records


def _selection(primary: str, question: str, glossary: str, max_records: int,
               request: Request):
    try:
        rows = StudyStore().summaries()
        for row in rows:
            if row.get("readable") and isinstance(row.get("study_id"), str):
                bundle, _ = StudyStore().load(row["study_id"])
                row["search_text"] = canonical_bytes(bundle.to_mapping()).decode("utf-8")
        ids = select_whole_records(rows, question, primary, max_records)
    except InvalidParameterError as refusal:
        raise HTTPException(status_code=400, detail=str(refusal))
    records = _whole_records(ids, glossary, request)
    receipt = grounding_receipt(records); verify_grounding(records, receipt)
    return ids, records, receipt


def _provider(provider_id: str, model_id: str) -> Dict[str, Any]:
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        raise HTTPException(status_code=400,
            detail="Provider %r is not registered; it was refused before any call." % provider_id)
    if not spec["supports_declared_json_schema"]:
        raise HTTPException(status_code=400,
            detail="Provider %r cannot honour the declared response schema." % provider_id)
    return dict(spec, model_id=model_id)


def _model_prompt(records: List[Mapping[str, Any]], turns: List[Mapping[str, Any]], question: str):
    scientific = json.dumps({"complete_source_records": records}, sort_keys=True, ensure_ascii=False)
    dialogue = json.dumps({"conversation_so_far": [
        {"role": turn["role"], "text": turn["text"]} for turn in turns],
        "researcher_question": question}, sort_keys=True, ensure_ascii=False)
    return scientific, dialogue


def _ask_model(*, key: str, model_id: str, records: List[Mapping[str, Any]],
               turns: List[Mapping[str, Any]], question: str) -> Dict[str, Any]:
    scientific, dialogue = _model_prompt(records, turns, question)
    prompt = ("Answer only from COMPLETE SOURCE RECORDS supplied afresh below. Dialogue is continuity, "
        "never scientific authority. Preserve refusals, limitations and round-table dissent. "
        "Distinguish measurements, recorded commentary and interpretation. If records do not answer, say so. "
        "Return exactly answer, records_used, cautions, suggested_questions as JSON.\n\n"
        "COMPLETE SOURCE RECORDS (authoritative):\n" + scientific +
        "\n\nDIALOGUE (non-authoritative):\n" + dialogue)
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "medium"},
        "maxOutputTokens": MAX_OUTPUT_TOKENS, "responseMimeType": "application/json",
        "responseJsonSchema": {"type": "object", "properties": {
            "answer": {"type": "string"}, "records_used": {"type": "array", "items": {"type": "string"}},
            "cautions": {"type": "array", "items": {"type": "string"}},
            "suggested_questions": {"type": "array", "items": {"type": "string"}}},
            "required": ["answer", "records_used", "cautions", "suggested_questions"],
            "additionalProperties": False}}}
    try:
        response = httpx.post("%s/models/%s:generateContent" % (GEMINI_API_ROOT, model_id),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"}, json=payload, timeout=180.0)
        response.raise_for_status(); raw = response.json()
        text = "".join(part.get("text", "") for part in raw["candidates"][0]["content"]["parts"]).strip()
        answer = json.loads(text)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise DataSourceError("the discussion model did not return a usable answer") from exc
    required = {"answer", "records_used", "cautions", "suggested_questions"}
    if not isinstance(answer, dict) or set(answer) != required or not str(answer.get("answer", "")).strip() \
            or any(not isinstance(answer.get(name), list) or any(not isinstance(x, str) for x in answer[name])
                   for name in required - {"answer"}):
        raise DataSourceError("the discussion model response did not match its declared shape")
    return {**answer, "usage": raw.get("usageMetadata", {})}


def _session(body: AskFindingRequest, study_id: str) -> _Session:
    provider = _provider(body.provider_id, body.model_id)
    with _SESSION_LOCK:
        if body.conversation_id:
            session = _SESSIONS.get(body.conversation_id)
            if session is None:
                raise HTTPException(status_code=410, detail="This private conversation expired; start a new one.")
            fixed = (study_id, body.glossary, body.provider_id, body.model_id)
            actual = (session.primary_study_id, session.glossary, session.provider_id, session.model_id)
            if fixed != actual:
                raise HTTPException(status_code=409,
                    detail="The conversation's study, glossary, provider and model are immutable.")
            return session
        session = _Session(uuid.uuid4().hex, study_id, body.glossary, provider["provider_id"],
            body.model_id, body.max_calls, body.max_total_tokens)
        _SESSIONS[session.conversation_id] = session
        return session


@router.get("/studies/{study_id}/context")
def conversation_context(study_id: str, request: Request,
                         glossary: str = Query(default="reanalysis")) -> Dict[str, Any]:
    grounding = _grounding(study_id, glossary, request)
    return {"schema": "finding-conversation-context/v2", "study_id": study_id,
        "bundle_sha256": grounding["bundle_sha256"], "bundle_revision": grounding["bundle_revision"],
        "glossary": glossary, "formal_review_count": len(grounding["formal_round_table"]["reviews"]),
        "run_count": len(grounding["experiment_runs"]), "model_id": DEFAULT_REVIEW_MODEL,
        "providers": list(PROVIDERS.values()), "key_present": _key() is not None,
        "key_variables": list(KEY_VARIABLES), "recording": "OFF", "whole_corpus_selection": True,
        "default_budget": {"max_calls": DEFAULT_MAX_CALLS, "max_total_tokens": DEFAULT_MAX_TOTAL_TOKENS},
        "what_is_loaded": ["final finding", "complete evidence bundle", "experiment runs",
                           "formal round-table transcripts, outcomes and usage"],
        "claim_boundary": CLAIM_BOUNDARY, "network_used": False}


@router.post("/corpus/select")
def select_corpus(body: CorpusSelectionRequest, request: Request) -> Dict[str, Any]:
    ids, records, receipt = _selection(body.primary_study_id, body.question, body.glossary,
                                       body.max_records, request)
    return {"schema": "whole-record-corpus-selection/v1", "selected_study_ids": ids,
        "selection_method": "deterministic term overlap; primary study always included",
        "record_granularity": "complete", "no_fragments": True, "grounding": receipt,
        "records": [{"study_id": r["study_id"], "bundle_sha256": r["bundle_sha256"],
                     "bundle_revision": r["bundle_revision"]} for r in records]}


@router.post("/studies/{study_id}/ask")
def ask_about_finding(study_id: str, body: AskFindingRequest, request: Request) -> Dict[str, Any]:
    if not body.i_authorise_paid_call:
        raise HTTPException(status_code=400, detail="Authorise one paid external model call. Nothing was sent.")
    key = _key()
    if key is None:
        raise HTTPException(status_code=400, detail="No discussion API key is configured. Nothing was sent.")
    session = _session(body, study_id)
    ids, records, receipt = _selection(study_id, body.question, body.glossary, body.max_records, request)
    if records[0]["bundle_sha256"] != body.expected_bundle_sha256:
        raise HTTPException(status_code=409, detail="The study changed. Reload it before continuing the discussion.")
    for record in records:
        pinned = session.bindings.get(record["study_id"])
        current = {"bundle_sha256": record["bundle_sha256"], "bundle_revision": record["bundle_revision"]}
        if pinned is not None and pinned != current:
            raise HTTPException(status_code=409,
                detail="A study touched by this conversation changed; start a new discussion.")
    scientific, dialogue = _model_prompt(records, session.turns, body.question)
    projected = session.tokens_used + (len(scientific) + len(dialogue) + 3) // 4 + MAX_OUTPUT_TOKENS
    if session.calls_used >= session.max_calls or projected > session.max_total_tokens:
        raise HTTPException(status_code=409, detail={"reason": "conversation budget exhausted",
            "calls_used": session.calls_used, "max_calls": session.max_calls,
            "tokens_used": session.tokens_used, "projected_tokens": projected,
            "max_total_tokens": session.max_total_tokens, "nothing_was_sent": True})
    try:
        result = _ask_model(key=key, model_id=body.model_id, records=records,
            turns=session.turns, question=body.question)
    except DataSourceError as refusal:
        raise HTTPException(status_code=502, detail=str(refusal))
    provider = _provider(body.provider_id, body.model_id)
    session.turns.append(append_turn(session.turns, role="researcher", text=body.question,
                                     grounding=receipt))
    session.turns.append(append_turn(session.turns, role="model", text=result["answer"],
        grounding=receipt, provider=provider, usage=result["usage"]))
    session.calls_used += 1
    actual = result["usage"].get("totalTokenCount") if isinstance(result["usage"], Mapping) else None
    session.tokens_used += int(actual) if isinstance(actual, (int, float)) else projected - session.tokens_used
    for record in records:
        session.bindings[record["study_id"]] = {"bundle_sha256": record["bundle_sha256"],
                                                "bundle_revision": record["bundle_revision"]}
    proof = verify_transcript_independence(session.turns,
        [StudyStore().load(sid)[0] for sid in sorted(session.bindings)])
    return {"schema": "ephemeral-finding-answer/v2", "conversation_id": session.conversation_id,
        "study_id": study_id, "bundle_sha256": records[0]["bundle_sha256"], "answer": result["answer"],
        "records_used": result["records_used"], "cautions": result["cautions"],
        "suggested_questions": result["suggested_questions"], "turns": session.turns,
        "selected_study_ids": ids, "grounding": receipt, "provider": provider, "usage": result["usage"],
        "budget": {"calls_used": session.calls_used, "max_calls": session.max_calls,
                   "tokens_used": session.tokens_used, "max_total_tokens": session.max_total_tokens},
        "independence": proof, "recorded": False, "claim_boundary": CLAIM_BOUNDARY}


@router.post("/studies/{study_id}/save")
def save_finding_discussion(study_id: str, body: SaveFindingDiscussionRequest,
                            request: Request) -> Dict[str, Any]:
    with _SESSION_LOCK:
        session = _SESSIONS.get(body.conversation_id)
    if session is None or session.primary_study_id != study_id:
        raise HTTPException(status_code=404, detail="No live private conversation matches this study.")
    bundles = []
    for bound_id, binding in sorted(session.bindings.items()):
        bundle, _ = StudyStore().load(bound_id)
        if {"bundle_sha256": bundle.bundle_sha256, "bundle_revision": bundle.revision} != binding:
            raise HTTPException(status_code=409, detail="A touched study changed. The discussion was not saved.")
        bundles.append(bundle)
    proof = verify_transcript_independence(session.turns, bundles)
    payload = {"schema": "saved-record-conversation/v2", "conversation_schema": CONVERSATION_SCHEMA,
        "conversation_id": session.conversation_id, "primary_study_id": study_id,
        "title": body.title or "Discussion of %s" % study_id,
        "saved_at": datetime.now(timezone.utc).isoformat(), "bindings": session.bindings,
        "provider_id": session.provider_id, "model_id": session.model_id,
        "budget": {"calls_used": session.calls_used, "max_calls": session.max_calls,
                   "tokens_used": session.tokens_used, "max_total_tokens": session.max_total_tokens},
        "turns": session.turns, "independence": proof, "recording_was_explicitly_requested": True,
        "claim_boundary": "Saved interpretation beside evidence; never evidence itself."}
    digest = sha256(payload); record = dict(payload, discussion_sha256=digest)
    root = _conversation_root(); root.mkdir(parents=True, exist_ok=True)
    path = root / ("%s.%s.json" % (study_id, digest[:16]))
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False); handle.write("\n")
    except FileExistsError:
        pass
    return {"saved": True, "file": path.name, "discussion_sha256": digest, "recorded": True,
        "independence": proof, "claim_boundary": payload["claim_boundary"]}


__all__ = ["router", "CONVERSATION_ROOT_ENV", "DEFAULT_CONVERSATION_ROOT"]
