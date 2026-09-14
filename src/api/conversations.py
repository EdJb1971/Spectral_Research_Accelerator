"""Ephemeral, source-grounded discussion of a published finding.

Every turn reloads the complete evidence bundle, derived finding and formal review. The client
supplies dialogue history for conversational continuity, but never scientific context. Nothing is
written unless the separate save route is called explicitly, and neither route can alter evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.api.findings import (
    StudyStore, _declaration_payload, _figures_for, _unadmitted_reading,
)
from src.api.reviews import DEFAULT_REVIEW_MODEL, KEY_VARIABLES, _key, study_review_payload
from src.core.errors import DataSourceError, InvalidParameterError
from src.core.experiment_run import RunStore
from src.core.five_outputs import summarise_evidence
from src.core.review_cost import GEMINI_API_ROOT
from src.core.translation import glossary_for, translate


router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])
CONVERSATION_ROOT_ENV = "SPECTRAL_CONVERSATION_ROOT"
DEFAULT_CONVERSATION_ROOT = Path("data") / "conversations"
MAX_HISTORY_TURNS = 20
MAX_TURN_CHARS = 8_000


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)


class AskFindingRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_TURN_CHARS)
    history: List[ConversationTurn] = Field(default_factory=list, max_items=MAX_HISTORY_TURNS)
    glossary: str = Field(default="reanalysis", min_length=1, max_length=100)
    model_id: str = Field(default=DEFAULT_REVIEW_MODEL, min_length=1, max_length=100)
    expected_bundle_sha256: str = Field(..., min_length=64, max_length=64)
    i_authorise_paid_call: bool = False


class SaveFindingDiscussionRequest(BaseModel):
    turns: List[ConversationTurn] = Field(..., min_items=2, max_items=100)
    glossary: str = Field(default="reanalysis", min_length=1, max_length=100)
    expected_bundle_sha256: str = Field(..., min_length=64, max_length=64)
    title: Optional[str] = Field(default=None, max_length=160)


def _run_root(request: Request) -> Path:
    root = getattr(request.app.state, "experiment_run_dir", None)
    return Path(root or os.getenv("EXPERIMENT_RUN_DIR", "data/experiment_runs"))


def _conversation_root() -> Path:
    return Path(os.getenv(CONVERSATION_ROOT_ENV) or DEFAULT_CONVERSATION_ROOT)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _grounding(study_id: str, glossary_name: str, request: Request) -> Dict[str, Any]:
    try:
        bundle, bundle_path = StudyStore().load(study_id)
    except (KeyError, FileNotFoundError, InvalidParameterError) as refusal:
        raise HTTPException(status_code=404, detail=str(refusal))
    try:
        glossary = glossary_for(glossary_name)
    except Exception:
        raise HTTPException(status_code=404, detail="no glossary registered as %r" % glossary_name)

    outputs = summarise_evidence(bundle)
    document = translate(outputs, glossary, figures=_figures_for(bundle))
    translated = dict(
        document.to_mapping(),
        rendered_text=document.render(),
        figures_text=None if document.figures is None else document.figures.render(),
        domain_limits=_declaration_payload(glossary_name),
        unadmitted_reading=_unadmitted_reading(outputs.rung, glossary_name),
    )
    runs = [row for row in RunStore(_run_root(request)).list_runs()
            if row.get("study_id") == study_id]
    review = study_review_payload(study_id)
    return {
        "study_id": study_id,
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_revision": bundle.revision,
        "evidence_bundle": bundle.to_mapping(),
        "derived_finding": dict(outputs.to_mapping(), summary_sha256=outputs.summary_sha256),
        "finding_in_selected_vocabulary": translated,
        "experiment_runs": runs,
        "formal_round_table": review,
    }


def _prompt(grounding: Mapping[str, Any], history: List[ConversationTurn], question: str) -> str:
    return json.dumps({
        "complete_source_records": grounding,
        "conversation_so_far": [turn.dict() for turn in history],
        "researcher_question": question,
    }, sort_keys=True, ensure_ascii=False)


def _ask_model(*, key: str, model_id: str, grounding: Mapping[str, Any],
               history: List[ConversationTurn], question: str) -> Dict[str, Any]:
    system = (
        "You help a researcher understand one completed scientific study. Answer only from the "
        "complete source records supplied on this turn, including their limitations and any formal "
        "round-table dissent. Distinguish measured facts, panel commentary, and your interpretation. "
        "Never promote the claim, invent missing evidence, or imply this conversation changes it. "
        "If the records do not answer the question, say so plainly. Return JSON with exactly answer, "
        "records_used, cautions, and suggested_questions."
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": _prompt(grounding, history, question)}]}],
        "generationConfig": {
            "thinkingConfig": {"thinkingLevel": "medium"},
            "responseMimeType": "application/json",
            "responseJsonSchema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "records_used": {"type": "array", "items": {"type": "string"}},
                    "cautions": {"type": "array", "items": {"type": "string"}},
                    "suggested_questions": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["answer", "records_used", "cautions", "suggested_questions"],
                "additionalProperties": False,
            },
        },
    }
    try:
        response = httpx.post(
            "%s/models/%s:generateContent" % (GEMINI_API_ROOT, model_id),
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=payload, timeout=180.0,
        )
        response.raise_for_status()
        raw = response.json()
        parts = raw["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        answer = json.loads(text)
    except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise DataSourceError("the discussion model did not return a usable answer") from exc
    required = {"answer", "records_used", "cautions", "suggested_questions"}
    list_fields = ("records_used", "cautions", "suggested_questions")
    if (not isinstance(answer, dict) or set(answer) != required
            or not isinstance(answer.get("answer"), str) or not answer["answer"].strip()
            or any(not isinstance(answer.get(field), list) for field in list_fields)
            or any(not isinstance(item, str) for field in list_fields for item in answer[field])):
        raise DataSourceError("the discussion model response did not match its declared shape")
    return {**answer, "usage": raw.get("usageMetadata", {})}


@router.get("/studies/{study_id}/context")
def conversation_context(study_id: str, request: Request,
                         glossary: str = Query(default="reanalysis")) -> Dict[str, Any]:
    grounding = _grounding(study_id, glossary, request)
    return {
        "schema": "finding-conversation-context/v1",
        "study_id": study_id,
        "bundle_sha256": grounding["bundle_sha256"],
        "bundle_revision": grounding["bundle_revision"],
        "glossary": glossary,
        "formal_review_count": len(grounding["formal_round_table"]["reviews"]),
        "run_count": len(grounding["experiment_runs"]),
        "model_id": DEFAULT_REVIEW_MODEL,
        "key_present": _key() is not None,
        "key_variables": list(KEY_VARIABLES),
        "recording": "OFF",
        "what_is_loaded": ["final finding", "complete evidence bundle", "experiment runs",
                           "formal round-table transcripts, outcomes and usage"],
        "claim_boundary": "Discussion helps interpret the record; it cannot change its evidence status.",
        "network_used": False,
    }


@router.post("/studies/{study_id}/ask")
def ask_about_finding(study_id: str, body: AskFindingRequest, request: Request) -> Dict[str, Any]:
    if not body.i_authorise_paid_call:
        raise HTTPException(status_code=400, detail="Authorise one paid external model call. Nothing was sent.")
    key = _key()
    if key is None:
        raise HTTPException(status_code=400, detail="No discussion API key is configured. Nothing was sent.")
    grounding = _grounding(study_id, body.glossary, request)
    if grounding["bundle_sha256"] != body.expected_bundle_sha256:
        raise HTTPException(status_code=409, detail="The study changed. Reload it before continuing the discussion.")
    try:
        result = _ask_model(key=key, model_id=body.model_id, grounding=grounding,
                            history=body.history, question=body.question)
    except DataSourceError as refusal:
        raise HTTPException(status_code=502, detail=str(refusal))
    return {
        "schema": "ephemeral-finding-answer/v1",
        "study_id": study_id,
        "bundle_sha256": grounding["bundle_sha256"],
        "answer": result["answer"],
        "records_used": result["records_used"],
        "cautions": result["cautions"],
        "suggested_questions": result["suggested_questions"],
        "usage": result["usage"],
        "recorded": False,
        "claim_boundary": "This is an interpretation, not evidence, and it changed no finding.",
    }


@router.post("/studies/{study_id}/save")
def save_finding_discussion(study_id: str, body: SaveFindingDiscussionRequest,
                            request: Request) -> Dict[str, Any]:
    grounding = _grounding(study_id, body.glossary, request)
    if grounding["bundle_sha256"] != body.expected_bundle_sha256:
        raise HTTPException(status_code=409, detail="The study changed. This discussion was not saved.")
    saved_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema": "saved-finding-discussion/v1",
        "study_id": study_id,
        "bundle_sha256": grounding["bundle_sha256"],
        "bundle_revision": grounding["bundle_revision"],
        "glossary": body.glossary,
        "title": body.title or "Discussion of %s" % study_id,
        "saved_at": saved_at,
        "turns": [turn.dict() for turn in body.turns],
        "recording_was_explicitly_requested": True,
        "claim_boundary": "Saved interpretation beside the evidence; never evidence itself.",
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    record = dict(payload, discussion_sha256=digest)
    root = _conversation_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("%s.%s.json" % (study_id, digest[:16]))
    if not path.exists():
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"saved": True, "file": path.name, "discussion_sha256": digest,
            "recorded": True, "claim_boundary": record["claim_boundary"]}


__all__ = ["router", "CONVERSATION_ROOT_ENV", "DEFAULT_CONVERSATION_ROOT"]
