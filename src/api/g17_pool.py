"""HTTP surface for the corrected G17 real-record pool and its human review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from src.core.adoption import REQUIRED_AFFIRMATION, AdoptionRefused, adoption_state, sign_declaration
from src.core.errors import SpectralEarthError
from src.core.real_pool_curation import (
    PoolCurationReview, build_curation_review_request, load_adopted_curation_review)


router = APIRouter(prefix="/api/v1/g17-pool", tags=["g17-pool"])


def _measurements(request: Request) -> Path:
    return Path(getattr(request.app.state, "g17_measurement_dir", "measurements"))


def _studies(request: Request) -> Path:
    return Path(getattr(request.app.state, "g17_study_dir", "data/studies"))


def _profiles(request: Request) -> tuple[Path, ...]:
    value = getattr(request.app.state, "g17_profile_roots", None)
    return tuple(Path(path) for path in value) if value else (
        Path("data/profile_collections"), Path("data/channels"))


def _read(path: Path) -> Dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise HTTPException(status_code=503, detail="%s is unavailable: %s" % (path, error))
    if not isinstance(body, dict):
        raise HTTPException(status_code=503, detail="%s does not contain a JSON object" % path)
    return body


def _review_state(request: Request) -> Dict[str, Any]:
    directory = _studies(request)
    name = "g17_pool_curation_review.json"
    path = directory / name
    if not path.exists():
        return {"status": "NOT_WRITTEN", "adopted": False}
    body = _read(path)
    state = adoption_state(directory, name)
    return {
        "status": "ADOPTED" if state.get("adopted") else "WRITTEN_NOT_ADOPTED",
        "declaration": body,
        "declaration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "adoption": state,
    }


def _current_request(request: Request) -> Dict[str, Any]:
    measurements = _measurements(request)
    path = measurements / "g17_pool_curation_review_request.json"
    recorded = _read(path)
    try:
        current = build_curation_review_request(
            packet_path=measurements / "g17_pool_curation_packet.json",
            roots=_profiles(request))
    except SpectralEarthError as error:
        raise HTTPException(status_code=409, detail=str(error))
    if recorded != current:
        raise HTTPException(status_code=409, detail="the G17 review request is stale")
    return current


@router.get("")
def read_pool(request: Request) -> Dict[str, Any]:
    """Return the complete corrected pool state and its deliberately separate operations."""
    measurements = _measurements(request)
    archived = measurements / "superseded" / "g17_tess_lineage_bug"
    return {
        "schema": "g17-pool-surface/v1",
        "readiness": _read(measurements / "g17_real_pool_readiness.json"),
        "qualification": _read(
            measurements / "g17_tess_periodic_acquisition_qualified_104.json"),
        "assessment": _read(measurements / "g17_tess_periodic_pool_assessment_104.json"),
        "packet": _read(measurements / "g17_pool_curation_packet.json"),
        "review_request": _current_request(request),
        "review": _review_state(request),
        "archived_lineage_bug_artifacts": sorted(
            path.name for path in archived.glob("*.json")) if archived.is_dir() else [],
        "operations": [
            {"operation": "inspect evidence and archived lineage", "available_in_ui": True},
            {"operation": "write the human curation review", "available_in_ui": True},
            {"operation": "adopt the written review", "available_in_ui": True},
            {"operation": "discover and acquire external TESS products", "available_in_ui": False,
             "reason": "Network and bounded bulk-transfer operations remain explicit CLI jobs."},
            {"operation": "rebuild and export source-bound evidence", "available_in_ui": False,
             "reason": "Immutable offline evidence publication remains an auditable CLI pipeline."},
        ],
        "network_used": False,
        "claim_boundary": (
            "This surface displays verified artifacts and records explicit human acts. It does "
            "not infer exchangeability or run external acquisition."),
    }


class ReviewBody(BaseModel):
    exchangeability: Literal["ESTABLISHED", "NOT_ESTABLISHED"]
    basis: str = Field(..., min_length=1)
    unmeasured_properties: List[str] = Field(..., min_items=1)
    limitations: str = Field(..., min_length=1)
    claim_boundary: str = Field(..., min_length=1)


@router.post("/review")
def write_review(request: Request, body: ReviewBody) -> Dict[str, Any]:
    """Write the reviewer's decision while supplying only machine-verifiable bindings."""
    measurements = _measurements(request)
    review_request = _current_request(request)
    try:
        declaration = PoolCurationReview.parse_obj({
            "schema": "g17-pool-curation-review/v2",
            **{name: review_request[name] for name in (
                "study_id", "successor_declaration_sha256", "readiness_assessment_sha256",
                "admission_contract_sha256", "curation_packet_sha256", "record_ids")},
            **body.dict(),
        })
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    directory = _studies(request)
    path = directory / "g17_pool_curation_review.json"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(json.loads(declaration.json(by_alias=True)), indent=2) + "\n")
    except FileExistsError:
        raise HTTPException(status_code=409, detail="the G17 curation review already exists")
    return {"status": "WRITTEN_NOT_ADOPTED", "declaration": json.loads(
        declaration.json(by_alias=True)), "declaration_sha256": hashlib.sha256(
            path.read_bytes()).hexdigest(), "network_used": False}


class AdoptBody(BaseModel):
    adopted_by: str
    adopted_as: str
    what_was_adopted: str
    affirmation: str
    why: Optional[str] = None


@router.post("/review/adopt")
def adopt_review(request: Request, body: AdoptBody) -> Dict[str, Any]:
    """Record a named person's typed adoption, then verify it against the live packet."""
    directory = _studies(request)
    try:
        review_request = _current_request(request)
        review = PoolCurationReview.parse_obj(_read(
            directory / "g17_pool_curation_review.json"))
        expected = {name: review_request[name] for name in (
            "study_id", "successor_declaration_sha256", "readiness_assessment_sha256",
            "admission_contract_sha256", "curation_packet_sha256", "record_ids")}
        if {name: getattr(review, name) for name in expected} != expected:
            raise ValueError("the written G17 review does not bind the current packet")
        adoption = sign_declaration(
            directory, "g17_pool_curation_review.json", adopted_by=body.adopted_by,
            adopted_as=body.adopted_as, what_was_adopted=body.what_was_adopted,
            affirmation=body.affirmation, why=body.why)
        verified = load_adopted_curation_review(
            directory / "g17_pool_curation_review.json",
            packet_path=_measurements(request) / "g17_pool_curation_packet.json",
            roots=_profiles(request))
    except (AdoptionRefused, SpectralEarthError, ValidationError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "status": "ADOPTED", "adoption": adoption.describe(),
        "binding": verified.binding, "decision": verified.review.exchangeability,
        "required_affirmation": REQUIRED_AFFIRMATION, "network_used": False,
    }


__all__ = ["router"]