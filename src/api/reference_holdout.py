"""HTTP surface for the complete T4E.39 temporal reference holdout and its human review.

Every stage of the holdout -- declaration, adoption, catalogue census, exact field plan,
acquisition, materialisation, measurement and review -- is readable here, and each one is
re-verified from its own bytes on the way out rather than being read back as a claim. Two
stages are deliberately not executable from this surface: acquisition reached a network under
a separate named authorization, and the reserved period has now been opened once, as declared.
A button that re-ran either would be offering a second holdout, which is exactly what the
pre-opening declaration exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from src.benchmarks.reference_holdout import DEFAULT_CENSUS, DEFAULT_DECLARATION
from src.benchmarks.reference_holdout_fields import DEFAULT_OUTPUT, plan_holdout_fields
from src.core.adoption import REQUIRED_AFFIRMATION, AdoptionRefused, sign_declaration
from src.core.errors import SpectralEarthError
from src.core.holdout_review import (
    PERMITTED_ASSESSMENTS, REQUIRED_VERDICT, HoldoutResultReview,
    build_holdout_review_request, load_adopted_holdout_review, review_state,
    verified_holdout_result)


router = APIRouter(prefix="/api/v1/reference-holdout", tags=["reference-holdout"])

REVIEW_FILE = "t4e39-holdout-result-review.json"


def _declaration(request: Request) -> Path:
    return Path(getattr(request.app.state, "holdout_declaration", DEFAULT_DECLARATION))


def _census(request: Request) -> Path:
    return Path(getattr(request.app.state, "holdout_census", DEFAULT_CENSUS))


def _measurement(request: Request) -> Path:
    return Path(getattr(request.app.state, "holdout_measurement", DEFAULT_OUTPUT))


def _review_dir(request: Request) -> Path:
    return Path(getattr(request.app.state, "holdout_review_dir",
                        _declaration(request).parent))


def _paths(request: Request) -> Dict[str, Path]:
    return {"declaration_path": _declaration(request), "census_path": _census(request),
            "measurement_path": _measurement(request)}


def _verified(request: Request) -> Dict[str, Any]:
    try:
        return verified_holdout_result(**_paths(request))
    except SpectralEarthError as error:
        raise HTTPException(status_code=409, detail=str(error))


def _read(path: Path) -> Dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise HTTPException(status_code=503, detail="%s is unavailable: %s" % (path, error))
    if not isinstance(body, dict):
        raise HTTPException(status_code=503, detail="%s does not contain a JSON object" % path)
    return body


def _artifact_paths(request: Request) -> Dict[str, Path]:
    declaration = _declaration(request)
    verified = _verified(request)
    adoption = declaration.parent / str(verified["adoption"].get("adoption_file"))
    return {
        "declaration": declaration,
        "adoption": adoption,
        "census": _census(request),
        "field-record": Path(verified["field_record_receipt"]),
        "measurement": _measurement(request),
    }


def _field_plan(request: Request) -> Dict[str, Any]:
    """Re-plan the four exact requests offline, so the UI shows a live plan, not a memory."""
    try:
        return plan_holdout_fields(_declaration(request), _census(request))
    except SpectralEarthError as error:
        raise HTTPException(status_code=409, detail=str(error))


def _stages(request: Request, verified: Dict[str, Any], plan: Dict[str, Any],
            review: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe the flow in the order it was actually performed, with what proves each step."""
    census = verified["census"]
    decision = verified["decision"]
    adoption = verified["adoption"]
    return [
        {
            "stage": "declaration",
            "title": "Freeze the holdout before opening it",
            "status": "FROZEN",
            "performed_by": "instrument",
            "artifact": "declaration",
            "digest": verified["declaration_sha256"],
            "facts": {
                "dates": census["selection"]["dates"],
                "latitude": census["selection"]["latitude"],
                "longitude": census["selection"]["longitude"],
                "synoptic_hours_utc": census["selection"]["synoptic_hours"],
                "minimum_agency_fixes": census["selection"]["minimum_agency_fixes"],
                "interior_margin_degrees": census["selection"]["interior_margin_degrees"],
                "minimum_selected_storms": census["minimum_selected_storms"],
                "strict_majority_formula": "floor(n / 2) + 1",
            },
            "boundary": "Selection, decision rule and verdict were fixed before any holdout "
                        "identity was parsed. Nothing downstream may re-select or reweight.",
        },
        {
            "stage": "adoption",
            "title": "A named person adopts the exact declaration",
            "status": "ADOPTED" if adoption.get("adopted") else "NOT_ADOPTED",
            "performed_by": "human",
            "artifact": "adoption",
            "digest": adoption.get("adopts_sha256"),
            "facts": {
                "adopted_by": adoption.get("adopted_by"),
                "adopted_on": adoption.get("adopted_on"),
                "required_affirmation": adoption.get("required_affirmation"),
                "signature_still_reaches_the_declaration": adoption.get(
                    "signature_still_reaches_the_declaration"),
            },
            "boundary": "Adopting the census design authorized no ERA5 access; the network act "
                        "required its own separate, experiment-specific authorization.",
        },
        {
            "stage": "census",
            "title": "Open the signed catalogue only",
            "status": census["status"],
            "performed_by": "instrument",
            "artifact": "census",
            "digest": verified["census_receipt_sha256"],
            "facts": {
                **census["census"],
                "selected_storms": census["selected_storms"],
                "signed_reference_sha256": census["signed_reference"]["expected_sha256"],
                "era5_record_opened": census["era5_record_opened"],
                "network_used": census["network_used"],
            },
            "boundary": census["claim_boundary"],
        },
        {
            "stage": "field-plan",
            "title": "Plan the four exact field requests offline",
            "status": "PLANNED",
            "performed_by": "instrument",
            "artifact": None,
            "digest": None,
            "facts": {
                "streams": len(plan["requests"]),
                "requests": [{"field": item["field"], "segment": item["segment"],
                              "shards": len(item["shards"])} for item in plan["requests"]],
                "total_shards": plan["total_shards"],
                "total_raw_value_bytes": plan["total_raw_value_bytes"],
                "record_opened": plan["record_opened"],
                "network_used": plan["network_used"],
            },
            "boundary": "Planning is metadata only. It opened no value and reached no network. "
                        "This plan is recomputed on every read and must still match what was "
                        "acquired.",
        },
        {
            "stage": "acquisition",
            "title": "Acquire the 48 exact-time shards",
            "status": "ACQUIRED",
            "performed_by": "human-authorized instrument",
            "artifact": "field-record",
            "digest": verified["field_record_receipt_sha256"],
            "facts": {
                "shards_per_stream": verified["shards_per_stream"],
                "every_shard_digest_recorded": True,
            },
            "boundary": "This is the one stage that reached a network, under a separate named "
                        "authorization. It is not repeatable from this surface.",
        },
        {
            "stage": "materialisation",
            "title": "Join the segments into one immutable record",
            "status": "PUBLISHED" if verified["seam_all_passed"] else "REFUSED",
            "performed_by": "instrument",
            "artifact": "field-record",
            "digest": verified["record_sha256"],
            "facts": {
                "shape": verified["record_shape"],
                "seam_checks": verified["seam_checks"],
                "seam_all_passed": verified["seam_all_passed"],
                "record_path": verified["record_path"],
            },
            "boundary": "Seam disagreement beyond source-encoding tolerance refuses the record "
                        "rather than averaging the two segments.",
        },
        {
            "stage": "measurement",
            "title": "Run the frozen basin comparison",
            "status": verified["outcome"],
            "performed_by": "instrument",
            "artifact": "measurement",
            "digest": verified["measurement_receipt_sha256"],
            "facts": {
                "counts": decision.get("counts", {}),
                "population_denominator": decision.get("population_denominator"),
                "strict_majority_needed": decision.get("strict_majority_needed"),
                "is_acceptance_verdict": decision.get("is_acceptance_verdict"),
                "VERDICT": verified["verdict"],
                "method": verified["measurement"].get("method", {}),
            },
            "boundary": verified["claim_boundary"],
        },
        {
            "stage": "review",
            "title": "A named person reads the result and its boundary",
            "status": review.get("status"),
            "performed_by": "human",
            "artifact": "review",
            "digest": review.get("declaration_sha256"),
            "facts": {
                "permitted_assessments": list(PERMITTED_ASSESSMENTS),
                "acceptance_available": False,
                "binds_current_result": review.get("binds_current_result"),
            },
            "boundary": "A review records a reading, not an acceptance. The verdict stays "
                        "%s however the review is written." % REQUIRED_VERDICT,
        },
    ]


@router.get("")
def read_holdout(request: Request) -> Dict[str, Any]:
    """Return every stage of the completed holdout, re-verified, with its human review state."""
    verified = _verified(request)
    review = review_state(_review_dir(request), REVIEW_FILE, verified=verified)
    plan = _field_plan(request)
    if plan["total_shards"] != sum(verified["shards_per_stream"].values()):
        raise HTTPException(
            status_code=409,
            detail="the recomputed T4E.39 field plan no longer matches what was acquired")
    try:
        review_request = build_holdout_review_request(**_paths(request))
    except SpectralEarthError as error:
        raise HTTPException(status_code=409, detail=str(error))
    decision = verified["decision"]
    counts = decision.get("counts", {})
    needed = int(decision.get("strict_majority_needed", 0))
    leading = max(int(value) for value in counts.values()) if counts else 0
    return {
        "schema": "t4e39-holdout-surface/v1",
        "task": "T4E.39",
        "outcome": verified["outcome"],
        "verdict": verified["verdict"],
        "decision": decision,
        "margin_over_strict_majority": leading - needed,
        "one_row_would_change_the_outcome": leading == needed,
        "stages": _stages(request, verified, plan, review),
        "review_request": review_request,
        "review": review,
        "artifacts": sorted(_artifact_paths(request)),
        "operations": [
            {"operation": "inspect every stage artifact and its digests",
             "available_in_ui": True},
            {"operation": "inspect all measured rows and both basin walks",
             "available_in_ui": True},
            {"operation": "write the human result review", "available_in_ui": True},
            {"operation": "adopt the written review", "available_in_ui": True},
            {"operation": "acquire ERA5 fields", "available_in_ui": False,
             "reason": "Acquisition reaches a network under a separate named authorization "
                       "and remains an explicit CLI job."},
            {"operation": "re-run the holdout", "available_in_ui": False,
             "reason": "The reserved 2022-2023 period has been opened once, as declared. "
                       "Re-running it would manufacture a second holdout from a spent one."},
        ],
        "network_used": False,
        "claim_boundary": verified["claim_boundary"],
    }


@router.get("/rows")
def read_rows(request: Request) -> Dict[str, Any]:
    """Return every measured storm row, including both complete basin walks."""
    verified = _verified(request)
    measurement = verified["measurement"]
    return {
        "schema": "t4e39-holdout-rows/v1",
        "measurement_receipt_sha256": verified["measurement_receipt_sha256"],
        "method": measurement.get("method", {}),
        "decision": verified["decision"],
        "rows": measurement.get("rows", []),
        "network_used": False,
        "claim_boundary": verified["claim_boundary"],
    }


@router.get("/artifacts/{name}")
def read_artifact(request: Request, name: str) -> Dict[str, Any]:
    """Return one complete stage artifact exactly as stored, beside its recomputed digest."""
    paths = _artifact_paths(request)
    review_path = _review_dir(request) / REVIEW_FILE
    if name == "review":
        if not review_path.exists():
            raise HTTPException(status_code=404, detail="the T4E.39 review is not written")
        path = review_path
    elif name in paths:
        path = paths[name]
    else:
        raise HTTPException(
            status_code=404,
            detail="unknown T4E.39 artifact %r; available: %s" % (
                name, ", ".join(sorted(paths) + ["review"])))
    body = _read(path)
    return {
        "schema": "t4e39-holdout-artifact/v1",
        "artifact": name,
        "path": str(path).replace("\\", "/"),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "body": body,
        "network_used": False,
    }


class ReviewBody(BaseModel):
    boundary_assessment: Literal["BOUNDARY_SOUND", "BOUNDARY_DISPUTED"]
    basis: str = Field(..., min_length=1)
    what_this_does_not_establish: List[str] = Field(..., min_items=1)
    limitations: str = Field(..., min_length=1)
    next_action: str = Field(..., min_length=1)
    claim_boundary: str = Field(..., min_length=1)


@router.post("/review")
def write_review(request: Request, body: ReviewBody) -> Dict[str, Any]:
    """Write the reviewer's reading while supplying only machine-verifiable bindings."""
    try:
        review_request = build_holdout_review_request(**_paths(request))
    except SpectralEarthError as error:
        raise HTTPException(status_code=409, detail=str(error))
    bound = {name: review_request[name] for name in (
        "declaration_sha256", "census_receipt_sha256", "field_record_receipt_sha256",
        "measurement_receipt_sha256", "reviewed_outcome", "reviewed_verdict",
        "population_denominator", "strict_majority_needed")}
    try:
        declaration = HoldoutResultReview.parse_obj(
            {"schema": review_request["review_schema"], "task": "T4E.39",
             **bound, **body.dict()})
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    directory = _review_dir(request)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REVIEW_FILE
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(json.loads(declaration.json(by_alias=True)), indent=2) + "\n")
    except FileExistsError:
        raise HTTPException(status_code=409, detail="the T4E.39 result review already exists")
    return {
        "status": "WRITTEN_NOT_ADOPTED",
        "declaration": json.loads(declaration.json(by_alias=True)),
        "declaration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verdict": REQUIRED_VERDICT,
        "network_used": False,
    }


class AdoptBody(BaseModel):
    adopted_by: str
    adopted_as: str
    what_was_adopted: str
    affirmation: str
    why: Optional[str] = None


@router.post("/review/adopt")
def adopt_review(request: Request, body: AdoptBody) -> Dict[str, Any]:
    """Record a named person's typed adoption, then verify it against the live result."""
    directory = _review_dir(request)
    try:
        adoption = sign_declaration(
            directory, REVIEW_FILE, adopted_by=body.adopted_by, adopted_as=body.adopted_as,
            what_was_adopted=body.what_was_adopted, affirmation=body.affirmation, why=body.why)
        verified = load_adopted_holdout_review(directory / REVIEW_FILE, **_paths(request))
    except (AdoptionRefused, SpectralEarthError, ValidationError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error))
    return {
        "status": "ADOPTED",
        "adoption": adoption.describe(),
        "binding": verified.binding,
        "boundary_assessment": verified.review.boundary_assessment,
        "verdict": REQUIRED_VERDICT,
        "required_affirmation": REQUIRED_AFFIRMATION,
        "network_used": False,
    }
