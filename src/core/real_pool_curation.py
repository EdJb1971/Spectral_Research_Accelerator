"""Human-adopted exchangeability review for an exact G17 profile inventory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

from pydantic import BaseModel, Field, validator

from src.core.adoption import REQUIRED_AFFIRMATION, adoption_state
from src.core.errors import UserInputError
from src.core.real_pool_curation_packet import build_curation_packet
from src.core.real_pool_readiness import DEFAULT_PROFILE_ROOTS, audit_real_pool_readiness
from src.core.scale_shape_successor import successor_scale_shape_manifest


SCHEMA = "g17-pool-curation-review/v2"
REQUEST_SCHEMA = "g17-pool-curation-review-request/v1"


class _FrozenModel(BaseModel):
    class Config:
        allow_mutation = False
        allow_population_by_field_name = True
        extra = "forbid"


class PoolCurationReview(_FrozenModel):
    """A person's conclusion about one exact, already review-ready inventory."""

    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")
    study_id: str = Field(..., min_length=1)
    successor_declaration_sha256: str
    readiness_assessment_sha256: str
    admission_contract_sha256: str
    curation_packet_sha256: str
    record_ids: List[str] = Field(..., min_items=1)
    exchangeability: Literal["ESTABLISHED", "NOT_ESTABLISHED"]
    basis: str = Field(..., min_length=1)
    unmeasured_properties: List[str] = Field(..., min_items=1)
    limitations: str = Field(..., min_length=1)
    claim_boundary: str = Field(..., min_length=1)

    @validator("successor_declaration_sha256", "readiness_assessment_sha256",
               "admission_contract_sha256", "curation_packet_sha256")
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value

    @validator("study_id", "basis", "limitations", "claim_boundary")
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty declaration")
        return value.strip()

    @validator("record_ids")
    def _unique_records(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("record_ids must be non-empty and unique")
        return cleaned

    @validator("unmeasured_properties")
    def _named_unknowns(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("unmeasured_properties must be non-empty and unique")
        return cleaned


@dataclass(frozen=True)
class AdoptedPoolCurationReview:
    review: PoolCurationReview
    binding: Dict[str, Any]


def build_curation_review_request(
    *, packet_path: Path, roots: Sequence[Path] = DEFAULT_PROFILE_ROOTS
) -> Dict[str, Any]:
    """Prepare exact review inputs without making or signing the human decision."""
    successor = successor_scale_shape_manifest()
    readiness = audit_real_pool_readiness(successor, roots=roots)
    expected_packet = build_curation_packet(roots=roots)
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("G17 curation packet is invalid: %s" % error) from error
    if packet != expected_packet:
        raise UserInputError("G17 curation packet does not match the exact current inventory")
    return {
        "schema": REQUEST_SCHEMA,
        "status": "AWAITING_HUMAN_REVIEW",
        "review_schema": SCHEMA,
        "study_id": successor.study_id,
        "successor_declaration_sha256": successor.digest,
        "readiness_assessment_sha256": readiness["assessment_sha256"],
        "admission_contract_sha256": successor.admission_contract["contract_sha256"],
        "curation_packet_sha256": packet["packet_sha256"],
        "record_ids": packet["recommended_record_ids"],
        "verified_facts": {
            "profile_count": packet["profile_count"],
            "selected_record_count": packet["recommended_size"],
            "alternatives_per_selected_record": packet[
                "alternatives_per_recommended_record"],
            "marginal_admission": "VERIFIED",
            "exchangeability": "NOT_ASSESSED",
        },
        "human_inputs_required": {
            "exchangeability": ["ESTABLISHED", "NOT_ESTABLISHED"],
            "basis": "A reviewer-authored scientific basis beyond inventory count.",
            "unmeasured_properties": "One or more explicitly named unknown properties.",
            "limitations": "The limits of the decision in the reviewer's own terms.",
            "attribution": "The reviewer's full name and role.",
            "affirmation": REQUIRED_AFFIRMATION,
        },
        "claim_boundary": (
            "This is a machine-prepared review request, not a curation decision or adoption. "
            "It cannot establish exchangeability and is not accepted by the review loader."),
    }


def load_adopted_curation_review(
    path: Path, *, packet_path: Path, roots: Sequence[Path] = DEFAULT_PROFILE_ROOTS
) -> AdoptedPoolCurationReview:
    """Load a human decision only while it binds the exact live, review-ready inventory."""
    successor = successor_scale_shape_manifest()
    readiness = audit_real_pool_readiness(successor, roots=roots)
    if readiness["status"] != "READY_FOR_CURATION_REVIEW":
        raise UserInputError(
            "G17 inventory must be READY_FOR_CURATION_REVIEW before a curation decision can "
            "be loaded; current status is %s" % readiness["status"])
    expected_packet = build_curation_packet(roots=roots)
    try:
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("G17 curation packet is invalid: %s" % error) from error
    if packet != expected_packet:
        raise UserInputError("G17 curation packet does not match the exact current inventory")

    try:
        raw = path.read_bytes()
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        review = PoolCurationReview.parse_obj(body)
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("G17 pool curation review is invalid: %s" % error) from error

    expected = {
        "study_id": successor.study_id,
        "successor_declaration_sha256": successor.digest,
        "readiness_assessment_sha256": readiness["assessment_sha256"],
        "admission_contract_sha256": successor.admission_contract["contract_sha256"],
        "curation_packet_sha256": packet["packet_sha256"],
        "record_ids": packet["recommended_record_ids"],
    }
    observed = {name: getattr(review, name) for name in expected}
    if observed != expected:
        raise UserInputError(
            "G17 pool curation review does not bind the exact current inventory and successor")
    if review.exchangeability == "ESTABLISHED" \
            and packet["recommended_size"] < packet["minimum_selected_records"]:
        raise UserInputError(
            "G17 exchangeability cannot be established: the curation packet contains fewer "
            "than 49 pairwise-admissible records, so each record lacks 48 alternatives")

    state = adoption_state(path.parent, path.name)
    if not state.get("adopted"):
        raise UserInputError(
            "G17 pool curation requires a named maintainer adoption before it can be used")
    if not state.get("signature_still_reaches_the_declaration"):
        raise UserInputError(
            "G17 pool curation adoption does not bind the current review declaration")
    adopted_by = state.get("adopted_by")
    adopted_on = state.get("adopted_on")
    if not isinstance(adopted_by, str) or not adopted_by.strip() \
            or not isinstance(adopted_on, str) or not adopted_on.strip():
        raise UserInputError("G17 pool curation adoption lacks reviewer attribution")

    adoption_path = path.parent / state["adoption_file"]
    binding = {
        "schema": SCHEMA,
        "declaration_file": path.name,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "adoption_file": adoption_path.name,
        "adoption_sha256": hashlib.sha256(adoption_path.read_bytes()).hexdigest(),
        "adopted_by": adopted_by.strip(),
        "adopted_on": adopted_on.strip(),
        "readiness_assessment_sha256": readiness["assessment_sha256"],
    }
    return AdoptedPoolCurationReview(review=review, binding=binding)


__all__ = [
    "REQUEST_SCHEMA", "SCHEMA", "AdoptedPoolCurationReview", "PoolCurationReview",
    "build_curation_review_request", "load_adopted_curation_review",
]