"""Human review of one completed, immutable T4E.39 temporal holdout result.

The holdout's verdict is fixed at `NOT_AN_ACCEPTANCE` by its pre-opening declaration, and
nothing a reviewer writes here can move it. What a reviewer can do is say, in their own words
and under their own name, whether the recorded claim boundary is an honest description of what
the measurement established -- and what they judge should happen next. That is a different act
from acceptance, and this module refuses to let it impersonate one: the review model has no
field in which an acceptance can be expressed, and the loader refuses any result whose verdict
is not still `NOT_AN_ACCEPTANCE`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import BaseModel, Field, validator

from src.benchmarks.reference_holdout import (
    DEFAULT_DECLARATION, load_holdout_declaration, require_current_adoption)
from src.benchmarks.reference_holdout_fields import (
    DEFAULT_CENSUS, DEFAULT_OUTPUT, HOLDOUT_EVALUATION_SCHEMA,
    HOLDOUT_FIELD_RECORD_SCHEMA, load_holdout_census, verify_receipt_identity)
from src.core.adoption import REQUIRED_AFFIRMATION, adoption_state
from src.core.errors import UserInputError


SCHEMA = "t4e39-holdout-result-review/v1"
REQUEST_SCHEMA = "t4e39-holdout-result-review-request/v1"
REQUIRED_VERDICT = "NOT_AN_ACCEPTANCE"
PERMITTED_ASSESSMENTS = ("BOUNDARY_SOUND", "BOUNDARY_DISPUTED")


class _FrozenModel(BaseModel):
    class Config:
        allow_mutation = False
        allow_population_by_field_name = True
        extra = "forbid"


class HoldoutResultReview(_FrozenModel):
    """A named person's reading of one exact holdout result and its stated boundary."""

    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")
    task: Literal["T4E.39"] = "T4E.39"
    declaration_sha256: str
    census_receipt_sha256: str
    field_record_receipt_sha256: str
    measurement_receipt_sha256: str
    reviewed_outcome: str = Field(..., min_length=1)
    reviewed_verdict: Literal["NOT_AN_ACCEPTANCE"] = REQUIRED_VERDICT
    population_denominator: int = Field(..., ge=1)
    strict_majority_needed: int = Field(..., ge=1)
    boundary_assessment: Literal["BOUNDARY_SOUND", "BOUNDARY_DISPUTED"]
    basis: str = Field(..., min_length=1)
    what_this_does_not_establish: List[str] = Field(..., min_items=1)
    limitations: str = Field(..., min_length=1)
    next_action: str = Field(..., min_length=1)
    claim_boundary: str = Field(..., min_length=1)

    @validator("declaration_sha256", "census_receipt_sha256", "field_record_receipt_sha256",
               "measurement_receipt_sha256")
    def _sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("must be a lowercase SHA-256 digest")
        return value

    @validator("reviewed_outcome", "basis", "limitations", "next_action", "claim_boundary")
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty declaration")
        return value.strip()

    @validator("what_this_does_not_establish")
    def _named_limits(cls, values: List[str]) -> List[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
            raise ValueError("what_this_does_not_establish must be non-empty and unique")
        return cleaned


@dataclass(frozen=True)
class AdoptedHoldoutResultReview:
    review: HoldoutResultReview
    binding: Dict[str, Any]


def _read(path: Path, label: str) -> Dict[str, Any]:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("%s is unreadable: %s" % (label, error)) from error
    if not isinstance(body, dict):
        raise UserInputError("%s does not contain a JSON object" % label)
    return body


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_holdout_result(
    *,
    declaration_path: Union[str, Path] = DEFAULT_DECLARATION,
    census_path: Union[str, Path] = DEFAULT_CENSUS,
    measurement_path: Union[str, Path] = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    """Re-verify the complete chain from declaration to measurement, opening no network.

    Every digest is recomputed here rather than believed: the census must bind the declaration
    bytes, the field record must bind both, the measurement must bind all three, and each
    artifact's own receipt identity must reproduce from its own content.
    """
    declaration_path = Path(declaration_path)
    load_holdout_declaration(declaration_path)
    declaration_sha256 = _file_sha256(declaration_path)
    adoption = require_current_adoption(declaration_path)
    census = load_holdout_census(census_path, declaration_path)

    measurement_path = Path(measurement_path)
    measurement = _read(measurement_path, "T4E.39 measurement")
    if measurement.get("schema") != HOLDOUT_EVALUATION_SCHEMA:
        raise UserInputError("T4E.39 measurement carries an unsupported schema")
    verify_receipt_identity(measurement, "T4E.39 measurement")
    sources = measurement.get("sources", {})
    if (measurement.get("declaration_sha256") != declaration_sha256
            or sources.get("census_receipt_sha256") != census["receipt_sha256"]):
        raise UserInputError(
            "T4E.39 measurement does not bind the current declaration and census")

    receipt_path = Path(str(sources.get("field_record_receipt", "")))
    record_receipt = _read(receipt_path, "T4E.39 field record receipt")
    if record_receipt.get("schema") != HOLDOUT_FIELD_RECORD_SCHEMA:
        raise UserInputError("T4E.39 field record receipt carries an unsupported schema")
    verify_receipt_identity(record_receipt, "T4E.39 field record")
    if (record_receipt.get("receipt_sha256") != sources.get("field_record_receipt_sha256")
            or record_receipt.get("declaration_sha256") != declaration_sha256
            or record_receipt.get("census_receipt_sha256") != census["receipt_sha256"]):
        raise UserInputError(
            "T4E.39 field record does not bind the current declaration and census")
    if measurement.get("VERDICT") != REQUIRED_VERDICT:
        raise UserInputError(
            "T4E.39 measurement no longer carries its mandatory %s verdict" % REQUIRED_VERDICT)

    seam = record_receipt.get("seam", {})
    shards = {name: len(source.get("shards", {}))
              for name, source in sorted(record_receipt.get("sources", {}).items())}
    decision = measurement.get("decision", {})
    return {
        "declaration_sha256": declaration_sha256,
        "adoption": adoption,
        "census": census,
        "census_receipt_sha256": census["receipt_sha256"],
        "field_record_receipt": str(receipt_path).replace("\\", "/"),
        "field_record_receipt_sha256": record_receipt["receipt_sha256"],
        "record_sha256": record_receipt.get("record_sha256"),
        "record_path": record_receipt.get("record_path"),
        "record_shape": record_receipt.get("shape", {}),
        "seam_checks": len(seam.get("checks", [])),
        "seam_all_passed": bool(seam.get("all_passed")),
        "shards_per_stream": shards,
        "measurement": measurement,
        "measurement_path": str(measurement_path).replace("\\", "/"),
        "measurement_receipt_sha256": measurement["receipt_sha256"],
        "measurement_file_sha256": _file_sha256(measurement_path),
        "outcome": measurement.get("OUTCOME"),
        "verdict": measurement.get("VERDICT"),
        "decision": decision,
        "claim_boundary": measurement.get("claim_boundary"),
        "network_used": False,
    }


def build_holdout_review_request(
    *,
    declaration_path: Union[str, Path] = DEFAULT_DECLARATION,
    census_path: Union[str, Path] = DEFAULT_CENSUS,
    measurement_path: Union[str, Path] = DEFAULT_OUTPUT,
) -> Dict[str, Any]:
    """Prepare exact review inputs without making, wording or signing the human decision."""
    verified = verified_holdout_result(
        declaration_path=declaration_path, census_path=census_path,
        measurement_path=measurement_path)
    decision = verified["decision"]
    counts = decision.get("counts", {})
    needed = int(decision.get("strict_majority_needed", 0))
    leading = max(int(value) for value in counts.values()) if counts else 0
    return {
        "schema": REQUEST_SCHEMA,
        "status": "AWAITING_HUMAN_REVIEW",
        "review_schema": SCHEMA,
        "task": "T4E.39",
        "declaration_sha256": verified["declaration_sha256"],
        "census_receipt_sha256": verified["census_receipt_sha256"],
        "field_record_receipt_sha256": verified["field_record_receipt_sha256"],
        "measurement_receipt_sha256": verified["measurement_receipt_sha256"],
        "reviewed_outcome": verified["outcome"],
        "reviewed_verdict": verified["verdict"],
        "population_denominator": int(decision.get("population_denominator", 0)),
        "strict_majority_needed": needed,
        "verified_facts": {
            "counts": counts,
            "rows_above_strict_majority": leading - needed,
            "one_row_would_change_the_outcome": leading == needed,
            "seam_checks_passed": verified["seam_all_passed"],
            "seam_checks": verified["seam_checks"],
            "shards_per_stream": verified["shards_per_stream"],
            "record_sha256": verified["record_sha256"],
            "acceptance_available": False,
        },
        "human_inputs_required": {
            "boundary_assessment": list(PERMITTED_ASSESSMENTS),
            "basis": "A reviewer-authored reading of this result beyond its counts.",
            "what_this_does_not_establish": (
                "One or more explicitly named limits, in the reviewer's own words."),
            "limitations": "The limits of this review itself.",
            "next_action": "What the reviewer judges should happen next.",
            "attribution": "The reviewer's full name and role.",
            "affirmation": REQUIRED_AFFIRMATION,
        },
        "claim_boundary": (
            "This is a machine-prepared review request, not a review, a decision or an "
            "adoption. Neither it nor the review it invites can turn a NOT_AN_ACCEPTANCE "
            "result into an acceptance, widen the spent holdout or license a re-selection."),
    }


def load_adopted_holdout_review(
    path: Union[str, Path],
    *,
    declaration_path: Union[str, Path] = DEFAULT_DECLARATION,
    census_path: Union[str, Path] = DEFAULT_CENSUS,
    measurement_path: Union[str, Path] = DEFAULT_OUTPUT,
) -> AdoptedHoldoutResultReview:
    """Load a human review only while it binds the exact verified result it claims to read."""
    verified = verified_holdout_result(
        declaration_path=declaration_path, census_path=census_path,
        measurement_path=measurement_path)
    path = Path(path)
    try:
        raw = path.read_bytes()
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        review = HoldoutResultReview.parse_obj(body)
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("T4E.39 holdout result review is invalid: %s" % error) from error

    decision = verified["decision"]
    expected = {
        "declaration_sha256": verified["declaration_sha256"],
        "census_receipt_sha256": verified["census_receipt_sha256"],
        "field_record_receipt_sha256": verified["field_record_receipt_sha256"],
        "measurement_receipt_sha256": verified["measurement_receipt_sha256"],
        "reviewed_outcome": verified["outcome"],
        "reviewed_verdict": verified["verdict"],
        "population_denominator": int(decision.get("population_denominator", 0)),
        "strict_majority_needed": int(decision.get("strict_majority_needed", 0)),
    }
    if {name: getattr(review, name) for name in expected} != expected:
        raise UserInputError(
            "T4E.39 holdout result review does not bind the exact current result")

    state = adoption_state(path.parent, path.name)
    if not state.get("adopted"):
        raise UserInputError(
            "T4E.39 holdout result review requires a named human adoption before it can be used")
    if not state.get("signature_still_reaches_the_declaration"):
        raise UserInputError(
            "T4E.39 holdout result review adoption does not bind the current review")
    adopted_by = state.get("adopted_by")
    adopted_on = state.get("adopted_on")
    if not isinstance(adopted_by, str) or not adopted_by.strip() \
            or not isinstance(adopted_on, str) or not adopted_on.strip():
        raise UserInputError("T4E.39 holdout result review adoption lacks reviewer attribution")

    adoption_path = path.parent / state["adoption_file"]
    binding = {
        "schema": SCHEMA,
        "declaration_file": path.name,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "adoption_file": adoption_path.name,
        "adoption_sha256": hashlib.sha256(adoption_path.read_bytes()).hexdigest(),
        "adopted_by": adopted_by.strip(),
        "adopted_on": adopted_on.strip(),
        "measurement_receipt_sha256": verified["measurement_receipt_sha256"],
        "verdict_after_review": REQUIRED_VERDICT,
    }
    return AdoptedHoldoutResultReview(review=review, binding=binding)


def review_state(
    directory: Union[str, Path], name: str, *,
    verified: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe whether the review exists, is adopted, and still binds the live result."""
    directory = Path(directory)
    path = directory / name
    if not path.exists():
        return {"status": "NOT_WRITTEN", "adopted": False}
    state = adoption_state(directory, name)
    described = {
        "status": "ADOPTED" if state.get("adopted") else "WRITTEN_NOT_ADOPTED",
        "declaration": _read(path, "T4E.39 holdout result review"),
        "declaration_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "adoption": state,
    }
    if verified is not None:
        described["binds_current_result"] = bool(
            described["declaration"].get("measurement_receipt_sha256")
            == verified.get("measurement_receipt_sha256"))
    return described


__all__ = [
    "PERMITTED_ASSESSMENTS", "REQUEST_SCHEMA", "REQUIRED_VERDICT", "SCHEMA",
    "AdoptedHoldoutResultReview", "HoldoutResultReview", "build_holdout_review_request",
    "load_adopted_holdout_review", "review_state", "verified_holdout_result",
]
