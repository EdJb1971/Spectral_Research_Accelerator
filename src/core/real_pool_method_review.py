"""Adopted scientific-method review for G17 partner-profile ingress."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field, validator

from src.core.adoption import adoption_state
from src.core.errors import UserInputError


SCHEMA = "g17-marginal-method-review/v1"


class _FrozenModel(BaseModel):
    class Config:
        allow_mutation = False
        allow_population_by_field_name = True
        extra = "forbid"


class MarginalMethodReview(_FrozenModel):
    """Methods a named maintainer has reviewed for one profile population."""

    schema_: Literal[SCHEMA] = Field(SCHEMA, alias="schema")
    study_id: str = Field(..., min_length=1)
    native_seconds_basis: str = Field(..., min_length=1)
    effective_sample_size_method: str = Field(..., min_length=1)
    noise_floor_method: str = Field(..., min_length=1)
    applicability: str = Field(..., min_length=1)
    limitations: str = Field(..., min_length=1)
    claim_boundary: str = Field(..., min_length=1)

    @validator("study_id", "native_seconds_basis", "effective_sample_size_method",
               "noise_floor_method", "applicability", "limitations", "claim_boundary")
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty declaration")
        return value.strip()


@dataclass(frozen=True)
class AdoptedMarginalMethodReview:
    review: MarginalMethodReview
    binding: Dict[str, Any]


def load_adopted_method_review(path: Path) -> AdoptedMarginalMethodReview:
    """Load a review only when a human adoption still binds its exact bytes."""
    try:
        raw = path.read_bytes()
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("expected a JSON object")
        review = MarginalMethodReview.parse_obj(body)
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("G17 marginal method review is invalid: %s" % error) from error

    state = adoption_state(path.parent, path.name)
    if not state.get("adopted"):
        raise UserInputError(
            "G17 marginal methods require a named maintainer adoption before profile emission")
    if not state.get("signature_still_reaches_the_declaration"):
        raise UserInputError(
            "G17 marginal method adoption does not bind the current review declaration")
    adopted_by = state.get("adopted_by")
    adopted_on = state.get("adopted_on")
    if not isinstance(adopted_by, str) or not adopted_by.strip() \
            or not isinstance(adopted_on, str) or not adopted_on.strip():
        raise UserInputError("G17 marginal method adoption lacks reviewer attribution")

    adoption_path = path.parent / state["adoption_file"]
    binding = {
        "schema": SCHEMA,
        "declaration_file": path.name,
        "declaration_sha256": hashlib.sha256(raw).hexdigest(),
        "adoption_file": adoption_path.name,
        "adoption_sha256": hashlib.sha256(adoption_path.read_bytes()).hexdigest(),
        "adopted_by": adopted_by.strip(),
        "adopted_on": adopted_on.strip(),
    }
    return AdoptedMarginalMethodReview(review=review, binding=binding)


__all__ = [
    "SCHEMA",
    "AdoptedMarginalMethodReview",
    "MarginalMethodReview",
    "load_adopted_method_review",
]