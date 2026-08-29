"""TG11.5: read-only access to recorded, non-reproducible review artifacts.

The review layer is deliberately separate from the claim surface.  This router reads immutable
review records, round-robin outcomes and cost receipts from their own store, verifies every
content digest through the core types, and binds them to the latest published evidence bundle.
It never runs a model, records a call, appends evidence or accepts a claim state from a client.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from fastapi import APIRouter, HTTPException

from src.api.findings import StudyStore, refuse_bare_confidence
from src.core.errors import InvalidParameterError
from src.core.recorded_call import (
    REVIEW_RECORD_SCHEMA,
    R23_DECLARATION,
    ReviewRecord,
    load_review_record,
)
from src.core.review_cost import COST_RECEIPT_SCHEMA, ReviewCostReceipt, load_cost_receipt
from src.core.round_robin import ROUND_ROBIN_SCHEMA, RoundRobinOutcome


REVIEW_ROOT_ENV = "SPECTRAL_REVIEW_ROOT"
DEFAULT_REVIEW_ROOT = Path("data") / "reviews"
REVIEW_SURFACE_SCHEMA = "review-surface/v1"
CLAIM_BOUNDARY = (
    "These are recorded arguments beside the evidence. They do not permit a claim, change a "
    "claim rung, establish consensus, or make a non-deterministic call reproducible (R22, R23)."
)


def review_root() -> Path:
    """The dedicated artifact directory, resolved at request time for tests and deployments."""
    return Path(os.environ.get(REVIEW_ROOT_ENV) or DEFAULT_REVIEW_ROOT)


class ReviewStore:
    """Verified review artifacts, classified by their declared schema rather than filename."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else review_root()

    def paths(self) -> List[Path]:
        if not self.root.is_dir():
            return []
        return sorted(path for path in self.root.glob("*.json") if path.is_file())

    @staticmethod
    def _mapping(path: Path) -> Mapping[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidParameterError("review artifact", path.name,
                                        "readable JSON carrying a known review schema") from exc
        if not isinstance(value, Mapping):
            raise InvalidParameterError("review artifact", path.name, "a JSON object")
        return value

    def read(self) -> Tuple[List[Tuple[str, ReviewRecord]],
                            List[Tuple[str, RoundRobinOutcome]],
                            List[Tuple[str, ReviewCostReceipt]], List[Dict[str, str]]]:
        records: List[Tuple[str, ReviewRecord]] = []
        outcomes: List[Tuple[str, RoundRobinOutcome]] = []
        receipts: List[Tuple[str, ReviewCostReceipt]] = []
        unreadable: List[Dict[str, str]] = []
        for path in self.paths():
            try:
                value = self._mapping(path)
                schema = value.get("schema")
                if schema == REVIEW_RECORD_SCHEMA:
                    records.append((path.name, load_review_record(path)))
                elif schema == ROUND_ROBIN_SCHEMA:
                    outcomes.append((path.name, RoundRobinOutcome.from_mapping(value)))
                elif schema == COST_RECEIPT_SCHEMA:
                    receipts.append((path.name, load_cost_receipt(path)))
                else:
                    raise InvalidParameterError(
                        "review artifact schema", schema,
                        "%s, %s, or %s" % (
                            REVIEW_RECORD_SCHEMA, ROUND_ROBIN_SCHEMA, COST_RECEIPT_SCHEMA))
            except InvalidParameterError as exc:
                unreadable.append({"file": path.name, "refused_because": str(exc)})
        return records, outcomes, receipts, unreadable


router = APIRouter(prefix="/api/v1/reviews", tags=["reviews"])


@router.get("/studies/{study_id}")
async def get_study_review(study_id: str) -> Dict[str, Any]:
    """All verified commentary bound to the latest published revision of one study."""
    try:
        bundle, bundle_path = StudyStore().load(study_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="no published study %r" % study_id)

    records, outcomes, receipts, unreadable = ReviewStore().read()
    bound_records = [
        (name, record) for name, record in records
        if record.study_id == bundle.study_id
        and record.bundle_sha256 == bundle.bundle_sha256
        and record.bundle_revision == bundle.revision
    ]
    rows = []
    for filename, record in sorted(bound_records, key=lambda item: (item[1].revision, item[0])):
        linked_outcomes = [
            {"file": name, "outcome": outcome.to_mapping(), "rendered": outcome.render()}
            for name, outcome in outcomes
            if outcome.study_id == bundle.study_id
            and outcome.bundle_sha256 == bundle.bundle_sha256
            and outcome.bundle_revision == bundle.revision
            and outcome.record_sha256 == record.record_sha256
        ]
        linked_receipts = [
            {"file": name, "receipt": receipt.to_mapping()}
            for name, receipt in receipts
            if receipt.review_record_sha256 == record.record_sha256
        ]
        rows.append({
            "file": filename,
            "record": record.to_mapping(),
            "rendered": record.render(),
            "outcomes": linked_outcomes,
            "cost_receipts": linked_receipts,
        })

    payload = {
        "schema": REVIEW_SURFACE_SCHEMA,
        "study_id": bundle.study_id,
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_revision": bundle.revision,
        "declaration": R23_DECLARATION,
        "claim_boundary": CLAIM_BOUNDARY,
        "reviews": rows,
        "unreadable": unreadable,
        "absence_note": (
            "No review is recorded for this exact bundle revision. That is not the same as no "
            "review having been performed or no criticism existing."
            if not rows else None),
    }
    return refuse_bare_confidence(payload, where="review surface")


__all__ = [
    "REVIEW_ROOT_ENV", "DEFAULT_REVIEW_ROOT", "REVIEW_SURFACE_SCHEMA", "CLAIM_BOUNDARY",
    "review_root", "ReviewStore", "router",
]
