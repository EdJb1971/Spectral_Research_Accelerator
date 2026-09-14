"""TG11.5 / T4E.33: recorded, non-reproducible review artifacts -- read, and now convened.

The review layer is deliberately separate from the claim surface.  This router reads immutable
review records, round-robin outcomes and cost receipts from their own store, verifies every
content digest through the core types, and binds them to the latest published evidence bundle.

**One route runs a model, and it is the only one.**  Until T4E.33 this module stated that it
never did, and that was true because there was no bundle to review and no way to convene a panel
except a terminal.  The statement is replaced rather than quietly deleted: `POST
/studies/{study_id}/round-robin` takes seven to eleven turns against an external service. It costs money,
it puts the bundle's contents in front of a third party, and it cannot be undone.

So the authorisation is moved into the request rather than removed from the system.  The caller
must confirm in the body, the key is read from the server's environment and never from the wire,
and the response reports what was spent.  **No route appends evidence, accepts a claim state, or
moves a rung** -- `verify_claim_independence` is re-derived after the exchange and the outcome
carries the bundle's own rung, copied and never set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.findings import StudyStore, refuse_bare_confidence
from src.core.errors import InvalidParameterError
from src.core.recorded_call import (
    REVIEW_RECORD_SCHEMA,
    R23_DECLARATION,
    REVIEW_ROLES,
    ReviewRecord,
    load_review_record,
    save_review_record,
)
from src.core.review_cost import (
    COST_RECEIPT_SCHEMA, GEMINI_35_FLASH, GeminiBatchTransport, ReviewCostReceipt,
    load_cost_receipt,
)
from src.core.round_robin import (
    MAX_ROUND_ROBIN_CALLS, ROUND_ROBIN_SCHEMA, RUBRICS, SCHEMA_FOR_ROLE, PanelSeat,
    RecordedTurnRefused, ReviewPanel, RoundRobinOutcome, advance, close_round_robin,
    open_round_robin,
)


REVIEW_ROOT_ENV = "SPECTRAL_REVIEW_ROOT"
DEFAULT_REVIEW_ROOT = Path("data") / "reviews"
REVIEW_SURFACE_SCHEMA = "review-surface/v1"
CLAIM_BOUNDARY = (
    "These are recorded arguments beside the evidence. They do not permit a claim, change a "
    "claim rung, establish consensus, or make a non-deterministic call reproducible (R22, R23)."
)


#: Read from the server's environment and never from a request body, so a key is never in a
#: browser, a log, or a payload that crosses the wire.
KEY_VARIABLES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
DEFAULT_REVIEW_MODEL = GEMINI_35_FLASH


def _key() -> Optional[str]:
    for name in KEY_VARIABLES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


def review_root() -> Path:
    """The dedicated artifact directory, resolved at request time for tests and deployments."""
    return Path(os.environ.get(REVIEW_ROOT_ENV) or DEFAULT_REVIEW_ROOT)


def _unused_partial_path(root: Path, study_id: str) -> Path:
    first = root / ("%s.review.partial.json" % study_id)
    if not first.exists():
        return first
    for attempt in range(2, 10_000):
        candidate = root / ("%s.review.attempt-%d.partial.json" % (study_id, attempt))
        if not candidate.exists():
            return candidate
    raise FileExistsError("no unused partial-review filename remains for %s" % study_id)


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

# ------------------------------------------------------ T4E.33: convening the panel from a surface
#
# The protocol, the seats, the dissent register and the transport were built at TG7.1 and TG7.2,
# and a terminal was the only way to reach them. For a single-maintainer research instrument that
# is a real barrier: the point of the surface is to run the experiment, not to describe one that
# must then be run somewhere else.
#
# What does NOT move into the surface is the decision. The request must carry `i_authorise_paid_calls`,
# the number of calls is reported before any is made, and a missing key is refused by name rather
# than silently producing nothing.


class RoundRobinRequest(BaseModel):
    """What convening a panel requires. The key is never one of these fields."""

    i_authorise_paid_calls: bool = False
    model_id: str = DEFAULT_REVIEW_MODEL
    effort: str = "high"
    seats: Optional[Dict[str, str]] = None


def _panel(body: "RoundRobinRequest") -> ReviewPanel:
    chosen = {role: body.model_id for role in REVIEW_ROLES}
    for role, model in (body.seats or {}).items():
        if role not in REVIEW_ROLES:
            raise HTTPException(status_code=400,
                                detail="%r is not a review role; the protocol has %s"
                                       % (role, ", ".join(REVIEW_ROLES)))
        chosen[role] = model
    try:
        return ReviewPanel(seats={role: PanelSeat(role=role, model_id=chosen[role],
                                                  effort=body.effort)
                                  for role in REVIEW_ROLES})
    except InvalidParameterError as refusal:
        raise HTTPException(status_code=400, detail=str(refusal))


@router.get("/panel-plan")
def read_panel_plan() -> Dict[str, Any]:
    """The seats, what each must return, and what convening one would cost. Sends nothing."""
    return {
        "roles": [{"role": role, "expects": sorted(SCHEMA_FOR_ROLE[role].fields),
                   "rubric": RUBRICS[role]} for role in REVIEW_ROLES],
        "calls_if_every_turn_is_taken": MAX_ROUND_ROBIN_CALLS,
        "calls_if_nothing_is_dissented_from": len(REVIEW_ROLES) - 1,
        "why_that_differs": (
            "`response_and_revision` is skipped when no challenger dissented and repeats once "
            "for each dissent otherwise. A response to no dissent is a rebuttal of nothing."),
        "default_model": DEFAULT_REVIEW_MODEL,
        "key_variables": list(KEY_VARIABLES),
        "key_present": _key() is not None,
        "cost_is_the_caller_s": (
            "Convening a panel makes paid calls to an external service on the maintainer's own "
            "account. This surface reports the count before making them and the usage after; it "
            "does not estimate a price, because a wrong estimate is worse than none."),
        "one_model_in_every_seat_is_recorded_not_refused": (
            "An independent reassessment by the model that wrote the candidate synthesis is not "
            "independent in the sense the word usually carries. The overlap is computed into the "
            "panel digest and stated wherever the outcome is shown, rather than making a "
            "single-provider panel impossible to run."),
        "claim_boundary": CLAIM_BOUNDARY,
        "network_used": False,
    }


@router.post("/studies/{study_id}/round-robin")
def convene_round_robin(study_id: str, body: RoundRobinRequest) -> Dict[str, Any]:
    """Take the exchange against the study's latest published bundle, and record it.

    Deliberately a synchronous handler. The transport is blocking and a panel takes up to
    eleven turns whose default batch timeout is twenty-four hours, so declaring this
    ``async def`` put that wait directly on the event loop: one convening made the whole API
    stop answering, including ``/health``, for as long as the panel ran. FastAPI runs a plain
    ``def`` route in a threadpool, which keeps the instrument readable while a panel is out.
    Measured: with ``async def``, every other request hung until the process was killed.
    """
    if not body.i_authorise_paid_calls:
        raise HTTPException(
            status_code=400,
            detail=("this would make up to %d calls to an external service on your own account, "
                    "which costs money and puts the bundle in front of a third party. Set "
                    "i_authorise_paid_calls to convene the panel. Nothing was sent."
                    % MAX_ROUND_ROBIN_CALLS))
    key = _key()
    if key is None:
        raise HTTPException(
            status_code=400,
            detail=("no API key is present in %s on the server. Nothing was sent. The key is "
                    "read from the environment and never from the request, so it is never in a "
                    "browser, a log or this payload." % " or ".join(KEY_VARIABLES)))
    try:
        bundle, bundle_path = StudyStore().load(study_id)
    except (FileNotFoundError, InvalidParameterError) as refusal:
        raise HTTPException(status_code=404, detail=str(refusal))

    panel = _panel(body)
    exchange = open_round_robin(bundle, panel)
    transport = GeminiBatchTransport(key)
    taken: List[str] = []
    try:
        for _ in range(MAX_ROUND_ROBIN_CALLS):
            turn = exchange.next_turn
            if turn is None:
                break
            taken.append(turn.role)
            exchange = advance(exchange, transport=transport,
                               requested_at=datetime.now(timezone.utc).isoformat())
    except RecordedTurnRefused as refusal:
        # The calls were made and cannot be regenerated (R23), so the partial record is written.
        partial = _unused_partial_path(review_root(), study_id)
        save_review_record(partial, refusal.reviewed.review, bundle_path=bundle_path)
        raise HTTPException(
            status_code=422,
            detail={"refused": str(refusal), "turns_taken": taken,
                    "partial_record": partial.name,
                    "note": ("the calls were made and are kept; nothing paid for is discarded "
                             "because it was disappointing")})

    outcome = close_round_robin(exchange)
    root = review_root()
    root.mkdir(parents=True, exist_ok=True)
    review_path = root / ("%s.review.json" % study_id)
    outcome_path = root / ("%s.round-robin.json" % study_id)
    save_review_record(review_path, exchange.reviewed.review, bundle_path=bundle_path)
    outcome_path.write_text(json.dumps(outcome.to_mapping(), indent=2), encoding="utf-8")

    usage = [dict(call.response.usage) for call in exchange.reviewed.review.calls]
    return {
        "study_id": study_id,
        "bundle_sha256": bundle.bundle_sha256,
        "bundle_revision": bundle.revision,
        "turns_taken": taken,
        "outcome": outcome.to_mapping(),
        "review_file": review_path.name,
        "outcome_file": outcome_path.name,
        "usage": usage,
        "total_tokens": sum(int(entry.get("total_tokens", 0)) for entry in usage),
        "the_rung_was_copied_not_set": (
            "The outcome carries the bundle's own rung. Claim independence was re-derived after "
            "the exchange: deleting every recorded call changes no claim level (R22)."),
        "claim_boundary": CLAIM_BOUNDARY,
        "network_used": True,
    }



def study_review_payload(study_id: str) -> Dict[str, Any]:
    """Build the verified formal-review context for one published study."""
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


@router.get("/studies/{study_id}")
async def get_study_review(study_id: str) -> Dict[str, Any]:
    """All verified commentary bound to the latest published revision of one study."""
    return study_review_payload(study_id)


__all__ = [
    "REVIEW_ROOT_ENV", "DEFAULT_REVIEW_ROOT", "REVIEW_SURFACE_SCHEMA", "CLAIM_BOUNDARY",
    "review_root", "ReviewStore", "study_review_payload", "router",
]
