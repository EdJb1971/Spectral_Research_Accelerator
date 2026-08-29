"""TG7.2: the adversarial round-robin.

Eight roles speak in a fixed order over one frozen evidence bundle: a candidate synthesis, four
challengers, a response to each dissent, an independent reassessment, and a bounded final
synthesis.  Every turn goes through the TG7.1 boundary, so the whole exchange is recorded
verbatim beside the bundle and none of it can reach a claim level (R22, R23).

Three things are enforced here that the boundary alone does not give.

**The order is the protocol, not a suggestion.**  A `RoundRobin` replays the recorded chain
against the plan and refuses a record whose roles, targets, models or efforts are not the ones
the protocol demanded at that point.  A challenge cannot be synthesised over before it has been
answered, and the panel is pinned turn by turn, because a bundle reviewed by a different model
is different evidence.

**This is not majority voting.**  Nothing here counts verdicts.  A dissent is retired only by
the candidate conceding it, or by a rebuttal that the independent reassessment declines to
reopen.  Three challengers agreeing does not retire the fourth's objection, and no arithmetic
over `VERDICTS` appears anywhere in this module.

**Unresolved dissent is retained, never reconciled.**  The final synthesis must carry every
unresolved dissent by name; a synthesis that drops one, or that reports no remaining dissent
while one stands, is refused.  The roadmap says dissent is retained *in the bundle*; R22 says
no LLM output may be in the bundle at all.  Both are honoured by retaining it in the review
record, which is published beside the bundle and travels with it - see `RoundRobinOutcome`.

Like the boundary below it, this module opens no socket.  The transport is injected.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dc_field
from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from src.core.errors import InvalidParameterError, UserInputError
from src.core.evidence import EvidenceBundle
from src.core.five_outputs import summarise_evidence
from src.core.recorded_call import (
    EFFORTS, R23_DECLARATION, REVIEW_ROLES, FieldSpec, ResponseSchema, ReviewedBundle,
    attach_review, record_call, verify_claim_independence,
)


ROUND_ROBIN_SCHEMA = "round-robin/v1"
PANEL_SCHEMA = "review-panel/v1"

#: The four challengers, in the order they speak.  Each gets exactly one turn.
CHALLENGE_ROLES = (
    "statistical_challenge",
    "confounder_challenge",
    "domain_plausibility_challenge",
    "provenance_challenge",
)

VERDICTS = ("supported", "unsupported", "unclear")

#: What the candidate may say about a dissent.  Only `conceded` retires one on its own; a
#: `rebutted` claim is the candidate marking its own homework, and stands only if the
#: independent reassessment declines to reopen it.
RESPONSE_OUTCOMES = ("conceded", "rebutted", "unresolved")

DISSENT_STATES = ("resolved", "unresolved")

#: Why a dissent was retired.  There is deliberately no value here meaning "outvoted".
CLOSURES = ("conceded by the candidate", "rebutted, and not reopened on reassessment")

R22_RUBRIC = (
    "You are commenting beside the evidence, never on it. Nothing you write can change a claim "
    "level, and no claim level will be changed because you asked for one. Do not describe a "
    "finding as promoted, demonstrated or established; the gates decide that and they do not "
    "read you. Answer only in the declared schema."
)

CANDIDATE_SCHEMA = ResponseSchema(name="candidate_synthesis", fields={
    "claim": FieldSpec(kind="string"),
    "basis": FieldSpec(kind="string_list"),
    "known_weaknesses": FieldSpec(kind="string_list"),
})

CHALLENGE_SCHEMA = ResponseSchema(name="challenge", fields={
    "verdict": FieldSpec(kind="string", enum=VERDICTS),
    "argument": FieldSpec(kind="string"),
    "alternatives": FieldSpec(kind="string_list"),
    "dissent": FieldSpec(kind="boolean"),
})

RESPONSE_SCHEMA = ResponseSchema(name="response_and_revision", fields={
    "answers_challenge": FieldSpec(kind="string", enum=CHALLENGE_ROLES),
    "outcome": FieldSpec(kind="string", enum=RESPONSE_OUTCOMES),
    "revision": FieldSpec(kind="string"),
})

REASSESSMENT_SCHEMA = ResponseSchema(name="independent_reassessment", fields={
    "verdict": FieldSpec(kind="string", enum=VERDICTS),
    "argument": FieldSpec(kind="string"),
    "standing_dissent": FieldSpec(kind="string_list"),
})

FINAL_SCHEMA = ResponseSchema(name="final_synthesis", fields={
    "finding": FieldSpec(kind="string"),
    "bounded_by": FieldSpec(kind="string_list"),
    "retained_dissent": FieldSpec(kind="string_list"),
    "dissent_remains": FieldSpec(kind="boolean"),
})

SCHEMA_FOR_ROLE: Mapping[str, ResponseSchema] = MappingProxyType({
    "candidate_synthesis": CANDIDATE_SCHEMA,
    "statistical_challenge": CHALLENGE_SCHEMA,
    "confounder_challenge": CHALLENGE_SCHEMA,
    "domain_plausibility_challenge": CHALLENGE_SCHEMA,
    "provenance_challenge": CHALLENGE_SCHEMA,
    "response_and_revision": RESPONSE_SCHEMA,
    "independent_reassessment": REASSESSMENT_SCHEMA,
    "final_synthesis": FINAL_SCHEMA,
})

RUBRICS: Mapping[str, str] = MappingProxyType({
    "candidate_synthesis":
        "State the strongest reading the evidence in front of you will actually bear, and the "
        "weaknesses you already know about. Claim no more than the record supports. " + R22_RUBRIC,
    "statistical_challenge":
        "Attack the statistics: power, multiplicity, the null used, the effect size against its "
        "own uncertainty, and whether the reported quantity is the one that was pre-registered. "
        "Raise a dissent if the claim does not survive. " + R22_RUBRIC,
    "confounder_challenge":
        "Attack the causal reading: shared drivers, selection, look-ahead, and any alternative "
        "that would produce the same numbers. Name the alternatives you cannot exclude. "
        + R22_RUBRIC,
    "domain_plausibility_challenge":
        "Attack the physical or domain plausibility: mechanism, magnitude, timescale, and "
        "whether the claim contradicts established results in the field. " + R22_RUBRIC,
    "provenance_challenge":
        "Attack the provenance and method: where the data came from, what was frozen and when, "
        "held-out discipline, and whether the record would let someone else check this. "
        + R22_RUBRIC,
    "response_and_revision":
        "Answer exactly one challenge. Concede it, rebut it, or record it as unresolved. "
        "Conceding is not a failure and an unresolved answer is not a defeat; a rebuttal you "
        "cannot support is worse than either. " + R22_RUBRIC,
    "independent_reassessment":
        "Read the exchange as someone who wrote none of it. Say which dissents you consider "
        "still standing despite the response given. You may reopen a dissent; you cannot "
        "originate one at this turn. " + R22_RUBRIC,
    "final_synthesis":
        "State the finding within its bounds, and carry every unresolved dissent by name. You "
        "are not counting votes and you are not producing agreement: a dissent nobody answered "
        "stays in the record. " + R22_RUBRIC,
})

_TURN_FIELDS = ("index", "role", "target")
_SEAT_FIELDS = ("role", "model_id", "effort")
_PANEL_FIELDS = ("schema", "seats", "panel_sha256")
_OUTCOME_FIELDS = ("schema", "study_id", "bundle_sha256", "bundle_revision", "panel",
                   "record_sha256", "finding", "bounded_by", "retained_dissent",
                   "claim_sha256", "rung", "outcome_sha256")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError("round-robin value", type(value).__name__,
                                    "finite, canonical JSON data") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(name, value, "a non-empty string")
    return value.strip()


def _strings(name: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise InvalidParameterError(name, type(value).__name__, "a list of strings")
    return tuple(str(item) for item in value)


class RecordedTurnRefused(UserInputError):
    """A turn was recorded verbatim and then refused by the protocol.

    The refusal must not destroy the record of the call: it was made, it cost money, and R23
    says an LLM output cannot be regenerated.  The reviewed bundle carried here therefore
    *includes* the offending turn, so a caller can publish it before deciding what to do.
    """

    def __init__(self, reviewed: ReviewedBundle, cause: Exception) -> None:
        super().__init__(
            "The recorded turn %d does not satisfy the round-robin protocol and the exchange "
            "cannot continue through it: %s. The call itself is kept - it was made and cannot "
            "be regenerated (R23) - and is available on this error as `.reviewed`."
            % (reviewed.review.revision, cause),
            turn=reviewed.review.revision, cause=str(cause))
        self.reviewed = reviewed
        self.cause = cause


# --------------------------------------------------------------------------------------------
# The panel: who speaks in which seat, pinned for the whole exchange
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelSeat:
    """One role and the model that will answer for it."""

    role: str
    model_id: str
    effort: str

    def __post_init__(self) -> None:
        if self.role not in REVIEW_ROLES:
            raise InvalidParameterError("seat.role", self.role,
                                        "one of %s" % ", ".join(REVIEW_ROLES))
        object.__setattr__(self, "model_id", _nonempty("seat.model_id", self.model_id))
        if self.effort not in EFFORTS:
            raise InvalidParameterError("seat.effort", self.effort,
                                        "one of %s" % ", ".join(EFFORTS))

    def to_mapping(self) -> Dict[str, Any]:
        return {"role": self.role, "model_id": self.model_id, "effort": self.effort}


@dataclass(frozen=True)
class ReviewPanel:
    """Every seat filled, digested, and pinned into the outcome.

    Reviewer overlap is *recorded, not refused*.  One model in several seats is a weaker
    exchange than several - an independent reassessment by the model that wrote the candidate
    synthesis is not independent in the sense the word usually carries - but refusing it would
    make a single-provider panel impossible to run at all.  So the overlap is computed, carried
    in the digest, and stated plainly by `render()` wherever the outcome is displayed.
    """

    seats: Mapping[str, PanelSeat]
    panel_sha256: str = ""
    schema: str = dc_field(default=PANEL_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.seats, Mapping):
            raise InvalidParameterError("panel.seats", type(self.seats).__name__,
                                        "a mapping of role to PanelSeat")
        seats = {}
        for role in REVIEW_ROLES:
            if role not in self.seats:
                raise InvalidParameterError(
                    "panel.seats", sorted(self.seats),
                    "a seat for every role; %r is unfilled and the exchange would stop there"
                    % role)
            seat = self.seats[role]
            if not isinstance(seat, PanelSeat):
                raise InvalidParameterError("panel.seats.%s" % role, type(seat).__name__,
                                            "a PanelSeat")
            if seat.role != role:
                raise InvalidParameterError("panel.seats.%s" % role, seat.role,
                                            "a seat for the role it is filed under")
            seats[role] = seat
        extra = sorted(set(self.seats) - set(REVIEW_ROLES))
        if extra:
            raise InvalidParameterError("panel.seats", extra,
                                        "only the roles the protocol has, %s"
                                        % ", ".join(REVIEW_ROLES))
        object.__setattr__(self, "seats", MappingProxyType(seats))
        computed = _digest(self.body())
        if not self.panel_sha256:
            object.__setattr__(self, "panel_sha256", computed)
        elif self.panel_sha256 != computed:
            raise InvalidParameterError("panel.panel_sha256", self.panel_sha256,
                                        "the digest recomputed from the seated panel")

    @classmethod
    def uniform(cls, model_id: str, effort: str = "high") -> "ReviewPanel":
        """One model in every seat.  Legal, digested, and reported as the overlap it is."""
        return cls(seats={role: PanelSeat(role=role, model_id=model_id, effort=effort)
                          for role in REVIEW_ROLES})

    def seat(self, role: str) -> PanelSeat:
        if role not in self.seats:
            raise InvalidParameterError("role", role, "one of %s" % ", ".join(REVIEW_ROLES))
        return self.seats[role]

    @property
    def models(self) -> Tuple[str, ...]:
        return tuple(sorted({seat.model_id for seat in self.seats.values()}))

    @property
    def shared_seats(self) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
        """Every model answering in more than one seat, with the seats it holds."""
        grouped: Dict[str, Tuple[str, ...]] = {}
        for role in REVIEW_ROLES:
            model = self.seats[role].model_id
            grouped[model] = grouped.get(model, ()) + (role,)
        return tuple(sorted((model, roles) for model, roles in grouped.items()
                            if len(roles) > 1))

    @property
    def reassessment_is_independent(self) -> bool:
        """Whether a different model reassesses the synthesis than wrote it."""
        return (self.seats["independent_reassessment"].model_id
                != self.seats["candidate_synthesis"].model_id)

    def body(self) -> Dict[str, Any]:
        return {"schema": self.schema,
                "seats": [self.seats[role].to_mapping() for role in REVIEW_ROLES]}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), panel_sha256=self.panel_sha256)

    def render(self) -> str:
        lines = ["panel %s" % self.panel_sha256]
        for role in REVIEW_ROLES:
            seat = self.seats[role]
            lines.append("   %-30s %s (effort %s)" % (role, seat.model_id, seat.effort))
        for model, roles in self.shared_seats:
            lines.append("   note: %s answered in %d seats (%s)"
                         % (model, len(roles), ", ".join(roles)))
        if not self.reassessment_is_independent:
            lines.append("   note: the reassessment was made by the model that wrote the "
                         "candidate synthesis; read 'independent' with that in mind")
        return "\n".join(lines)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewPanel":
        if not isinstance(value, Mapping) or value.get("schema") != PANEL_SCHEMA:
            raise InvalidParameterError("panel", value, "a %s record" % PANEL_SCHEMA)
        seats = value.get("seats")
        if not isinstance(seats, list):
            raise InvalidParameterError("panel.seats", type(seats).__name__, "a JSON array")
        built = {}
        for item in seats:
            if not isinstance(item, Mapping) or sorted(item) != sorted(_SEAT_FIELDS):
                raise InvalidParameterError("panel.seat", item,
                                            "an object with %s" % ", ".join(_SEAT_FIELDS))
            built[item["role"]] = PanelSeat(role=item["role"], model_id=item["model_id"],
                                            effort=item["effort"])
        return cls(seats=built, panel_sha256=value.get("panel_sha256", ""))


# --------------------------------------------------------------------------------------------
# The protocol: a fixed order, replayed against the record
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One position in the exchange: who speaks next, and about what."""

    index: int
    role: str
    target: str = ""

    def __post_init__(self) -> None:
        if self.role not in REVIEW_ROLES:
            raise InvalidParameterError("turn.role", self.role,
                                        "one of %s" % ", ".join(REVIEW_ROLES))
        if self.target and self.target not in CHALLENGE_ROLES:
            raise InvalidParameterError("turn.target", self.target,
                                        "one of %s" % ", ".join(CHALLENGE_ROLES))

    @property
    def schema(self) -> ResponseSchema:
        return SCHEMA_FOR_ROLE[self.role]

    def to_mapping(self) -> Dict[str, Any]:
        return {"index": self.index, "role": self.role, "target": self.target}

    def describe(self) -> str:
        return "%d. %s%s" % (self.index, self.role,
                             " answering %s" % self.target if self.target else "")


def _answer(call: Any) -> Dict[str, Any]:
    """The structured response as plain JSON data."""
    return _thaw(call.response.structured)


def _dissents_raised(calls: Tuple[Any, ...]) -> Tuple[str, ...]:
    """The challengers who recorded a dissent, in the order they spoke."""
    return tuple(call.request.role for call in calls
                 if call.request.role in CHALLENGE_ROLES and _answer(call)["dissent"])


def _answered(calls: Tuple[Any, ...]) -> Tuple[str, ...]:
    return tuple(_answer(call)["answers_challenge"] for call in calls
                 if call.request.role == "response_and_revision")


def _spoken(calls: Tuple[Any, ...], role: str) -> bool:
    return any(call.request.role == role for call in calls)


def _planned(calls: Tuple[Any, ...]) -> Optional[Turn]:
    """The turn the protocol demands next, given everything recorded so far.

    The order is total, so this is a function of the record alone.  There is no branch here in
    which a challenge is skipped: the only variable part is how many dissents were raised, and
    every one of them is answered before anyone reassesses or synthesises.
    """
    index = len(calls) + 1
    if not calls:
        return Turn(index=index, role="candidate_synthesis")
    if len(calls) <= len(CHALLENGE_ROLES):
        return Turn(index=index, role=CHALLENGE_ROLES[len(calls) - 1])
    answered = _answered(calls)
    for challenger in _dissents_raised(calls):
        if challenger not in answered:
            return Turn(index=index, role="response_and_revision", target=challenger)
    if not _spoken(calls, "independent_reassessment"):
        return Turn(index=index, role="independent_reassessment")
    if not _spoken(calls, "final_synthesis"):
        return Turn(index=index, role="final_synthesis")
    return None


# --------------------------------------------------------------------------------------------
# Dissent: retired by argument, never by arithmetic
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dissent:
    """One challenger's standing objection, and what became of it."""

    challenger: str
    verdict: str
    argument: str
    alternatives: Tuple[str, ...]
    outcome: str
    state: str
    closed_by: str

    def to_mapping(self) -> Dict[str, Any]:
        return {"challenger": self.challenger, "verdict": self.verdict,
                "argument": self.argument, "alternatives": list(self.alternatives),
                "outcome": self.outcome, "state": self.state, "closed_by": self.closed_by}


def _standing(calls: Tuple[Any, ...]) -> Optional[Tuple[str, ...]]:
    """What the independent reassessment left open, or None if it has not spoken yet."""
    for call in calls:
        if call.request.role == "independent_reassessment":
            return tuple(_answer(call)["standing_dissent"])
    return None


def _register(calls: Tuple[Any, ...]) -> Tuple[Dissent, ...]:
    """Every dissent raised, with its state under the closure rule.

    The rule, in full: conceding retires a dissent; recording it unresolved keeps it open; a
    rebuttal is the candidate marking its own homework and holds only if the independent
    reassessment does not list the challenge as still standing.  Until that reassessment has
    happened, a rebuttal is provisional and the dissent counts as open.  No step in this
    function looks at how many challengers agreed with anything.
    """
    outcomes = {_answer(call)["answers_challenge"]: _answer(call)["outcome"]
                for call in calls if call.request.role == "response_and_revision"}
    standing = _standing(calls)
    register = []
    for call in calls:
        if call.request.role not in CHALLENGE_ROLES:
            continue
        answer = _answer(call)
        if not answer["dissent"]:
            continue
        challenger = call.request.role
        outcome = outcomes.get(challenger, "")
        if outcome == "conceded":
            state, closed_by = "resolved", CLOSURES[0]
        elif outcome == "rebutted" and standing is not None and challenger not in standing:
            state, closed_by = "resolved", CLOSURES[1]
        else:
            state, closed_by = "unresolved", ""
        register.append(Dissent(challenger=challenger, verdict=answer["verdict"],
                                argument=answer["argument"],
                                alternatives=tuple(answer["alternatives"]),
                                outcome=outcome, state=state, closed_by=closed_by))
    return tuple(register)


# --------------------------------------------------------------------------------------------
# The exchange
# --------------------------------------------------------------------------------------------


def _check_turn(turn: Turn, call: Any, panel: ReviewPanel, before: Tuple[Any, ...]) -> None:
    """Refuse a recorded call that is not the one the protocol asked for at this point."""
    if call.request.role != turn.role:
        raise InvalidParameterError(
            "call.request.role", call.request.role,
            "%r at turn %d. The order is the protocol: a challenge cannot be synthesised over "
            "before it has been answered" % (turn.role, turn.index))
    if call.request.response_schema.digest() != turn.schema.digest():
        raise InvalidParameterError("call.request.response_schema",
                                    call.request.response_schema.name,
                                    "the %r schema this turn declares" % turn.schema.name)
    seat = panel.seat(turn.role)
    if call.request.model_id != seat.model_id or call.request.effort != seat.effort:
        raise InvalidParameterError(
            "call.request.model_id", "%s at effort %s" % (call.request.model_id,
                                                          call.request.effort),
            "the seated %s at effort %s. The panel is pinned for the whole exchange, because a "
            "bundle reviewed by a different model is different evidence"
            % (seat.model_id, seat.effort))
    answer = _answer(call)
    if turn.role in CHALLENGE_ROLES:
        if answer["verdict"] == "unsupported" and not answer["dissent"]:
            raise InvalidParameterError(
                "response.dissent", answer["dissent"],
                "true. A challenge that finds the claim unsupported is a dissent, whatever it "
                "calls itself, and it must be answered before anyone synthesises")
    elif turn.role == "response_and_revision":
        if answer["answers_challenge"] in _answered(before):
            raise InvalidParameterError(
                "response.answers_challenge", answer["answers_challenge"],
                "a dissent that has not been answered yet. Each challenge is answered exactly "
                "once; answering one twice would leave another open for ever and the exchange "
                "would never reach a synthesis")
        if answer["answers_challenge"] != turn.target:
            raise InvalidParameterError(
                "response.answers_challenge", answer["answers_challenge"],
                "%r, the oldest dissent still unanswered. Dissents are answered in the order "
                "they were raised" % turn.target)
    elif turn.role == "independent_reassessment":
        raised = _dissents_raised(before)
        seen = _strings("response.standing_dissent", answer["standing_dissent"])
        if len(set(seen)) != len(seen):
            raise InvalidParameterError("response.standing_dissent", list(seen),
                                        "each standing dissent named once")
        stray = tuple(name for name in seen if name not in raised)
        if stray:
            raise InvalidParameterError(
                "response.standing_dissent", list(stray),
                "only challenges actually raised (%s). The reassessment may reopen a dissent; "
                "it cannot originate one at this turn, because nothing would answer it"
                % (", ".join(raised) if raised else "none were"))
    elif turn.role == "final_synthesis":
        open_now = tuple(item.challenger for item in _register(before)
                         if item.state == "unresolved")
        retained = _strings("response.retained_dissent", answer["retained_dissent"])
        if set(retained) != set(open_now):
            raise InvalidParameterError(
                "response.retained_dissent", list(retained),
                "exactly the unresolved dissents (%s). Unresolved dissent is retained, never "
                "reconciled into consensus, and no count of agreeing challengers retires one"
                % (", ".join(open_now) if open_now else "none"))
        if answer["dissent_remains"] != bool(open_now):
            raise InvalidParameterError(
                "response.dissent_remains", answer["dissent_remains"],
                "%r, which is whether any dissent is actually unresolved" % bool(open_now))


@dataclass(frozen=True)
class RoundRobin:
    """A reviewed bundle whose recorded chain is a valid partial or complete exchange."""

    reviewed: ReviewedBundle
    panel: ReviewPanel

    def __post_init__(self) -> None:
        if not isinstance(self.reviewed, ReviewedBundle):
            raise InvalidParameterError("reviewed", type(self.reviewed).__name__,
                                        "a ReviewedBundle")
        if not isinstance(self.panel, ReviewPanel):
            raise InvalidParameterError("panel", type(self.panel).__name__, "a ReviewPanel")
        calls = self.reviewed.review.calls
        for position in range(len(calls)):
            before = calls[:position]
            turn = _planned(before)
            if turn is None:
                raise InvalidParameterError(
                    "review.calls", len(calls),
                    "no more than the %d turns the protocol has. The exchange was already "
                    "complete and a further call cannot be part of it" % position)
            _check_turn(turn, calls[position], self.panel, before)

    @property
    def turns(self) -> Tuple[Turn, ...]:
        calls = self.reviewed.review.calls
        return tuple(_planned(calls[:position]) for position in range(len(calls)))

    @property
    def next_turn(self) -> Optional[Turn]:
        return _planned(self.reviewed.review.calls)

    @property
    def complete(self) -> bool:
        return self.next_turn is None

    @property
    def dissent(self) -> Tuple[Dissent, ...]:
        return _register(self.reviewed.review.calls)

    @property
    def unresolved_dissent(self) -> Tuple[Dissent, ...]:
        return tuple(item for item in self.dissent if item.state == "unresolved")

    @property
    def claim_sha256(self) -> str:
        """A function of the bundle only.  No turn of this exchange appears in it (R22)."""
        return self.reviewed.claim_sha256

    @property
    def rung(self) -> str:
        return self.reviewed.rung

    def answer(self, role: str) -> Optional[Dict[str, Any]]:
        """The structured answer given in a seat, or None if that seat has not spoken."""
        if role not in REVIEW_ROLES:
            raise InvalidParameterError("role", role, "one of %s" % ", ".join(REVIEW_ROLES))
        for call in self.reviewed.review.calls:
            if call.request.role == role:
                return _answer(call)
        return None


def _context(reviewed: ReviewedBundle, turn: Turn) -> Dict[str, Any]:
    """What the seat is shown: the claim state, and everything said so far.

    The claim state is a read-only projection of the bundle.  Showing a reviewer the rung it
    stands on is deliberate - a challenger cannot attack a claim it has not been told - and it
    is exactly what makes the acceptance test worth running: the model can see the rung, argue
    at length that it is wrong, and still not move it.
    """
    outputs = reviewed.outputs
    return {
        "study_id": reviewed.bundle.study_id,
        "bundle_sha256": reviewed.bundle.bundle_sha256,
        "bundle_revision": reviewed.bundle.revision,
        "claim_state": outputs.to_mapping(),
        "transcript": [{"turn": call.sequence, "role": call.request.role,
                        "answer": _answer(call)} for call in reviewed.review.calls],
        "answering": turn.target,
    }


def _instruction(turn: Turn) -> str:
    if turn.role == "response_and_revision":
        return ("Turn %d. Answer the %s in the transcript, and only that one. Concede it, "
                "rebut it, or record it as unresolved." % (turn.index, turn.target))
    if turn.role == "final_synthesis":
        return ("Turn %d. State the finding within its bounds and name every dissent left "
                "unresolved by the exchange above." % turn.index)
    return "Turn %d. Speak in the seat of %s over the evidence shown." % (turn.index, turn.role)


def open_round_robin(bundle: EvidenceBundle, panel: ReviewPanel) -> RoundRobin:
    """Seat a panel beside a frozen bundle, with nothing yet said."""
    if not isinstance(panel, ReviewPanel):
        raise InvalidParameterError("panel", type(panel).__name__, "a ReviewPanel")
    return RoundRobin(reviewed=attach_review(bundle), panel=panel)


def advance(exchange: RoundRobin, *, transport: Callable[[Mapping[str, Any]], Any],
            requested_at: str) -> RoundRobin:
    """Take exactly one turn, in the seat the protocol demands, and record it verbatim.

    A turn that comes back malformed is still a turn that was taken.  It is recorded first and
    refused second, and the refusal carries the record, so nothing paid for is thrown away
    because it was disappointing (R23).
    """
    if not isinstance(exchange, RoundRobin):
        raise InvalidParameterError("exchange", type(exchange).__name__, "a RoundRobin")
    turn = exchange.next_turn
    if turn is None:
        raise InvalidParameterError(
            "exchange", "complete",
            "an exchange with a turn left to take. The final synthesis has been given and "
            "adding to it would be revising a conclusion after the fact")
    seat = exchange.panel.seat(turn.role)
    after = record_call(exchange.reviewed, role=turn.role, model_id=seat.model_id,
                        effort=seat.effort, system=RUBRICS[turn.role],
                        instruction=_instruction(turn),
                        context=_context(exchange.reviewed, turn),
                        response_schema=turn.schema, transport=transport,
                        requested_at=requested_at)
    try:
        return RoundRobin(reviewed=after, panel=exchange.panel)
    except UserInputError as exc:
        raise RecordedTurnRefused(after, exc) from None


# --------------------------------------------------------------------------------------------
# The outcome: a bounded finding, and every dissent that outlived the argument
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundRobinOutcome:
    """What a completed exchange leaves behind.  Commentary, digested, beside the bundle.

    `claim_sha256` and `rung` are carried so a reader can see which claim state was under
    discussion.  They are copied from the bundle, never set here, and `close_round_robin`
    re-derives them through `verify_claim_independence` before this record is built.
    """

    study_id: str
    bundle_sha256: str
    bundle_revision: int
    panel: ReviewPanel
    record_sha256: str
    finding: str
    bounded_by: Tuple[str, ...]
    retained_dissent: Tuple[Dissent, ...]
    claim_sha256: str
    rung: str
    outcome_sha256: str = ""
    schema: str = dc_field(default=ROUND_ROBIN_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bounded_by", tuple(self.bounded_by))
        object.__setattr__(self, "retained_dissent", tuple(self.retained_dissent))
        computed = _digest(self.body())
        if not self.outcome_sha256:
            object.__setattr__(self, "outcome_sha256", computed)
        elif self.outcome_sha256 != computed:
            raise InvalidParameterError("outcome_sha256", self.outcome_sha256,
                                        "the digest recomputed from the complete outcome")

    @property
    def dissent_remains(self) -> bool:
        return bool(self.retained_dissent)

    def body(self) -> Dict[str, Any]:
        return {"schema": self.schema, "study_id": self.study_id,
                "bundle_sha256": self.bundle_sha256, "bundle_revision": self.bundle_revision,
                "panel": self.panel.to_mapping(), "record_sha256": self.record_sha256,
                "finding": self.finding, "bounded_by": list(self.bounded_by),
                "retained_dissent": [item.to_mapping() for item in self.retained_dissent],
                "claim_sha256": self.claim_sha256, "rung": self.rung}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), outcome_sha256=self.outcome_sha256)

    def render(self) -> str:
        lines = [R23_DECLARATION, "",
                 "round-robin over bundle %s at revision %d (study %s)"
                 % (self.bundle_sha256, self.bundle_revision, self.study_id),
                 "claim state under discussion: %s, digest %s (set by the gates, not by this "
                 "exchange)" % (self.rung, self.claim_sha256), "",
                 self.panel.render(), "",
                 "finding: %s" % self.finding]
        for bound in self.bounded_by:
            lines.append("   bounded by: %s" % bound)
        lines.append("")
        if not self.retained_dissent:
            lines.append("no dissent was left unresolved. That is not a vote and not a "
                         "consensus: every challenge raised was conceded or rebutted on the "
                         "record above.")
            return "\n".join(lines)
        lines.append("unresolved dissent retained (%d), not reconciled:"
                     % len(self.retained_dissent))
        for item in self.retained_dissent:
            lines.append("   %s [%s]: %s" % (item.challenger, item.verdict, item.argument))
            for alternative in item.alternatives:
                lines.append("      alternative not excluded: %s" % alternative)
        return "\n".join(lines)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RoundRobinOutcome":
        if not isinstance(value, Mapping) or sorted(value) != sorted(_OUTCOME_FIELDS):
            raise InvalidParameterError("outcome", value,
                                        "an object with %s" % ", ".join(_OUTCOME_FIELDS))
        if value["schema"] != ROUND_ROBIN_SCHEMA:
            raise InvalidParameterError("schema", value["schema"], ROUND_ROBIN_SCHEMA)
        dissent = tuple(
            Dissent(challenger=item["challenger"], verdict=item["verdict"],
                    argument=item["argument"], alternatives=tuple(item["alternatives"]),
                    outcome=item["outcome"], state=item["state"], closed_by=item["closed_by"])
            for item in value["retained_dissent"])
        return cls(study_id=value["study_id"], bundle_sha256=value["bundle_sha256"],
                   bundle_revision=value["bundle_revision"],
                   panel=ReviewPanel.from_mapping(value["panel"]),
                   record_sha256=value["record_sha256"], finding=value["finding"],
                   bounded_by=tuple(value["bounded_by"]), retained_dissent=dissent,
                   claim_sha256=value["claim_sha256"], rung=value["rung"],
                   outcome_sha256=value["outcome_sha256"])


def close_round_robin(exchange: RoundRobin) -> RoundRobinOutcome:
    """Close a completed exchange, after proving it moved nothing (R22)."""
    if not isinstance(exchange, RoundRobin):
        raise InvalidParameterError("exchange", type(exchange).__name__, "a RoundRobin")
    if not exchange.complete:
        turn = exchange.next_turn
        raise InvalidParameterError(
            "exchange", turn.describe() if turn else "",
            "a complete exchange. A synthesis over an unfinished round-robin would be a "
            "conclusion drawn before the challenges were heard")
    claim = verify_claim_independence(exchange.reviewed)
    final = exchange.answer("final_synthesis") or {}
    bundle = exchange.reviewed.bundle
    return RoundRobinOutcome(
        study_id=bundle.study_id, bundle_sha256=bundle.bundle_sha256,
        bundle_revision=bundle.revision, panel=exchange.panel,
        record_sha256=exchange.reviewed.review.record_sha256,
        finding=final.get("finding", ""), bounded_by=tuple(final.get("bounded_by", ())),
        retained_dissent=exchange.unresolved_dissent, claim_sha256=claim,
        rung=summarise_evidence(bundle).rung)


__all__ = [
    "ROUND_ROBIN_SCHEMA", "PANEL_SCHEMA", "CHALLENGE_ROLES", "VERDICTS", "RESPONSE_OUTCOMES",
    "DISSENT_STATES", "CLOSURES", "R22_RUBRIC", "RUBRICS", "SCHEMA_FOR_ROLE",
    "CANDIDATE_SCHEMA", "CHALLENGE_SCHEMA", "RESPONSE_SCHEMA", "REASSESSMENT_SCHEMA",
    "FINAL_SCHEMA", "RecordedTurnRefused", "PanelSeat", "ReviewPanel", "Turn", "Dissent",
    "RoundRobin", "RoundRobinOutcome", "open_round_robin", "advance", "close_round_robin",
]
