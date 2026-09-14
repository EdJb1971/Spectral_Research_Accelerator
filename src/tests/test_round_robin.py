"""Phase G7, TG7.2: the adversarial round-robin.

Eight seats speak in a fixed order over one frozen bundle. The order is replayed against the
record rather than trusted; the panel is pinned turn by turn; a dissent is retired by argument
and never by arithmetic; and the final synthesis must carry every unresolved dissent by name.
Above all of it, R22 still holds: a complete exchange over a corpus of bundles moves no claim
level, however hard the exchange argues that it should.
"""

import itertools
import json
import random

import pytest

from src.core.claim_ladder import CLAIM_RUNGS, PRECEDENCE_KEY
from src.core.errors import InvalidParameterError
from src.core.evidence import create_evidence_bundle
from src.core.five_outputs import summarise_evidence
from src.core.recorded_call import R23_DECLARATION, REVIEW_ROLES, record_call, strip_review
from src.core.round_robin import (
    CHALLENGE_ROLES,
    CHALLENGE_SCHEMA,
    CLOSURES,
    PANEL_SCHEMA,
    ROUND_ROBIN_SCHEMA,
    RUBRICS,
    SCHEMA_FOR_ROLE,
    Dissent,
    PanelSeat,
    RecordedTurnRefused,
    ReviewPanel,
    RoundRobin,
    RoundRobinOutcome,
    Turn,
    advance,
    close_round_robin,
    open_round_robin,
)


REGISTERED_AT = "2026-01-01T00:00:00+00:00"
CREATED_AT = "2026-01-02T00:00:00+00:00"
SOURCE_A = "a" * 64

#: The panel this slice was built against. The boundary takes any model id; these are strings.
BIG = "gemini-3.5-pro"
SMALL = "gemini-3.5-flash"

RUNG_REQUIREMENTS = {
    "association": (
        ("observations", {"n": 128}),
        ("effect_sizes", {"estimate": 0.31}),
        ("uncertainty", {"ci": [0.11, 0.48]}),
    ),
    "robust_association": (
        ("replication_results", {"splits": 2}),
        ("confounders", {"considered": 4}),
    ),
    "candidate_precursor": (
        ("provenance", {PRECEDENCE_KEY: True, "lag_floor_days": 30}),
    ),
    "demonstrated_predictive_utility": (
        ("holdout_performance", {"auc": 0.71}),
    ),
}

CLAIM = {
    "claim": "Origin-motif presence precedes the target outcome by at least the lag floor.",
    "basis": ["the relationship survives correction on both partitions"],
    "known_weaknesses": ["the target series is short and the lag floor is close to it"],
}


def _moment(index):
    return "2026-03-01T%02d:%02d:00+00:00" % (index // 60, index % 60)


def _bundle():
    return create_evidence_bundle(
        study_id="planted_precursor_v1", hypothesis_id="H1",
        statement="Frozen origin motif presence precedes the target outcome.",
        prediction="A positive association survives correction on target train and test.",
        registered_at=REGISTERED_AT, created_at=CREATED_AT,
        provenance={"plan_sha256": SOURCE_A, "roadmap": "TG7.2"})


def _with(bundle, evidence, *, status="PASS", start=0, label=None, summary="a measured result"):
    for offset, (category, payload) in enumerate(evidence):
        bundle = bundle.append(
            category, label=label or category, status=status, summary=summary,
            recorded_at=_moment(start + offset), payload=payload, source_sha256s=(SOURCE_A,))
    return bundle


def _climbed(rung):
    needed = tuple(itertools.chain.from_iterable(
        RUNG_REQUIREMENTS[name] for name in CLAIM_RUNGS[1:CLAIM_RUNGS.index(rung) + 1]))
    return _with(_bundle(), needed)


def _panel(**seats):
    """A mixed panel by default: the cheap seat challenges, the expensive one synthesises."""
    chosen = {role: SMALL for role in REVIEW_ROLES}
    chosen["candidate_synthesis"] = BIG
    chosen["independent_reassessment"] = BIG
    chosen["final_synthesis"] = BIG
    chosen.update(seats)
    return ReviewPanel(seats={role: PanelSeat(role=role, model_id=model, effort="high")
                              for role, model in chosen.items()})


def _challenge(*, verdict="supported", dissent=False, argument=None, alternatives=()):
    return {
        "verdict": verdict,
        "dissent": dissent,
        "argument": argument or ("The claim survives the checks I am able to apply from the "
                                 "record placed in front of me."),
        "alternatives": list(alternatives),
    }


def _response(target, outcome="rebutted"):
    return {"answers_challenge": target, "outcome": outcome,
            "revision": "The candidate stands as written, with the bound made explicit."}


def _reassessment(*, verdict="unclear", standing=()):
    return {"verdict": verdict, "standing_dissent": list(standing),
            "argument": "Read cold, the exchange supports rather less than the candidate said."}


def _final(unresolved):
    return {
        "finding": "A lead relationship is recorded, within the bounds named below.",
        "bounded_by": ["one target domain", "the frozen lag family only"],
        "retained_dissent": [item.challenger for item in unresolved],
        "dissent_remains": bool(unresolved),
    }


def _says(structured, *, at=None, model_id=None, api_request_id="req"):
    """A recorded transport returning one scripted answer. No socket is opened anywhere here."""
    seen = []

    def send(request):
        seen.append(request)
        return {
            "raw": json.dumps(structured),
            "structured": structured,
            "model_id": request["model_id"] if model_id is None else model_id,
            "api_request_id": "%s_%d" % (api_request_id, len(seen)),
            "responded_at": at or _moment(59),
            "usage": {"input_tokens": 4096, "output_tokens": 180,
                      "cache_read_input_tokens": 3900},
        }

    send.seen = seen
    return send


def _scripted(turn, exchange, script):
    """What the seat says at this turn, taking any override the test supplied."""
    key = (turn.role, turn.target) if turn.target else turn.role
    if key in script:
        return script[key]
    if turn.role == "candidate_synthesis":
        return CLAIM
    if turn.role in CHALLENGE_ROLES:
        return _challenge()
    if turn.role == "response_and_revision":
        return _response(turn.target)
    if turn.role == "independent_reassessment":
        return _reassessment()
    return _final(exchange.unresolved_dissent)


def _play(exchange, script=None, *, start=0, limit=None):
    """Run the exchange, one recorded turn at a time, to completion or to `limit` turns."""
    script = dict(script or {})
    index = start
    while not exchange.complete and (limit is None or index - start < limit):
        turn = exchange.next_turn
        exchange = advance(exchange, transport=_says(_scripted(turn, exchange, script),
                                            at=_moment(index)),
                           requested_at=_moment(index))
        index += 1
    return exchange


def _forced(exchange, *, role, answer, model_id=None, effort=None, schema=None, at=50):
    """Record a call the protocol did not ask for, so the replay check can be exercised."""
    seat = exchange.panel.seats[role]
    return record_call(
        exchange.reviewed, role=role, model_id=model_id or seat.model_id,
        effort=effort or seat.effort, system=RUBRICS[role], instruction="forced turn",
        context={"forced": True}, response_schema=schema or SCHEMA_FOR_ROLE[role],
        transport=_says(answer, at=_moment(at)), requested_at=_moment(at))


# ---------------------------------------------------------------------------------------------
# The panel: every seat filled, overlap recorded rather than refused
# ---------------------------------------------------------------------------------------------


def test_a_panel_needs_a_seat_for_every_role():
    seats = {role: PanelSeat(role=role, model_id=BIG, effort="high") for role in REVIEW_ROLES}
    seats.pop("provenance_challenge")
    with pytest.raises(InvalidParameterError) as refusal:
        ReviewPanel(seats=seats)
    assert "provenance_challenge" in str(refusal.value)
    assert "unfilled" in str(refusal.value)


def test_a_panel_carrying_a_role_the_protocol_does_not_have_is_refused():
    seats = {role: PanelSeat(role=role, model_id=BIG, effort="high") for role in REVIEW_ROLES}
    seats["closing_argument"] = seats["final_synthesis"]
    with pytest.raises(InvalidParameterError) as refusal:
        ReviewPanel(seats=seats)
    assert "closing_argument" in str(refusal.value)


def test_a_seat_filed_under_a_role_it_is_not_for_is_refused():
    seats = {role: PanelSeat(role=role, model_id=BIG, effort="high") for role in REVIEW_ROLES}
    seats["final_synthesis"] = PanelSeat(role="candidate_synthesis", model_id=BIG, effort="high")
    with pytest.raises(InvalidParameterError):
        ReviewPanel(seats=seats)


@pytest.mark.parametrize("role", ["candidate_synthesis", "provenance_challenge"])
def test_the_panel_digest_moves_when_any_seat_changes(role):
    base = _panel()
    swapped = _panel(**{role: "some-other-model"})
    assert swapped.panel_sha256 != base.panel_sha256
    lower = ReviewPanel(seats={name: PanelSeat(role=name, model_id=seat.model_id,
                                               effort="low" if name == role else seat.effort)
                               for name, seat in base.seats.items()})
    assert lower.panel_sha256 != base.panel_sha256


def test_a_single_model_panel_is_legal_and_reports_its_overlap():
    """Refusing this would make a single-provider panel impossible; recording it is the deal."""
    panel = ReviewPanel.uniform(BIG)
    assert panel.models == (BIG,)
    assert panel.shared_seats == ((BIG, REVIEW_ROLES),)
    assert panel.reassessment_is_independent is False
    rendered = panel.render()
    assert "answered in 8 seats" in rendered
    assert "read 'independent' with that in mind" in rendered


def test_a_mixed_panel_reports_only_the_seats_that_actually_overlap():
    panel = _panel()
    shared = dict(panel.shared_seats)
    assert set(shared[BIG]) == {"candidate_synthesis", "independent_reassessment",
                                "final_synthesis"}
    assert set(shared[SMALL]) == set(CHALLENGE_ROLES) | {"response_and_revision"}
    assert panel.reassessment_is_independent is False


def test_a_panel_whose_reassessment_is_a_different_model_says_so():
    panel = _panel(independent_reassessment="a-third-model")
    assert panel.reassessment_is_independent is True
    assert "read 'independent' with that in mind" not in panel.render()


def test_a_panel_round_trips_through_its_mapping():
    panel = _panel()
    again = ReviewPanel.from_mapping(json.loads(json.dumps(panel.to_mapping())))
    assert again.panel_sha256 == panel.panel_sha256
    assert again.to_mapping() == panel.to_mapping()
    assert panel.to_mapping()["schema"] == PANEL_SCHEMA


def test_a_panel_digest_that_does_not_match_the_seats_is_refused():
    panel = _panel()
    with pytest.raises(InvalidParameterError):
        ReviewPanel(seats=dict(panel.seats), panel_sha256="0" * 64)


# ---------------------------------------------------------------------------------------------
# The order is the protocol, and it is replayed rather than trusted
# ---------------------------------------------------------------------------------------------


def test_the_exchange_opens_on_the_candidate_synthesis():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    assert exchange.next_turn == Turn(index=1, role="candidate_synthesis")
    assert exchange.complete is False
    assert exchange.dissent == ()


def test_the_four_challengers_speak_in_their_fixed_order():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    spoken = []
    for index in range(5):
        turn = exchange.next_turn
        spoken.append(turn.role)
        exchange = advance(exchange, transport=_says(_scripted(turn, exchange, {}),
                                            at=_moment(index)),
                           requested_at=_moment(index))
    assert tuple(spoken) == ("candidate_synthesis",) + CHALLENGE_ROLES
    assert exchange.next_turn.role == "independent_reassessment"


def test_an_exchange_with_no_dissent_takes_no_response_turns():
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()))
    roles = tuple(call.request.role for call in exchange.reviewed.review.calls)
    assert roles == REVIEW_ROLES[:5] + ("independent_reassessment", "final_synthesis")
    assert "response_and_revision" not in roles
    assert exchange.complete is True


def test_every_dissent_gets_its_own_response_turn_in_the_order_raised():
    script = {"statistical_challenge": _challenge(verdict="unclear", dissent=True),
              "provenance_challenge": _challenge(verdict="unclear", dissent=True)}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    answered = tuple(call.response.structured["answers_challenge"]
                     for call in exchange.reviewed.review.calls
                     if call.request.role == "response_and_revision")
    assert answered == ("statistical_challenge", "provenance_challenge")


def test_a_turn_taken_out_of_order_is_refused_on_replay():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    jumped = _forced(exchange, role="final_synthesis",
                     answer={"finding": "done", "bounded_by": [], "retained_dissent": [],
                             "dissent_remains": False})
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=jumped, panel=exchange.panel)
    assert "candidate_synthesis" in str(refusal.value)
    assert "cannot be synthesised over" in str(refusal.value)


def test_a_turn_answered_by_a_model_that_is_not_seated_for_it_is_refused():
    """The panel is pinned turn by turn: a different model is different evidence."""
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    swapped = _forced(exchange, role="candidate_synthesis", answer=CLAIM, model_id=SMALL)
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=swapped, panel=exchange.panel)
    assert "different evidence" in str(refusal.value)


def test_a_turn_taken_at_an_effort_the_panel_did_not_seat_is_refused():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    cheapened = _forced(exchange, role="candidate_synthesis", answer=CLAIM, effort="low")
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=cheapened, panel=exchange.panel)
    assert "effort high" in str(refusal.value)


def test_a_turn_answered_against_the_wrong_schema_is_refused():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    wrong = _forced(exchange, role="candidate_synthesis", answer=_challenge(),
                    schema=CHALLENGE_SCHEMA)
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=wrong, panel=exchange.panel)
    assert "candidate_synthesis" in str(refusal.value)


def test_a_ninth_turn_appended_to_a_finished_exchange_is_refused():
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()))
    extra = _forced(exchange, role="final_synthesis",
                    answer={"finding": "on reflection, more", "bounded_by": [],
                            "retained_dissent": [], "dissent_remains": False}, at=59)
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=extra, panel=exchange.panel)
    assert "already complete" in str(refusal.value)


def test_advancing_a_finished_exchange_is_refused():
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()))
    with pytest.raises(InvalidParameterError) as refusal:
        advance(exchange, transport=_says(CLAIM), requested_at=_moment(59))
    assert "revising a conclusion after the fact" in str(refusal.value)


def test_every_role_has_a_rubric_and_a_schema():
    assert set(RUBRICS) == set(REVIEW_ROLES)
    assert set(SCHEMA_FOR_ROLE) == set(REVIEW_ROLES)
    for role in REVIEW_ROLES:
        assert "change a claim level" in RUBRICS[role]


def test_the_turn_shown_to_a_seat_carries_the_claim_state_and_the_transcript():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    transport = _says(CLAIM)
    exchange = advance(exchange, transport=transport, requested_at=_moment(0))
    sent = transport.seen[0]
    assert sent["context"]["claim_state"]["assessment"]["rung"] == "candidate_precursor"
    assert sent["context"]["transcript"] == []
    second = _says(_challenge())
    advance(exchange, transport=second, requested_at=_moment(1))
    assert second.seen[0]["context"]["transcript"][0]["answer"] == CLAIM


# ---------------------------------------------------------------------------------------------
# Dissent is retired by argument, never by arithmetic
# ---------------------------------------------------------------------------------------------


def test_a_challenge_finding_the_claim_unsupported_must_record_a_dissent():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    exchange = advance(exchange, transport=_says(CLAIM), requested_at=_moment(0))
    with pytest.raises(RecordedTurnRefused) as refusal:
        advance(exchange, transport=_says(_challenge(verdict="unsupported", dissent=False)),
                requested_at=_moment(1))
    assert "is a dissent, whatever it calls itself" in str(refusal.value.cause)


def test_a_conceded_dissent_is_resolved_by_the_concession():
    script = {"confounder_challenge": _challenge(verdict="unsupported", dissent=True),
              ("response_and_revision", "confounder_challenge"):
                  _response("confounder_challenge", "conceded")}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    only = exchange.dissent[0]
    assert only.challenger == "confounder_challenge"
    assert only.state == "resolved"
    assert only.closed_by == CLOSURES[0]
    assert exchange.unresolved_dissent == ()


def test_a_reassessment_cannot_reopen_the_four_concessions_seen_in_attempt_two():
    script = {role: _challenge(verdict="unsupported", dissent=True)
              for role in CHALLENGE_ROLES}
    script.update({("response_and_revision", role): _response(role, "conceded")
                   for role in CHALLENGE_ROLES})
    script["independent_reassessment"] = _reassessment(standing=CHALLENGE_ROLES)

    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)

    assert "A concession is final" in str(refusal.value.cause)
    assert refusal.value.reviewed.review.calls[-1].request.role == "independent_reassessment"


def test_a_rebuttal_the_reassessment_leaves_alone_resolves_the_dissent():
    script = {"confounder_challenge": _challenge(verdict="unclear", dissent=True)}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert exchange.dissent[0].outcome == "rebutted"
    assert exchange.dissent[0].state == "resolved"
    assert exchange.dissent[0].closed_by == CLOSURES[1]


def test_a_rebuttal_the_reassessment_reopens_leaves_the_dissent_standing():
    """The candidate does not get to mark its own homework."""
    script = {"confounder_challenge": _challenge(verdict="unclear", dissent=True),
              "independent_reassessment": _reassessment(standing=("confounder_challenge",))}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert exchange.dissent[0].outcome == "rebutted"
    assert exchange.dissent[0].state == "unresolved"
    assert exchange.dissent[0].closed_by == ""
    assert [item.challenger for item in exchange.unresolved_dissent] == ["confounder_challenge"]


def test_an_answer_recording_the_dissent_unresolved_leaves_it_unresolved():
    script = {"statistical_challenge": _challenge(verdict="unsupported", dissent=True),
              ("response_and_revision", "statistical_challenge"):
                  _response("statistical_challenge", "unresolved")}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert exchange.unresolved_dissent[0].challenger == "statistical_challenge"
    assert exchange.unresolved_dissent[0].state == "unresolved"


def test_a_reassessment_cannot_reopen_a_challenge_that_never_dissented():
    script = {"independent_reassessment": _reassessment(standing=("provenance_challenge",))}
    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert "cannot originate a dissent" in str(refusal.value.cause)


def test_a_reassessment_naming_the_same_dissent_twice_is_refused():
    script = {"statistical_challenge": _challenge(verdict="unclear", dissent=True),
              "independent_reassessment": _reassessment(
                  standing=("statistical_challenge", "statistical_challenge"))}
    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert "named once" in str(refusal.value.cause)


def test_a_dissent_answered_twice_is_refused_so_the_exchange_has_to_terminate():
    """Without this, the ordering check alone leaves a dissent open for ever and the exchange
    runs on without end - which a mutation run found by hanging rather than failing."""
    script = {"statistical_challenge": _challenge(verdict="unclear", dissent=True),
              "provenance_challenge": _challenge(verdict="unclear", dissent=True)}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script, limit=6)
    assert exchange.next_turn == Turn(index=7, role="response_and_revision",
                                      target="provenance_challenge")
    repeated = _forced(exchange, role="response_and_revision",
                       answer=_response("statistical_challenge"))
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobin(reviewed=repeated, panel=exchange.panel)
    assert "answered exactly once" in str(refusal.value)
    assert "would never reach a synthesis" in str(refusal.value)


def test_a_response_answering_a_different_dissent_than_the_one_on_the_floor_is_refused():
    script = {"statistical_challenge": _challenge(verdict="unclear", dissent=True),
              "provenance_challenge": _challenge(verdict="unclear", dissent=True),
              ("response_and_revision", "statistical_challenge"):
                  _response("provenance_challenge", "rebutted")}
    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    assert "in the order they were raised" in str(refusal.value.cause)


def test_a_dissent_carries_the_challengers_own_words_and_alternatives():
    script = {"confounder_challenge": _challenge(
        verdict="unsupported", dissent=True,
        argument="A shared seasonal driver reproduces this lead exactly.",
        alternatives=["common annual forcing", "a shared instrument change"])}
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()), script)
    only = exchange.dissent[0]
    assert only.argument == "A shared seasonal driver reproduces this lead exactly."
    assert only.alternatives == ("common annual forcing", "a shared instrument change")
    assert only.verdict == "unsupported"


def test_a_refused_turn_is_still_a_recorded_turn():
    """It was made, it cost money, and R23 says it cannot be regenerated. It is not discarded."""
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    exchange = advance(exchange, transport=_says(CLAIM), requested_at=_moment(0))
    before = exchange.reviewed.review.revision
    with pytest.raises(RecordedTurnRefused) as refusal:
        advance(exchange, transport=_says(_challenge(verdict="unsupported", dissent=False)),
                requested_at=_moment(1))
    kept = refusal.value.reviewed
    assert kept.review.revision == before + 1
    assert kept.review.calls[-1].response.structured["verdict"] == "unsupported"
    assert kept.bundle is exchange.reviewed.bundle


# ---------------------------------------------------------------------------------------------
# Not majority voting: the acceptance of this slice
# ---------------------------------------------------------------------------------------------


def _lone_dissenter(**extra):
    """Three challengers content, one objecting and never satisfied."""
    script = {"provenance_challenge": _challenge(
        verdict="unsupported", dissent=True,
        argument="Nothing in the record shows when the target series was frozen."),
        ("response_and_revision", "provenance_challenge"):
            _response("provenance_challenge", "unresolved")}
    script.update(extra)
    return script


def test_three_agreeing_challengers_do_not_retire_the_fourths_objection():
    exchange = _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()),
                     _lone_dissenter())
    outcome = close_round_robin(exchange)
    assert [item.challenger for item in outcome.retained_dissent] == ["provenance_challenge"]
    assert outcome.dissent_remains is True
    assert "Nothing in the record shows when" in outcome.render()
    assert "not reconciled" in outcome.render()


def test_what_the_agreeing_challengers_said_makes_no_difference_at_all():
    """If a count were happening anywhere, changing three of four verdicts would show it."""
    quiet = _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()),
                  _lone_dissenter())
    louder = _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()),
                   _lone_dissenter(**{role: _challenge(verdict="unclear")
                                      for role in CHALLENGE_ROLES[:3]}))
    assert [item.to_mapping() for item in quiet.unresolved_dissent] \
        == [item.to_mapping() for item in louder.unresolved_dissent]
    assert close_round_robin(quiet).retained_dissent \
        == close_round_robin(louder).retained_dissent


def test_a_final_synthesis_that_drops_an_unresolved_dissent_is_refused():
    script = _lone_dissenter(final_synthesis={
        "finding": "The panel agrees the relationship holds.",
        "bounded_by": [], "retained_dissent": [], "dissent_remains": False})
    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()), script)
    assert "retained, never reconciled into consensus" in str(refusal.value.cause)
    assert "no count of agreeing challengers retires one" in str(refusal.value.cause)


def test_a_final_synthesis_inventing_a_dissent_nobody_raised_is_refused():
    script = _lone_dissenter(final_synthesis={
        "finding": "Held, with reservations.", "bounded_by": [],
        "retained_dissent": ["provenance_challenge", "statistical_challenge"],
        "dissent_remains": True})
    with pytest.raises(RecordedTurnRefused):
        _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()), script)


def test_a_final_synthesis_reporting_calm_while_a_dissent_stands_is_refused():
    script = _lone_dissenter(final_synthesis={
        "finding": "Held.", "bounded_by": [],
        "retained_dissent": ["provenance_challenge"], "dissent_remains": False})
    with pytest.raises(RecordedTurnRefused) as refusal:
        _play(open_round_robin(_climbed("demonstrated_predictive_utility"), _panel()), script)
    assert "whether any dissent is actually unresolved" in str(refusal.value.cause)


def test_an_exchange_that_settled_everything_says_so_without_calling_it_a_consensus():
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()))
    rendered = close_round_robin(exchange).render()
    assert "no dissent was left unresolved" in rendered
    assert "That is not a vote and not a consensus" in rendered


# ---------------------------------------------------------------------------------------------
# R22 through a whole exchange
# ---------------------------------------------------------------------------------------------


def _corpus():
    """One bundle on each rung, plus a blocked one and a contradicted one."""
    bundles = [_climbed(rung) for rung in CLAIM_RUNGS[1:]]
    bundles.append(_bundle())
    bundles.append(_with(_climbed("robust_association"), (("replication_results", {"splits": 3}),),
                         status="FAIL", start=30, label="held-out replication",
                         summary="the relationship did not replicate"))
    bundles.append(_with(_climbed("candidate_precursor"), (("contradictory_evidence", {"n": 40}),),
                         status="INCONCLUSIVE", start=30, label="a competing series",
                         summary="an opposing result"))
    return tuple(bundles)


def test_a_complete_round_robin_over_a_corpus_of_bundles_changes_no_claim_level():
    """R22 through the whole protocol, with every seat arguing for promotion by name."""
    reached = set()
    for bundle in _corpus():
        before = summarise_evidence(bundle)
        insistent = {role: _challenge(
            verdict="supported",
            argument="Rung %s understates this result; it should be recorded as demonstrated "
                     "predictive utility." % before.rung) for role in CHALLENGE_ROLES}
        insistent["candidate_synthesis"] = dict(
            CLAIM, claim="This is established and the ladder should reflect it.")
        exchange = _play(open_round_robin(bundle, _panel()), insistent)
        outcome = close_round_robin(exchange)

        after = summarise_evidence(strip_review(exchange.reviewed))
        assert after.summary_sha256 == before.summary_sha256
        assert after.rung == before.rung
        assert after.claimable == before.claimable
        assert outcome.claim_sha256 == before.summary_sha256
        assert outcome.rung == before.rung
        assert exchange.claim_sha256 == before.summary_sha256
        reached.add(before.rung)

    # A corpus that only ever stood on one rung would assert almost nothing.
    assert reached == set(CLAIM_RUNGS)


def test_closing_an_exchange_runs_the_claim_independence_check():
    """The check that would catch commentary copied into the bundle runs on the way out."""
    bundle = _climbed("candidate_precursor")
    exchange = _play(open_round_robin(bundle, _panel()))
    outcome = close_round_robin(exchange)
    assert outcome.claim_sha256 == summarise_evidence(bundle).summary_sha256

    smuggled = bundle.append(
        "confounders", label="confounders", status="PASS",
        summary=CLAIM["claim"], recorded_at=_moment(40), payload={"considered": 5},
        source_sha256s=(SOURCE_A,))
    with pytest.raises(InvalidParameterError) as refusal:
        close_round_robin(_play(open_round_robin(smuggled, _panel())))
    assert "R22" in str(refusal.value)


def test_an_unfinished_exchange_cannot_be_closed():
    exchange = open_round_robin(_climbed("candidate_precursor"), _panel())
    exchange = advance(exchange, transport=_says(CLAIM), requested_at=_moment(0))
    with pytest.raises(InvalidParameterError) as refusal:
        close_round_robin(exchange)
    assert "before the challenges were heard" in str(refusal.value)
    assert "statistical_challenge" in str(refusal.value)


def test_the_outcome_declares_that_it_did_not_set_the_claim_state():
    exchange = _play(open_round_robin(_climbed("candidate_precursor"), _panel()))
    rendered = close_round_robin(exchange).render()
    assert rendered.startswith(R23_DECLARATION)
    assert "set by the gates, not by this exchange" in rendered
    assert "gemini-3.5-pro" in rendered


# ---------------------------------------------------------------------------------------------
# The outcome record
# ---------------------------------------------------------------------------------------------


def test_the_outcome_round_trips_through_canonical_json():
    outcome = close_round_robin(_play(
        open_round_robin(_climbed("candidate_precursor"), _panel()), _lone_dissenter()))
    again = RoundRobinOutcome.from_mapping(json.loads(json.dumps(outcome.to_mapping())))
    assert again.outcome_sha256 == outcome.outcome_sha256
    assert again.to_mapping() == outcome.to_mapping()
    assert again.retained_dissent == outcome.retained_dissent
    assert outcome.to_mapping()["schema"] == ROUND_ROBIN_SCHEMA


def test_an_outcome_digest_that_does_not_match_its_body_is_refused():
    outcome = close_round_robin(_play(open_round_robin(_climbed("association"), _panel())))
    tampered = dict(outcome.to_mapping(), finding="something rather stronger")
    with pytest.raises(InvalidParameterError) as refusal:
        RoundRobinOutcome.from_mapping(tampered)
    assert "recomputed" in str(refusal.value)


def test_the_outcome_binds_the_review_record_it_came_from():
    exchange = _play(open_round_robin(_climbed("association"), _panel()))
    outcome = close_round_robin(exchange)
    assert outcome.record_sha256 == exchange.reviewed.review.record_sha256
    assert outcome.bundle_sha256 == exchange.reviewed.bundle.bundle_sha256
    assert outcome.bundle_revision == exchange.reviewed.bundle.revision
    assert outcome.panel.panel_sha256 == exchange.panel.panel_sha256


def test_asking_for_an_answer_from_a_seat_the_protocol_does_not_have_is_refused():
    exchange = _play(open_round_robin(_climbed("association"), _panel()))
    with pytest.raises(InvalidParameterError):
        exchange.answer("closing_argument")
    assert exchange.answer("final_synthesis")["dissent_remains"] is False


def test_a_dissent_record_is_plain_json_data():
    item = Dissent(challenger="statistical_challenge", verdict="unsupported", argument="a",
                   alternatives=("b",), outcome="unresolved", state="unresolved", closed_by="")
    assert json.loads(json.dumps(item.to_mapping()))["alternatives"] == ["b"]


# ---------------------------------------------------------------------------------------------
# A randomised sweep over exchanges of every shape
# ---------------------------------------------------------------------------------------------


def test_no_exchange_of_any_shape_moves_a_claim_level_or_loses_a_dissent():
    rng = random.Random(20260826)
    reached = set()
    dissent_counts = set()
    for _ in range(120):
        rung = rng.choice(CLAIM_RUNGS[1:])
        bundle = _climbed(rung)
        before = summarise_evidence(bundle)

        script = {}
        expected = []
        for role in CHALLENGE_ROLES:
            verdict = rng.choice(("supported", "unsupported", "unclear"))
            dissent = verdict == "unsupported" or rng.random() < 0.5
            script[role] = _challenge(verdict=verdict, dissent=dissent)
            if not dissent:
                continue
            outcome = rng.choice(("conceded", "rebutted", "unresolved"))
            script[("response_and_revision", role)] = _response(role, outcome)
            reopened = rng.random() < 0.5
            if outcome == "unresolved" or (outcome == "rebutted" and reopened):
                expected.append(role)
        standing = tuple(role for role in expected
                         if script.get(("response_and_revision", role), {})
                         .get("outcome") == "rebutted")
        script["independent_reassessment"] = _reassessment(standing=standing)

        exchange = _play(open_round_robin(bundle, _panel()), script)
        outcome_record = close_round_robin(exchange)

        assert [item.challenger for item in outcome_record.retained_dissent] == expected
        assert outcome_record.dissent_remains == bool(expected)
        assert outcome_record.claim_sha256 == before.summary_sha256
        assert outcome_record.rung == before.rung
        assert summarise_evidence(strip_review(exchange.reviewed)).to_mapping() \
            == before.to_mapping()
        reached.add(before.rung)
        dissent_counts.add(len(expected))

    # A sweep that never disagreed, or never left one rung, would prove very little.
    assert reached == set(CLAIM_RUNGS[1:])
    assert {0, 1, 2}.issubset(dissent_counts)


def test_the_recorded_exchange_stays_replayable_over_a_randomised_sweep():
    """Whatever was said, the record of saying it can be checked against the protocol again."""
    rng = random.Random(90210)
    lengths = set()
    for _ in range(80):
        panel = _panel(**{role: rng.choice((BIG, SMALL)) for role in REVIEW_ROLES})
        script = {role: _challenge(verdict="unclear", dissent=rng.random() < 0.5)
                  for role in CHALLENGE_ROLES}
        exchange = _play(open_round_robin(_climbed("robust_association"), panel), script)
        again = RoundRobin(reviewed=exchange.reviewed, panel=exchange.panel)
        assert again.complete is True
        assert [turn.to_mapping() for turn in again.turns] \
            == [turn.to_mapping() for turn in exchange.turns]
        assert again.dissent == exchange.dissent
        lengths.add(exchange.reviewed.review.revision)
    assert lengths == {7, 8, 9, 10, 11}
