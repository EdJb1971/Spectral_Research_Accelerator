"""T4E.31: the round-robin runner, exercised over the real published bundle.

The protocol, the seats, the dissent register and the transport were all built and tested before
this. What had never happened was a complete exchange over a bundle from this programme's own
work, because until T4E.30 there were no bundles.

**These tests fabricate the panel's answers, and that is only acceptable here.** A test needs to
drive eight turns without paying for them. `tools/review_join_rerun.py` deliberately has no stub:
a review record asserts that a panel said something, and a fabricated one sitting in
`data/reviews/` beside real ones would be the worst artefact this programme could produce.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.evidence import load_evidence_bundle
from src.core.recorded_call import REVIEW_ROLES, verify_claim_independence
from src.core.round_robin import (
    CHALLENGE_ROLES, MAX_ROUND_ROBIN_CALLS, RecordedTurnRefused, advance,
    close_round_robin, open_round_robin,
)
from tools.review_join_rerun import panel_from, resolve_key

BUNDLE = Path("data/studies/t4e28-join-rerun.r7.json")


def _answer(role: str, *, dissent: bool = False):
    if role == "candidate_synthesis":
        return {"claim": "The T4E.18 figures regenerate from this repository.",
                "basis": ["both gates reproduced", "18 of 18 SWT rows regenerated"],
                "known_weaknesses": ["a reproduction of a failing measurement still fails"]}
    if role in CHALLENGE_ROLES:
        return {"verdict": "unclear" if dissent else "supported",
                "argument": "A stated argument about %s." % role,
                "alternatives": ["an alternative reading"], "dissent": dissent}
    if role == "response_and_revision":
        return {"answers_challenge": "statistical_challenge", "outcome": "rebutted",
                "revision": "The claim is unchanged; the objection is answered."}
    if role == "independent_reassessment":
        return {"verdict": "supported", "argument": "Reassessed against the bundle alone.",
                "standing_dissent": []}
    return {"finding": "The regeneration holds; the acceptance it reproduces still fails.",
            "bounded_by": ["says nothing about the atmosphere"],
            "retained_dissent": [], "dissent_remains": False}


def _transport(dissent_on=()):
    """A fabricated panel. Legitimate in a test; refused by the tool."""
    def send(request):
        role = request["role"]
        structured = _answer(role, dissent=role in dissent_on)
        return {"raw": json.dumps(structured), "structured": structured,
                "model_id": request["model_id"],
                "api_request_id": "test-%s" % role,
                "responded_at": "2026-09-11T12:30:00+00:00",
                "usage": {"provider": "test", "service_mode": "test", "input_tokens": 10,
                          "output_tokens": 5, "cached_input_tokens": 0, "total_tokens": 15,
                          "batch_name": "batches/test"}}
    return send


def _run(transport):
    exchange = open_round_robin(load_evidence_bundle(BUNDLE), panel_from("a-model", "high"))
    for index in range(MAX_ROUND_ROBIN_CALLS):
        if exchange.next_turn is None:
            break
        exchange = advance(exchange, transport=transport,
                           requested_at="2026-09-11T12:00:%02d+00:00" % index)
    return exchange


# ---------------- the exchange, over the real bundle


def test_an_exchange_with_no_dissent_takes_seven_turns_not_eight():
    """`response_and_revision` is SKIPPED when nothing was dissented from.

    Eight roles exist; eight turns are not always taken. A response to no dissent would be a
    rebuttal of nothing, and the protocol declines to spend a call on it. Asserting eight here
    would have been asserting a number rather than reading the protocol.
    """
    exchange = _run(_transport())
    spoken = [call.request.role for call in exchange.reviewed.review.calls]

    assert exchange.complete is True
    assert spoken == [role for role in REVIEW_ROLES if role != "response_and_revision"]
    assert len(spoken) == 7


def test_the_review_moves_no_claim_level_which_is_r22_made_executable():
    """The whole point of the separation: deleting every recorded call changes nothing."""
    before = load_evidence_bundle(BUNDLE)
    exchange = _run(_transport())

    assert verify_claim_independence(exchange.reviewed)
    assert exchange.reviewed.bundle.bundle_sha256 == before.bundle_sha256
    assert exchange.reviewed.bundle.revision == before.revision


def test_the_outcome_copies_the_rung_from_the_bundle_and_never_sets_one():
    exchange = _run(_transport())
    outcome = close_round_robin(exchange)

    assert outcome.rung == "observation"
    assert outcome.bundle_sha256 == load_evidence_bundle(BUNDLE).bundle_sha256
    assert outcome.finding


def test_one_dissent_adds_the_response_turn_and_must_be_answered_by_name():
    """Three challengers agreeing does not retire the fourth's objection."""
    def answering(request):
        if request["role"] != "response_and_revision":
            return _transport(dissent_on=("provenance_challenge",))(request)
        structured = {"answers_challenge": "provenance_challenge", "outcome": "conceded",
                      "revision": "The provenance objection is accepted."}
        return {"raw": json.dumps(structured), "structured": structured,
                "model_id": request["model_id"], "api_request_id": "x",
                "responded_at": "2026-09-11T12:30:00+00:00",
                "usage": {"provider": "test", "service_mode": "test", "input_tokens": 1,
                          "output_tokens": 1, "cached_input_tokens": 0, "total_tokens": 2,
                          "batch_name": "batches/test"}}

    exchange = _run(answering)
    spoken = [call.request.role for call in exchange.reviewed.review.calls]

    assert "response_and_revision" in spoken and len(spoken) == 8
    outcome = close_round_robin(exchange)
    # Conceded retires it, so nothing is retained -- and the concession is in the record.
    assert outcome.retained_dissent == ()


def test_four_dissents_take_eleven_calls_and_each_response_answers_the_current_target():
    def answering(request):
        role = request["role"]
        if role != "response_and_revision":
            return _transport(dissent_on=CHALLENGE_ROLES)(request)
        target = request["context"]["answering"]
        structured = {"answers_challenge": target, "outcome": "conceded",
                      "revision": "The %s objection is accepted." % target}
        return {"raw": json.dumps(structured), "structured": structured,
                "model_id": request["model_id"], "api_request_id": "response-%s" % target,
                "responded_at": "2026-09-11T12:30:00+00:00",
                "usage": {"provider": "test", "service_mode": "test", "input_tokens": 1,
                          "output_tokens": 1, "cached_input_tokens": 0, "total_tokens": 2,
                          "batch_name": "batches/test"}}

    exchange = _run(answering)
    spoken = [call.request.role for call in exchange.reviewed.review.calls]
    answered = [call.response.structured["answers_challenge"]
                for call in exchange.reviewed.review.calls
                if call.request.role == "response_and_revision"]

    assert exchange.complete is True
    assert len(spoken) == MAX_ROUND_ROBIN_CALLS == 11
    assert answered == list(CHALLENGE_ROLES)


def test_closing_an_unfinished_exchange_is_refused():
    from src.core.errors import InvalidParameterError

    exchange = open_round_robin(load_evidence_bundle(BUNDLE), panel_from("a-model", "high"))
    exchange = advance(exchange, transport=_transport(),
                       requested_at="2026-09-11T12:00:00+00:00")
    with pytest.raises(InvalidParameterError) as caught:
        close_round_robin(exchange)
    assert "before the challenges were heard" in str(caught.value)


def test_a_dissent_must_be_answered_by_name_and_oldest_first():
    """A protocol violation is recorded and THEN refused, so the paid call survives (R23).

    The response here names `statistical_challenge` while the outstanding dissent is
    `provenance_challenge`. That is refused, and the refusal carries the call.
    """
    def answers_the_wrong_one(request):
        if request["role"] != "response_and_revision":
            return _transport(dissent_on=("provenance_challenge",))(request)
        structured = {"answers_challenge": "statistical_challenge", "outcome": "rebutted",
                      "revision": "answering a challenge that raised no dissent"}
        return {"raw": json.dumps(structured), "structured": structured,
                "model_id": request["model_id"], "api_request_id": "x",
                "responded_at": "2026-09-11T12:30:00+00:00",
                "usage": {"provider": "test", "service_mode": "test", "input_tokens": 1,
                          "output_tokens": 1, "cached_input_tokens": 0, "total_tokens": 2,
                          "batch_name": "batches/test"}}

    exchange = open_round_robin(load_evidence_bundle(BUNDLE), panel_from("a-model", "high"))
    with pytest.raises(RecordedTurnRefused) as caught:
        for index in range(MAX_ROUND_ROBIN_CALLS):
            if exchange.next_turn is None:
                break
            exchange = advance(exchange, transport=answers_the_wrong_one,
                               requested_at="2026-09-11T12:00:%02d+00:00" % index)

    assert "the oldest dissent still unanswered" in str(caught.value)
    # The call was made and cannot be regenerated, so it is kept on the refusal.
    assert caught.value.reviewed.review.calls
    assert caught.value.reviewed.review.calls[-1].request.role == "response_and_revision"


def test_a_response_that_does_not_match_its_schema_is_refused_before_it_is_recorded():
    """Observed rather than assumed, and NOT the same path as a protocol violation.

    A response whose fields are not the declared schema is refused inside `record_call`, so
    unlike a protocol violation it does not come back as `RecordedTurnRefused` carrying the
    call. That asymmetry is a property of the machinery as it stands and is pinned here so a
    change to it is a deliberate change rather than a surprise.
    """
    from src.core.errors import InvalidParameterError

    def wrong_shape(request):
        structured = {"not": "the schema"}
        return {"raw": json.dumps(structured), "structured": structured,
                "model_id": request["model_id"], "api_request_id": "x",
                "responded_at": "2026-09-11T12:30:00+00:00",
                "usage": {"provider": "test", "service_mode": "test", "input_tokens": 1,
                          "output_tokens": 1, "cached_input_tokens": 0, "total_tokens": 2,
                          "batch_name": "batches/test"}}

    exchange = open_round_robin(load_evidence_bundle(BUNDLE), panel_from("a-model", "high"))
    with pytest.raises(InvalidParameterError) as caught:
        advance(exchange, transport=wrong_shape, requested_at="2026-09-11T12:00:00+00:00")
    assert "exactly the fields in the versioned TG7.1 schema" in str(caught.value)


# ---------------- the runner's own refusals


def test_the_runner_sends_nothing_without_explicit_authorisation(capsys, monkeypatch):
    """Nothing reaches the network without the maintainer saying so in the command."""
    from tools.review_join_rerun import main

    monkeypatch.setattr("sys.argv", ["review_join_rerun", "--bundle", str(BUNDLE)])
    assert main() == 2
    printed = capsys.readouterr().out
    assert "--send-to-the-network" in printed
    assert "costs money" in printed


def test_authorisation_without_a_key_refuses_and_says_nothing_was_sent(capsys, monkeypatch):
    from tools.review_join_rerun import KEY_VARIABLES, main

    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.argv", ["review_join_rerun", "--bundle", str(BUNDLE),
                                     "--send-to-the-network"])
    assert main() == 2
    printed = capsys.readouterr().out
    assert "Nothing was sent" in printed
    assert "never from an argument" in printed


def test_a_key_is_read_from_the_environment_and_not_from_an_argument(monkeypatch):
    from tools.review_join_rerun import KEY_VARIABLES

    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert resolve_key() is None
    monkeypatch.setenv(KEY_VARIABLES[0], "  a-key  ")
    assert resolve_key() == "a-key"


def test_a_dry_run_names_every_turn_and_states_that_nothing_was_sent(capsys, monkeypatch):
    from tools.review_join_rerun import main

    monkeypatch.setattr("sys.argv", ["review_join_rerun", "--bundle", str(BUNDLE), "--dry-run"])
    assert main() == 0
    printed = capsys.readouterr().out
    for role in REVIEW_ROLES:
        assert role in printed
    assert "NOTHING WAS SENT" in printed


def test_a_missing_bundle_is_refused_by_name(capsys, monkeypatch):
    from tools.review_join_rerun import main

    monkeypatch.setattr("sys.argv", ["review_join_rerun", "--bundle", "data/studies/none.json",
                                     "--dry-run"])
    assert main() == 2
    assert "no bundle at" in capsys.readouterr().out
