"""Phase G7, TG7.1: the recorded-call boundary.

Every call captures the verbatim request, the verbatim response, the exact model id, the effort
setting, the API request id and the timestamps, because an LLM output cannot be regenerated
(R23). Structured outputs constrain every response to a declared schema, so commentary can never
arrive as free text that later code parses hopefully. And the acceptance test of the whole phase
is here: deleting every LLM output from a corpus of bundles changes no claim level (R22).
"""

import itertools
import json
import random

import pytest

from src.core.errors import InvalidParameterError
from src.core.evidence import create_evidence_bundle
from src.core.claim_ladder import CLAIM_RUNGS, PRECEDENCE_KEY, assess_claim_ladder
from src.core.five_outputs import summarise_evidence
from src.core.recorded_call import (
    DETERMINISM,
    EFFORTS,
    FORBIDDEN_SAMPLING_PARAMETERS,
    R23_DECLARATION,
    RECORDED_CALL_SCHEMA,
    REVIEW_RECORD_SCHEMA,
    REVIEW_ROLES,
    SMUGGLING_FLOOR,
    CallRequest,
    CallResponse,
    FieldSpec,
    RecordedCall,
    ResponseSchema,
    ReviewRecord,
    ReviewedBundle,
    attach_review,
    load_review_record,
    record_call,
    save_review_record,
    strip_review,
    verify_claim_independence,
)


REGISTERED_AT = "2026-01-01T00:00:00+00:00"
CREATED_AT = "2026-01-02T00:00:00+00:00"
SOURCE_A = "a" * 64
MODEL = "claude-opus-4-5-20260101"
SENT_AT = "2026-04-01T09:00:00+00:00"
BACK_AT = "2026-04-01T09:00:12+00:00"

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

ALL_REQUIREMENTS = tuple(itertools.chain.from_iterable(
    RUNG_REQUIREMENTS[rung] for rung in CLAIM_RUNGS[1:]))

CHALLENGE = ResponseSchema(name="challenge", fields={
    "verdict": FieldSpec(kind="string", enum=("supported", "unsupported", "unclear")),
    "argument": FieldSpec(kind="string"),
    "alternatives": FieldSpec(kind="string_list"),
    "residual_dissent": FieldSpec(kind="boolean"),
})

#: Deliberately assertive commentary. If prose could move a rung, this prose would.
ANSWER = {
    "verdict": "supported",
    "argument": "The measured lead is unambiguous and in my judgement establishes that the "
                "origin motif causes the target outcome; treat this as demonstrated.",
    "alternatives": ["a shared seasonal driver has not been excluded to my satisfaction"],
    "residual_dissent": True,
}


def _moment(index):
    return "2026-03-01T%02d:%02d:00+00:00" % (index // 60, index % 60)


def _bundle():
    return create_evidence_bundle(
        study_id="planted_precursor_v1", hypothesis_id="H1",
        statement="Frozen origin motif presence precedes the target outcome.",
        prediction="A positive association survives correction on target train and test.",
        registered_at=REGISTERED_AT, created_at=CREATED_AT,
        provenance={"plan_sha256": SOURCE_A, "roadmap": "TG7.1"})


def _with(bundle, evidence, *, status="PASS", start=0, label=None, summary="a measured result"):
    for offset, (category, payload) in enumerate(evidence):
        bundle = bundle.append(
            category, label=label or category, status=status, summary=summary,
            recorded_at=_moment(start + offset), payload=payload, source_sha256s=(SOURCE_A,))
    return bundle


def _climbed(rung):
    """A bundle carrying exactly the evidence needed to stand on `rung`."""
    needed = tuple(itertools.chain.from_iterable(
        RUNG_REQUIREMENTS[name] for name in CLAIM_RUNGS[1:CLAIM_RUNGS.index(rung) + 1]))
    return _with(_bundle(), needed)


def _transport(structured=None, *, raw=None, model_id=None, api_request_id="req_01",
               responded_at=BACK_AT, usage=None):
    """A recorded transport. The boundary never opens a socket; this is where one would go."""
    seen = []

    def send(request):
        seen.append(request)
        payload = ANSWER if structured is None else structured
        return {
            "raw": json.dumps(payload) if raw is None else raw,
            "structured": payload,
            "model_id": request["model_id"] if model_id is None else model_id,
            "api_request_id": "%s_%d" % (api_request_id, len(seen)),
            "responded_at": responded_at,
            "usage": {"input_tokens": 4096, "output_tokens": 180,
                      "cache_read_input_tokens": 3900} if usage is None else usage,
        }

    send.seen = seen
    return send


def _record(reviewed, *, role="statistical_challenge", effort="medium", transport=None,
            context=None, requested_at=SENT_AT, schema=CHALLENGE):
    return record_call(
        reviewed, role=role, model_id=MODEL, effort=effort,
        system="You are a statistical challenger. You may not change a claim level.",
        instruction="State whether the recorded evidence supports the hypothesis.",
        context={"bundle_sha256": reviewed.bundle.bundle_sha256} if context is None else context,
        response_schema=schema, transport=_transport() if transport is None else transport,
        requested_at=requested_at)


# --------------------------------------------------------------------------------------------
# Recording a call verbatim (R23)
# --------------------------------------------------------------------------------------------


def test_a_recorded_call_captures_everything_that_cannot_be_regenerated():
    reviewed = _record(attach_review(_climbed("association")))
    call = reviewed.review.calls[0]
    assert call.request.model_id == MODEL
    assert call.request.effort == "medium"
    assert call.request.role == "statistical_challenge"
    assert call.response.model_id == MODEL
    assert call.response.api_request_id == "req_01_1"
    assert call.requested_at == SENT_AT
    assert call.response.responded_at == BACK_AT
    assert call.response.raw == json.dumps(ANSWER)
    assert call.response.to_mapping()["structured"] == ANSWER
    assert call.response.usage["cache_read_input_tokens"] == 3900


def test_the_transport_receives_exactly_the_recorded_request():
    transport = _transport()
    reviewed = _record(attach_review(_climbed("association")), transport=transport)
    assert transport.seen[0] == reviewed.review.calls[0].request.to_mapping()


def test_a_recorded_call_declares_itself_a_recording_rather_than_a_computation():
    call = _record(attach_review(_bundle())).review.calls[0]
    assert call.determinism == DETERMINISM
    assert call.body()["determinism"] == "recorded-not-reproducible"
    assert call.schema == RECORDED_CALL_SCHEMA


def test_a_loaded_call_claiming_to_be_reproducible_is_refused():
    """The honest label is part of the record, so a rewritten one is a refusal, not a nuance."""
    call = _record(attach_review(_bundle())).review.calls[0]
    forged = dict(call.to_mapping(), determinism="deterministic")
    with pytest.raises(InvalidParameterError) as refusal:
        RecordedCall.from_mapping(forged)
    assert "R23" in str(refusal.value)


@pytest.mark.parametrize("parameter", FORBIDDEN_SAMPLING_PARAMETERS)
def test_pinning_a_sampling_parameter_is_refused(parameter):
    reviewed = attach_review(_bundle())
    with pytest.raises(InvalidParameterError) as refusal:
        _record(reviewed, context={"bundle_sha256": reviewed.bundle.bundle_sha256,
                                   parameter: 0})
    assert "R23" in str(refusal.value)


def test_a_sampling_parameter_buried_in_a_nested_object_is_refused_too():
    reviewed = attach_review(_bundle())
    with pytest.raises(InvalidParameterError):
        _record(reviewed, context={"options": {"decoding": {"temperature": 0.0}}})


def test_an_answer_from_a_different_model_is_refused():
    """A bundle reviewed by a different model is different evidence."""
    reviewed = attach_review(_bundle())
    with pytest.raises(InvalidParameterError) as refusal:
        _record(reviewed, transport=_transport(model_id="some-other-model"))
    assert "different evidence" in str(refusal.value)


def test_an_answer_dated_before_its_own_request_is_refused():
    reviewed = attach_review(_bundle())
    with pytest.raises(InvalidParameterError):
        _record(reviewed, transport=_transport(responded_at="2026-04-01T08:59:00+00:00"))


def test_a_response_without_an_api_request_id_is_refused():
    """Without it the call cannot be looked up at the provider, so the record is not a record."""
    def anonymous(request):
        return dict(_transport()(request), api_request_id="   ")

    with pytest.raises(InvalidParameterError):
        _record(attach_review(_bundle()), transport=anonymous)


def test_a_transport_returning_the_wrong_fields_is_refused():
    def careless(request):
        return {"raw": json.dumps(ANSWER), "structured": ANSWER}

    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()), transport=careless)
    assert "api_request_id" in str(refusal.value)


@pytest.mark.parametrize("role", ["referee", "", "STATISTICAL_CHALLENGE"])
def test_a_call_cannot_invent_a_role_for_itself(role):
    with pytest.raises(InvalidParameterError):
        _record(attach_review(_bundle()), role=role)


def test_an_undeclared_effort_setting_is_refused():
    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()), effort="maximum")
    assert all(effort in str(refusal.value) for effort in EFFORTS)


# --------------------------------------------------------------------------------------------
# Structured outputs: a declared shape, or nothing
# --------------------------------------------------------------------------------------------


def test_the_declared_schema_is_sent_as_the_output_format():
    transport = _transport()
    _record(attach_review(_bundle()), transport=transport)
    sent = transport.seen[0]["response_schema"]
    assert sent["type"] == "json_schema"
    assert sent["name"] == "challenge"
    assert sent["schema"]["additionalProperties"] is False
    assert sent["schema"]["required"] == ["alternatives", "argument", "residual_dissent",
                                          "verdict"]
    assert sent["schema"]["properties"]["verdict"]["enum"] == ["supported", "unsupported",
                                                               "unclear"]


def test_free_text_where_a_schema_was_declared_is_refused():
    """Not parsed hopefully, not stored as a string: refused."""
    prose = "Frankly, the effect looks causal to me."
    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()), transport=_transport(raw=prose))
    assert "free text" in str(refusal.value).lower()


def test_a_response_missing_a_declared_field_is_refused():
    short = {key: value for key, value in ANSWER.items() if key != "alternatives"}
    with pytest.raises(InvalidParameterError):
        _record(attach_review(_bundle()), transport=_transport(short))


def test_a_response_carrying_an_undeclared_field_is_refused():
    """An extra field is where a claim level would be smuggled in, so the schema is closed."""
    extra = dict(ANSWER, claim_level="demonstrated_predictive_utility")
    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()), transport=_transport(extra))
    assert "claim_level" in str(refusal.value)


def test_a_value_outside_its_declared_enumeration_is_refused():
    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()),
                transport=_transport(dict(ANSWER, verdict="proven")))
    assert "supported" in str(refusal.value)


def test_a_field_of_the_wrong_type_is_refused():
    with pytest.raises(InvalidParameterError):
        _record(attach_review(_bundle()),
                transport=_transport(dict(ANSWER, alternatives="one long string")))


def test_a_structured_parse_that_disagrees_with_the_verbatim_bytes_is_refused():
    """The record keeps the bytes so the parse can always be re-checked; here it does not agree."""
    with pytest.raises(InvalidParameterError) as refusal:
        _record(attach_review(_bundle()),
                transport=_transport(dict(ANSWER, verdict="unsupported"),
                                     raw=json.dumps(ANSWER)))
    assert "verbatim response bytes" in str(refusal.value)


def test_the_schema_check_survives_a_round_trip_rather_than_holding_only_at_record_time():
    call = _record(attach_review(_bundle())).review.calls[0]
    forged = call.to_mapping()
    forged["response"]["raw"] = "not json at all"
    with pytest.raises(InvalidParameterError):
        RecordedCall.from_mapping(forged)


def test_an_enumeration_on_a_non_string_field_is_refused():
    with pytest.raises(InvalidParameterError):
        FieldSpec(kind="number", enum=("high", "low"))


def test_a_schema_with_no_fields_is_refused():
    with pytest.raises(InvalidParameterError):
        ResponseSchema(name="empty", fields={})


def test_a_response_schema_round_trips_through_the_format_block():
    assert ResponseSchema.from_mapping(CHALLENGE.to_mapping()).digest() == CHALLENGE.digest()


# --------------------------------------------------------------------------------------------
# The review record: commentary beside the bundle, bound to it from outside
# --------------------------------------------------------------------------------------------


def test_calls_are_appended_in_order_and_chained_to_their_predecessor():
    reviewed = attach_review(_climbed("robust_association"))
    anchor = reviewed.review.head_sha256
    for role in REVIEW_ROLES[:3]:
        reviewed = _record(reviewed, role=role)
    assert [call.sequence for call in reviewed.review.calls] == [1, 2, 3]
    assert reviewed.review.calls[0].previous_sha256 == anchor
    assert reviewed.review.calls[1].previous_sha256 == reviewed.review.calls[0].call_sha256
    assert reviewed.review.calls[2].previous_sha256 == reviewed.review.calls[1].call_sha256
    assert reviewed.review.head_sha256 == reviewed.review.calls[-1].call_sha256
    assert reviewed.review.schema == REVIEW_RECORD_SCHEMA


def test_recording_leaves_the_previous_review_byte_for_byte_unchanged():
    first = _record(attach_review(_bundle()))
    digest = first.review.record_sha256
    _record(first, role="confounder_challenge")
    assert first.review.record_sha256 == digest
    assert first.review.revision == 1


def test_by_role_selects_the_calls_that_role_made():
    reviewed = attach_review(_bundle())
    reviewed = _record(reviewed, role="statistical_challenge")
    reviewed = _record(reviewed, role="confounder_challenge")
    reviewed = _record(reviewed, role="statistical_challenge")
    assert [call.sequence for call in reviewed.review.by_role("statistical_challenge")] == [1, 3]
    assert reviewed.review.by_role("final_synthesis") == ()


def test_commentary_on_one_revision_is_not_commentary_on_another():
    """Appending evidence produces a new bundle, and the old review does not follow it across."""
    reviewed = _record(attach_review(_climbed("association")))
    later = _with(reviewed.bundle, (("confounders", {"considered": 2}),), start=40)
    with pytest.raises(InvalidParameterError) as refusal:
        ReviewedBundle(bundle=later, review=reviewed.review)
    assert "not review of this one" in str(refusal.value)


def test_a_call_recorded_against_a_different_bundle_cannot_be_spliced_into_this_record():
    other = _record(attach_review(_climbed("association"))).review.calls[0]
    mine = attach_review(_bundle()).review
    with pytest.raises(InvalidParameterError):
        mine.append(other)


def test_a_call_bound_to_another_bundle_is_refused_even_when_the_chain_would_accept_it():
    """The chain check would not catch this: the digest links up, but the subject does not."""
    mine = _record(attach_review(_climbed("association")))
    elsewhere = _climbed("robust_association")
    assert elsewhere.bundle_sha256 != mine.bundle.bundle_sha256
    request = CallRequest(
        role="final_synthesis", model_id=MODEL, effort="high", system="s", instruction="i",
        context={"bundle_sha256": elsewhere.bundle_sha256}, response_schema=CHALLENGE,
        bundle_sha256=elsewhere.bundle_sha256, bundle_revision=elsewhere.revision)
    response = CallResponse(raw=json.dumps(ANSWER), structured=ANSWER, model_id=MODEL,
                            api_request_id="req_elsewhere", responded_at=BACK_AT)
    stray = RecordedCall(sequence=2, request=request, response=response, requested_at=SENT_AT,
                         previous_sha256=mine.review.head_sha256)
    assert stray.previous_sha256 == mine.review.head_sha256
    with pytest.raises(InvalidParameterError) as refusal:
        mine.review.append(stray)
    assert "not commentary on another" in str(refusal.value)


def test_a_review_record_round_trips_through_canonical_json():
    reviewed = attach_review(_climbed("candidate_precursor"))
    for role in REVIEW_ROLES[:4]:
        reviewed = _record(reviewed, role=role)
    restored = ReviewRecord.from_mapping(json.loads(json.dumps(reviewed.review.to_mapping())))
    assert restored.record_sha256 == reviewed.review.record_sha256
    assert restored.to_mapping() == reviewed.review.to_mapping()


def test_editing_a_recorded_response_is_caught_by_the_digest():
    reviewed = _record(attach_review(_bundle()))
    tampered = reviewed.review.to_mapping()
    tampered["calls"][0]["response"]["structured"]["verdict"] = "unsupported"
    with pytest.raises(InvalidParameterError):
        ReviewRecord.from_mapping(tampered)


def test_render_states_the_recorded_limitation_wherever_commentary_is_displayed():
    reviewed = _record(attach_review(_climbed("association")))
    text = reviewed.review.render()
    assert text.startswith(R23_DECLARATION)
    assert "R22, R23" in text
    assert "statistical_challenge" in text and MODEL in text and "req_01_1" in text
    assert "verdict: supported" in text


def test_render_says_plainly_when_nothing_was_recorded():
    text = attach_review(_bundle()).review.render()
    assert "not the same as none having been made" in text


def test_render_is_deterministic():
    reviewed = _record(attach_review(_bundle()))
    assert reviewed.review.render() == reviewed.review.render()


def test_a_review_record_is_published_in_its_own_file_never_over_the_bundle(tmp_path):
    reviewed = _record(attach_review(_climbed("association")))
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
    with pytest.raises(InvalidParameterError) as refusal:
        save_review_record(bundle_path, reviewed.review, bundle_path=bundle_path)
    assert "beside" in str(refusal.value)

    review_path = tmp_path / "review.json"
    digest = save_review_record(review_path, reviewed.review, bundle_path=bundle_path)
    assert load_review_record(review_path, published_sha256=digest).to_mapping() == \
        reviewed.review.to_mapping()


def test_publishing_a_review_record_never_overwrites_an_earlier_one(tmp_path):
    reviewed = _record(attach_review(_bundle()))
    path = tmp_path / "review.json"
    save_review_record(path, reviewed.review)
    with pytest.raises(FileExistsError):
        save_review_record(path, _record(reviewed, role="final_synthesis").review)


def test_loading_a_review_record_under_the_wrong_published_digest_is_refused(tmp_path):
    reviewed = _record(attach_review(_bundle()))
    path = tmp_path / "review.json"
    save_review_record(path, reviewed.review)
    with pytest.raises(InvalidParameterError):
        load_review_record(path, published_sha256="f" * 64)


# --------------------------------------------------------------------------------------------
# R22 made executable: deleting every LLM output changes no claim level
# --------------------------------------------------------------------------------------------


def _corpus():
    """One bundle standing on each rung, plus a blocked one and a contradicted one."""
    bundles = [_climbed(rung) for rung in CLAIM_RUNGS[1:]]
    bundles.append(_bundle())
    blocked = _with(_climbed("robust_association"),
                    (("replication_results", {"splits": 3}),), status="FAIL", start=30,
                    label="held-out replication", summary="the relationship did not replicate")
    bundles.append(blocked)
    contradicted = _with(_climbed("candidate_precursor"),
                         (("contradictory_evidence", {"n": 40}),), status="INCONCLUSIVE",
                         start=30, label="a competing series", summary="an opposing result")
    bundles.append(contradicted)
    return tuple(bundles)


def test_deleting_every_llm_output_from_a_corpus_of_bundles_changes_no_claim_level():
    """The acceptance test of the phase. If this fails, the review layer is built wrongly."""
    reached = set()
    for bundle in _corpus():
        before = summarise_evidence(bundle)
        reviewed = attach_review(bundle)
        for index, role in enumerate(REVIEW_ROLES):
            reviewed = _record(reviewed, role=role, requested_at=_moment(index),
                               transport=_transport(dict(
                                   ANSWER, verdict="supported",
                                   argument="Rung %s understates this result and it should be "
                                            "promoted to demonstrated predictive utility."
                                            % before.rung)))
        assert reviewed.review.revision == len(REVIEW_ROLES)

        after = summarise_evidence(strip_review(reviewed))
        assert after.summary_sha256 == before.summary_sha256
        assert after.rung == before.rung
        assert after.claimable == before.claimable
        assert reviewed.claim_sha256 == before.summary_sha256
        assert reviewed.without_review().claim_sha256 == before.summary_sha256
        assert verify_claim_independence(reviewed) == before.summary_sha256
        reached.add(before.rung)

    # A corpus that only ever stood on one rung would assert almost nothing.
    assert reached == set(CLAIM_RUNGS)


def test_the_claim_state_does_not_depend_on_what_the_review_said():
    """Two opposite reviews of one bundle, and one claim digest between them."""
    bundle = _climbed("robust_association")
    approving = _record(attach_review(bundle), transport=_transport(dict(
        ANSWER, verdict="supported", residual_dissent=False)))
    damning = _record(attach_review(bundle), transport=_transport(dict(
        ANSWER, verdict="unsupported",
        argument="Nothing here survives scrutiny and the finding should be withdrawn at once.")))
    assert approving.review.record_sha256 != damning.review.record_sha256
    assert approving.claim_sha256 == damning.claim_sha256
    assert approving.rung == damning.rung == "robust_association"


def test_recording_a_call_hands_back_the_same_bundle_object():
    """Not an equal bundle: the same one. There is no route from a response to the chain."""
    reviewed = attach_review(_climbed("association"))
    after = _record(reviewed)
    assert after.bundle is reviewed.bundle
    assert after.bundle.revision == reviewed.bundle.revision
    assert after.bundle.bundle_sha256 == reviewed.bundle.bundle_sha256


def test_recorded_commentary_found_inside_the_evidence_chain_is_refused():
    """The one way R22 could actually be broken: someone appends the response as evidence."""
    bundle = _climbed("association")
    smuggled = bundle.append(
        "confounders", label="reviewer opinion", status="PASS",
        summary=ANSWER["argument"], recorded_at=_moment(40),
        payload={"considered": 1}, source_sha256s=(SOURCE_A,))
    reviewed = _record(attach_review(smuggled))
    with pytest.raises(InvalidParameterError) as refusal:
        verify_claim_independence(reviewed)
    assert "R22" in str(refusal.value)


def test_commentary_shorter_than_the_smuggling_floor_is_not_treated_as_evidence_of_smuggling():
    """A short phrase can coincide with honest wording, and a false alarm here would be noise."""
    assert len("supported") < SMUGGLING_FLOOR
    bundle = _with(_bundle(), (("observations", {"n": 12}),), label="supported",
                   summary="an ordinary recorded observation")
    verify_claim_independence(_record(attach_review(bundle)))


def test_the_bundle_has_no_category_a_recorded_call_could_be_appended_to():
    """Commentary is not one of the ten scientific fields, and there is no eleventh."""
    from src.core.evidence import EVIDENCE_FIELDS
    assert not any(name in EVIDENCE_FIELDS
                   for name in ("commentary", "review", "llm_output", "opinion"))
    with pytest.raises(InvalidParameterError):
        _bundle().append("commentary", label="reviewer", status="PASS", summary="prose",
                         recorded_at=_moment(0), payload={"text": "prose"})


# --------------------------------------------------------------------------------------------
# A randomised sweep over reviewed corpora
# --------------------------------------------------------------------------------------------


def _swept_bundle(rng):
    bundle = _bundle()
    offset = 0
    for category, payload in ALL_REQUIREMENTS:
        if rng.random() < 0.75:
            bundle = bundle.append(category, label=category, status="PASS", summary="swept",
                                   recorded_at=_moment(offset), payload=payload,
                                   source_sha256s=(SOURCE_A,))
            offset += 1
    for _ in range(rng.randint(0, 3)):
        bundle = bundle.append(
            rng.choice(("contradictory_evidence", "failure_states", "null_results")),
            label="swept adverse entry",
            status=rng.choice(("FAIL", "INVALID", "INCONCLUSIVE", "PASS")),
            summary="swept", recorded_at=_moment(offset), payload={"n": rng.randint(1, 200)},
            source_sha256s=(SOURCE_A,))
        offset += 1
    return bundle


def _swept_answer(rng):
    return {
        "verdict": rng.choice(("supported", "unsupported", "unclear")),
        "argument": rng.choice((
            "This is in my view a demonstrated causal mechanism and should be reported as one.",
            "The whole result is an artefact of the representation and must be withdrawn.",
            "I would place this two rungs higher than the deterministic gates allow.")),
        "alternatives": rng.sample(["a shared seasonal driver remains open",
                                    "the partitions may not be independent",
                                    "the lag floor may be unjustified for this domain"],
                                   rng.randint(0, 3)),
        "residual_dissent": rng.random() < 0.5,
    }


def test_no_review_of_any_shape_moves_any_claim_level_over_a_randomised_sweep():
    rng = random.Random(20260826)
    reached = set()
    reviewed_at_all = 0
    for _ in range(300):
        bundle = _swept_bundle(rng)
        before = summarise_evidence(bundle)
        assessment = assess_claim_ladder(bundle)
        reviewed = attach_review(bundle)
        for index in range(rng.randint(0, 4)):
            reviewed = _record(reviewed, role=rng.choice(REVIEW_ROLES),
                               effort=rng.choice(EFFORTS), requested_at=_moment(index),
                               transport=_transport(_swept_answer(rng)))
        reviewed_at_all += 1 if reviewed.review.calls else 0

        assert verify_claim_independence(reviewed) == before.summary_sha256
        assert reviewed.rung == assessment.rung
        assert reviewed.assessment.blocking_entries == assessment.blocking_entries
        assert summarise_evidence(strip_review(reviewed)).to_mapping() == before.to_mapping()
        assert reviewed.review.revision == len(reviewed.review.calls)
        reached.add(before.rung)

    # A sweep in which nothing was ever reviewed, or that never left one rung, proves nothing.
    assert reached == set(CLAIM_RUNGS)
    assert reviewed_at_all > 0


def test_the_recorded_chain_verifies_over_a_randomised_sweep():
    """Whatever was said, the record of saying it stays self-checking and reloadable."""
    rng = random.Random(90210)
    lengths = set()
    for _ in range(200):
        reviewed = attach_review(_swept_bundle(rng))
        for index in range(rng.randint(1, 5)):
            reviewed = _record(reviewed, role=rng.choice(REVIEW_ROLES),
                               effort=rng.choice(EFFORTS), requested_at=_moment(index),
                               transport=_transport(_swept_answer(rng)))
        record = reviewed.review
        lengths.add(record.revision)
        restored = ReviewRecord.from_mapping(json.loads(json.dumps(record.to_mapping())))
        assert restored.record_sha256 == record.record_sha256
        assert restored.head_sha256 == record.calls[-1].call_sha256
        assert all(call.request.bundle_sha256 == reviewed.bundle.bundle_sha256
                   for call in record.calls)
        assert R23_DECLARATION in record.render()
    assert lengths == {1, 2, 3, 4, 5}


# --------------------------------------------------------------------------------------------
# Direct construction is held to the same standard as the recording path
# --------------------------------------------------------------------------------------------


def test_a_call_request_refuses_a_digest_that_is_not_the_request_it_describes():
    reviewed = _record(attach_review(_bundle()))
    body = reviewed.review.calls[0].request.to_mapping()
    with pytest.raises(InvalidParameterError):
        CallRequest.from_mapping(dict(body, request_sha256="0" * 64))


def test_a_call_response_refuses_a_digest_that_is_not_the_response_it_describes():
    reviewed = _record(attach_review(_bundle()))
    body = reviewed.review.calls[0].response.to_mapping()
    with pytest.raises(InvalidParameterError):
        CallResponse.from_mapping(dict(body, response_sha256="0" * 64))


def test_a_reviewed_bundle_refuses_a_review_of_a_different_study():
    bundle = _bundle()
    foreign = ReviewRecord(study_id="another_study", bundle_sha256=bundle.bundle_sha256,
                           bundle_revision=bundle.revision)
    with pytest.raises(InvalidParameterError):
        ReviewedBundle(bundle=bundle, review=foreign)


def test_the_boundary_refuses_a_transport_that_is_not_callable():
    with pytest.raises(InvalidParameterError):
        record_call(attach_review(_bundle()), role="final_synthesis", model_id=MODEL,
                    effort="high", system="s", instruction="i", context={"k": "v"},
                    response_schema=CHALLENGE, transport="not a callable",
                    requested_at=SENT_AT)


def test_strip_review_and_attach_review_refuse_the_wrong_type():
    with pytest.raises(InvalidParameterError):
        strip_review(_bundle())
    with pytest.raises(InvalidParameterError):
        attach_review(attach_review(_bundle()))
