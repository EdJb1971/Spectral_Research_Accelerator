"""TG7.3: provider-neutral cost controls and the first Gemini Batch transport."""

import json
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from src.core.errors import DataSourceError, InvalidParameterError
from src.core.evidence import create_evidence_bundle
from src.core.recorded_call import FieldSpec, ResponseSchema, attach_review, record_call
from src.core.review_cost import (
    COST_POLICY_SCHEMA,
    COST_RECEIPT_SCHEMA,
    DEFAULT_ROLE_EFFORTS,
    GEMINI_35_FLASH,
    GEMINI_BATCH_MODE,
    GEMINI_PROVIDER,
    GeminiBatchTransport,
    ReviewCostPolicy,
    ReviewCostReceipt,
    audit_review_cost,
    build_gemini_batch_request,
    load_cost_receipt,
    save_cost_receipt,
)


SOURCE = "a" * 64
MOMENT = "2026-08-26T01:00:00+00:00"
ANSWER_SCHEMA = ResponseSchema(
    name="cost_acceptance",
    fields={"finding": FieldSpec("string"), "dissent": FieldSpec("boolean")},
)
ANSWER = {"finding": "The claim remains bounded by its deterministic gates.",
          "dissent": False}


def _bundle():
    return create_evidence_bundle(
        study_id="tg7_3_cost", hypothesis_id="H1",
        statement="A review can be cost-controlled without changing a claim.",
        prediction="The route and measured usage are independently auditable.",
        registered_at="2026-08-25T00:00:00+00:00", created_at=MOMENT,
        provenance={"source_sha256": SOURCE})


def _request(effort="low", model_id=GEMINI_35_FLASH):
    return {
        "role": "statistical_challenge", "model_id": model_id, "effort": effort,
        "system": "Apply the statistical challenge rubric and do not change any claim gate.",
        "instruction": "Challenge the evidence shown.",
        "context": {"bundle_sha256": SOURCE, "claim_state": {"rung": "observation"}},
        "response_schema": ANSWER_SCHEMA.to_mapping(), "bundle_sha256": SOURCE,
        "bundle_revision": 0, "request_sha256": "b" * 64,
    }


def _usage(index=1, *, cached=4500, mode=GEMINI_BATCH_MODE):
    raw = {"promptTokenCount": 5000, "candidatesTokenCount": 100,
           "cachedContentTokenCount": cached, "totalTokenCount": 5100}
    return {
        "provider": GEMINI_PROVIDER, "service_mode": mode,
        "input_tokens": 5000, "output_tokens": 100,
        "cached_input_tokens": cached, "total_tokens": 5100,
        "batch_name": "batches/review-%d" % index, "provider_usage": raw,
    }


def _recorded_review(*, usage_factory=_usage, policy=None):
    policy = policy or ReviewCostPolicy()
    reviewed = attach_review(_bundle())
    for index, role in enumerate(DEFAULT_ROLE_EFFORTS, 1):
        usage = usage_factory(index)

        def transport(request, usage=usage):
            return {"raw": json.dumps(ANSWER, separators=(",", ":")),
                    "structured": ANSWER, "model_id": request["model_id"],
                    "api_request_id": usage["batch_name"], "responded_at": MOMENT,
                    "usage": usage}

        reviewed = record_call(
            reviewed, role=role, model_id=policy.model_id,
            effort=policy.role_efforts[role], system="A stable review rubric.",
            instruction="Speak only in the assigned seat.", context={"turn": index},
            response_schema=ANSWER_SCHEMA, transport=transport, requested_at=MOMENT)
    return reviewed


class _Response:
    def __init__(self, value, status_code=200):
        self.value = value
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP failure whose body must not be repeated")

    def json(self):
        return self.value


class _Client:
    def __init__(self, final, *, status_code=200):
        self.final = final
        self.status_code = status_code
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if self.status_code >= 400:
            return _Response({"error": {"message": "secret provider detail"}}, self.status_code)
        return _Response({"name": "batches/abc", "metadata": {
            "state": "BATCH_STATE_PENDING"}})

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return _Response(self.final)


def _final_batch(*, cached=4500, state="BATCH_STATE_SUCCEEDED"):
    raw = json.dumps(ANSWER, separators=(",", ":"))
    return {"name": "batches/abc", "done": True, "response": {
        "name": "batches/abc", "state": state, "endTime": MOMENT,
        "output": {"inlinedResponses": {"inlinedResponses": [{"response": {
            "candidates": [{"content": {"parts": [{"text": raw}]}}],
            "usageMetadata": {"promptTokenCount": 5000, "candidatesTokenCount": 100,
                              "cachedContentTokenCount": cached,
                              "totalTokenCount": 5100},
        }}]}}}}


def _live_operation_shape():
    """Shape returned by the first real Gemini batch on 2026-08-26 (text removed)."""
    raw = json.dumps(ANSWER, separators=(",", ":"))
    response = {
        "candidates": [{"content": {"parts": [{"text": raw}]}}],
        "usageMetadata": {"promptTokenCount": 52, "candidatesTokenCount": 28,
                          "thoughtsTokenCount": 305, "totalTokenCount": 385},
    }
    return {"name": "batches/abc", "done": True,
            "metadata": {"name": "batches/abc", "state": "BATCH_STATE_SUCCEEDED",
                         "endTime": MOMENT, "batchStats": {"requestCount": "1",
                                                            "successfulRequestCount": "1"}},
            "response": {"inlinedResponses": {"inlinedResponses": [
                {"metadata": {"request_sha256": "b" * 64}, "response": response}]}}}


def test_default_policy_pins_one_real_model_and_cheaper_challenger_effort():
    policy = ReviewCostPolicy()
    panel = policy.panel()
    assert policy.schema == COST_POLICY_SCHEMA
    assert all(seat.model_id == GEMINI_35_FLASH for seat in panel.seats.values())
    assert panel.seat("statistical_challenge").effort == "low"
    assert panel.seat("response_and_revision").effort == "medium"
    assert panel.seat("final_synthesis").effort == "high"
    assert policy.required_service_mode == "batch"
    assert policy.batch_discount_fraction == 0.5
    assert ReviewCostPolicy.from_mapping(policy.to_mapping()) == policy


def test_policy_digest_catches_route_and_documentation_tampering():
    value = ReviewCostPolicy().to_mapping()
    value["role_efforts"]["final_synthesis"] = "low"
    with pytest.raises(InvalidParameterError, match="policy_sha256"):
        ReviewCostPolicy.from_mapping(value)


def test_batch_builder_preserves_schema_effort_cache_and_stable_request_identity():
    payload = build_gemini_batch_request(_request(), cached_content="cachedContents/shared")
    row = payload["batch"]["inputConfig"]["requests"]["requests"][0]
    generated = row["request"]
    assert payload["batch"]["displayName"] == "spectralearth-" + "b" * 20
    assert row["metadata"] == {"request_sha256": "b" * 64}
    assert generated["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    assert generated["generationConfig"]["responseFormat"]["text"]["mimeType"] \
        == "APPLICATION_JSON"
    assert generated["generationConfig"]["responseFormat"]["text"]["schema"] \
        == ANSWER_SCHEMA.to_mapping()["schema"]
    assert generated["cachedContent"] == "cachedContents/shared"
    assert json.loads(generated["contents"][0]["parts"][0]["text"])["instruction"] \
        == "Challenge the evidence shown."


@pytest.mark.parametrize("change,pattern", [
    ({"model_id": "gemini-made-up"}, "model_id"),
    ({"effort": "minimal"}, "effort"),
])
def test_batch_builder_refuses_drift_from_the_accepted_provider_contract(change, pattern):
    request = dict(_request(), **change)
    with pytest.raises(InvalidParameterError, match=pattern):
        build_gemini_batch_request(request)
    with pytest.raises(InvalidParameterError, match="cached_content"):
        build_gemini_batch_request(_request(), cached_content="shared")


def test_gemini_transport_uses_batch_polling_and_returns_tg7_1_contract():
    client = _Client(_final_batch())
    transport = GeminiBatchTransport(
        "key-that-must-never-be-recorded", cached_content="cachedContents/shared",
        client=client, poll_interval_seconds=0, sleeper=lambda _: None)
    answer = transport(_request())

    assert client.posts[0][0].endswith("/models/gemini-3.5-flash:batchGenerateContent")
    assert client.gets[0][0].endswith("/batches/abc")
    assert client.posts[0][1]["headers"]["x-goog-api-key"] \
        == "key-that-must-never-be-recorded"
    assert "key-that-must-never-be-recorded" not in json.dumps(client.posts[0][1]["json"])
    assert "key-that-must-never-be-recorded" not in repr(transport)
    assert "key-that-must-never-be-recorded" not in json.dumps(answer)
    assert answer["structured"] == ANSWER
    assert answer["api_request_id"] == "batches/abc"
    assert answer["usage"]["service_mode"] == "batch"
    assert answer["usage"]["cached_input_tokens"] == 4500
    assert answer["usage"]["provider_usage"]["cachedContentTokenCount"] == 4500


def test_gemini_transport_accepts_the_real_operation_shape_and_bills_thinking_as_output():
    client = _Client(_live_operation_shape())
    transport = GeminiBatchTransport("secret", client=client, poll_interval_seconds=0,
                                     sleeper=lambda _: None)
    answer = transport(_request())
    assert answer["structured"] == ANSWER
    assert answer["responded_at"] == MOMENT
    assert answer["usage"]["input_tokens"] == 52
    assert answer["usage"]["output_tokens"] == 333  # 28 visible + 305 thinking
    assert answer["usage"]["total_tokens"] == 385


def test_gemini_transport_reports_safe_http_and_terminal_failures():
    failed_http = GeminiBatchTransport("secret", client=_Client({}, status_code=403),
                                       poll_interval_seconds=0)
    with pytest.raises(DataSourceError, match="HTTP 403") as caught:
        failed_http(_request())
    assert "secret" not in str(caught.value)

    failed_batch = GeminiBatchTransport(
        "secret", client=_Client(_final_batch(state="BATCH_STATE_FAILED")),
        poll_interval_seconds=0, sleeper=lambda _: None)
    with pytest.raises(DataSourceError, match="BATCH_STATE_FAILED"):
        failed_batch(_request())


def test_cost_audit_binds_every_route_measured_cache_and_raw_usage():
    policy = ReviewCostPolicy()
    reviewed = _recorded_review(policy=policy)
    receipt = audit_review_cost(reviewed, policy)
    assert receipt.schema == COST_RECEIPT_SCHEMA
    assert receipt.review_record_sha256 == reviewed.review.record_sha256
    assert receipt.policy_sha256 == policy.policy_sha256
    assert receipt.call_count == 8
    assert receipt.input_tokens == 40000
    assert receipt.output_tokens == 800
    assert receipt.cached_input_tokens == 36000
    assert receipt.total_tokens == 40800
    assert receipt.cache_hit_fraction == 0.9
    assert len(receipt.batch_names) == len(set(receipt.batch_names)) == 8


def test_enabling_cache_without_a_measured_hit_is_a_refusal():
    reviewed = _recorded_review(usage_factory=lambda index: _usage(index, cached=0))
    with pytest.raises(InvalidParameterError, match="Enabling caching is not evidence"):
        audit_review_cost(reviewed, ReviewCostPolicy())


def test_standard_service_and_effort_drift_are_refused_even_with_good_token_counts():
    standard = _recorded_review(
        usage_factory=lambda index: _usage(index, mode="standard"))
    with pytest.raises(InvalidParameterError, match="service_mode"):
        audit_review_cost(standard, ReviewCostPolicy())

    policy = ReviewCostPolicy()
    wrong = ReviewCostPolicy(role_efforts=dict(policy.role_efforts,
                                                statistical_challenge="medium"))
    reviewed = _recorded_review(policy=wrong)
    with pytest.raises(InvalidParameterError, match="call.effort"):
        audit_review_cost(reviewed, policy)


@pytest.mark.parametrize("mutate,pattern", [
    (lambda value: value.update(cached_input_tokens=5001), "cached_input_tokens"),
    (lambda value: value.update(total_tokens=5000), "total_tokens"),
    (lambda value: value.pop("provider_usage"), "versioned cost-control fields"),
    (lambda value: value["provider_usage"].update(cachedContentTokenCount=1),
     "independently parsed"),
])
def test_usage_arithmetic_and_identity_are_rechecked_not_trusted(mutate, pattern):
    def usage_factory(index):
        value = _usage(index)
        mutate(value)
        return value

    reviewed = _recorded_review(usage_factory=usage_factory)
    with pytest.raises(InvalidParameterError, match=pattern):
        audit_review_cost(reviewed, ReviewCostPolicy())


def test_batch_identity_must_match_the_recorded_api_request_id():
    policy = ReviewCostPolicy(minimum_total_cached_tokens=0)
    reviewed = attach_review(_bundle())
    usage = _usage()

    def transport(request):
        return {"raw": json.dumps(ANSWER), "structured": ANSWER,
                "model_id": request["model_id"], "api_request_id": "batches/other",
                "responded_at": MOMENT, "usage": usage}

    reviewed = record_call(
        reviewed, role="statistical_challenge", model_id=policy.model_id,
        effort=policy.role_efforts["statistical_challenge"], system="Stable rubric.",
        instruction="Challenge.", context={}, response_schema=ANSWER_SCHEMA,
        transport=transport, requested_at=MOMENT)
    with pytest.raises(InvalidParameterError, match="API request id"):
        audit_review_cost(reviewed, policy)


def test_cost_receipt_round_trip_no_overwrite_and_tamper_detection(tmp_path):
    receipt = audit_review_cost(_recorded_review(), ReviewCostPolicy())
    path = tmp_path / "cost.json"
    save_cost_receipt(path, receipt)
    assert load_cost_receipt(path) == receipt
    with pytest.raises(InvalidParameterError, match="overwrite=True"):
        save_cost_receipt(path, receipt)

    value = json.loads(path.read_text(encoding="utf-8"))
    value["cached_input_tokens"] -= 1
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(InvalidParameterError):
        load_cost_receipt(path)


def test_receipt_constructor_refuses_a_forged_fraction_or_digest():
    receipt = audit_review_cost(_recorded_review(), ReviewCostPolicy())
    value = deepcopy(receipt.to_mapping())
    value["cache_hit_fraction"] = 0.1
    with pytest.raises(InvalidParameterError, match="cache_hit_fraction"):
        ReviewCostReceipt.from_mapping(value)
    value = receipt.to_mapping()
    value["receipt_sha256"] = "0" * 64
    with pytest.raises(InvalidParameterError, match="receipt_sha256"):
        ReviewCostReceipt.from_mapping(value)

    value = receipt.to_mapping()
    value["batch_names"][1] = value["batch_names"][0]
    with pytest.raises(InvalidParameterError, match="unique provider batch"):
        ReviewCostReceipt.from_mapping(value)


def test_offline_acceptance_uses_no_gemini_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    receipt = audit_review_cost(_recorded_review(), ReviewCostPolicy())
    assert receipt.call_count == 8
    assert datetime.now(timezone.utc).tzinfo is not None
