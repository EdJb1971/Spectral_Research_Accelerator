"""G19: private-by-default conversation with one complete study record."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.conversations import CONVERSATION_ROOT_ENV
from src.api.findings import STUDY_ROOT_ENV
from src.api.main import app
from src.api.reviews import REVIEW_ROOT_ENV
from src.core.evidence import save_evidence_bundle
from src.tests.test_review_cost import _recorded_review


@pytest.fixture()
def conversation_stores(tmp_path, monkeypatch):
    studies = tmp_path / "studies"
    reviews = tmp_path / "reviews"
    conversations = tmp_path / "conversations"
    runs = tmp_path / "runs"
    monkeypatch.setenv(STUDY_ROOT_ENV, str(studies))
    monkeypatch.setenv(REVIEW_ROOT_ENV, str(reviews))
    monkeypatch.setenv(CONVERSATION_ROOT_ENV, str(conversations))
    monkeypatch.setattr(app.state, "experiment_run_dir", runs, raising=False)
    reviewed = _recorded_review()
    studies.mkdir(parents=True)
    save_evidence_bundle(studies / "bundle.json", reviewed.bundle)
    return reviewed, conversations


@pytest.fixture()
def client(conversation_stores):
    return TestClient(app)


def _context(client, reviewed):
    response = client.get(
        "/api/v1/conversations/studies/%s/context" % reviewed.bundle.study_id,
        params={"glossary": "reanalysis"},
    )
    assert response.status_code == 200
    return response.json()


def test_context_declares_complete_grounding_and_recording_off(client, conversation_stores):
    reviewed, conversations = conversation_stores

    body = _context(client, reviewed)

    assert body["bundle_sha256"] == reviewed.bundle.bundle_sha256
    assert body["recording"] == "OFF"
    assert body["network_used"] is False
    assert body["run_count"] == 0
    assert body["formal_review_count"] == 0
    assert "complete evidence bundle" in body["what_is_loaded"]
    assert "formal round-table transcripts, outcomes and usage" in body["what_is_loaded"]
    assert not conversations.exists()


def test_ask_requires_explicit_paid_call_authorisation_before_any_model_use(
        client, conversation_stores, monkeypatch):
    reviewed, _conversations = conversation_stores
    context = _context(client, reviewed)
    called = []
    monkeypatch.setattr("src.api.conversations._ask_model", lambda **kwargs: called.append(kwargs))

    response = client.post(
        "/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "What matters?", "history": [], "glossary": "reanalysis",
              "expected_bundle_sha256": context["bundle_sha256"],
              "i_authorise_paid_call": False},
    )

    assert response.status_code == 400
    assert "Nothing was sent" in response.json()["detail"]
    assert called == []


def test_answer_is_not_recorded_and_receives_the_complete_current_record(
        client, conversation_stores, monkeypatch):
    reviewed, conversations = conversation_stores
    context = _context(client, reviewed)
    seen = {}

    def fake_model(**kwargs):
        seen.update(kwargs)
        return {"answer": "The limitation is recorded.", "records_used": ["evidence_bundle"],
                "cautions": ["Interpretation only."], "suggested_questions": ["What next?"],
                "usage": {"totalTokenCount": 12}}

    monkeypatch.setattr("src.api.conversations._key", lambda: "test-key")
    monkeypatch.setattr("src.api.conversations._ask_model", fake_model)
    response = client.post(
        "/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "What matters?", "history": [], "glossary": "reanalysis",
              "expected_bundle_sha256": context["bundle_sha256"],
              "i_authorise_paid_call": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is False
    assert seen["records"][0]["evidence_bundle"]["study_id"] == reviewed.bundle.study_id
    assert "derived_finding" in seen["records"][0]
    assert "experiment_runs" in seen["records"][0]
    assert "formal_round_table" in seen["records"][0]
    assert body["grounding"]["record_granularity"] == "complete"
    assert body["grounding"]["contains_dialogue"] is False
    assert body["independence"]["verified"] is True
    assert body["turns"][1]["claim_boundary"]
    assert not conversations.exists()


def test_stale_conversation_is_refused_before_model_use(client, conversation_stores, monkeypatch):
    reviewed, _conversations = conversation_stores
    monkeypatch.setattr("src.api.conversations._key", lambda: "test-key")
    called = []
    monkeypatch.setattr("src.api.conversations._ask_model", lambda **kwargs: called.append(kwargs))

    response = client.post(
        "/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "What matters?", "history": [], "glossary": "reanalysis",
              "expected_bundle_sha256": "0" * 64, "i_authorise_paid_call": True},
    )

    assert response.status_code == 409
    assert called == []


def test_only_the_explicit_save_route_writes_a_bound_interpretation(
        client, conversation_stores, monkeypatch):
    reviewed, conversations = conversation_stores
    context = _context(client, reviewed)
    monkeypatch.setattr("src.api.conversations._key", lambda: "test-key")
    monkeypatch.setattr("src.api.conversations._ask_model", lambda **kwargs: {
        "answer": "The recorded limitation.", "records_used": ["evidence_bundle"],
        "cautions": [], "suggested_questions": [], "usage": {"totalTokenCount": 12}})
    asked = client.post(
        "/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "What matters?", "glossary": "reanalysis",
              "expected_bundle_sha256": context["bundle_sha256"],
              "i_authorise_paid_call": True},
    ).json()

    response = client.post(
        "/api/v1/conversations/studies/%s/save" % reviewed.bundle.study_id,
        json={"conversation_id": asked["conversation_id"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is True
    files = list(conversations.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert [turn["text"] for turn in record["turns"]] == [
        "What matters?", "The recorded limitation."]
    assert record["bindings"][reviewed.bundle.study_id]["bundle_sha256"] == reviewed.bundle.bundle_sha256
    assert record["recording_was_explicitly_requested"] is True
    assert record["independence"]["verified"] is True
    assert "never evidence" in record["claim_boundary"]


def test_corpus_route_selects_complete_related_studies(client, conversation_stores):
    reviewed, _ = conversation_stores
    response = client.post("/api/v1/conversations/corpus/select", json={
        "primary_study_id": reviewed.bundle.study_id,
        "question": reviewed.bundle.hypothesis.statement,
        "glossary": "reanalysis", "max_records": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["record_granularity"] == "complete"
    assert body["no_fragments"] is True
    assert body["grounding"]["derivable_from_named_records"] is True


def test_unknown_provider_is_refused_before_model_use(client, conversation_stores, monkeypatch):
    reviewed, _ = conversation_stores
    context = _context(client, reviewed)
    monkeypatch.setattr("src.api.conversations._key", lambda: "test-key")
    called = []
    monkeypatch.setattr("src.api.conversations._ask_model", lambda **kwargs: called.append(kwargs))
    response = client.post(
        "/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "What matters?", "provider_id": "opaque-provider",
              "expected_bundle_sha256": context["bundle_sha256"],
              "i_authorise_paid_call": True})
    assert response.status_code == 400
    assert "refused before any call" in response.json()["detail"]
    assert called == []


def test_server_enforces_conversation_budget_not_browser_history(
        client, conversation_stores, monkeypatch):
    reviewed, _ = conversation_stores
    context = _context(client, reviewed)
    calls = []
    monkeypatch.setattr("src.api.conversations._key", lambda: "test-key")
    monkeypatch.setattr("src.api.conversations._ask_model", lambda **kwargs: calls.append(kwargs) or {
        "answer": "Bound answer", "records_used": [], "cautions": [],
        "suggested_questions": [], "usage": {"totalTokenCount": 20}})
    first = client.post("/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "First?", "expected_bundle_sha256": context["bundle_sha256"],
              "max_calls": 1, "i_authorise_paid_call": True})
    assert first.status_code == 200
    second = client.post("/api/v1/conversations/studies/%s/ask" % reviewed.bundle.study_id,
        json={"question": "Second?", "expected_bundle_sha256": context["bundle_sha256"],
              "conversation_id": first.json()["conversation_id"],
              "i_authorise_paid_call": True})
    assert second.status_code == 409
    assert second.json()["detail"]["nothing_was_sent"] is True
    assert len(calls) == 1
