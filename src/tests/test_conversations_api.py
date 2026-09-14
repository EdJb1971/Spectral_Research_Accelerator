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
    assert seen["grounding"]["evidence_bundle"]["study_id"] == reviewed.bundle.study_id
    assert "derived_finding" in seen["grounding"]
    assert "experiment_runs" in seen["grounding"]
    assert "formal_round_table" in seen["grounding"]
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


def test_only_the_explicit_save_route_writes_a_bound_interpretation(client, conversation_stores):
    reviewed, conversations = conversation_stores
    context = _context(client, reviewed)
    turns = [{"role": "user", "text": "What matters?"},
             {"role": "assistant", "text": "The recorded limitation."}]

    response = client.post(
        "/api/v1/conversations/studies/%s/save" % reviewed.bundle.study_id,
        json={"turns": turns, "glossary": "reanalysis",
              "expected_bundle_sha256": context["bundle_sha256"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recorded"] is True
    files = list(conversations.glob("*.json"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["turns"] == turns
    assert record["bundle_sha256"] == reviewed.bundle.bundle_sha256
    assert record["recording_was_explicitly_requested"] is True
    assert "never evidence" in record["claim_boundary"]
