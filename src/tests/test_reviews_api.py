"""TG11.5: the recorded-not-reproducible review surface."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.findings import STUDY_ROOT_ENV
from src.api.main import app
from src.api.reviews import CLAIM_BOUNDARY, REVIEW_ROOT_ENV, REVIEW_SURFACE_SCHEMA
from src.core.evidence import save_evidence_bundle
from src.core.five_outputs import summarise_evidence
from src.core.recorded_call import R23_DECLARATION, save_review_record
from src.core.review_cost import ReviewCostPolicy, audit_review_cost, save_cost_receipt
from src.core.round_robin import RoundRobinOutcome
from src.tests.test_review_cost import _recorded_review


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    studies = tmp_path / "studies"
    reviews = tmp_path / "reviews"
    monkeypatch.setenv(STUDY_ROOT_ENV, str(studies))
    monkeypatch.setenv(REVIEW_ROOT_ENV, str(reviews))
    return studies, reviews


@pytest.fixture()
def client(stores):
    return TestClient(app)


def _publish(stores):
    studies, reviews = stores
    reviewed = _recorded_review()
    studies.mkdir(parents=True, exist_ok=True)
    reviews.mkdir(parents=True, exist_ok=True)
    save_evidence_bundle(studies / "bundle.json", reviewed.bundle)
    save_review_record(reviews / "record.json", reviewed.review)
    receipt = audit_review_cost(reviewed, ReviewCostPolicy())
    save_cost_receipt(reviews / "cost.json", receipt)
    claim = summarise_evidence(reviewed.bundle)
    outcome = RoundRobinOutcome(
        study_id=reviewed.bundle.study_id,
        bundle_sha256=reviewed.bundle.bundle_sha256,
        bundle_revision=reviewed.bundle.revision,
        panel=ReviewCostPolicy().panel(),
        record_sha256=reviewed.review.record_sha256,
        finding="The panel recorded an argument; the record permits no stronger claim.",
        bounded_by=("the evidence bundle the panel was shown",),
        retained_dissent=(),
        claim_sha256=claim.summary_sha256,
        rung=claim.rung,
    )
    (reviews / "outcome.json").write_text(
        json.dumps(outcome.to_mapping(), sort_keys=True), encoding="utf-8")
    return reviewed, outcome, receipt


def test_an_unreviewed_published_study_reports_absence_without_inventing_reassurance(client,
                                                                                     stores):
    reviewed = _recorded_review()
    stores[0].mkdir(parents=True)
    save_evidence_bundle(stores[0] / "bundle.json", reviewed.bundle)

    response = client.get("/api/v1/reviews/studies/%s" % reviewed.bundle.study_id)

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == REVIEW_SURFACE_SCHEMA
    assert body["reviews"] == []
    assert "not the same as" in body["absence_note"]
    assert body["declaration"] == R23_DECLARATION
    assert body["claim_boundary"] == CLAIM_BOUNDARY


def test_the_whole_verified_review_record_outcome_and_cost_receipt_are_served(client, stores):
    reviewed, outcome, receipt = _publish(stores)

    body = client.get("/api/v1/reviews/studies/%s" % reviewed.bundle.study_id).json()

    assert body["bundle_sha256"] == reviewed.bundle.bundle_sha256
    assert body["bundle_revision"] == reviewed.bundle.revision
    assert body["absence_note"] is None
    assert len(body["reviews"]) == 1
    row = body["reviews"][0]
    assert row["record"] == reviewed.review.to_mapping()
    assert row["rendered"] == reviewed.review.render()
    assert row["outcomes"][0]["outcome"] == outcome.to_mapping()
    assert row["outcomes"][0]["rendered"] == outcome.render()
    assert row["cost_receipts"][0]["receipt"] == receipt.to_mapping()
    assert "price" not in json.dumps(row["cost_receipts"]).lower()


def test_commentary_is_bound_to_the_exact_latest_bundle_revision(client, stores):
    reviewed, _outcome, _receipt = _publish(stores)
    newer = reviewed.bundle.append(
        "observations", label="new observation", status="PASS", summary="recorded later",
        recorded_at="2026-08-27T00:00:00+00:00", payload={"n": 1},
        source_sha256s=("a" * 64,))
    save_evidence_bundle(stores[0] / "newer.json", newer)

    body = client.get("/api/v1/reviews/studies/%s" % newer.study_id).json()

    assert body["bundle_sha256"] == newer.bundle_sha256
    assert body["reviews"] == []
    assert body["absence_note"] is not None


def test_unbound_outcomes_and_receipts_cannot_attach_to_another_record(client, stores):
    reviewed, _outcome, _receipt = _publish(stores)
    other = _recorded_review()
    # Same study fixture but a distinct empty review digest; neither artifact binds to it.
    from src.core.recorded_call import attach_review
    empty = attach_review(other.bundle).review
    save_review_record(stores[1] / "empty.json", empty)

    body = client.get("/api/v1/reviews/studies/%s" % reviewed.bundle.study_id).json()
    empty_row = next(row for row in body["reviews"] if row["file"] == "empty.json")
    assert empty_row["outcomes"] == []
    assert empty_row["cost_receipts"] == []


def test_tampered_and_unknown_artifacts_are_reported_not_silently_skipped(client, stores):
    reviewed, _outcome, _receipt = _publish(stores)
    (stores[1] / "broken.json").write_text("{not json", encoding="utf-8")
    (stores[1] / "stray.json").write_text(
        json.dumps({"schema": "something-else/v1"}), encoding="utf-8")

    body = client.get("/api/v1/reviews/studies/%s" % reviewed.bundle.study_id).json()

    assert {row["file"] for row in body["unreadable"]} == {"broken.json", "stray.json"}
    assert all(row["refused_because"] for row in body["unreadable"])


def test_an_unknown_study_is_404_even_when_review_artifacts_exist(client, stores):
    _publish(stores)
    response = client.get("/api/v1/reviews/studies/not-published")
    assert response.status_code == 404
    assert "no published study" in response.json()["detail"]


def test_the_review_router_is_read_only():
    routes = [route for route in app.routes if getattr(route, "path", "").startswith(
        "/api/v1/reviews")]
    assert [(route.path, route.methods) for route in routes] == [
        ("/api/v1/reviews/studies/{study_id}", {"GET"})]


def test_reading_a_review_does_not_change_the_published_bundle(client, stores):
    reviewed, _outcome, _receipt = _publish(stores)
    path = stores[0] / "bundle.json"
    before = path.read_bytes()

    client.get("/api/v1/reviews/studies/%s" % reviewed.bundle.study_id)

    assert path.read_bytes() == before
