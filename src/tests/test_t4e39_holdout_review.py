"""T4E.39: the whole holdout must be inspectable, and its review must not become acceptance.

These tests hold two lines at once. The first is that a researcher can reach every stage of the
holdout and the bytes behind it from the surface, including the stages that refused and the
operations the surface will not perform. The second is that the human act at the end of the
flow records a reading and cannot be dressed up as an acceptance, however it is written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.benchmarks.reference_holdout import DEFAULT_DECLARATION
from src.benchmarks.reference_holdout_fields import DEFAULT_CENSUS, DEFAULT_OUTPUT
from src.core.errors import UserInputError
from src.core.holdout_review import (
    HoldoutResultReview, build_holdout_review_request, load_adopted_holdout_review,
    verified_holdout_result)


REVIEW_FILE = "t4e39-holdout-result-review.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")


def _resigned(body: dict) -> dict:
    """Re-seal a tampered artifact so its receipt identity is valid and only the claim differs."""
    unsigned = {name: value for name, value in body.items() if name != "receipt_sha256"}
    return {**unsigned, "receipt_sha256": hashlib.sha256(_canonical(unsigned)).hexdigest()}


def _written_review(directory: Path) -> Path:
    request = build_holdout_review_request()
    bound = {name: request[name] for name in (
        "declaration_sha256", "census_receipt_sha256", "field_record_receipt_sha256",
        "measurement_receipt_sha256", "reviewed_outcome", "reviewed_verdict",
        "population_denominator", "strict_majority_needed")}
    review = HoldoutResultReview.parse_obj({
        "schema": request["review_schema"], "task": "T4E.39", **bound,
        "boundary_assessment": "BOUNDARY_SOUND",
        "basis": "The declared rule was fixed before opening and the margin is the minimum.",
        "what_this_does_not_establish": ["independence from ERA5 assimilation"],
        "limitations": "One period, one deterministic rule, twelve storms.",
        "next_action": "Close the identity criterion rather than repeat this comparison.",
        "claim_boundary": "This review records a reading and accepts nothing.",
    })
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REVIEW_FILE
    path.write_text(json.dumps(json.loads(review.json(by_alias=True)), indent=2) + "\n",
                    encoding="utf-8")
    return path


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_the_surface_publishes_every_performed_stage_and_refuses_to_re_run_the_holdout():
    response = TestClient(app).get("/api/v1/reference-holdout")
    assert response.status_code == 200
    body = response.json()

    assert [stage["stage"] for stage in body["stages"]] == [
        "declaration", "adoption", "census", "field-plan", "acquisition", "materialisation",
        "measurement", "review"]
    assert body["verdict"] == "NOT_AN_ACCEPTANCE"
    assert body["decision"]["is_acceptance_verdict"] is False
    assert body["network_used"] is False
    # The declared boundary travels with the numbers rather than being left in a document.
    assert all(stage["boundary"].strip() for stage in body["stages"])

    operations = {row["operation"]: row for row in body["operations"]}
    assert operations["write the human result review"]["available_in_ui"] is True
    assert operations["acquire ERA5 fields"]["available_in_ui"] is False
    assert operations["re-run the holdout"]["available_in_ui"] is False
    assert "spent" in operations["re-run the holdout"]["reason"]


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_the_smallest_admissible_majority_is_reported_as_fragile_rather_than_as_a_win():
    body = TestClient(app).get("/api/v1/reference-holdout").json()
    counts = body["decision"]["counts"]
    needed = body["decision"]["strict_majority_needed"]

    assert max(counts.values()) - needed == body["margin_over_strict_majority"]
    assert body["one_row_would_change_the_outcome"] is (max(counts.values()) == needed)
    # Ties and refusals are published beside the two candidate labels, not summarised away.
    assert set(counts) == {
        "VERTICAL_QUANTITY_SEPARATION_CANDIDATE", "CATALOGUE_REANALYSIS_ALIGNMENT_CANDIDATE",
        "EXACT_TIE", "REFUSED"}


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_every_stage_artifact_and_measured_row_can_be_examined_from_the_surface():
    client = TestClient(app)
    body = client.get("/api/v1/reference-holdout").json()

    for name in body["artifacts"]:
        artifact = client.get("/api/v1/reference-holdout/artifacts/%s" % name)
        assert artifact.status_code == 200
        described = artifact.json()
        path = Path(described["path"])
        assert described["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert described["body"]

    assert client.get("/api/v1/reference-holdout/artifacts/nothing").status_code == 404

    rows = client.get("/api/v1/reference-holdout/rows").json()
    assert len(rows["rows"]) == body["decision"]["population_denominator"]
    for row in rows["rows"]:
        assert row["mslp_basin"]["path"] and row["negated_vorticity_basin"]["path"]
        assert row["classification"] in body["decision"]["counts"]
    assert rows["method"]["search_radius_used"] is False


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_a_review_cannot_express_an_acceptance_and_a_request_is_not_a_review(tmp_path):
    request = build_holdout_review_request()
    assert request["status"] == "AWAITING_HUMAN_REVIEW"
    assert request["verified_facts"]["acceptance_available"] is False
    assert request["human_inputs_required"]["boundary_assessment"] == [
        "BOUNDARY_SOUND", "BOUNDARY_DISPUTED"]

    review = json.loads(_written_review(tmp_path).read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        HoldoutResultReview.parse_obj({**review, "reviewed_verdict": "ACCEPTED"})
    with pytest.raises(ValidationError):
        HoldoutResultReview.parse_obj({**review, "accepted": True})
    with pytest.raises(ValidationError):
        HoldoutResultReview.parse_obj({**review, "boundary_assessment": "ACCEPTED"})

    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(UserInputError, match="review is invalid"):
        load_adopted_holdout_review(request_path)


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_a_result_that_lost_its_boundary_is_refused_before_any_review_can_bind_it(tmp_path):
    measurement = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    tampered = tmp_path / "measurement.json"
    tampered.write_text(json.dumps(
        _resigned({**measurement, "VERDICT": "ACCEPTED"}), indent=2), encoding="utf-8")

    # The forged receipt is internally valid, so only the boundary guard can refuse it.
    with pytest.raises(UserInputError, match="NOT_AN_ACCEPTANCE"):
        verified_holdout_result(
            declaration_path=DEFAULT_DECLARATION, census_path=DEFAULT_CENSUS,
            measurement_path=tampered)

    detached = tmp_path / "detached.json"
    detached.write_text(json.dumps(
        _resigned({**measurement, "receipt_sha256": "0" * 64,
                   "declaration_sha256": "0" * 64}), indent=2), encoding="utf-8")
    with pytest.raises(UserInputError, match="does not bind"):
        verified_holdout_result(
            declaration_path=DEFAULT_DECLARATION, census_path=DEFAULT_CENSUS,
            measurement_path=detached)


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_the_surface_writes_a_review_without_adopting_it(tmp_path):
    app.state.holdout_review_dir = tmp_path
    try:
        response = TestClient(app).post("/api/v1/reference-holdout/review", json={
            "boundary_assessment": "BOUNDARY_DISPUTED",
            "basis": "A twelve-row majority of one is too fragile to describe as support.",
            "what_this_does_not_establish": ["cyclone identity", "vertical tilt"],
            "limitations": "The reviewer has not re-derived the basin rule.",
            "next_action": "State what a passing identity criterion would have to establish.",
            "claim_boundary": "This review accepts nothing and licenses no further acquisition.",
        })
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "WRITTEN_NOT_ADOPTED"
        assert body["verdict"] == "NOT_AN_ACCEPTANCE"
        assert not (tmp_path / "t4e39-holdout-result-review-adoption.json").exists()

        with pytest.raises(UserInputError, match="requires a named human adoption"):
            load_adopted_holdout_review(tmp_path / REVIEW_FILE)

        duplicate = TestClient(app).post("/api/v1/reference-holdout/review", json={
            "boundary_assessment": "BOUNDARY_SOUND", "basis": "second attempt",
            "what_this_does_not_establish": ["anything"], "limitations": "none",
            "next_action": "none", "claim_boundary": "none",
        })
        assert duplicate.status_code == 409
    finally:
        del app.state.holdout_review_dir


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_adoption_requires_the_exact_affirmation_and_binds_the_live_result(tmp_path):
    _written_review(tmp_path)
    app.state.holdout_review_dir = tmp_path
    client = TestClient(app)
    try:
        refused = client.post("/api/v1/reference-holdout/review/adopt", json={
            "adopted_by": "Ed Bentley", "adopted_as": "REVIEWED",
            "what_was_adopted": "the T4E.39 result review", "affirmation": "looks fine",
        })
        assert refused.status_code == 400
        assert not (tmp_path / "t4e39-holdout-result-review-adoption.json").exists()

        adopted = client.post("/api/v1/reference-holdout/review/adopt", json={
            "adopted_by": "Ed Bentley", "adopted_as": "REVIEWED",
            "what_was_adopted": "the T4E.39 result review",
            "affirmation": "I have read this declaration and I adopt it",
        })
        assert adopted.status_code == 200
        body = adopted.json()
        assert body["status"] == "ADOPTED"
        assert body["verdict"] == "NOT_AN_ACCEPTANCE"
        assert body["binding"]["verdict_after_review"] == "NOT_AN_ACCEPTANCE"
        assert body["binding"]["adopted_by"] == "Ed Bentley"

        surface = client.get("/api/v1/reference-holdout").json()
        assert surface["review"]["status"] == "ADOPTED"
        assert surface["review"]["binds_current_result"] is True
        assert surface["verdict"] == "NOT_AN_ACCEPTANCE"
        assert client.get("/api/v1/reference-holdout/artifacts/review").status_code == 200
    finally:
        del app.state.holdout_review_dir


@pytest.mark.skipif(not DEFAULT_OUTPUT.exists(), reason="the T4E.39 measurement is not present")
def test_an_adopted_review_that_no_longer_binds_the_result_is_refused_and_shown_as_stale(tmp_path):
    path = _written_review(tmp_path)
    review = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(
        {**review, "measurement_receipt_sha256": "1" * 64}, indent=2), encoding="utf-8")

    with pytest.raises(UserInputError, match="does not bind the exact current result"):
        load_adopted_holdout_review(path)

    app.state.holdout_review_dir = tmp_path
    try:
        body = TestClient(app).get("/api/v1/reference-holdout").json()
        assert body["review"]["binds_current_result"] is False
    finally:
        del app.state.holdout_review_dir
