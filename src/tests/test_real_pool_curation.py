import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.benchmarks.pool_calibration import CALIBRATION_CONTRACT
from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import UserInputError
from src.core.real_pool_curation import (
    build_curation_review_request, load_adopted_curation_review)
from src.core.real_pool_curation_packet import build_curation_packet
from src.core.real_pool_readiness import audit_real_pool_readiness
from src.core.scale_shape_successor import successor_scale_shape_manifest
from tools.audit_g17_pool_curation import main


def _write_profiles(root, count=49):
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        body = {
            "schema": "correspondence-record-profile/v3",
            "source": {
                "filename": "source-%02d.csv" % index,
                "content_sha256": hashlib.sha256(
                    ("source-%02d" % index).encode()).hexdigest(),
                "format": "delimited_text",
                "time_column": "time",
                "value_column": "value",
                "time_units": "seconds",
                "window_seconds": 20.0,
            },
            "derivation": {
                "n_samples": "finite declared time/value rows",
                "cadence_seconds": "median positive adjacent time difference",
                "coverage_fraction": "declared test derivation",
                "native_seconds": "catalogue period",
                "effective_sample_size": "declared test method",
                "noise_floor": "declared test method",
            },
            "method_review": {
                "schema": "g17-marginal-method-review/v1",
                "declaration_file": "review.json",
                "declaration_sha256": "1" * 64,
                "adoption_file": "review-adoption.json",
                "adoption_sha256": "2" * 64,
                "adopted_by": "Ada Reviewer",
                "adopted_on": "2026-09-12",
            },
            "profile": {
                "record_id": "record-%02d" % index,
                "provenance_key": "archive/product-%02d" % index,
                "n_samples": 100,
                "effective_sample_size": 50.0,
                "native_seconds": 3600.0 + index,
                "cadence_seconds": 10.0,
                "coverage_fraction": 1.0,
                "noise_floor": 0.1,
            },
        }
        (root / ("record-%02d.partner-profile.json" % index)).write_text(
            json.dumps(body), encoding="utf-8")


def _write_review(root, profiles, decision="ESTABLISHED"):
    successor = successor_scale_shape_manifest()
    readiness = audit_real_pool_readiness(successor, roots=(profiles,))
    packet = build_curation_packet(roots=(profiles,))
    packet_path = root / "g17-pool-curation-packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    path = root / "g17-pool-curation-review.json"
    path.write_text(json.dumps({
        "schema": "g17-pool-curation-review/v2",
        "study_id": successor.study_id,
        "successor_declaration_sha256": successor.digest,
        "readiness_assessment_sha256": readiness["assessment_sha256"],
        "admission_contract_sha256": CALIBRATION_CONTRACT.digest,
        "curation_packet_sha256": packet["packet_sha256"],
        "record_ids": packet["recommended_record_ids"],
        "exchangeability": decision,
        "basis": "The named records were reviewed against the declared marginal contract.",
        "unmeasured_properties": ["archive-specific selection effects"],
        "limitations": "This review does not establish any correspondence or test result.",
        "claim_boundary": "A human inventory decision, not an inference result.",
    }), encoding="utf-8")
    sign_declaration(
        root, path.name, adopted_by="Ada Reviewer", adopted_as="G17 pool curator",
        what_was_adopted="The exact inventory and its exchangeability assessment.",
        affirmation=REQUIRED_AFFIRMATION, today="2026-09-12")
    return path, packet_path


@pytest.mark.parametrize("decision", ["ESTABLISHED", "NOT_ESTABLISHED"])
def test_named_human_can_adopt_either_curation_decision(tmp_path, decision):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    review_path, packet_path = _write_review(tmp_path, profiles, decision)

    adopted = load_adopted_curation_review(
        review_path, packet_path=packet_path, roots=(profiles,))

    assert adopted.review.exchangeability == decision
    assert len(adopted.review.record_ids) == 49
    assert adopted.binding["adopted_by"] == "Ada Reviewer"
    assert adopted.binding["readiness_assessment_sha256"] == \
        audit_real_pool_readiness(
            successor_scale_shape_manifest(), roots=(profiles,))["assessment_sha256"]
    assert successor_scale_shape_manifest().real_pool_inventory.status == "UNRESOLVED"
    assert main([str(review_path), "--packet", str(packet_path),
                 "--profile-root", str(profiles)]) == 0


def test_curation_review_is_refused_before_inventory_is_review_ready(tmp_path):
    profiles = tmp_path / "profiles"
    review_path, packet_path = _write_review(tmp_path, profiles, "NOT_ESTABLISHED")

    with pytest.raises(UserInputError, match="READY_FOR_CURATION_REVIEW"):
        load_adopted_curation_review(
            review_path, packet_path=packet_path, roots=(profiles,))


def test_profile_change_invalidates_an_adopted_curation_review(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    review_path, packet_path = _write_review(tmp_path, profiles)
    changed = profiles / "record-00.partner-profile.json"
    body = json.loads(changed.read_text(encoding="utf-8"))
    body["profile"]["noise_floor"] = 0.2
    changed.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(UserInputError, match="exact current inventory"):
        load_adopted_curation_review(
            review_path, packet_path=packet_path, roots=(profiles,))


@pytest.mark.parametrize("mutation", ["remove-adoption", "change-after-adoption"])
def test_curation_review_requires_a_current_named_adoption(tmp_path, mutation):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    review_path, packet_path = _write_review(tmp_path, profiles)
    if mutation == "remove-adoption":
        (tmp_path / "g17-pool-curation-review-adoption.json").unlink()
    else:
        review_path.write_text(
            review_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="adoption"):
        load_adopted_curation_review(
            review_path, packet_path=packet_path, roots=(profiles,))


def test_curation_packet_requires_49_pairwise_admissible_records(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles, count=48)
    packet = build_curation_packet(roots=(profiles,))
    assert packet["recommended_size"] == 0
    assert packet["alternatives_per_recommended_record"] == 0

    _write_profiles(profiles, count=49)
    packet = build_curation_packet(roots=(profiles,))
    assert packet["recommended_size"] == 49
    assert packet["alternatives_per_recommended_record"] == 48


def test_review_request_prepares_inputs_without_making_a_decision(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    packet = build_curation_packet(roots=(profiles,))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    request = build_curation_review_request(packet_path=packet_path, roots=(profiles,))

    assert request["status"] == "AWAITING_HUMAN_REVIEW"
    assert request["record_ids"] == packet["recommended_record_ids"]
    assert request["verified_facts"]["exchangeability"] == "NOT_ASSESSED"
    assert request["human_inputs_required"]["exchangeability"] == [
        "ESTABLISHED", "NOT_ESTABLISHED"]
    assert "exchangeability" in request["claim_boundary"]

    response = TestClient(app).get("/api/v1/g17-pool")
    assert response.status_code == 200
    body = response.json()
    assert body["readiness"]["profile_count"] == 104
    assert body["qualification"]["refused_target_count"] == 8
    assert body["packet"]["recommended_size"] == 57
    assert body["review"]["status"] == "NOT_WRITTEN"
    assert len(body["archived_lineage_bug_artifacts"]) == 8
    operations = {row["operation"]: row for row in body["operations"]}
    assert operations["write the human curation review"]["available_in_ui"] is True
    assert operations["discover and acquire external TESS products"]["available_in_ui"] is False


def test_review_request_is_not_accepted_as_a_human_review(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    packet = build_curation_packet(roots=(profiles,))
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(build_curation_review_request(
        packet_path=packet_path, roots=(profiles,))), encoding="utf-8")

    with pytest.raises(UserInputError, match="curation review is invalid"):
        load_adopted_curation_review(
            request_path, packet_path=packet_path, roots=(profiles,))


def test_g17_pool_api_writes_but_does_not_adopt_a_human_review(tmp_path):
    app.state.g17_study_dir = tmp_path
    try:
        response = TestClient(app).post("/api/v1/g17-pool/review", json={
            "exchangeability": "NOT_ESTABLISHED",
            "basis": "The reviewer does not have evidence beyond marginal agreement.",
            "unmeasured_properties": ["target independence"],
            "limitations": "This decision applies only to the exact named subset.",
            "claim_boundary": "No correspondence result is established by this review.",
        })
    finally:
        del app.state.g17_study_dir

    assert response.status_code == 200
    assert response.json()["status"] == "WRITTEN_NOT_ADOPTED"
    assert not (tmp_path / "g17_pool_curation_review-adoption.json").exists()