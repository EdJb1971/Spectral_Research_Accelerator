import json

import pytest

from src.core.adoption import REQUIRED_AFFIRMATION, sign_declaration
from src.core.errors import UserInputError
from src.core.real_pool_readiness import (
    MEASUREMENT_FILE,
    audit_real_pool_readiness,
)
from src.core.real_pool_ingress import RecordProfileDeclaration, build_source_bound_profile
from src.core.real_pool_method_review import load_adopted_method_review
from src.core.scale_shape_successor import successor_scale_shape_manifest


def _write_profile(root, index):
    root.mkdir(parents=True, exist_ok=True)
    review_path = root / "g17-marginal-method-review.json"
    if not review_path.exists():
        review_path.write_text(json.dumps({
            "schema": "g17-marginal-method-review/v1",
            "study_id": "g17_scale_shape_successor_v1",
            "native_seconds_basis": "catalogue period",
            "effective_sample_size_method": "declared test method",
            "noise_floor_method": "declared test method",
            "applicability": "Synthetic test records.",
            "limitations": "Test fixture only.",
            "claim_boundary": "No scientific conclusion.",
        }), encoding="utf-8")
        sign_declaration(
            root, review_path.name, adopted_by="Ada Reviewer",
            adopted_as="G17 method reviewer",
            what_was_adopted="Test fixture methods.", affirmation=REQUIRED_AFFIRMATION,
            today="2026-09-12")
    method_review = load_adopted_method_review(review_path)
    payload = ("time,value\n0,%d\n10,%d\n20,%d\n" % (index, index + 1, index + 2)).encode()
    body = build_source_bound_profile(
        payload, filename="source-%02d.csv" % index, delimiter=",",
        declaration=RecordProfileDeclaration(
            record_id="record-%02d" % index,
            provenance_key="source-%02d" % index,
            time_column="time",
            value_column="value",
            time_units="seconds",
            window_seconds=20.0,
            native_seconds=1000.0 + index,
            native_seconds_basis="catalogue period",
            effective_sample_size=2.0,
            effective_sample_size_method="declared test method",
            noise_floor=0.1,
            noise_floor_method="declared test method"),
        method_review=method_review)
    (root / ("record-%02d.partner-profile.json" % index)).write_text(
        json.dumps(body), encoding="utf-8")


def test_empty_repository_inventory_is_a_bounded_absence(tmp_path):
    result = audit_real_pool_readiness(successor_scale_shape_manifest(), roots=(tmp_path,))

    assert result["status"] == "NO_INVENTORY"
    assert result["profile_count"] == 0
    assert result["shortfall"] == 48
    assert result["exchangeability"] == "NOT_ASSESSED"
    assert "absence of evidence" in result["claim_boundary"]


def test_committed_assessment_is_the_current_repository_audit():
    recorded = json.loads(MEASUREMENT_FILE.read_text(encoding="utf-8"))
    assert recorded == audit_real_pool_readiness(successor_scale_shape_manifest())


def test_profile_count_can_be_sufficient_without_establishing_exchangeability(tmp_path):
    for index in range(48):
        _write_profile(tmp_path, index)

    result = audit_real_pool_readiness(successor_scale_shape_manifest(), roots=(tmp_path,))
    assert result["status"] == "READY_FOR_CURATION_REVIEW"
    assert result["shortfall"] == 0
    assert result["exchangeability"] == "NOT_ASSESSED"
    assert len({row["file_sha256"] for row in result["profiles"]}) == 48


@pytest.mark.parametrize("duplicate_field", ["record_id", "provenance_key"])
def test_duplicate_record_or_provenance_identity_is_refused(tmp_path, duplicate_field):
    _write_profile(tmp_path, 1)
    _write_profile(tmp_path, 2)
    duplicate = tmp_path / "record-02.partner-profile.json"
    body = json.loads(duplicate.read_text(encoding="utf-8"))
    body["profile"][duplicate_field] = body["profile"][duplicate_field].replace("02", "01")
    duplicate.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(UserInputError, match="identity|identities"):
        audit_real_pool_readiness(successor_scale_shape_manifest(), roots=(tmp_path,))


@pytest.mark.parametrize("body", ["{}", "[]", "not-json"])
def test_malformed_explicit_profile_is_refused_instead_of_skipped(tmp_path, body):
    (tmp_path / "broken.partner-profile.json").write_text(body, encoding="utf-8")
    with pytest.raises(UserInputError, match="not a valid G17 partner profile"):
        audit_real_pool_readiness(successor_scale_shape_manifest(), roots=(tmp_path,))


@pytest.mark.parametrize("mutation", [
    "legacy", "bad-source-digest", "bad-source-units", "bad-review-digest"])
def test_unbound_or_invalid_source_evidence_is_refused(tmp_path, mutation):
    _write_profile(tmp_path, 1)
    path = tmp_path / "record-01.partner-profile.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "legacy":
        body = {"schema": "correspondence-record-profile/v1", "profile": body["profile"]}
    elif mutation == "bad-source-digest":
        body["source"]["content_sha256"] = "not-a-digest"
    elif mutation == "bad-review-digest":
        body["method_review"]["adoption_sha256"] = "not-a-digest"
    else:
        body["source"]["time_units"] = "days"
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(UserInputError, match="not a valid G17 partner profile"):
        audit_real_pool_readiness(successor_scale_shape_manifest(), roots=(tmp_path,))