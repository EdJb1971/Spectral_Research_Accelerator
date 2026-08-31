"""TG17.9: portable replay must preserve identity without becoming evidence."""

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.errors import InvalidParameterError
from src.core.experiment_manifest import flagship_recipe
from src.core.experiment_receipt import (BUNDLE_FIELDS, BUNDLE_SCHEMA, RECEIPT_FIELDS,
                                         build_bundle, capability_snapshot, publish_bundle,
                                         verify_bundle)
from src.core.experiment_run import RunStore
from src.core.run_workers import build_suite


def _spec():
    recipe = flagship_recipe()
    return recipe.copy(update={"observations": [item for item in recipe.observations
                                                if item.domain != "order_book"]})


def _complete(tmp_path):
    run = RunStore(tmp_path).open(_spec())
    run.execute(build_suite("fixture_dry_run", run=run))
    return run


def _rehash(bundle):
    body = dict(bundle)
    body.pop("bundle_sha256", None)
    def normal(value):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: normal(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normal(item) for item in value]
        return value
    return hashlib.sha256(json.dumps(normal(body), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def test_only_a_completed_run_can_be_exported(tmp_path):
    run = RunStore(tmp_path).open(_spec())
    run.preflight()
    with pytest.raises(InvalidParameterError) as caught:
        build_bundle(run)
    assert "COMPLETE" in str(caught.value)


def test_bundle_has_exactly_the_registered_explained_fields(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    assert set(bundle) == set(BUNDLE_FIELDS)
    assert {row["name"] for row in RECEIPT_FIELDS} == set(bundle)
    assert all(row["label"] and row["meaning"] for row in RECEIPT_FIELDS)


def test_bundle_is_self_hashed_over_every_other_field(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert bundle["bundle_sha256"] == _rehash(bundle)


def test_exact_manifest_and_run_identity_survive_replay(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    replay = verify_bundle(json.loads(json.dumps(bundle)))
    assert replay["integrity"] == "VERIFIED"
    assert replay["manifest"] == bundle["manifest"]
    assert replay["run_identity"] == bundle["run_identity"]
    assert replay["run_receipt"] == bundle["run_receipt"]
    assert replay["results"] == bundle["results"]
    assert replay["refusals"] == bundle["refusals"]


def test_browser_style_integral_float_roundtrip_does_not_change_identity(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    # JSON.parse/stringify in a browser emits 1.0 as 1. Simulate that spelling loss explicitly.
    browser = json.loads(json.dumps(bundle))
    value = browser["manifest"]["alignment"]["minimum_effective_samples"]
    assert isinstance(value, float) and value.is_integer()
    browser["manifest"]["alignment"]["minimum_effective_samples"] = int(value)
    assert verify_bundle(browser)["bundle_sha256"] == bundle["bundle_sha256"]


def test_replay_needs_no_run_store_or_ui_state(tmp_path):
    bundle = build_bundle(_complete(tmp_path / "run"))
    exported = json.loads(json.dumps(bundle))
    replay = verify_bundle(exported)
    assert not (tmp_path / "replay" / "runs").exists()
    assert replay["bundle_sha256"] == bundle["bundle_sha256"]


def test_a_changed_field_is_detected(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    bundle["claim_boundary"] = "completed means proven"
    with pytest.raises(InvalidParameterError) as caught:
        verify_bundle(bundle)
    assert "bundle_sha256" in str(caught.value)


def test_rehashing_a_forged_receipt_does_not_defeat_semantic_replay(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    bundle["run_receipt"]["state"] = "REFUSED"
    bundle["bundle_sha256"] = _rehash(bundle)
    with pytest.raises(InvalidParameterError) as caught:
        verify_bundle(bundle)
    assert "reconstructed" in str(caught.value)


def test_rehashing_an_impossible_transition_is_refused(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    transition = next(event for event in bundle["events"] if event["kind"] == "transition")
    transition["to_state"] = "COMPLETE"
    bundle["bundle_sha256"] = _rehash(bundle)
    with pytest.raises(InvalidParameterError) as caught:
        verify_bundle(bundle)
    assert "transition" in str(caught.value)


def test_unknown_top_level_fields_are_refused_even_when_rehashed(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    bundle["unexplained_claim"] = True
    bundle["bundle_sha256"] = _rehash(bundle)
    with pytest.raises(InvalidParameterError) as caught:
        verify_bundle(bundle)
    assert "unexpected" in str(caught.value)


def test_sources_keep_acquisition_identity_and_plan_separate(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    assert len(bundle["sources"]) == 3
    for row in bundle["sources"]:
        assert row["acquisition_identity"]["source_id"] == row["source_plan"]["source_id"]
        assert "identity" in row["acquisition_identity"]
        assert row["source_plan"]["opens_measurement_values"] is False


def test_adapter_contracts_carry_versions_definition_digests_and_config(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    for row in bundle["adapters"]:
        binding, contract = row["manifest_binding"], row["contract"]
        assert binding["adapter_id"] == contract["adapter_id"]
        assert binding["adapter_version"] == contract["adapter_version"]
        assert len(contract["definition_sha256"]) == 64
        assert row["translator_config"] == {"ddof": 0}


def test_artefacts_are_separated_by_scientific_role(tmp_path):
    bundle = build_bundle(_complete(tmp_path))
    assert set(bundle["artefacts"]) == {"native", "canonical", "mining", "confirmation"}
    assert set(bundle["artefacts"]["native"]) == {"reanalysis", "argo_float", "tess_lightcurve"}
    assert set(bundle["artefacts"]["canonical"]) == set(bundle["artefacts"]["native"])
    assert bundle["results"]["measured"] is False
    assert bundle["results"]["artefacts"] == {}
    assert len(bundle["results"]["fixture_artefacts"]) == 2
    statuses = {row["category"]: row["status"]
                for row in bundle["evidence_handoff"]["categories"]}
    assert statuses["measured_results"] == "ABSENT"


def test_inference_keeps_family_null_correction_confirmation_and_seeds(tmp_path):
    inference = build_bundle(_complete(tmp_path))["inference"]
    assert inference["family"]["maximum_members"] > 0
    assert inference["nulls"]
    assert inference["correction"] == "benjamini_yekutieli"
    assert inference["confirmation"]["stage"] == "generate_then_confirm"
    assert inference["seeds"]


def test_environment_was_captured_at_freeze_not_invented_by_export(tmp_path):
    run = _complete(tmp_path)
    frozen = next(event for event in run.events if event["kind"] == "freeze")
    assert build_bundle(run)["environment"] == frozen["environment"]
    assert frozen["environment"]["software_version"].endswith("tg17.9")
    assert frozen["archival_context"]["preflight"]["manifest_sha256"] == run.manifest_sha256
    assert len(frozen["archival_context"]["adapters"]) == 3


def test_methods_report_is_hashed_and_states_the_claim_limit(tmp_path):
    report = build_bundle(_complete(tmp_path))["methods_report"]
    assert report["sha256"] == hashlib.sha256(report["text"].encode()).hexdigest()
    assert "not automatically a study" in report["text"]
    assert "independent-replication claim" in report["text"]
    assert "benjamini_yekutieli" in report["text"]


def test_handoff_has_no_automatic_action_and_names_absent_categories(tmp_path):
    handoff = build_bundle(_complete(tmp_path))["evidence_handoff"]
    assert handoff["automatic_actions"] == []
    assert "open_reviewable_study_draft" in handoff["eligible_actions"]
    statuses = {row["category"]: row["status"] for row in handoff["categories"]}
    assert statuses["registered_hypothesis"] == "ABSENT"
    assert statuses["admitted_evidence"] == "ABSENT"
    assert statuses["claim_promotion"] == "ABSENT"


def test_publication_is_content_addressed_and_idempotent(tmp_path):
    run = _complete(tmp_path / "runs")
    first = publish_bundle(run, tmp_path / "exports")
    assert first["reused"] is False
    before = Path(first["bundle_path"]).read_bytes()
    second = publish_bundle(run, tmp_path / "exports")
    assert second["reused"] is True
    assert first["bundle"]["bundle_sha256"] == second["bundle"]["bundle_sha256"]
    assert Path(second["bundle_path"]).read_bytes() == before


def test_capability_snapshot_is_generated_from_registry_and_field_contract():
    snapshot = capability_snapshot()
    assert {row["name"] for row in snapshot["receipt_fields"]} == set(BUNDLE_FIELDS)
    assert {row["contract"]["adapter_id"] if "contract" in row else row["adapter_id"]
            for row in snapshot["adapters"]} >= {
                "reanalysis.standardized-level", "argo_float.standardized-level",
                "tess_lightcurve.standardized-level", "order_book.bespoke_record"}
    assert all(operation["writes_evidence"] is False for operation in snapshot["operations"])
    assert snapshot["refusals"]


def test_http_export_replay_methods_and_capabilities(tmp_path):
    run = _complete(tmp_path)
    app.state.experiment_run_dir = tmp_path
    client = TestClient(app)
    capabilities = client.get("/api/v1/experiment-receipts").json()
    assert capabilities["receipt_fields"]
    exported = client.post("/api/v1/experiment-receipts/runs/%s/export" % run.run_id)
    assert exported.status_code == 200
    body = exported.json()
    assert body["integrity"] == "VERIFIED"
    replayed = client.post("/api/v1/experiment-receipts/replay", json=body["bundle"])
    assert replayed.status_code == 200
    assert replayed.json()["run_receipt"] == body["bundle"]["run_receipt"]
    methods = client.get("/api/v1/experiment-receipts/runs/%s/methods" % run.run_id)
    assert methods.status_code == 200
    assert methods.headers["content-type"].startswith("text/plain")
    assert "Methods and limitations".lower() in methods.text.lower()


def test_http_refuses_export_of_a_non_complete_run(tmp_path):
    run = RunStore(tmp_path).open(_spec())
    run.preflight()
    app.state.experiment_run_dir = tmp_path
    response = TestClient(app).post(
        "/api/v1/experiment-receipts/runs/%s/export" % run.run_id)
    assert response.status_code == 409
    assert "COMPLETE" in response.json()["detail"]
