"""T5.6g verified receipt reporting and API/UI evidence boundary."""

from __future__ import annotations

import json

import pytest

from src.data_layer.zarr_source import CATALOGUE, manifest_path
from src.forecasting.adapter import ForecastContractError
from src.forecasting.evaluation_report import (
    EVALUATION_REPORT_SCHEMA,
    EvaluationReceiptStore,
    build_evaluation_report,
    decode_receipt,
)
from src.forecasting.evaluation_run import load_evaluation_run_receipt, run_external_evaluation
from src.tests.test_evaluation_run import _config, _inputs


@pytest.fixture()
def receipt_payloads(tmp_path):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    synthetic_path = tmp_path / "synthetic.json"
    run_external_evaluation(
        forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
        era5_crop=crop, era5_cache_dir=cache_dir, config=_config(), receipt_path=synthetic_path)

    manifest_file = manifest_path(crop, str(cache_dir))
    manifest = json.loads(open(manifest_file, encoding="utf-8").read())
    manifest.pop("source_route")
    with open(manifest_file, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, sort_keys=True)
    old = CATALOGUE.get(crop.store)
    CATALOGUE[crop.store] = {"uri": crop.store}
    official_path = tmp_path / "official.json"
    run_external_evaluation(
        forecast_artifact_path=artifact, forecast_request=request, forecast_result=result,
        era5_crop=crop, era5_cache_dir=cache_dir, config=_config(), receipt_path=official_path)
    try:
        yield synthetic_path.read_bytes(), official_path.read_bytes()
    finally:
        if old is None:
            CATALOGUE.pop(crop.store, None)
        else:
            CATALOGUE[crop.store] = old


def test_report_flattens_exact_metrics_scope_provenance_and_claim_boundaries(receipt_payloads):
    _, payload = receipt_payloads
    report = build_evaluation_report(decode_receipt(payload))
    assert report["schema"] == EVALUATION_REPORT_SCHEMA
    assert report["readiness"] == {
        "receipt_integrity": "VERIFIED",
        "forecast_artifact": "AUTHENTICATED_AND_VALIDATED",
        "truth_source": "DECLARED_OFFICIAL_ERA5",
        "display_eligible": True,
        "independent_holdout": True,
        "scientific_skill": "NOT_ESTABLISHED",
        "reason": "Verified receipt over a declared official ERA5 catalogue source.",
    }
    assert report["scope"]["variables"] == ["t850"]
    assert report["scope"]["lead_durations_hours"] == [6.0]
    assert report["metrics"][0]["persistence"]["rmse"] > 0
    assert report["metrics"][0]["unit"] == "K"
    assert report["rank_diagnostics"][0]["diagnostic_only"] is True
    assert sum(report["rank_diagnostics"][0]["area_weighted_frequency"]) == pytest.approx(1.0)
    assert report["provenance"]["forecast"]["model"] == "FourCastNet 3"
    assert len(report["provenance"]["forecast"]["validation_sha256"]) == 64
    assert len(report["claim_boundaries"]) == 2
    assert "cache_path" not in json.dumps(report)
    assert "artifact_reference" not in json.dumps(report)


def test_store_refuses_hash_valid_synthetic_receipt_and_remains_empty(tmp_path, receipt_payloads):
    synthetic, _ = receipt_payloads
    store = EvaluationReceiptStore(tmp_path / "reports")
    with pytest.raises(ForecastContractError, match="not an accepted official ERA5"):
        store.import_bytes(synthetic)
    assert store.list() == []
    assert not (tmp_path / "reports").exists()


def test_store_is_content_addressed_idempotent_and_reverifies_on_read(tmp_path, receipt_payloads):
    _, payload = receipt_payloads
    store = EvaluationReceiptStore(tmp_path / "reports")
    first = store.import_bytes(payload)
    second = store.import_bytes(payload)
    assert first == second == store.get(first["report_id"])
    assert len(store.list()) == 1
    target = tmp_path / "reports" / (first["report_id"] + ".json")
    record = json.loads(target.read_text(encoding="utf-8"))
    record["evaluation"]["initialization_count"] = 99
    target.write_text(json.dumps(record), encoding="utf-8")
    assert store.list() == []
    with pytest.raises(ForecastContractError, match="SHA-256 mismatch"):
        store.get(first["report_id"])


def test_api_empty_state_import_list_and_get_without_server_paths(client, tmp_path, receipt_payloads):
    _, payload = receipt_payloads
    from src.api.main import app

    app.state.evaluation_receipt_dir = tmp_path / "api-reports"
    assert client.get("/api/v1/evaluation/receipts").json() == []
    imported = client.post(
        "/api/v1/evaluation/receipts/import",
        files={"file": ("evaluation.json", payload, "application/json")})
    assert imported.status_code == 200, imported.text
    report = imported.json()
    listed = client.get("/api/v1/evaluation/receipts").json()
    fetched = client.get("/api/v1/evaluation/receipts/%s" % report["report_id"]).json()
    assert listed == [report] and fetched == report
    encoded = json.dumps(report)
    assert str(tmp_path) not in encoded and "cache_path" not in encoded
    del app.state.evaluation_receipt_dir


def test_api_refuses_tamper_synthetic_wrong_extension_and_unknown_id(client, tmp_path,
                                                                      receipt_payloads):
    synthetic, official = receipt_payloads
    from src.api.main import app

    app.state.evaluation_receipt_dir = tmp_path / "api-refusals"
    refused = client.post("/api/v1/evaluation/receipts/import",
                          files={"file": ("fixture.json", synthetic, "application/json")})
    assert refused.status_code == 422
    tampered = bytearray(official)
    tampered[-3] = ord("0") if tampered[-3] != ord("0") else ord("1")
    assert client.post("/api/v1/evaluation/receipts/import",
                       files={"file": ("bad.json", bytes(tampered), "application/json")}).status_code == 422
    assert client.post("/api/v1/evaluation/receipts/import",
                       files={"file": ("receipt.txt", official, "text/plain")}).status_code == 415
    assert client.get("/api/v1/evaluation/receipts/%s" % ("a" * 64)).status_code == 404
    del app.state.evaluation_receipt_dir
