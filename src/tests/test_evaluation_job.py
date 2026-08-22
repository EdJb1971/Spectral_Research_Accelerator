"""Acceptance tests for T5.6f portable external-evaluation jobs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.forecasting import ForecastContractError, load_evaluation_run_receipt
from src.forecasting.evaluation_job import (
    EVALUATION_JOB_SCHEMA,
    EVALUATION_PATH_BINDINGS_SCHEMA,
    EvaluationJob,
    EvaluationPathBindings,
    load_evaluation_job,
    load_evaluation_path_bindings,
    main,
    preflight_evaluation_job,
    resolve_evaluation_paths,
    run_evaluation_job,
    save_evaluation_job,
    save_evaluation_path_bindings,
    verify_evaluation_receipt_for_job,
)
from src.tests.test_evaluation_run import _config, _inputs


def _job(tmp_path: Path):
    request, result, artifact, crop, cache_dir = _inputs(tmp_path)
    job = EvaluationJob(
        schema=EVALUATION_JOB_SCHEMA,
        forecast_request=request,
        forecast_result=result,
        era5_crop=crop,
        evaluation_config=_config(),
    )
    return job, artifact, cache_dir


def _bindings(job, artifact, cache_dir, receipt="receipts/evaluation.json"):
    return EvaluationPathBindings(
        schema=EVALUATION_PATH_BINDINGS_SCHEMA,
        job_sha256=job.fingerprint(),
        forecast_artifact_path=str(artifact),
        era5_cache_dir=str(cache_dir),
        receipt_path=receipt,
    )


def test_job_is_versioned_hashable_canonical_and_tamper_evident(tmp_path):
    job, _, _ = _job(tmp_path)
    path = tmp_path / "job.json"
    assert save_evaluation_job(path, job) == job.fingerprint()
    loaded = load_evaluation_job(path)
    assert loaded == job
    assert hash(loaded) == hash(job)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_evaluation_job(path, job)

    record = json.loads(path.read_text(encoding="utf-8"))
    record["job"]["evaluation_config"]["split"] = "different_test"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ForecastContractError, match="SHA-256 mismatch"):
        load_evaluation_job(path)


def test_bindings_are_separate_relocatable_and_bound_to_one_job(tmp_path):
    job, artifact, cache_dir = _job(tmp_path)
    first_root = tmp_path / "laptop"
    second_root = tmp_path / "hpc"
    first = _bindings(job, "../forecast.zarr", "../era5-cache")
    second = _bindings(job, "/shared/fcn/forecast.zarr", "/shared/era5")
    first_path = first_root / "bindings.json"
    second_path = second_root / "bindings.json"
    save_evaluation_path_bindings(first_path, first)
    save_evaluation_path_bindings(second_path, second)

    assert load_evaluation_path_bindings(first_path) == first
    assert load_evaluation_path_bindings(second_path) == second
    assert resolve_evaluation_paths(first, base_dir=first_root).forecast_artifact_path == artifact
    assert job.fingerprint() == load_evaluation_job(_save_job(tmp_path, job)).fingerprint()
    wrong = EvaluationPathBindings(
        schema=EVALUATION_PATH_BINDINGS_SCHEMA, job_sha256="0" * 64,
        forecast_artifact_path=str(artifact), era5_cache_dir=str(cache_dir),
        receipt_path="receipt.json")
    with pytest.raises(ForecastContractError, match="different evaluation job"):
        preflight_evaluation_job(job, wrong, base_dir=tmp_path)


def _save_job(tmp_path: Path, job: EvaluationJob) -> Path:
    path = tmp_path / "portable-job.json"
    if not path.exists():
        save_evaluation_job(path, job)
    return path


def test_preflight_authenticates_inputs_without_writing_or_network(tmp_path):
    job, artifact, cache_dir = _job(tmp_path)
    bindings = _bindings(job, artifact, cache_dir)
    report = preflight_evaluation_job(job, bindings, base_dir=tmp_path)
    assert report["ready"] is True
    assert report["network_used"] is False
    assert report["model_inference_used"] is False
    assert report["job_sha256"] == job.fingerprint()
    assert "forecast_artifact_authenticated" in report["checks"]
    assert not (tmp_path / "receipts" / "evaluation.json").exists()


@pytest.mark.parametrize("failure", ["missing_cache", "tampered_artifact", "existing_receipt"])
def test_preflight_refuses_unready_or_unsafe_local_bindings(tmp_path, failure):
    job, artifact, cache_dir = _job(tmp_path)
    bindings = _bindings(job, artifact, cache_dir)
    if failure == "missing_cache":
        bindings = _bindings(job, artifact, tmp_path / "absent-cache")
        match = "not present"
    elif failure == "tampered_artifact":
        (artifact / ".tamper").write_text("changed", encoding="utf-8")
        match = "SHA-256 mismatch"
    else:
        receipt = tmp_path / "receipts" / "evaluation.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text("keep", encoding="utf-8")
        match = "refusing to overwrite"
    with pytest.raises((ForecastContractError, FileExistsError), match=match):
        preflight_evaluation_job(job, bindings, base_dir=tmp_path)


def test_run_executes_exact_pipeline_and_receipt_verifies_against_job(tmp_path):
    job, artifact, cache_dir = _job(tmp_path)
    bindings = _bindings(job, artifact, cache_dir)
    receipt = run_evaluation_job(job, bindings, base_dir=tmp_path)
    saved_path = tmp_path / "receipts" / "evaluation.json"
    saved = load_evaluation_run_receipt(saved_path)
    verify_evaluation_receipt_for_job(job, saved)
    assert receipt.fingerprint() == saved["receipt_sha256"]
    assert saved["evaluation"]["initialization_count"] == 1

    other = EvaluationJob(
        schema=EVALUATION_JOB_SCHEMA, forecast_request=job.forecast_request,
        forecast_result=job.forecast_result, era5_crop=job.era5_crop,
        evaluation_config=_config(max_evaluation_tile_bytes=512))
    with pytest.raises(ForecastContractError, match="does not correspond"):
        verify_evaluation_receipt_for_job(other, saved)


def test_cli_create_bind_preflight_and_run_workflow(tmp_path, capsys):
    job, artifact, cache_dir = _job(tmp_path)
    request_path = tmp_path / "request.json"
    result_path = tmp_path / "result.json"
    crop_path = tmp_path / "crop.json"
    config_path = tmp_path / "config.json"
    job_path = tmp_path / "portable" / "job.json"
    bindings_path = tmp_path / "portable" / "bindings.json"

    request_path.write_text(json.dumps({
        "request": job.forecast_request.to_mapping(),
        "request_sha256": job.forecast_request.fingerprint(),
    }), encoding="utf-8")
    result_path.write_text(json.dumps(job.forecast_result.to_mapping()), encoding="utf-8")
    crop_path.write_text(json.dumps(job.era5_crop.to_provenance()), encoding="utf-8")
    config_path.write_text(json.dumps(job.evaluation_config.to_mapping()), encoding="utf-8")

    assert main([
        "create", "--forecast-request", str(request_path), "--forecast-result", str(result_path),
        "--era5-crop", str(crop_path), "--evaluation-config", str(config_path),
        "--output", str(job_path)]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["job_sha256"] == job.fingerprint()

    assert main([
        "bind", "--job", str(job_path), "--forecast-artifact", str(artifact),
        "--era5-cache-dir", str(cache_dir), "--receipt", "result/evaluation.json",
        "--output", str(bindings_path)]) == 0
    capsys.readouterr()
    assert main(["preflight", "--job", str(job_path), "--bindings", str(bindings_path)]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True
    assert main(["run", "--job", str(job_path), "--bindings", str(bindings_path)]) == 0
    completed = json.loads(capsys.readouterr().out)
    assert completed["job_sha256"] == job.fingerprint()
    assert Path(completed["receipt_path"]).exists()


def test_cli_reports_contract_failure_without_traceback(tmp_path, capsys):
    missing = tmp_path / "missing.json"
    assert main(["preflight", "--job", str(missing), "--bindings", str(missing)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cannot read evaluation job" in captured.err
