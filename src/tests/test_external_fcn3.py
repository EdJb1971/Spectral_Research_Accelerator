"""Acceptance tests for the offline T5.6a FCN3 request/result contract."""

from __future__ import annotations

import copy
import json
import sys

import pytest

from src.forecasting import (
    FCN3_PARAMETER_COUNT,
    FCN3_VARIABLES,
    NZ_850_VARIABLES,
    ExternalForecastResult,
    ExternalForecastRun,
    ForecastContractError,
    load_external_forecast_result,
    load_external_forecast_run,
    save_external_forecast_result,
    save_external_forecast_run,
    seal_external_forecast_result,
    verify_external_forecast_artifact,
    verify_external_forecast_result,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64


def complete_request(*, worker: bool = False):
    return {
        "schema": "external-forecast-run/fcn3-v1",
        "request_revision": 1,
        "run_id": "fcn3-nz-independent-evaluation-fixture",
        "evidence_reference": "evidence/fcn3-reviewed-contract.json@fixture-v1",
        "evidence_sha256": SHA_A,
        "model": {
            "provider": "NVIDIA", "name": "FourCastNet 3", "model_version": "v1",
            "parameter_count": FCN3_PARAMETER_COUNT,
            "package_reference": "ngc://nvidia/earth-2/fourcastnet3:0.1.0",
            "package_sha256": SHA_B, "checkpoint_sha256": SHA_C,
            "model_card_reference": (
                "https://catalog.ngc.nvidia.com/orgs/nvidia/earth-2/models/fourcastnet3"),
            "model_card_sha256": SHA_D, "license_spdx": "Apache-2.0",
        },
        "software": {
            "runner": "earth2studio.models.px.FCN3", "earth2studio_version": "0.14.0",
            "torch_version": "2.8.0", "torch_harmonics_version": "0.8.0",
            "environment_lock_reference": "workers/fcn3/requirements.lock@fixture-v1",
            "environment_lock_sha256": SHA_E,
        },
        "input": {
            "source_name": "NCAR ERA5", "source_version": "fixture-snapshot-v1",
            "source_reference": "ncar-era5://2020-02-11T00:00:00Z",
            "source_sha256": SHA_F, "variables": list(FCN3_VARIABLES), "scope": "global",
            "grid_shape": [721, 1440], "grid_spacing_degrees": 0.25,
            "latitude_order": "descending", "longitude_convention": "0..360",
            "grid_coordinates_sha256": SHA_A,
            "initialization_times": ["2020-02-11T00:00:00+00:00"],
            "normalisation_reference": "ngc-package://normalisation/v1",
            "normalisation_sha256": SHA_B,
        },
        "rollout": {
            "cadence_hours": 6.0, "steps": 16, "ensemble_size": 4,
            "member_seeds": [7103, 7104, 7105, 7106],
            "stochastic_process": "FCN3 hidden-Markov spherical latent diffusion",
            "perturbation": "none",
        },
        "output": {
            "variables": list(NZ_850_VARIABLES), "artifact_format": "netcdf4",
            "artifact_reference": "artifacts/fcn3/global-ensemble.nc",
            "crop_policy": "global_then_crop",
        },
        "runtime": ({
            "mode": "external_worker", "operating_system": "linux",
            "accelerator": "nvidia_cuda", "hardware_model": "H100 fixture",
            "precision": "bf16",
        } if worker else {
            "mode": "prepare_only", "operating_system": None, "accelerator": None,
            "hardware_model": None, "precision": "bf16",
        }),
    }


def test_complete_request_is_canonical_hashable_offline_and_round_trips(tmp_path):
    before = set(sys.modules)
    request = ExternalForecastRun.from_mapping(complete_request())
    same = ExternalForecastRun.from_mapping(json.loads(json.dumps(complete_request())))
    imported = set(sys.modules) - before

    assert request == same and hash(request) == hash(same)
    assert len(request.fingerprint()) == 64
    assert request.rollout.lead_hours[-1] == 96.0
    assert len(request.input.variables) == 72
    assert not any(name.startswith(("earth2studio", "torch_harmonics", "makani"))
                   for name in imported)
    path = tmp_path / "request.json"
    assert save_external_forecast_run(path, request) == request.fingerprint()
    assert load_external_forecast_run(path) == request
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_external_forecast_run(path, request)


@pytest.mark.parametrize("mutation,match", [
    (lambda record: record["input"].update(scope="regional"), "scope must be global"),
    (lambda record: record["input"].update(grid_shape=[120, 80]), "global 721x1440"),
    (lambda record: record["input"]["variables"].reverse(), "canonical model-card order"),
    (lambda record: record["input"].update(variables=list(FCN3_VARIABLES[:-1])), "all 72"),
    (lambda record: record["output"].update(crop_policy="crop_then_run"), "global_then_crop"),
])
def test_request_refuses_regional_or_semantically_changed_model_inputs(mutation, match):
    record = complete_request()
    mutation(record)
    with pytest.raises(ForecastContractError, match=match):
        ExternalForecastRun.from_mapping(record)


@pytest.mark.parametrize("mutation,match", [
    (lambda record: record["rollout"].update(cadence_hours=1), "exactly 6 hours"),
    (lambda record: record["rollout"].update(member_seeds=[1, 2]), "one seed per"),
    (lambda record: record["rollout"].update(member_seeds=[1, 1, 2, 3]), "must be unique"),
    (lambda record: record["rollout"].update(perturbation="gaussian"), "must be none"),
    (lambda record: record["input"].update(
        initialization_times=["2020-02-11T00:00:00"]), "UTC offset"),
])
def test_request_refuses_ambiguous_time_or_ensemble_semantics(mutation, match):
    record = complete_request()
    mutation(record)
    with pytest.raises(ForecastContractError, match=match):
        ExternalForecastRun.from_mapping(record)


def test_request_refuses_model_drift_placeholders_and_unknown_fields():
    changed = complete_request()
    changed["model"]["parameter_count"] -= 1
    with pytest.raises(ForecastContractError, match="parameter_count"):
        ExternalForecastRun.from_mapping(changed)
    placeholder = complete_request()
    placeholder["software"]["earth2studio_version"] = "TBD"
    with pytest.raises(ForecastContractError, match="non-placeholder"):
        ExternalForecastRun.from_mapping(placeholder)
    unknown = complete_request()
    unknown["runtime"]["allow_fallback"] = True
    with pytest.raises(ForecastContractError, match="unknown"):
        ExternalForecastRun.from_mapping(unknown)


@pytest.mark.parametrize("runtime,match", [
    ({"mode": "prepare_only", "operating_system": "linux", "accelerator": None,
      "hardware_model": None, "precision": "bf16"}, "must leave"),
    ({"mode": "external_worker", "operating_system": "windows",
      "accelerator": "nvidia_cuda", "hardware_model": "RTX fixture",
      "precision": "bf16"}, "Linux and NVIDIA CUDA"),
    ({"mode": "external_worker", "operating_system": "linux",
      "accelerator": "amd_rocm", "hardware_model": "AMD fixture",
      "precision": "bf16"}, "Linux and NVIDIA CUDA"),
])
def test_runtime_never_silently_promotes_unverified_hardware(runtime, match):
    record = complete_request()
    record["runtime"] = runtime
    with pytest.raises(ForecastContractError, match=match):
        ExternalForecastRun.from_mapping(record)


def test_completed_result_binds_request_file_resources_and_detects_tampering(tmp_path):
    request = ExternalForecastRun.from_mapping(complete_request(worker=True))
    artifact = tmp_path / "global-ensemble.nc"
    artifact.write_bytes(b"deterministic synthetic NetCDF fixture bytes")
    result = seal_external_forecast_result(
        request, artifact, peak_vram_bytes=12_000_000_000,
        peak_ram_bytes=8_000_000_000, wall_time_seconds=41.25,
        worker_log_sha256=SHA_C)
    assert result.request_sha256 == request.fingerprint()
    assert result.output_variables == NZ_850_VARIABLES
    assert result.grid_shape == (721, 1440)
    assert len(result.result_sha256) == len(result.artifact_sha256) == 64
    verify_external_forecast_artifact(result, artifact)
    verify_external_forecast_result(request, result, artifact)

    manifest = tmp_path / "result.json"
    assert save_external_forecast_result(manifest, result) == result.result_sha256
    assert load_external_forecast_result(manifest) == result
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_external_forecast_result(manifest, result)
    artifact.write_bytes(b"changed")
    with pytest.raises(ForecastContractError, match="artifact SHA-256 mismatch"):
        verify_external_forecast_artifact(result, artifact)


def test_result_refuses_prepare_only_claim_and_manifest_tampering(tmp_path):
    artifact = tmp_path / "forecast.nc"
    artifact.write_bytes(b"fixture")
    request = ExternalForecastRun.from_mapping(complete_request())
    with pytest.raises(ForecastContractError, match="prepare-only"):
        seal_external_forecast_result(
            request, artifact, peak_vram_bytes=1, peak_ram_bytes=1,
            wall_time_seconds=1, worker_log_sha256=SHA_A)

    worker = ExternalForecastRun.from_mapping(complete_request(worker=True))
    result = seal_external_forecast_result(
        worker, artifact, peak_vram_bytes=1, peak_ram_bytes=1,
        wall_time_seconds=1, worker_log_sha256=SHA_A)
    changed = copy.deepcopy(result.to_mapping())
    changed["ensemble_size"] = 99
    with pytest.raises(ForecastContractError, match="result SHA-256 mismatch"):
        ExternalForecastResult.from_mapping(changed)


def test_zarr_tree_identity_is_path_order_independent_and_value_sensitive(tmp_path):
    first, second = tmp_path / "first.zarr", tmp_path / "second.zarr"
    first.mkdir()
    second.mkdir()
    (first / "zarr.json").write_text("metadata", encoding="utf-8")
    (first / "c1").write_bytes(b"one")
    (second / "c1").write_bytes(b"one")
    (second / "zarr.json").write_text("metadata", encoding="utf-8")
    record = complete_request(worker=True)
    record["output"]["artifact_format"] = "zarr"
    record["output"]["artifact_reference"] = "artifacts/fcn3/global-ensemble.zarr"
    request = ExternalForecastRun.from_mapping(record)
    result = seal_external_forecast_result(
        request, first, peak_vram_bytes=1, peak_ram_bytes=1,
        wall_time_seconds=1, worker_log_sha256=SHA_A)
    verify_external_forecast_artifact(result, second)
    (second / "c1").write_bytes(b"two")
    with pytest.raises(ForecastContractError, match="artifact SHA-256 mismatch"):
        verify_external_forecast_artifact(result, second)


def test_result_refuses_format_kind_mismatch_and_cross_request_substitution(tmp_path):
    directory = tmp_path / "not-netcdf.zarr"
    directory.mkdir()
    (directory / "zarr.json").write_text("{}", encoding="utf-8")
    request = ExternalForecastRun.from_mapping(complete_request(worker=True))
    with pytest.raises(ForecastContractError, match="netcdf4 output requires a file"):
        seal_external_forecast_result(
            request, directory, peak_vram_bytes=1, peak_ram_bytes=1,
            wall_time_seconds=1, worker_log_sha256=SHA_A)

    artifact = tmp_path / "forecast.nc"
    artifact.write_bytes(b"fixture")
    result = seal_external_forecast_result(
        request, artifact, peak_vram_bytes=1, peak_ram_bytes=1,
        wall_time_seconds=1, worker_log_sha256=SHA_A)
    changed = complete_request(worker=True)
    changed["rollout"]["steps"] = 8
    another_request = ExternalForecastRun.from_mapping(changed)
    with pytest.raises(ForecastContractError, match="differs from its request"):
        verify_external_forecast_result(another_request, result)
