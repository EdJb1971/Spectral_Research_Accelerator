"""Acceptance tests for T5.0b protocol-to-runtime binding."""

from __future__ import annotations

import copy

import pytest

from src.forecasting import (
    ExperimentProtocolBinding,
    ForecastContractError,
    MotivatingExperimentProtocol,
    TinyResidualCoefficientModel,
    bind_artifact_to_protocol,
    bind_dataset_to_protocol,
    evaluate_against_persistence,
    load_laboratory_forecaster,
    protocol_training_provenance,
    provenance_sha256,
    save_laboratory_artifact,
)
from src.tests.test_forecasting_adapter import _forecast_batch
from src.transform_engine.training import RawRepresentation


def _records():
    latitude, longitude = [-40.0, -41.0], [170.0, 171.0]
    transform = {
        "name": "raw", "package": "SpectralEarth", "package_version": "test-fixture-v1",
        "filters": ["identity"], "decomposition_levels": 0, "boundary_mode": "none",
        "coefficient_packing": "BCHW identity", "transform_normalisation": "none",
        "shift_invariant": True, "directional": False,
    }
    normalisation = {
        "variables": ["t", "q"], "mean": [0.0, 0.0], "std": [1.0, 1.0],
        "n_values_per_variable": 128, "fitted_split": "train",
        "first_timestamp": "1969-01-01T00:00:00",
        "last_timestamp": "1969-03-01T00:00:00", "source_content_hash": "fixture",
        "computation_method": "eager float64 population moments", "chunk_frames": None,
        "method": "eager float64 population moments", "ddof": 0,
    }
    normalisation_sha = provenance_sha256(normalisation)
    normalisation["artifact_hash"] = normalisation_sha
    protocol_record = {
        "schema": "motivating-forecast-protocol/v1", "protocol_revision": 1,
        "experiment_id": "runtime-binding-fixture",
        "evidence_reference": "tests/fixtures/runtime-binding.json@v1",
        "evidence_sha256": "a" * 64,
        "domain": {
            "name": "two-by-two fixture", "latitude_bounds": [-41.0, -40.0],
            "longitude_bounds": [170.0, 171.0], "grid_shape": [2, 2],
            "latitude_spacing_degrees": 1.0, "longitude_spacing_degrees": 1.0,
            "longitude_convention": "0..360", "latitude_order": "descending",
            "grid_coordinates_sha256": provenance_sha256(
                {"latitude": latitude, "longitude": longitude}),
        },
        "data": {
            "source_dataset": "ERA5 fixture", "source_version": "fixture-v1",
            "variables": ["t", "q"], "pressure_levels_hpa": [850], "cadence_hours": 6.0,
        },
        "timeline": {"history_frames": 2, "lead_frames": [1, 2],
                     "lead_durations_hours": [6.0, 12.0]},
        "transform": transform,
        "normalisation": {
            "method": "per-variable population z-score",
            "scope": ("each variable over all training times and spatial grid points at the "
                      "selected pressure level"),
            "fitted_split": "train", "ddof": 0,
            "statistics_artifact_sha256": normalisation_sha,
        },
        "splits": {
            "train_start": "1969-01-01T00:00:00+00:00",
            "train_end": "1969-03-01T00:00:00+00:00",
            "validation_start": "1969-06-01T00:00:00+00:00",
            "validation_end": "1969-09-01T00:00:00+00:00",
            "test_start": "1970-01-01T00:00:00+00:00",
            "test_end": "1970-01-03T18:00:00+00:00", "embargo_frames": 2,
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW", "package_version": "fixture-v1",
            "kwargs": {"lr": 0.001}, "scheduler_class_path": None,
            "scheduler_kwargs": {}, "training_steps": 10, "batch_size": 3, "seed": 22,
            "gradient_clip_norm": None,
        },
        "rollout": {
            "mode": "autoregressive", "history_usage": "final_frame",
            "feedback": "predicted_state", "training_strategy": "one-step fixture",
            "loss_lead_frames": [1],
        },
        "model": {
            "class_path": "src.forecasting.adapter.TinyResidualCoefficientModel",
            "model_version": "fixture-model-v1", "config": {"channels": 2},
            "parameter_count": 6, "trainable_parameter_count": 6,
        },
    }
    dataset_record = {
        "schema": "regional_forecast_dataset/v2",
        "source": {"dataset_identity": {"name": "ERA5 fixture", "version": "fixture-v1"}},
        "config": {"variables": ["t", "q"], "level_hpa": 850, "history_frames": 2,
                   "lead_frames": [1, 2], "embargo_frames": 2},
        "level_hpa": 850,
        "cadence": {"is_regular": True, "observed_cadence_hours": 6.0},
        "grid": {"latitude": latitude, "longitude": longitude, "shape": [2, 2],
                 "sha256": protocol_record["domain"]["grid_coordinates_sha256"],
                 "longitude_convention": "0..360"},
        "timestamps": {"timezone": "UTC"},
        "normalisation_contract": protocol_record["normalisation"],
        "normalisation": normalisation,
        "temporal_split": {
            "train": {"first_timestamp": "1969-01-01T00:00:00",
                      "last_timestamp": "1969-03-01T00:00:00"},
            "val": {"first_timestamp": "1969-06-01T00:00:00",
                    "last_timestamp": "1969-09-01T00:00:00"},
            "test": {"first_timestamp": "1970-01-01T00:00:00",
                     "last_timestamp": "1970-01-03T18:00:00"},
        },
    }
    return MotivatingExperimentProtocol.from_mapping(protocol_record), dataset_record, transform


def _bound_artifact(tmp_path):
    protocol, dataset_record, transform = _records()
    dataset_binding = bind_dataset_to_protocol(protocol, dataset_record)
    training = protocol_training_provenance(protocol, dataset_binding)
    artifact = save_laboratory_artifact(
        tmp_path / "bound", TinyResidualCoefficientModel(2),
        model_config={"channels": 2}, representation_config=transform,
        training_provenance=training)
    binding = bind_artifact_to_protocol(protocol, dataset_binding, artifact)
    return protocol, dataset_record, transform, artifact, binding


def test_exact_dataset_and_artifact_create_one_content_addressed_binding(tmp_path):
    protocol, dataset_record, _transform, artifact, binding = _bound_artifact(tmp_path)
    assert binding.protocol_sha256 == protocol.fingerprint()
    assert binding.dataset_provenance_sha256 == provenance_sha256(dataset_record)
    assert binding.artifact_checkpoint_sha256 == artifact.checkpoint_sha256
    assert len(binding.binding_sha256) == 64
    assert "skill" in binding.claim_boundary
    assert ExperimentProtocolBinding.from_provenance(binding.to_provenance()) == binding
    tampered_binding = binding.to_provenance()
    tampered_binding["lead_frames"] = [1]
    with pytest.raises(ForecastContractError, match="binding SHA-256 mismatch"):
        ExperimentProtocolBinding.from_provenance(tampered_binding)

    changed_coordinate = copy.deepcopy(dataset_record)
    changed_coordinate["grid"]["latitude"][0] = -39.5
    with pytest.raises(ForecastContractError, match="does not match the recorded"):
        bind_dataset_to_protocol(protocol, changed_coordinate)
    changed_statistics = copy.deepcopy(dataset_record)
    changed_statistics["normalisation"]["mean"][0] = 99.0
    with pytest.raises(ForecastContractError, match="training statistics"):
        bind_dataset_to_protocol(protocol, changed_statistics)


@pytest.mark.parametrize("path", ["grid", "cadence", "normalisation", "split", "source"])
def test_dataset_binding_refuses_each_scientific_drift_class(path):
    protocol, dataset_record, _ = _records()
    changed = copy.deepcopy(dataset_record)
    if path == "grid":
        changed["grid"]["shape"] = [3, 2]
    elif path == "cadence":
        changed["cadence"]["observed_cadence_hours"] = 3.0
    elif path == "normalisation":
        changed["normalisation_contract"]["fitted_split"] = "all"
    elif path == "split":
        changed["temporal_split"]["test"]["first_timestamp"] = "1970-01-02T00:00:00"
    else:
        changed["source"]["dataset_identity"]["version"] = "replacement"
    with pytest.raises(ForecastContractError, match="does not conform"):
        bind_dataset_to_protocol(protocol, changed)


def test_artifact_binding_refuses_unbound_training_and_semantic_drift(tmp_path):
    protocol, dataset_record, transform = _records()
    dataset_binding = bind_dataset_to_protocol(protocol, dataset_record)
    artifact = save_laboratory_artifact(
        tmp_path / "unbound", TinyResidualCoefficientModel(2),
        model_config={"channels": 2}, representation_config=transform,
        training_provenance={"seed": 22})
    with pytest.raises(ForecastContractError, match="training_provenance.protocol_sha256"):
        bind_artifact_to_protocol(protocol, dataset_binding, artifact)


def test_bound_evaluation_records_identity_and_refuses_cross_run_mixing(tmp_path):
    _protocol, dataset_record, transform, _artifact, binding = _bound_artifact(tmp_path)
    forecaster, _ = load_laboratory_forecaster(
        tmp_path / "bound", TinyResidualCoefficientModel(2), RawRepresentation(),
        expected_model_config={"channels": 2}, expected_representation_config=transform)
    result = evaluate_against_persistence(
        forecaster, [_forecast_batch()], variables=("t", "q"), lead_frames=(1, 2),
        split="test", dataset_provenance=dataset_record, experiment_binding=binding)
    assert result.schema == "forecast-evaluation/v3"
    assert result.experiment_binding["binding_sha256"] == binding.binding_sha256

    changed = copy.deepcopy(dataset_record)
    changed["source"]["dataset_identity"]["version"] = "another-run"
    with pytest.raises(ForecastContractError, match="bound dataset"):
        evaluate_against_persistence(
            forecaster, [_forecast_batch()], variables=("t", "q"), lead_frames=(1, 2),
            split="test", dataset_provenance=changed, experiment_binding=binding)
    outside = _forecast_batch()
    offset = 366 * 24 * 3_600_000_000_000
    outside["input_times_ns"] += offset
    outside["target_times_ns"] += offset
    with pytest.raises(ForecastContractError, match="outside the bound test split"):
        evaluate_against_persistence(
            forecaster, [outside], variables=("t", "q"), lead_frames=(1, 2), split="test",
            dataset_provenance=dataset_record, experiment_binding=binding)
