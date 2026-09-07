"""Acceptance tests for the strict T5.0a motivating-experiment protocol."""

from __future__ import annotations

import json

import pytest

from src.forecasting import (
    ForecastContractError,
    MotivatingExperimentProtocol,
    load_experiment_protocol,
    save_experiment_protocol,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def complete_protocol():
    return {
        "schema": "motivating-forecast-protocol/v1",
        "protocol_revision": 1,
        "experiment_id": "external-wavelet-forecast-replication",
        "evidence_reference": "laboratory/configs/nz-wavelet.yaml@4d3c2b1",
        "evidence_sha256": SHA_A,
        "domain": {
            "name": "New Zealand limited area",
            "latitude_bounds": [-50.0, -20.25],
            "longitude_bounds": [160.0, 179.75],
            "grid_shape": [120, 80],
            "latitude_spacing_degrees": 0.25,
            "longitude_spacing_degrees": 0.25,
            "longitude_convention": "0..360",
            "latitude_order": "descending",
            "grid_coordinates_sha256": SHA_B,
        },
        "data": {
            "source_dataset": "ERA5 pressure-level reanalysis",
            "source_version": "CDS dataset revision recorded in acquisition manifest",
            "variables": ["t", "q", "u", "v", "z"],
            "pressure_levels_hpa": [850],
            "cadence_hours": 6.0,
        },
        "timeline": {
            "history_frames": 2,
            "lead_frames": [1, 2, 4],
            "lead_durations_hours": [6.0, 12.0, 24.0],
        },
        "transform": {
            "name": "DTCWT",
            "package": "pytorch_wavelets",
            "package_version": "1.3.0",
            "filters": ["near_sym_a", "qshift_a"],
            "decomposition_levels": 3,
            "boundary_mode": "symmetric",
            "coefficient_packing": "yl plus ordered complex yh tuple",
            "transform_normalisation": "package default analysis/synthesis scaling",
            "shift_invariant": True,
            "directional": True,
        },
        "normalisation": {
            "method": "per-variable population z-score",
            "scope": "all training times and spatial points at 850 hPa",
            "fitted_split": "train",
            "ddof": 0,
            "statistics_artifact_sha256": SHA_C,
        },
        "splits": {
            "train_start": "2010-01-01T00:00:00+00:00",
            "train_end": "2017-12-31T18:00:00+00:00",
            "validation_start": "2018-01-02T00:00:00+00:00",
            "validation_end": "2018-12-31T18:00:00+00:00",
            "test_start": "2019-01-02T00:00:00+00:00",
            "test_end": "2019-12-31T18:00:00+00:00",
            "embargo_frames": 4,
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "package_version": "2.8.0",
            "kwargs": {"lr": 0.0003, "weight_decay": 0.01},
            "scheduler_class_path": None,
            "scheduler_kwargs": {},
            "training_steps": 1000,
            "batch_size": 16,
            "seed": 7103,
            "gradient_clip_norm": None,
        },
        "rollout": {
            "mode": "autoregressive",
            "history_usage": "final_frame",
            "feedback": "predicted_state",
            "training_strategy": "one-step teacher-forced training; recursive evaluation",
            "loss_lead_frames": [1],
        },
        "model": {
            "class_path": "laboratory.models.WaveletForecaster",
            "model_version": "git:4d3c2b1",
            "config": {"hidden_channels": 32, "activation": "gelu"},
            "parameter_count": 412000,
            "trainable_parameter_count": 412000,
        },
    }


def test_complete_protocol_is_canonical_versioned_and_stably_hashable(tmp_path):
    protocol = MotivatingExperimentProtocol.from_mapping(complete_protocol())
    same = MotivatingExperimentProtocol.from_mapping(
        json.loads(json.dumps(complete_protocol(), sort_keys=True)))

    assert protocol.schema == "motivating-forecast-protocol/v1"
    assert protocol == same
    assert protocol.fingerprint() == same.fingerprint()
    assert len(protocol.fingerprint()) == 64
    assert hash(protocol) == hash(same)
    assert protocol.to_provenance()["protocol_sha256"] == protocol.fingerprint()

    path = tmp_path / "experiment-protocol.json"
    assert save_experiment_protocol(path, protocol) == protocol.fingerprint()
    assert load_experiment_protocol(path) == protocol
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_experiment_protocol(path, protocol)


@pytest.mark.parametrize("section,field", [
    ("domain", "grid_shape"),
    ("data", "cadence_hours"),
    ("timeline", "history_frames"),
    ("transform", "boundary_mode"),
    ("normalisation", "statistics_artifact_sha256"),
    ("splits", "validation_start"),
    ("optimizer", "kwargs"),
    ("rollout", "history_usage"),
    ("model", "parameter_count"),
])
def test_every_scientific_section_refuses_missing_information(section, field):
    record = complete_protocol()
    del record[section][field]
    with pytest.raises(ForecastContractError, match="missing %s" % field):
        MotivatingExperimentProtocol.from_mapping(record)


def test_protocol_refuses_unknown_fields_placeholders_and_unsupported_versions():
    extra = complete_protocol()
    extra["transform"]["assumed_default"] = True
    with pytest.raises(ForecastContractError, match="unknown assumed_default"):
        MotivatingExperimentProtocol.from_mapping(extra)

    placeholder = complete_protocol()
    placeholder["transform"]["boundary_mode"] = "TBD"
    with pytest.raises(ForecastContractError, match="placeholders"):
        MotivatingExperimentProtocol.from_mapping(placeholder)

    version = complete_protocol()
    version["schema"] = "motivating-forecast-protocol/v2"
    with pytest.raises(ForecastContractError, match="unsupported"):
        MotivatingExperimentProtocol.from_mapping(version)


def test_cross_section_temporal_and_rollout_contradictions_are_refused():
    duration = complete_protocol()
    duration["timeline"]["lead_durations_hours"][2] = 18.0
    with pytest.raises(ForecastContractError, match="cadence_hours"):
        MotivatingExperimentProtocol.from_mapping(duration)

    embargo = complete_protocol()
    embargo["splits"]["embargo_frames"] = 2
    with pytest.raises(ForecastContractError, match="at least the longest lead"):
        MotivatingExperimentProtocol.from_mapping(embargo)

    rollout = complete_protocol()
    rollout["rollout"]["feedback"] = "none"
    with pytest.raises(ForecastContractError, match="requires predicted_state"):
        MotivatingExperimentProtocol.from_mapping(rollout)


def test_protocol_hash_detects_material_drift_and_envelope_tampering(tmp_path):
    protocol = MotivatingExperimentProtocol.from_mapping(complete_protocol())
    changed_record = complete_protocol()
    changed_record["optimizer"]["kwargs"]["lr"] = 0.001
    changed = MotivatingExperimentProtocol.from_mapping(changed_record)
    assert changed.fingerprint() != protocol.fingerprint()

    path = tmp_path / "experiment-protocol.json"
    save_experiment_protocol(path, protocol)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["protocol"]["model"]["parameter_count"] += 1
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ForecastContractError, match="SHA-256 mismatch"):
        load_experiment_protocol(path)


def test_protocol_freezes_nested_free_form_configuration():
    record = complete_protocol()
    protocol = MotivatingExperimentProtocol.from_mapping(record)
    original_hash = protocol.fingerprint()
    record["optimizer"]["kwargs"]["lr"] = 99.0
    record["model"]["config"]["hidden_channels"] = 999

    assert protocol.optimizer.kwargs["lr"] == 0.0003
    assert protocol.model.config["hidden_channels"] == 32
    assert protocol.fingerprint() == original_hash
    with pytest.raises(TypeError):
        protocol.optimizer.kwargs["lr"] = 0.1


def test_protocol_requires_real_evidence_identity_not_poster_placeholders():
    missing_hash = complete_protocol()
    missing_hash["evidence_sha256"] = "poster estimate"
    with pytest.raises(ForecastContractError, match="evidence_sha256"):
        MotivatingExperimentProtocol.from_mapping(missing_hash)

    missing_reference = complete_protocol()
    missing_reference["evidence_reference"] = "unknown"
    with pytest.raises(ForecastContractError, match="placeholders"):
        MotivatingExperimentProtocol.from_mapping(missing_reference)
