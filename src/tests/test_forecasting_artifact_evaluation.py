"""Acceptance tests for T5.3b model artefacts and persistence-relative evaluation."""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from src.forecasting import (
    ForecasterAdapter,
    ForecastContractError,
    TinyResidualCoefficientModel,
    evaluate_against_persistence,
    load_laboratory_artifact,
    load_laboratory_forecaster,
    save_laboratory_artifact,
)
from src.tests.test_forecasting_adapter import _forecast_batch
from src.transform_engine.training import RawRepresentation


MODEL_CONFIG = {"architecture": "tiny-residual-fixture", "channels": 2}
REPRESENTATION_CONFIG = {"name": "raw"}
TRAINING_PROVENANCE = {
    "dataset_contract_hash": "fixture-dataset-contract",
    "split": "train",
    "seed": 7103,
    "optimizer": {"name": "not-run-fixture"},
}


class ConstantIncrement(nn.Module):
    def __init__(self, channels: int, increment: float) -> None:
        super().__init__()
        self.increment = nn.Parameter(torch.full((1, channels, 1, 1), increment))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.increment


def test_laboratory_artifact_round_trip_restores_exact_weights_and_identity(tmp_path):
    torch.manual_seed(22)
    original = TinyResidualCoefficientModel(2)
    artifact = save_laboratory_artifact(
        tmp_path / "run-a", original, model_config=MODEL_CONFIG,
        representation_config=REPRESENTATION_CONFIG,
        training_provenance=TRAINING_PROVENANCE, model_name="external-model-fixture")
    restored = TinyResidualCoefficientModel(2)
    loaded = load_laboratory_artifact(
        tmp_path / "run-a", restored, expected_model_config=MODEL_CONFIG,
        expected_representation_config=REPRESENTATION_CONFIG)

    assert loaded == artifact
    assert loaded.model_name == "external-model-fixture"
    assert len(loaded.checkpoint_sha256) == len(loaded.config_sha256) == 64
    assert loaded.training_provenance["dataset_contract_hash"] == "fixture-dataset-contract"
    for expected, actual in zip(original.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)

    adapter, bound_artifact = load_laboratory_forecaster(
        tmp_path / "run-a", TinyResidualCoefficientModel(2), RawRepresentation(),
        expected_model_config=MODEL_CONFIG,
        expected_representation_config=REPRESENTATION_CONFIG)
    assert bound_artifact == artifact
    assert adapter.to_provenance()["model_artifact"]["checkpoint_sha256"] == artifact.checkpoint_sha256


def test_artifact_refuses_checkpoint_tampering_before_deserialisation(tmp_path):
    root = tmp_path / "run-a"
    save_laboratory_artifact(
        root, TinyResidualCoefficientModel(2), model_config=MODEL_CONFIG,
        representation_config=REPRESENTATION_CONFIG, training_provenance=TRAINING_PROVENANCE)
    with (root / "state_dict.pt").open("ab") as handle:
        handle.write(b"replaced")
    with pytest.raises(ForecastContractError, match="SHA-256 mismatch"):
        load_laboratory_artifact(
            root, TinyResidualCoefficientModel(2), expected_model_config=MODEL_CONFIG,
            expected_representation_config=REPRESENTATION_CONFIG)


def test_artifact_refuses_config_drift_manifest_tampering_and_overwrite(tmp_path):
    root = tmp_path / "run-a"
    model = TinyResidualCoefficientModel(2)
    save_laboratory_artifact(
        root, model, model_config=MODEL_CONFIG,
        representation_config=REPRESENTATION_CONFIG, training_provenance=TRAINING_PROVENANCE)
    with pytest.raises(ForecastContractError, match="configuration does not match"):
        load_laboratory_artifact(
            root, TinyResidualCoefficientModel(2),
            expected_model_config={**MODEL_CONFIG, "channels": 3},
            expected_representation_config=REPRESENTATION_CONFIG)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_laboratory_artifact(
            root, model, model_config=MODEL_CONFIG,
            representation_config=REPRESENTATION_CONFIG,
            training_provenance=TRAINING_PROVENANCE)

    manifest_path = root / "manifest.json"
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["representation_config"]["name"] = "fft"
    manifest_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ForecastContractError, match="config hash"):
        load_laboratory_artifact(
            root, TinyResidualCoefficientModel(2), expected_model_config=MODEL_CONFIG,
            expected_representation_config=REPRESENTATION_CONFIG)


def _perfect_increment_adapter(batch):
    # The fixed fixture is linear over time, so this is an exact autoregressive forecast.
    increment = float((batch["targets"][:, 0] - batch["inputs"][:, -1]).mean())
    return ForecasterAdapter(
        ConstantIncrement(2, increment), RawRepresentation(), model_name="exact-fixture-model")


def test_evaluation_reports_exact_matched_persistence_skill_and_physical_units():
    batch = _forecast_batch()
    model = _perfect_increment_adapter(batch)
    model.train()
    result = evaluate_against_persistence(
        model, [batch], variables=("t", "q"), lead_frames=(1, 2), split="test",
        channel_std=(2.0, 0.5), channel_units={"t": "K", "q": "kg kg-1"},
        dataset_provenance={"split_contract_hash": "fixed-test-split"})

    assert result.sample_count == 3 and result.batch_count == 1
    assert result.schema == "forecast-evaluation/v2"
    assert result.lead_durations_hours == (6.0, 12.0)
    assert model.training  # evaluation restores caller state
    assert result.metrics["1"]["t"]["standardized_rmse"] < 1e-6
    assert result.metrics["2"]["q"]["standardized_rmse"] < 1e-6
    assert result.metrics["1"]["t"]["mse_skill_score_vs_persistence"] > 0.999999
    assert result.metrics["1"]["t"]["physical_unit"] == "K"
    assert result.metrics["2"]["t"]["lead_duration_hours"] == 12.0
    assert result.metrics["1"]["t"]["physical_rmse"] == pytest.approx(
        2.0 * result.metrics["1"]["t"]["standardized_rmse"])
    assert result.aggregate_standardized["1"]["aggregation"].startswith("equal weight")
    assert result.dataset_provenance["split_contract_hash"] == "fixed-test-split"
    assert "no uncertainty estimate" in result.claim_boundary
    assert len(result.prediction_sha256) == len(result.target_sha256) == 64


def test_evaluation_marks_skill_undefined_when_persistence_is_perfect():
    batch = _forecast_batch()
    stationary = dict(batch)
    stationary["targets"] = batch["inputs"][:, -1:].expand(-1, 2, -1, -1, -1).clone()
    model = ForecasterAdapter(ConstantIncrement(2, 0.1), RawRepresentation())
    result = evaluate_against_persistence(
        model, [stationary], variables=("t", "q"), lead_frames=(1, 2), split="validation")
    assert result.metrics["1"]["t"]["mse_skill_score_vs_persistence"] is None
    assert result.aggregate_standardized["2"]["mse_skill_score_vs_persistence"] is None
    assert "physical_rmse" not in result.metrics["1"]["t"]


def test_evaluation_refuses_training_empty_and_misaligned_contracts():
    batch = _forecast_batch()
    model = _perfect_increment_adapter(batch)
    with pytest.raises(ForecastContractError, match="must not be 'train'"):
        evaluate_against_persistence(
            model, [batch], variables=("t", "q"), lead_frames=(1, 2), split="train")
    with pytest.raises(ForecastContractError, match="no batches"):
        evaluate_against_persistence(
            model, [], variables=("t", "q"), lead_frames=(1, 2), split="test")
    with pytest.raises(ForecastContractError, match="declared contract"):
        evaluate_against_persistence(
            model, [batch], variables=("t",), lead_frames=(1, 2), split="test")


def test_evaluation_refuses_missing_or_variable_physical_lead_durations():
    batch = _forecast_batch()
    model = _perfect_increment_adapter(batch)
    missing = dict(batch)
    del missing["lead_durations_ns"]
    with pytest.raises(ForecastContractError, match="frame offsets alone"):
        evaluate_against_persistence(
            model, [missing], variables=("t", "q"), lead_frames=(1, 2), split="test")
    irregular = dict(batch)
    irregular["lead_durations_ns"] = batch["lead_durations_ns"].clone()
    irregular["target_times_ns"] = batch["target_times_ns"].clone()
    irregular["lead_durations_ns"][1, 0] += 3_600_000_000_000
    irregular["target_times_ns"][1, 0] += 3_600_000_000_000
    with pytest.raises(ForecastContractError, match="varies between samples"):
        evaluate_against_persistence(
            model, [irregular], variables=("t", "q"), lead_frames=(1, 2), split="test")
    inconsistent = dict(batch)
    inconsistent["lead_durations_ns"] = batch["lead_durations_ns"].clone()
    inconsistent["lead_durations_ns"][:, 1] += 3_600_000_000_000
    with pytest.raises(ForecastContractError, match="timing provenance is inconsistent"):
        evaluate_against_persistence(
            model, [inconsistent], variables=("t", "q"), lead_frames=(1, 2), split="test")
    irregular_axis = dict(batch)
    irregular_axis["time_axis_cadence_ns"] = torch.zeros_like(
        batch["time_axis_cadence_ns"])
    with pytest.raises(ForecastContractError, match="complete dataset time axis is irregular"):
        evaluate_against_persistence(
            model, [irregular_axis], variables=("t", "q"), lead_frames=(1, 2), split="test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm accelerator not available")
def test_evaluation_agrees_on_cpu_and_vendor_neutral_accelerator():
    batch = _forecast_batch()
    cpu = _perfect_increment_adapter(batch)
    accelerator = _perfect_increment_adapter(batch).to("cuda")
    accelerator.load_state_dict(cpu.state_dict())
    kwargs = {"variables": ("t", "q"), "lead_frames": (1, 2), "split": "test"}
    cpu_result = evaluate_against_persistence(cpu, [batch], device=torch.device("cpu"), **kwargs)
    accelerator_result = evaluate_against_persistence(
        accelerator, [batch], device=torch.device("cuda"), **kwargs)
    for lead in ("1", "2"):
        for variable in ("t", "q"):
            assert accelerator_result.metrics[lead][variable]["standardized_rmse"] == pytest.approx(
                cpu_result.metrics[lead][variable]["standardized_rmse"], rel=2e-5, abs=2e-5)
