"""Acceptance tests for the T5.3a forecasting integration seam."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_layer.regional_forecast import (
    NormalisationArtifact,
    RegionalForecastConfig,
    RegionalForecastDataset,
)
from src.forecasting.adapter import (
    ForecastContractError,
    ForecasterAdapter,
    PersistenceForecaster,
    TinyResidualCoefficientModel,
    run_tiny_deterministic_step,
)
from src.transform_engine.training import HaarRepresentation, RawRepresentation


def _forecast_batch():
    config = RegionalForecastConfig(
        variables=("t", "q"), history_frames=2, lead_frames=(1, 2), embargo_frames=2)
    values = torch.linspace(-1.0, 1.0, 12 * 2 * 8 * 8, dtype=torch.float32).reshape(12, 2, 8, 8)
    normalisation = NormalisationArtifact(
        variables=("t", "q"), mean=(0.0, 0.0), std=(1.0, 1.0),
        n_values_per_variable=12 * 8 * 8, fitted_split="train",
        first_timestamp="2020-01-01T00:00:00", last_timestamp="2020-01-03T18:00:00",
        source_content_hash="fixture-content-hash")
    dataset = RegionalForecastDataset(
        values=values, times_ns=(torch.arange(12, dtype=torch.int64)
                                * 6 * 3_600_000_000_000),
        frame_indices=tuple(range(12)), split="train", config=config,
        normalisation=normalisation, provenance={"fixture": "T5.3a"})
    return next(iter(DataLoader(dataset, batch_size=3, shuffle=False)))


class WrongShapeModel(nn.Module):
    def forward(self, values):
        return values[:, :-1]


class NonFiniteModel(nn.Module):
    def forward(self, values):
        return values * torch.tensor(float("nan"), device=values.device)


def test_persistence_is_an_exact_parameter_free_physical_baseline():
    batch = _forecast_batch()
    baseline = PersistenceForecaster()
    prediction = baseline.predict(batch["inputs"], lead_count=2)
    expected = batch["inputs"][:, -1:].repeat(1, 2, 1, 1, 1)
    assert torch.equal(prediction, expected)
    assert tuple(prediction.shape) == tuple(batch["targets"].shape)
    assert sum(parameter.numel() for parameter in baseline.parameters()) == 0
    record = baseline.to_provenance()
    assert record["kind"] == "physical_space_baseline"
    assert record["parameter_count"] == 0


def test_adapter_runs_representation_model_inverse_rollout_and_backward():
    batch = _forecast_batch()
    representation = HaarRepresentation(levels=1)
    encoded_channels = representation(batch["inputs"][:, -1]).values.shape[1]
    model = TinyResidualCoefficientModel(encoded_channels)
    adapter = ForecasterAdapter(model, representation, model_name="test_model")
    prediction = adapter.predict(batch["inputs"], lead_count=batch["targets"].shape[1])
    loss = torch.nn.functional.mse_loss(prediction, batch["targets"])
    loss.backward()
    assert tuple(prediction.shape) == tuple(batch["targets"].shape)
    assert loss.item() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    record = adapter.to_provenance()
    assert record["representation"] == "haar"
    assert record["history_policy"] == "final frame only"
    assert record["rollout_policy"].startswith("autoregressive")
    assert record["trainable_parameter_count"] > 0


def test_fixed_dataset_fixture_reproduces_tiny_run_exactly_and_bounds_its_claim():
    batch = _forecast_batch()
    first = run_tiny_deterministic_step(batch, HaarRepresentation(levels=1), seed=7103)
    second = run_tiny_deterministic_step(batch, HaarRepresentation(levels=1), seed=7103)
    assert first == second
    assert first.prediction_shape == tuple(batch["targets"].shape)
    assert first.gradient_l2 > 0
    assert len(first.prediction_sha256) == len(first.gradient_sha256) == 32
    assert "not trained forecast skill" in first.claim_boundary
    assert first.to_provenance()["input_shape"] == list(batch["inputs"].shape)


def test_adapter_refuses_ambiguous_or_silent_contract_breaks():
    batch = _forecast_batch()
    with pytest.raises(ForecastContractError, match="positive integer"):
        PersistenceForecaster().predict(batch["inputs"], lead_count=0)
    with pytest.raises(ForecastContractError, match="shape"):
        PersistenceForecaster().predict(batch["inputs"][:, -1], lead_count=1)
    with pytest.raises(ForecastContractError, match="output shape"):
        ForecasterAdapter(WrongShapeModel(), RawRepresentation()).predict(batch["inputs"])
    with pytest.raises(ForecastContractError, match="non-finite coefficients"):
        ForecasterAdapter(NonFiniteModel(), RawRepresentation()).predict(batch["inputs"])
    broken = dict(batch)
    broken["targets"] = batch["targets"][:, :, :-1]
    with pytest.raises(ForecastContractError, match="incompatible"):
        run_tiny_deterministic_step(broken)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm accelerator not available")
def test_adapter_prediction_and_gradient_agree_on_vendor_neutral_accelerator():
    batch = _forecast_batch()
    torch.manual_seed(31415)
    cpu_model = TinyResidualCoefficientModel(2)
    accelerator_model = TinyResidualCoefficientModel(2).to("cuda")
    accelerator_model.load_state_dict(cpu_model.state_dict())
    cpu_adapter = ForecasterAdapter(cpu_model, HaarRepresentation(levels=1))
    accelerator_adapter = ForecasterAdapter(
        accelerator_model, HaarRepresentation(levels=1).to("cuda"))

    cpu_prediction = cpu_adapter.predict(batch["inputs"], lead_count=2)
    cpu_loss = torch.nn.functional.mse_loss(cpu_prediction, batch["targets"])
    cpu_loss.backward()
    accelerator_prediction = accelerator_adapter.predict(
        batch["inputs"].to("cuda"), lead_count=2)
    accelerator_loss = torch.nn.functional.mse_loss(
        accelerator_prediction, batch["targets"].to("cuda"))
    accelerator_loss.backward()

    assert torch.allclose(cpu_prediction, accelerator_prediction.cpu(), rtol=2e-5, atol=2e-5)
    for cpu_parameter, accelerator_parameter in zip(
            cpu_model.parameters(), accelerator_model.parameters()):
        assert torch.allclose(cpu_parameter.grad, accelerator_parameter.grad.cpu(),
                              rtol=2e-5, atol=2e-5)
