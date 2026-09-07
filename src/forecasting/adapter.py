"""Minimal, explicit forecasting bridge for the Phase 5 laboratory workflow.

This module does not contain a weather model.  It defines the narrow contract by which an external
existing model can become the downstream judge of an accepted training representation:

``dataset history -> representation -> step model -> inverse -> physical prediction``.

The adapter is deliberately autoregressive and Markovian in T5.3a: the wrapped step model maps
the latest represented state to the next represented state.  That matches the motivating
``(B,C,H,W)`` inner loop.  A future history-aware adapter can implement the same ``Forecaster``
contract without changing the dataset, persistence baseline, loss or evaluation code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from src.transform_engine.training import RepresentationModule, make_representation


class ForecastContractError(ValueError):
    """A model, batch or prediction violates the training-facing forecast contract."""


def _validate_history(history: torch.Tensor) -> None:
    if not isinstance(history, torch.Tensor):
        raise ForecastContractError("history must be a torch.Tensor")
    if history.ndim != 5:
        raise ForecastContractError(
            "history must have shape (batch, history, channel, height, width), got %r"
            % (tuple(history.shape),))
    if any(int(size) < 1 for size in history.shape):
        raise ForecastContractError("history cannot contain an empty axis")
    if history.dtype not in (torch.float32, torch.float64):
        raise ForecastContractError("history dtype must be float32 or float64")
    if not bool(torch.isfinite(history).all()):
        raise ForecastContractError("history contains non-finite values; no implicit imputation is allowed")


def _validate_lead_count(lead_count: int) -> int:
    if isinstance(lead_count, bool) or int(lead_count) != lead_count or lead_count < 1:
        raise ForecastContractError("lead_count must be a positive integer")
    return int(lead_count)


class Forecaster(nn.Module, ABC):
    """Common physical-space forecast contract over dataset-shaped histories."""

    @abstractmethod
    def predict(self, history: torch.Tensor, lead_count: int = 1) -> torch.Tensor:
        """Return ``(B,lead,C,H,W)`` predictions in normalised physical space."""

    def forward(self, history: torch.Tensor, lead_count: int = 1) -> torch.Tensor:
        return self.predict(history, lead_count)

    @abstractmethod
    def to_provenance(self) -> Mapping[str, Any]:
        """Return the model/adapter facts needed to interpret a run."""


class PersistenceForecaster(Forecaster):
    """Exact no-change baseline: repeat the final observed physical state at every lead."""

    def predict(self, history: torch.Tensor, lead_count: int = 1) -> torch.Tensor:
        _validate_history(history)
        leads = _validate_lead_count(lead_count)
        return history[:, -1:].expand(-1, leads, -1, -1, -1).clone()

    def to_provenance(self) -> Mapping[str, Any]:
        return {
            "schema": "forecaster/v1",
            "name": "persistence",
            "kind": "physical_space_baseline",
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "definition": "repeat the final observed normalised physical state at every lead",
        }


class ForecasterAdapter(Forecaster):
    """Wrap a one-step coefficient model with differentiable analysis and synthesis.

    The wrapped model must map an encoded 4D tensor to a tensor of exactly the same shape,
    dtype and device.  Multi-lead predictions are autoregressive: each reconstructed physical
    prediction becomes the next step's state.  Earlier history frames remain in the public
    contract for interchangeability with the persistence baseline, but T5.3a's step model uses
    only the final frame; provenance states that limitation explicitly.
    """

    def __init__(self, step_model: nn.Module, representation: RepresentationModule,
                 model_name: Optional[str] = None,
                 model_artifact: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        if not isinstance(step_model, nn.Module):
            raise ForecastContractError("step_model must be a torch.nn.Module")
        if not isinstance(representation, RepresentationModule):
            raise ForecastContractError("representation must be an accepted RepresentationModule")
        self.step_model = step_model
        self.representation = representation
        self.model_name = model_name or type(step_model).__name__
        self.model_artifact = None if model_artifact is None else dict(model_artifact)

    def _step(self, state: torch.Tensor) -> torch.Tensor:
        encoded = self.representation(state)
        predicted_values = self.step_model(encoded.values)
        if not isinstance(predicted_values, torch.Tensor):
            raise ForecastContractError("step_model must return a torch.Tensor")
        if tuple(predicted_values.shape) != tuple(encoded.values.shape):
            raise ForecastContractError(
                "step_model output shape %r does not match encoded state shape %r"
                % (tuple(predicted_values.shape), tuple(encoded.values.shape)))
        if predicted_values.dtype != encoded.values.dtype:
            raise ForecastContractError(
                "step_model changed dtype from %s to %s" %
                (encoded.values.dtype, predicted_values.dtype))
        if predicted_values.device != encoded.values.device:
            raise ForecastContractError(
                "step_model changed device from %s to %s" %
                (encoded.values.device, predicted_values.device))
        if not bool(torch.isfinite(predicted_values).all()):
            raise ForecastContractError("step_model produced non-finite coefficients")
        prediction = self.representation.inverse(encoded.with_values(predicted_values))
        if tuple(prediction.shape) != tuple(state.shape):
            raise ForecastContractError(
                "inverse prediction shape %r does not match physical state shape %r"
                % (tuple(prediction.shape), tuple(state.shape)))
        return prediction

    def predict(self, history: torch.Tensor, lead_count: int = 1) -> torch.Tensor:
        _validate_history(history)
        leads = _validate_lead_count(lead_count)
        state = history[:, -1]
        predictions = []
        for _ in range(leads):
            state = self._step(state)
            predictions.append(state)
        return torch.stack(predictions, dim=1)

    def to_provenance(self) -> Mapping[str, Any]:
        parameters = tuple(self.step_model.parameters())
        record = {
            "schema": "forecaster/v1",
            "name": self.model_name,
            "kind": "represented_autoregressive_step_adapter",
            "representation": self.representation.name,
            "history_policy": "final frame only",
            "rollout_policy": "autoregressive physical-state feedback",
            "coefficient_contract": "model output shape/dtype/device exactly match encoded input",
            "parameter_count": int(sum(p.numel() for p in parameters)),
            "trainable_parameter_count": int(sum(p.numel() for p in parameters if p.requires_grad)),
        }
        if self.model_artifact is not None:
            record["model_artifact"] = dict(self.model_artifact)
        else:
            record["model_artifact"] = None
        return record


class TinyResidualCoefficientModel(nn.Module):
    """One 1x1 residual convolution used only as an importable integration fixture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if isinstance(channels, bool) or int(channels) != channels or channels < 1:
            raise ForecastContractError("channels must be a positive integer")
        self.correction = nn.Conv2d(int(channels), int(channels), kernel_size=1)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients + self.correction(coefficients)


@dataclass(frozen=True)
class TinyRunEvidence:
    """Reproducibility evidence from one plumbing test—not a forecast evaluation."""

    seed: int
    representation: str
    device: str
    dtype: str
    input_shape: Tuple[int, ...]
    target_shape: Tuple[int, ...]
    prediction_shape: Tuple[int, ...]
    parameter_count: int
    loss: float
    gradient_l2: float
    prediction_sha256: str
    gradient_sha256: str
    claim_boundary: str
    forecaster: Mapping[str, Any]

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["input_shape"] = list(self.input_shape)
        record["target_shape"] = list(self.target_shape)
        record["prediction_shape"] = list(self.prediction_shape)
        return record


def _tensor_hash(tensor: torch.Tensor) -> str:
    values = tensor.detach().to(device="cpu").contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()[:32]


def run_tiny_deterministic_step(
    batch: Mapping[str, torch.Tensor],
    representation: Union[str, RepresentationModule] = "raw",
    seed: int = 20260822,
) -> TinyRunEvidence:
    """Run dataset -> representation -> model -> inverse -> MSE -> backward once.

    This helper exists so installation checks and examples exercise the real seam instead of a
    parallel toy path.  It intentionally performs no optimiser step and makes no skill claim.
    Determinism is asserted for a fixed CPU fixture by acceptance tests; accelerator bitwise
    reproducibility remains dependent on the selected PyTorch kernels and runtime.
    """
    if "inputs" not in batch or "targets" not in batch:
        raise ForecastContractError("batch must contain 'inputs' and 'targets' tensors")
    inputs, targets = batch["inputs"], batch["targets"]
    _validate_history(inputs)
    if not isinstance(targets, torch.Tensor) or targets.ndim != 5:
        raise ForecastContractError("targets must have shape (batch, lead, channel, height, width)")
    expected = (inputs.shape[0], targets.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
    if tuple(targets.shape) != tuple(expected):
        raise ForecastContractError(
            "targets shape %r is incompatible with history shape %r" %
            (tuple(targets.shape), tuple(inputs.shape)))
    if targets.dtype != inputs.dtype or targets.device != inputs.device:
        raise ForecastContractError("inputs and targets must share dtype and device")
    if not bool(torch.isfinite(targets).all()):
        raise ForecastContractError("targets contain non-finite values")

    module = make_representation(representation) if isinstance(representation, str) else representation
    if not isinstance(module, RepresentationModule):
        raise ForecastContractError("representation must be a name or RepresentationModule")
    module = module.to(device=inputs.device, dtype=inputs.dtype)
    with torch.no_grad():
        encoded_channels = int(module(inputs[:, -1]).values.shape[1])

    devices = []
    if inputs.device.type == "cuda":
        devices = [inputs.device.index if inputs.device.index is not None
                   else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        model = TinyResidualCoefficientModel(encoded_channels).to(
            device=inputs.device, dtype=inputs.dtype)
    adapter = ForecasterAdapter(model, module, model_name="tiny_residual_fixture")
    prediction = adapter.predict(inputs, lead_count=int(targets.shape[1]))
    loss = F.mse_loss(prediction, targets)
    loss.backward()

    gradients = [parameter.grad.reshape(-1) for parameter in model.parameters()
                 if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("tiny integration run produced no parameter gradients")
    gradient_vector = torch.cat(gradients)
    if not bool(torch.isfinite(gradient_vector).all()):
        raise RuntimeError("tiny integration run produced non-finite gradients")
    return TinyRunEvidence(
        seed=int(seed), representation=module.name, device=str(inputs.device),
        dtype=str(inputs.dtype), input_shape=tuple(inputs.shape), target_shape=tuple(targets.shape),
        prediction_shape=tuple(prediction.shape),
        parameter_count=sum(p.numel() for p in model.parameters()),
        loss=float(loss.detach().cpu()), gradient_l2=float(torch.linalg.vector_norm(gradient_vector).cpu()),
        prediction_sha256=_tensor_hash(prediction), gradient_sha256=_tensor_hash(gradient_vector),
        claim_boundary=("Deterministic tiny integration evidence only; not trained forecast skill, "
                        "not a comparison between representations, and not laboratory-model validation."),
        forecaster=adapter.to_provenance())
