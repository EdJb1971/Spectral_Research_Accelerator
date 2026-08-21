"""Streaming, persistence-relative evaluation for regional forecasters."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from src.forecasting.adapter import (
    Forecaster, ForecastContractError, PersistenceForecaster, _validate_history,
)
from src.forecasting.binding import ExperimentProtocolBinding, validate_evaluation_binding


@dataclass(frozen=True)
class ForecastEvaluation:
    """One deterministic evaluation pass; uncertainty requires repeated trained runs."""

    schema: str
    split: str
    variables: Tuple[str, ...]
    lead_frames: Tuple[int, ...]
    lead_durations_hours: Tuple[float, ...]
    sample_count: int
    batch_count: int
    device: str
    metrics: Mapping[str, Any]
    aggregate_standardized: Mapping[str, Any]
    forecaster: Mapping[str, Any]
    baseline: Mapping[str, Any]
    dataset_provenance: Mapping[str, Any]
    experiment_binding: Optional[Mapping[str, Any]]
    prediction_sha256: str
    target_sha256: str
    claim_boundary: str

    def to_provenance(self) -> Dict[str, Any]:
        record = asdict(self)
        record["variables"] = list(self.variables)
        record["lead_frames"] = list(self.lead_frames)
        record["lead_durations_hours"] = list(self.lead_durations_hours)
        return record


def _validate_labels(variables: Sequence[str], lead_frames: Sequence[int]) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    variables = tuple(str(value) for value in variables)
    leads = tuple(int(value) for value in lead_frames)
    if not variables or len(set(variables)) != len(variables):
        raise ForecastContractError("variables must be a non-empty unique sequence")
    if not leads or any(value < 1 for value in leads) or len(set(leads)) != len(leads):
        raise ForecastContractError("lead_frames must be unique positive frame offsets")
    return variables, leads


def _safe_skill(model_mse: float, baseline_mse: float) -> Optional[float]:
    if baseline_mse == 0.0:
        return None
    return 1.0 - model_mse / baseline_mse


def _utc_ns(value: Any) -> int:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ForecastContractError("dataset split provenance contains an invalid timestamp") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def evaluate_against_persistence(
    forecaster: Forecaster,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    variables: Sequence[str],
    lead_frames: Sequence[int],
    split: str,
    channel_std: Optional[Sequence[float]] = None,
    channel_units: Optional[Mapping[str, str]] = None,
    dataset_provenance: Optional[Mapping[str, Any]] = None,
    experiment_binding: Optional[ExperimentProtocolBinding] = None,
    device: Optional[torch.device] = None,
) -> ForecastEvaluation:
    """Evaluate exactly matched predictions and persistence targets in bounded memory.

    MSE skill score is ``1 - model_MSE / persistence_MSE``; higher is better and zero means
    equal squared error.  A perfect persistence denominator is reported as ``None`` because
    skill is undefined.  Cross-variable aggregation is performed only after train-standardised
    normalisation.  Physical-unit metrics are per variable and require explicit standard
    deviations; no unit is guessed.
    """
    if not isinstance(forecaster, Forecaster):
        raise ForecastContractError("forecaster must implement the Forecaster contract")
    variables, leads = _validate_labels(variables, lead_frames)
    if not split or split == "train":
        raise ForecastContractError("evaluation split must be declared and must not be 'train'")
    std = None if channel_std is None else tuple(float(value) for value in channel_std)
    if std is not None and (len(std) != len(variables) or any(not math.isfinite(v) or v <= 0 for v in std)):
        raise ForecastContractError("channel_std must contain one finite positive value per variable")
    units = dict(channel_units or {})
    if set(units) - set(variables):
        raise ForecastContractError("channel_units contains variables outside the evaluation contract")
    target_device = device or next(forecaster.parameters(), torch.empty(0)).device
    target_device = torch.device(target_device)
    dataset_record = dict(dataset_provenance or {})
    forecaster_record = dict(forecaster.to_provenance())
    if experiment_binding is not None:
        if not isinstance(experiment_binding, ExperimentProtocolBinding):
            raise ForecastContractError(
                "experiment_binding must be a validated ExperimentProtocolBinding")
        if not dataset_record:
            raise ForecastContractError(
                "a bound evaluation requires the complete bound dataset provenance")
        validate_evaluation_binding(
            experiment_binding, dataset_record, forecaster_record, variables, leads)

    shape = (len(leads), len(variables))
    squared = torch.zeros(shape, dtype=torch.float64)
    absolute = torch.zeros(shape, dtype=torch.float64)
    signed = torch.zeros(shape, dtype=torch.float64)
    baseline_squared = torch.zeros(shape, dtype=torch.float64)
    baseline_absolute = torch.zeros(shape, dtype=torch.float64)
    value_count = torch.zeros(shape, dtype=torch.int64)
    prediction_digest, target_digest = hashlib.sha256(), hashlib.sha256()
    sample_count = batch_count = 0
    lead_durations_ns: Optional[Tuple[int, ...]] = None
    baseline = PersistenceForecaster().to(target_device)
    was_training = forecaster.training
    forecaster.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if "inputs" not in batch or "targets" not in batch:
                    raise ForecastContractError("every evaluation batch needs inputs and targets")
                timing_fields = {"input_times_ns", "target_times_ns", "lead_durations_ns",
                                 "time_axis_cadence_ns"}
                if not timing_fields.issubset(batch):
                    raise ForecastContractError(
                        "every evaluation batch needs input_times_ns, target_times_ns, "
                        "lead_durations_ns and time_axis_cadence_ns; frame offsets alone do not "
                        "declare physical forecast time")
                history = batch["inputs"].to(target_device)
                targets = batch["targets"].to(target_device)
                _validate_history(history)
                expected = (history.shape[0], len(leads), len(variables), history.shape[3], history.shape[4])
                if tuple(targets.shape) != tuple(expected):
                    raise ForecastContractError(
                        "evaluation targets shape %r does not match declared contract %r"
                        % (tuple(targets.shape), expected))
                if targets.dtype != history.dtype or not bool(torch.isfinite(targets).all()):
                    raise ForecastContractError("evaluation targets must be finite and share history dtype")
                durations = torch.as_tensor(batch["lead_durations_ns"], dtype=torch.int64)
                expected_duration_shape = (history.shape[0], len(leads))
                if tuple(durations.shape) != expected_duration_shape or bool((durations <= 0).any()):
                    raise ForecastContractError(
                        "lead_durations_ns must have shape %r with positive durations"
                        % (expected_duration_shape,))
                input_times = torch.as_tensor(batch["input_times_ns"], dtype=torch.int64)
                target_times = torch.as_tensor(batch["target_times_ns"], dtype=torch.int64)
                if tuple(input_times.shape) != (history.shape[0], history.shape[1]) \
                        or tuple(target_times.shape) != expected_duration_shape:
                    raise ForecastContractError(
                        "timestamp tensor shapes do not match the evaluation history/target contract")
                if experiment_binding is not None:
                    split_key = "val" if split == "validation" else split
                    split_record = dataset_record.get("temporal_split", {}).get(split_key)
                    if not isinstance(split_record, Mapping):
                        raise ForecastContractError(
                            "bound dataset provenance has no declared %r split" % split_key)
                    first_ns = _utc_ns(split_record.get("first_timestamp"))
                    last_ns = _utc_ns(split_record.get("last_timestamp"))
                    observed_min = min(int(input_times.min()), int(target_times.min()))
                    observed_max = max(int(input_times.max()), int(target_times.max()))
                    if observed_min < first_ns or observed_max > last_ns:
                        raise ForecastContractError(
                            "evaluation batch timestamps fall outside the bound %s split" % split_key)
                derived_durations = target_times - input_times[:, -1:]
                if not torch.equal(durations, derived_durations):
                    raise ForecastContractError(
                        "lead_durations_ns does not equal target timestamp minus final input "
                        "timestamp; timing provenance is inconsistent")
                if not bool((durations == durations[0:1]).all()):
                    raise ForecastContractError(
                        "physical lead duration varies between samples; irregular cadence cannot "
                        "be reported as one frame-offset forecast lead")
                observed_durations = tuple(int(value) for value in durations[0].tolist())
                if any(right <= left for left, right in zip(observed_durations,
                                                            observed_durations[1:])):
                    raise ForecastContractError(
                        "physical lead durations must be strictly increasing with lead_frames")
                cadence_candidates = []
                for duration, lead in zip(observed_durations, leads):
                    if duration % lead:
                        raise ForecastContractError(
                            "physical lead duration is not an integer cadence multiple of its "
                            "declared lead_frames offset")
                    cadence_candidates.append(duration // lead)
                if len(set(cadence_candidates)) != 1:
                    raise ForecastContractError(
                        "lead_frames do not map to one regular physical cadence")
                axis_cadence = torch.as_tensor(batch["time_axis_cadence_ns"], dtype=torch.int64)
                if tuple(axis_cadence.shape) != (history.shape[0],) \
                        or bool((axis_cadence <= 0).any()) \
                        or not bool((axis_cadence == cadence_candidates[0]).all()):
                    raise ForecastContractError(
                        "the complete dataset time axis is irregular or does not match the "
                        "physical cadence implied by lead_frames")
                if lead_durations_ns is None:
                    lead_durations_ns = observed_durations
                elif observed_durations != lead_durations_ns:
                    raise ForecastContractError(
                        "physical lead duration changed between evaluation batches; cadence is "
                        "irregular or batches do not share one temporal contract")
                predictions = forecaster.predict(history, lead_count=len(leads))
                if tuple(predictions.shape) != tuple(targets.shape):
                    raise ForecastContractError("forecaster prediction shape does not match evaluation targets")
                if predictions.dtype != targets.dtype or predictions.device != targets.device:
                    raise ForecastContractError("predictions and targets must share dtype and device")
                if not bool(torch.isfinite(predictions).all()):
                    raise ForecastContractError("forecaster produced non-finite physical predictions")
                baseline_predictions = baseline.predict(history, lead_count=len(leads))
                error = (predictions - targets).to(torch.float64)
                baseline_error = (baseline_predictions - targets).to(torch.float64)
                reduce_dims = (0, 3, 4)
                squared += error.square().sum(dim=reduce_dims).cpu()
                absolute += error.abs().sum(dim=reduce_dims).cpu()
                signed += error.sum(dim=reduce_dims).cpu()
                baseline_squared += baseline_error.square().sum(dim=reduce_dims).cpu()
                baseline_absolute += baseline_error.abs().sum(dim=reduce_dims).cpu()
                count = int(history.shape[0] * history.shape[3] * history.shape[4])
                value_count += count
                prediction_digest.update(predictions.detach().cpu().contiguous().numpy().tobytes())
                target_digest.update(targets.detach().cpu().contiguous().numpy().tobytes())
                sample_count += int(history.shape[0])
                batch_count += 1
    finally:
        forecaster.train(was_training)
    if batch_count == 0:
        raise ForecastContractError("evaluation received no batches")
    assert lead_durations_ns is not None
    lead_durations_hours = tuple(value / 3_600_000_000_000 for value in lead_durations_ns)

    metrics: Dict[str, Any] = {}
    for lead_index, lead in enumerate(leads):
        lead_record: Dict[str, Any] = {}
        for channel, variable in enumerate(variables):
            count = int(value_count[lead_index, channel])
            mse = float(squared[lead_index, channel] / count)
            mae = float(absolute[lead_index, channel] / count)
            bias = float(signed[lead_index, channel] / count)
            baseline_mse = float(baseline_squared[lead_index, channel] / count)
            baseline_mae = float(baseline_absolute[lead_index, channel] / count)
            record: Dict[str, Any] = {
                "lead_frames": lead,
                "lead_duration_hours": lead_durations_hours[lead_index],
                "standardized_rmse": math.sqrt(mse),
                "standardized_mae": mae,
                "standardized_bias": bias,
                "persistence_standardized_rmse": math.sqrt(baseline_mse),
                "persistence_standardized_mae": baseline_mae,
                "mse_skill_score_vs_persistence": _safe_skill(mse, baseline_mse),
                "skill_definition": "1 - model_MSE / persistence_MSE; higher is better",
                "value_count": count,
            }
            if std is not None:
                record.update({
                    "physical_rmse": math.sqrt(mse) * std[channel],
                    "physical_mae": mae * std[channel],
                    "physical_bias": bias * std[channel],
                    "physical_unit": units.get(variable, "original data unit (not declared)"),
                })
            lead_record[variable] = record
        metrics[str(lead)] = lead_record

    aggregate: Dict[str, Any] = {}
    for lead_index, lead in enumerate(leads):
        count = int(value_count[lead_index].sum())
        mse = float(squared[lead_index].sum() / count)
        baseline_mse = float(baseline_squared[lead_index].sum() / count)
        aggregate[str(lead)] = {
            "standardized_rmse": math.sqrt(mse),
            "persistence_standardized_rmse": math.sqrt(baseline_mse),
            "mse_skill_score_vs_persistence": _safe_skill(mse, baseline_mse),
            "aggregation": "equal weight per standardized grid-cell value",
            "value_count": count,
        }
    return ForecastEvaluation(
        schema=("forecast-evaluation/v3" if experiment_binding is not None
                else "forecast-evaluation/v2"),
        split=split, variables=variables, lead_frames=leads,
        lead_durations_hours=lead_durations_hours,
        sample_count=sample_count, batch_count=batch_count, device=str(target_device),
        metrics=metrics, aggregate_standardized=aggregate,
        forecaster=forecaster_record, baseline=baseline.to_provenance(),
        dataset_provenance=dataset_record,
        experiment_binding=(None if experiment_binding is None
                            else experiment_binding.to_provenance()),
        prediction_sha256=prediction_digest.hexdigest(), target_sha256=target_digest.hexdigest(),
        claim_boundary=("Single-checkpoint deterministic evaluation against persistence; no "
                        "uncertainty estimate, multiple-seed inference, significance test, or "
                        "scientific skill claim is implied."),
    )
