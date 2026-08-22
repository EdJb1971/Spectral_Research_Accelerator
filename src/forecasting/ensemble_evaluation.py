"""Matched-truth, persistence-relative ensemble verification (roadmap T5.6c).

The evaluator consumes the lazy regional view accepted by :mod:`external_cube`, an exact
verifying-truth cube, and the corresponding analysis at forecast initialization.  It computes
physical-variable scores a bounded spatial tile at a time; it never loads a global ensemble and
never combines variables with unlike units.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import xarray as xr

from src.forecasting.adapter import ForecastContractError
from src.forecasting.external_cube import RegionalForecastCube, fcn3_variable_unit, grid_coordinates_sha256


MATCHED_TRUTH_SCHEMA = "matched-regional-truth/v1"
INITIAL_STATE_SCHEMA = "matched-regional-initial-state/v1"
ENSEMBLE_EVALUATION_SCHEMA = "matched-ensemble-evaluation/v1"
DEFAULT_EVALUATION_TILE_BYTES = 64 * 1024 * 1024


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ForecastContractError("%s must be a 64-character SHA-256" % label)
    try:
        int(value, 16)
    except ValueError:
        raise ForecastContractError("%s must be hexadecimal" % label) from None
    return value.lower()


@dataclass(frozen=True)
class EnsembleEvaluation:
    schema: str
    forecast_provenance: Mapping[str, Any]
    truth_provenance: Mapping[str, Any]
    initial_state_provenance: Mapping[str, Any]
    variables: Tuple[str, ...]
    initialization_count: int
    ensemble_members: Tuple[int, ...]
    lead_durations_hours: Tuple[float, ...]
    grid_shape: Tuple[int, int]
    grid_coordinates_sha256: str
    area_weighting: str
    metrics: Mapping[str, Any]
    forecast_values_sha256: str
    truth_values_sha256: str
    initial_state_values_sha256: str
    maximum_observed_tile_bytes: int
    maximum_allowed_tile_bytes: int
    claim_boundary: str

    def to_mapping(self) -> Dict[str, Any]:
        value = asdict(self)
        value["variables"] = list(self.variables)
        value["ensemble_members"] = list(self.ensemble_members)
        value["lead_durations_hours"] = list(self.lead_durations_hours)
        value["grid_shape"] = list(self.grid_shape)
        return value

    def fingerprint(self) -> str:
        return _canonical_hash(self.to_mapping())


def _duration_ns(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[ns]").astype("int64")
    raise ForecastContractError("lead_time must use exact timedelta64 durations")


def _validate_provenance(dataset: xr.Dataset, schema: str, label: str) -> Dict[str, Any]:
    required = {"schema", "source_sha256", "split", "write_complete"}
    missing = sorted(required - set(dataset.attrs))
    if missing:
        raise ForecastContractError("%s provenance is missing %s" % (label, missing))
    if dataset.attrs["schema"] != schema:
        raise ForecastContractError("%s schema must be %r" % (label, schema))
    if dataset.attrs["write_complete"] != "true":
        raise ForecastContractError("%s write_complete must be the string 'true'" % label)
    split = dataset.attrs["split"]
    if not isinstance(split, str) or not split.strip() or split.lower() == "train":
        raise ForecastContractError("%s split must be declared and held out from training" % label)
    source = _require_sha256(dataset.attrs["source_sha256"], "%s source_sha256" % label)
    record = {"schema": schema, "source_sha256": source, "split": split,
              "write_complete": "true"}
    builder_fields = {
        "builder_schema", "builder_sha256", "source_content_hash", "split_start", "split_end",
        "evaluation_role", "fcn3_periods",
    }
    supplied = builder_fields & set(dataset.attrs)
    if supplied and supplied != builder_fields:
        raise ForecastContractError(
            "%s builder provenance is incomplete; missing %s" %
            (label, sorted(builder_fields - supplied)))
    if supplied:
        record["builder_sha256"] = _require_sha256(
            dataset.attrs["builder_sha256"], "%s builder_sha256" % label)
        content_hash = dataset.attrs["source_content_hash"]
        if not isinstance(content_hash, str) or len(content_hash) not in (32, 64):
            raise ForecastContractError(
                "%s source_content_hash must be a 32- or 64-character digest" % label)
        try:
            int(content_hash, 16)
        except ValueError:
            raise ForecastContractError("%s source_content_hash must be hexadecimal" % label) from None
        for name in builder_fields - {"builder_sha256", "source_content_hash"}:
            value = dataset.attrs[name]
            if not isinstance(value, str) or not value.strip():
                raise ForecastContractError("%s %s must be a non-empty string" % (label, name))
            record[name] = value
        record["source_content_hash"] = content_hash.lower()
    return record


def _same_coord(left: xr.Dataset, right: xr.Dataset, name: str, label: str) -> None:
    if name not in right.coords or not np.array_equal(left.coords[name].values,
                                                       right.coords[name].values):
        raise ForecastContractError("%s coordinate %r does not exactly match the forecast" %
                                    (label, name))


def _validate_inputs(forecast: RegionalForecastCube, truth: xr.Dataset,
                     initial_state: xr.Dataset) -> Tuple[Tuple[str, ...], Dict[str, Any], Dict[str, Any]]:
    if not isinstance(forecast, RegionalForecastCube):
        raise ForecastContractError("forecast must be a validated RegionalForecastCube")
    if not isinstance(truth, xr.Dataset) or not isinstance(initial_state, xr.Dataset):
        raise ForecastContractError("truth and initial_state must be xarray Datasets")
    required_forecast_provenance = {
        "schema", "request_sha256", "result_sha256", "artifact_sha256", "validation_sha256",
        "grid_shape", "grid_coordinates_sha256",
    }
    missing_forecast = sorted(required_forecast_provenance - set(forecast.provenance))
    if missing_forecast:
        raise ForecastContractError("regional forecast provenance is missing %s" % missing_forecast)
    if forecast.provenance["schema"] != "regional-external-forecast-cube/fcn3-v1":
        raise ForecastContractError("regional forecast provenance schema is not accepted")
    for name in ("request_sha256", "result_sha256", "artifact_sha256", "validation_sha256"):
        _require_sha256(forecast.provenance[name], "regional forecast %s" % name)
    expected_forecast_dims = ("time", "ensemble", "lead_time", "lat", "lon")
    if set(forecast.dataset.dims) != set(expected_forecast_dims):
        raise ForecastContractError("regional forecast dimension set must be exactly %r" %
                                    (expected_forecast_dims,))
    variables = tuple(forecast.dataset.data_vars)
    if not variables:
        raise ForecastContractError("regional forecast contains no variables")
    if set(truth.data_vars) != set(variables) or set(initial_state.data_vars) != set(variables):
        raise ForecastContractError("forecast, truth and initial_state variable sets must match exactly")
    if set(truth.dims) != {"time", "lead_time", "lat", "lon"}:
        raise ForecastContractError(
            "truth dimension set must be exactly ('time', 'lead_time', 'lat', 'lon')")
    if set(initial_state.dims) != {"time", "lat", "lon"}:
        raise ForecastContractError(
            "initial_state dimension set must be exactly ('time', 'lat', 'lon')")
    for name in ("time", "lead_time", "lat", "lon"):
        _same_coord(forecast.dataset, truth, name, "truth")
    for name in ("time", "lat", "lon"):
        _same_coord(forecast.dataset, initial_state, name, "initial_state")
    if set(truth.coords) != {"time", "lead_time", "lat", "lon"}:
        raise ForecastContractError("truth must contain only canonical coordinates")
    if set(initial_state.coords) != {"time", "lat", "lon"}:
        raise ForecastContractError("initial_state must contain only canonical coordinates")
    grid_shape = [int(forecast.dataset.sizes["lat"]), int(forecast.dataset.sizes["lon"])]
    if list(forecast.provenance["grid_shape"]) != grid_shape:
        raise ForecastContractError("regional forecast grid shape differs from its provenance")
    grid_hash = grid_coordinates_sha256(forecast.dataset.lat.values, forecast.dataset.lon.values)
    if forecast.provenance["grid_coordinates_sha256"] != grid_hash:
        raise ForecastContractError("regional forecast coordinates differ from their provenance")
    truth_record = _validate_provenance(truth, MATCHED_TRUTH_SCHEMA, "truth")
    initial_record = _validate_provenance(initial_state, INITIAL_STATE_SCHEMA, "initial_state")
    if truth_record["split"] != initial_record["split"]:
        raise ForecastContractError("truth and initial_state held-out split labels differ")
    if truth_record["source_sha256"] != initial_record["source_sha256"]:
        raise ForecastContractError(
            "truth and initial_state must bind the same verifying source provenance SHA-256")
    # Schemas intentionally differ; every other provenance field must be identical.
    truth_common = {key: value for key, value in truth_record.items() if key != "schema"}
    initial_common = {key: value for key, value in initial_record.items() if key != "schema"}
    if truth_common != initial_common:
        raise ForecastContractError("truth and initial_state builder provenance differs")
    for variable in variables:
        unit = fcn3_variable_unit(variable)
        arrays = (forecast.dataset[variable], truth[variable], initial_state[variable])
        expected_dims = (expected_forecast_dims,
                         ("time", "lead_time", "lat", "lon"),
                         ("time", "lat", "lon"))
        for array, dims, label in zip(arrays, expected_dims,
                                      ("forecast", "truth", "initial_state")):
            if tuple(array.dims) != dims:
                raise ForecastContractError("%s %s dimensions must be exactly %r" %
                                            (label, variable, dims))
            if not np.issubdtype(array.dtype, np.floating):
                raise ForecastContractError("%s %s must have a floating dtype" % (label, variable))
            if array.attrs.get("units") != unit:
                raise ForecastContractError("%s %s units must be %r" % (label, variable, unit))
    return variables, truth_record, initial_record


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def _safe_skill(model_mse: float, persistence_mse: float) -> float | None:
    return None if persistence_mse == 0.0 else 1.0 - model_mse / persistence_mse


def _tile_shape(height: int, width: int, member_count: int, itemsize: int,
                max_bytes: int) -> Tuple[int, int]:
    # Forecast, truth and initial state coexist; conservatively budget their input bytes.
    values_per_cell = member_count + 2
    cells = max_bytes // (values_per_cell * itemsize)
    if cells < 1:
        raise ForecastContractError("maximum evaluation tile budget cannot hold one grid cell")
    tile_width = min(width, cells)
    tile_height = min(height, max(1, cells // tile_width))
    return tile_height, tile_width


def _crps_ensemble(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Exact empirical CRPS in O(M log M), with members on axis zero."""
    members = values.shape[0]
    first = np.mean(np.abs(values - target[None, ...]), axis=0)
    ordered = np.sort(values, axis=0)
    coefficients = 2.0 * np.arange(1, members + 1, dtype="float64") - members - 1.0
    pair_term = np.sum(ordered * coefficients[:, None, None], axis=0) / (members * members)
    return first - pair_term


def evaluate_regional_ensemble(
    forecast: RegionalForecastCube,
    truth: xr.Dataset,
    initial_state: xr.Dataset,
    *,
    max_tile_bytes: int = DEFAULT_EVALUATION_TILE_BYTES,
) -> EnsembleEvaluation:
    """Evaluate a regional ensemble and exact persistence over matched physical samples.

    RMSE/MAE/bias, empirical CRPS and spread are cosine-latitude area weighted. Rank-bin counts
    use deterministic fractional allocation when truth ties ensemble members; their area-weighted
    frequencies are a calibration diagnostic only. No interpolation or missing-value deletion is
    performed.
    """
    if isinstance(max_tile_bytes, bool) or not isinstance(max_tile_bytes, int) or max_tile_bytes <= 0:
        raise ForecastContractError("max_tile_bytes must be a positive integer")
    variables, truth_provenance, initial_provenance = _validate_inputs(
        forecast, truth, initial_state)
    ds = forecast.dataset
    member_labels = tuple(int(value) for value in np.asarray(ds.ensemble.values).tolist())
    member_count = len(member_labels)
    if member_count < 2:
        raise ForecastContractError("ensemble diagnostics require at least two members")
    lead_ns = _duration_ns(np.asarray(ds.lead_time.values))
    if np.any(lead_ns <= 0) or np.any(np.diff(lead_ns) <= 0):
        raise ForecastContractError("lead_time must contain positive strictly increasing durations")
    lead_hours = tuple(float(value) / 3_600_000_000_000 for value in lead_ns)
    latitude = np.asarray(ds.lat.values, dtype="float64")
    longitude = np.asarray(ds.lon.values, dtype="float64")
    lat_weights = np.cos(np.deg2rad(latitude))
    if not np.isfinite(lat_weights).all() or np.any(lat_weights <= 0):
        raise ForecastContractError("regional cosine-latitude weights must be finite and positive")
    height, width = len(latitude), len(longitude)
    # Tiles are promoted to float64 for stable accumulation.
    itemsize = max(8, max(int(ds[name].dtype.itemsize) for name in variables))
    tile_height, tile_width = _tile_shape(height, width, member_count, itemsize, max_tile_bytes)

    metric_result: Dict[str, Any] = {}
    forecast_digest, truth_digest, initial_digest = (hashlib.sha256(), hashlib.sha256(),
                                                     hashlib.sha256())
    observed_max = 0
    for lead_index, hours in enumerate(lead_hours):
        lead_record: Dict[str, Any] = {}
        for variable in variables:
            member_sq = np.zeros(member_count, dtype="float64")
            member_abs = np.zeros(member_count, dtype="float64")
            member_signed = np.zeros(member_count, dtype="float64")
            mean_sq = mean_abs = mean_signed = persistence_sq = persistence_abs = 0.0
            persistence_signed = crps_sum = spread_variance_sum = weight_sum = 0.0
            rank_counts = np.zeros(member_count + 1, dtype="float64")
            rank_area = np.zeros(member_count + 1, dtype="float64")
            raw_count = 0
            for time_index in range(int(ds.sizes["time"])):
                for y0 in range(0, height, tile_height):
                    for x0 in range(0, width, tile_width):
                        region = {"lat": slice(y0, min(height, y0 + tile_height)),
                                  "lon": slice(x0, min(width, x0 + tile_width))}
                        predicted = np.asarray(ds[variable].isel(
                            time=time_index, lead_time=lead_index, **region).compute(),
                                               dtype="float64")
                        observed = np.asarray(truth[variable].isel(
                            time=time_index, lead_time=lead_index, **region).compute(),
                                              dtype="float64")
                        initial = np.asarray(initial_state[variable].isel(
                            time=time_index, **region).compute(), dtype="float64")
                        tile_bytes = predicted.nbytes + observed.nbytes + initial.nbytes
                        if tile_bytes > max_tile_bytes:
                            raise ForecastContractError("materialised evaluation tile exceeded max_tile_bytes")
                        observed_max = max(observed_max, tile_bytes)
                        if (not np.isfinite(predicted).all() or not np.isfinite(observed).all()
                                or not np.isfinite(initial).all()):
                            raise ForecastContractError(
                                "forecast, truth and initial_state must be finite; missing-value deletion is refused")
                        forecast_digest.update(np.ascontiguousarray(predicted).tobytes())
                        truth_digest.update(np.ascontiguousarray(observed).tobytes())
                        initial_digest.update(np.ascontiguousarray(initial).tobytes())
                        weights = lat_weights[region["lat"]][:, None] * np.ones(
                            (1, predicted.shape[2]), dtype="float64")
                        error = predicted - observed[None, ...]
                        ensemble_mean_error = predicted.mean(axis=0) - observed
                        persistence_error = initial - observed
                        member_sq += np.sum(error * error * weights[None, ...], axis=(1, 2))
                        member_abs += np.sum(np.abs(error) * weights[None, ...], axis=(1, 2))
                        member_signed += np.sum(error * weights[None, ...], axis=(1, 2))
                        mean_sq += float(np.sum(ensemble_mean_error ** 2 * weights))
                        mean_abs += float(np.sum(np.abs(ensemble_mean_error) * weights))
                        mean_signed += float(np.sum(ensemble_mean_error * weights))
                        persistence_sq += float(np.sum(persistence_error ** 2 * weights))
                        persistence_abs += float(np.sum(np.abs(persistence_error) * weights))
                        persistence_signed += float(np.sum(persistence_error * weights))
                        crps_sum += float(np.sum(_crps_ensemble(predicted, observed) * weights))
                        spread_variance_sum += float(np.sum(np.var(predicted, axis=0, ddof=0) * weights))
                        less = np.sum(predicted < observed[None, ...], axis=0)
                        equal = np.sum(predicted == observed[None, ...], axis=0)
                        for rank in range(member_count + 1):
                            allocation = ((rank >= less) & (rank <= less + equal)) / (equal + 1.0)
                            rank_counts[rank] += float(np.sum(allocation))
                            rank_area[rank] += float(np.sum(allocation * weights))
                        weight_sum += float(np.sum(weights))
                        raw_count += int(observed.size)
            mean_mse = mean_sq / weight_sum
            persistence_mse = persistence_sq / weight_sum
            member_metrics = {}
            for index, member in enumerate(member_labels):
                member_metrics[str(member)] = {
                    "rmse": math.sqrt(float(member_sq[index] / weight_sum)),
                    "mae": float(member_abs[index] / weight_sum),
                    "bias": float(member_signed[index] / weight_sum),
                }
            rank_total = float(rank_area.sum())
            lead_record[variable] = {
                "unit": fcn3_variable_unit(variable),
                "member_errors": member_metrics,
                "ensemble_mean": {
                    "rmse": math.sqrt(mean_mse), "mae": mean_abs / weight_sum,
                    "bias": mean_signed / weight_sum,
                    "mse_skill_score_vs_persistence": _safe_skill(mean_mse, persistence_mse),
                },
                "persistence": {
                    "rmse": math.sqrt(persistence_mse), "mae": persistence_abs / weight_sum,
                    "bias": persistence_signed / weight_sum,
                },
                "crps": crps_sum / weight_sum,
                "spread_rms": math.sqrt(spread_variance_sum / weight_sum),
                "spread_skill_ratio": _safe_ratio(
                    math.sqrt(spread_variance_sum / weight_sum), math.sqrt(mean_mse)),
                "spread_definition": "population ensemble SD RMS / ensemble-mean RMSE; no finite-member correction",
                "rank_histogram": {
                    "bin_definition": "truth rank 0..M among sorted members",
                    "fractional_tie_counts": rank_counts.tolist(),
                    "area_weighted_frequency": (rank_area / rank_total).tolist(),
                    "tie_policy": "uniform fractional allocation over every admissible tied rank",
                    "diagnostic_only": True,
                },
                "initialization_count": int(ds.sizes["time"]),
                "gridpoint_count": raw_count,
                "area_weight_sum": weight_sum,
            }
        metric_result[str(hours)] = lead_record

    forecast_record = dict(forecast.provenance)
    return EnsembleEvaluation(
        schema=ENSEMBLE_EVALUATION_SCHEMA,
        forecast_provenance=forecast_record,
        truth_provenance=truth_provenance,
        initial_state_provenance=initial_provenance,
        variables=variables,
        initialization_count=int(ds.sizes["time"]),
        ensemble_members=member_labels,
        lead_durations_hours=lead_hours,
        grid_shape=(height, width),
        grid_coordinates_sha256=grid_coordinates_sha256(latitude, longitude),
        area_weighting="cos(latitude), normalized independently for each variable and lead",
        metrics=metric_result,
        forecast_values_sha256=forecast_digest.hexdigest(),
        truth_values_sha256=truth_digest.hexdigest(),
        initial_state_values_sha256=initial_digest.hexdigest(),
        maximum_observed_tile_bytes=observed_max,
        maximum_allowed_tile_bytes=max_tile_bytes,
        claim_boundary=(
            "Deterministic matched-sample verification of one supplied ensemble against exact "
            "persistence. No sampling uncertainty, independence, significance, calibration, "
            "generalisation or scientific superiority claim is implied."),
    )
