"""Reproducible offline external-forecast evaluation orchestration (T5.6e).

This module composes the already accepted T5.6b--d boundaries.  It performs no download and
no model inference: both the sealed forecast artifact and the content-addressed ERA5 cache
must exist before a run starts.  A successful run is written as one canonical JSON receipt
using an atomic, no-overwrite filesystem operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from src.data_layer.zarr_source import CropSpec, open_cached_lazy
from src.forecasting.adapter import ForecastContractError
from src.forecasting.ensemble_evaluation import (
    DEFAULT_EVALUATION_TILE_BYTES,
    EnsembleEvaluation,
    evaluate_regional_ensemble,
)
from src.forecasting.external_cube import (
    DEFAULT_MAX_CHUNK_BYTES,
    GeographicBounds,
    open_canonical_forecast_cube,
)
from src.forecasting.external_fcn3 import ExternalForecastResult, ExternalForecastRun
from src.forecasting.matched_truth import (
    FRESH_HOLDOUT_ROLE,
    PUBLISHED_OVERLAP_ROLE,
    build_matched_truth,
)


EVALUATION_RUN_CONFIG_SCHEMA = "external-evaluation-run-config/fcn3-era5-v1"
EVALUATION_RUN_RECEIPT_SCHEMA = "external-evaluation-run-receipt/fcn3-era5-v1"


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError("evaluation run receipt must be finite JSON: %s" % exc) from exc


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ForecastContractError("%s must be a positive integer" % label)
    return value


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Machine-independent scientific and memory controls for one evaluation run."""

    schema: str
    bounds: GeographicBounds
    split: str
    split_start: str
    split_end: str
    evaluation_role: str
    level_hpa: int = 850
    max_forecast_chunk_bytes: int = DEFAULT_MAX_CHUNK_BYTES
    max_evaluation_tile_bytes: int = DEFAULT_EVALUATION_TILE_BYTES

    def __post_init__(self) -> None:
        if self.schema != EVALUATION_RUN_CONFIG_SCHEMA:
            raise ForecastContractError("unsupported evaluation run config schema")
        if not isinstance(self.bounds, GeographicBounds):
            raise ForecastContractError("evaluation bounds must be validated GeographicBounds")
        if not isinstance(self.split, str) or not self.split.strip() or self.split.lower() == "train":
            raise ForecastContractError("evaluation split must be explicit and held out from training")
        if not isinstance(self.split_start, str) or not self.split_start.strip():
            raise ForecastContractError("split_start must be an explicit ISO timestamp")
        if not isinstance(self.split_end, str) or not self.split_end.strip():
            raise ForecastContractError("split_end must be an explicit ISO timestamp")
        if self.evaluation_role not in (FRESH_HOLDOUT_ROLE, PUBLISHED_OVERLAP_ROLE):
            raise ForecastContractError("evaluation_role is not an accepted FCN3-period role")
        if isinstance(self.level_hpa, bool) or self.level_hpa != 850:
            raise ForecastContractError("T5.6e currently requires the declared 850-hPa bridge")
        _positive_int(self.max_forecast_chunk_bytes, "max_forecast_chunk_bytes")
        _positive_int(self.max_evaluation_tile_bytes, "max_evaluation_tile_bytes")

    def to_mapping(self) -> Dict[str, Any]:
        value = asdict(self)
        value["bounds"] = self.bounds.to_mapping()
        return value

    def fingerprint(self) -> str:
        return _canonical_hash(self.to_mapping())

    def __hash__(self) -> int:
        return int(self.fingerprint()[:16], 16)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRunConfig":
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != fields:
            supplied = set(value) if isinstance(value, Mapping) else set()
            raise ForecastContractError(
                "evaluation run config fields differ: missing=%r unknown=%r" %
                (sorted(fields - supplied), sorted(supplied - fields)))
        record = dict(value)
        bounds_fields = set(GeographicBounds.__dataclass_fields__)
        bounds = record.get("bounds")
        if not isinstance(bounds, Mapping) or set(bounds) != bounds_fields:
            supplied_bounds = set(bounds) if isinstance(bounds, Mapping) else set()
            raise ForecastContractError(
                "evaluation bounds fields differ: missing=%r unknown=%r" %
                (sorted(bounds_fields - supplied_bounds),
                 sorted(supplied_bounds - bounds_fields)))
        record["bounds"] = GeographicBounds(**dict(bounds))
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid evaluation run config: %s" % exc) from exc


@dataclass(frozen=True)
class EvaluationRunReceipt:
    """Complete metric result and the identities needed to audit how it was produced."""

    schema: str
    run_request: Mapping[str, Any]
    run_request_sha256: str
    forecast_validation: Mapping[str, Any]
    regional_forecast_provenance: Mapping[str, Any]
    era5_manifest: Mapping[str, Any]
    matched_truth_provenance: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    evaluation_sha256: str
    claim_boundary: str

    def unsigned_mapping(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return _canonical_hash(self.unsigned_mapping())

    def to_mapping(self) -> Dict[str, Any]:
        return {**self.unsigned_mapping(), "receipt_sha256": self.fingerprint()}


def _run_request(
    config: EvaluationRunConfig,
    forecast_request: ExternalForecastRun,
    forecast_result: ExternalForecastResult,
    era5_crop: CropSpec,
) -> Dict[str, Any]:
    return {
        "schema": "external-evaluation-run-request/fcn3-era5-v1",
        "config": config.to_mapping(),
        "config_sha256": config.fingerprint(),
        "forecast_request_sha256": forecast_request.fingerprint(),
        "forecast_result_sha256": forecast_result.result_sha256,
        "forecast_artifact_sha256": forecast_result.artifact_sha256,
        "forecast_artifact_reference": forecast_result.artifact_reference,
        # Embedded as well as hashed from T5.6g onward so a report can show the actual model,
        # checkpoint, software and runtime rather than reverse-label a digest.
        "forecast_request": forecast_request.to_mapping(),
        "forecast_result": forecast_result.to_mapping(),
        "era5_crop": era5_crop.to_provenance(),
        "era5_crop_content_key": era5_crop.content_key(),
        "execution": (
            "offline local-only composition: authenticate/validate forecast; exact crop; "
            "open existing ERA5 cache; exact time/grid/level match; bounded evaluation"),
    }


def _atomic_write_new(path: Union[str, os.PathLike[str]], payload: bytes) -> None:
    """Publish complete bytes atomically and refuse an existing target, including races."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name, suffix=".tmp", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing target. Unlike a hard link,
                # it also works on common removable-drive filesystems that support atomic rename.
                os.rename(temporary, target)
            else:
                # POSIX rename replaces its target, so publish with a same-filesystem hard link.
                os.link(temporary, target)
        except FileExistsError:
            raise FileExistsError(
                "refusing to overwrite evaluation run receipt at %s" % target) from None
        except OSError as exc:
            raise ForecastContractError(
                "cannot atomically publish evaluation run receipt at %s: %s" % (target, exc)) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def save_evaluation_run_receipt(
    path: Union[str, os.PathLike[str]], receipt: EvaluationRunReceipt,
) -> str:
    if not isinstance(receipt, EvaluationRunReceipt):
        raise ForecastContractError("receipt must be an EvaluationRunReceipt")
    if receipt.schema != EVALUATION_RUN_RECEIPT_SCHEMA:
        raise ForecastContractError("unsupported evaluation run receipt schema")
    record = receipt.to_mapping()
    _verify_receipt_record(record)
    payload = _canonical_json(record) + b"\n"
    _atomic_write_new(path, payload)
    return receipt.fingerprint()


def _verify_receipt_record(record: Any) -> None:
    fields = {
        "schema", "run_request", "run_request_sha256", "forecast_validation",
        "regional_forecast_provenance", "era5_manifest", "matched_truth_provenance",
        "evaluation", "evaluation_sha256", "claim_boundary", "receipt_sha256",
    }
    if not isinstance(record, dict) or set(record) != fields:
        raise ForecastContractError("evaluation run receipt fields are incomplete or unknown")
    if record.get("schema") != EVALUATION_RUN_RECEIPT_SCHEMA:
        raise ForecastContractError("unsupported evaluation run receipt schema")
    supplied = record.get("receipt_sha256")
    unsigned = {key: value for key, value in record.items() if key != "receipt_sha256"}
    if not isinstance(supplied, str) or supplied != _canonical_hash(unsigned):
        raise ForecastContractError("evaluation run receipt SHA-256 mismatch")
    run_request = record["run_request"]
    evaluation = record["evaluation"]
    if record["run_request_sha256"] != _canonical_hash(run_request):
        raise ForecastContractError("evaluation run request SHA-256 mismatch")
    if record["evaluation_sha256"] != _canonical_hash(evaluation):
        raise ForecastContractError("embedded ensemble evaluation SHA-256 mismatch")
    if run_request.get("config_sha256") != _canonical_hash(run_request.get("config")):
        raise ForecastContractError("embedded evaluation config SHA-256 mismatch")
    embedded_request = run_request.get("forecast_request")
    if embedded_request is not None:
        if _canonical_hash(embedded_request) != run_request.get("forecast_request_sha256"):
            raise ForecastContractError("embedded forecast request SHA-256 mismatch")
    embedded_result = run_request.get("forecast_result")
    if embedded_result is not None:
        result_without_hash = dict(embedded_result)
        result_sha256 = result_without_hash.pop("result_sha256", None)
        if (result_sha256 != run_request.get("forecast_result_sha256")
                or _canonical_hash(result_without_hash) != result_sha256):
            raise ForecastContractError("embedded forecast result SHA-256 mismatch")

    validation = record["forecast_validation"]
    regional = record["regional_forecast_provenance"]
    matched = record["matched_truth_provenance"]
    identities = {
        "request_sha256": "forecast_request_sha256",
        "result_sha256": "forecast_result_sha256",
        "artifact_sha256": "forecast_artifact_sha256",
    }
    for provenance_name, request_name in identities.items():
        expected = run_request.get(request_name)
        if validation.get(provenance_name) != expected or regional.get(provenance_name) != expected:
            raise ForecastContractError("forecast identity differs across evaluation receipt sections")
        if matched.get("forecast_%s" % provenance_name) != expected:
            raise ForecastContractError("matched truth differs from the forecast run identity")
    if regional.get("validation_sha256") != _canonical_hash(validation):
        raise ForecastContractError("regional crop does not bind the embedded forecast validation")
    if matched.get("source_sha256") != _canonical_hash(record["era5_manifest"]):
        raise ForecastContractError("matched truth does not bind the embedded ERA5 manifest")
    if evaluation.get("forecast_provenance") != regional:
        raise ForecastContractError("evaluation does not bind the embedded regional forecast")
    truth_provenance = evaluation.get("truth_provenance", {})
    if truth_provenance.get("builder_sha256") != matched.get("builder_sha256"):
        raise ForecastContractError("evaluation does not bind the matched-truth builder")


def verify_evaluation_run_receipt(record: Any) -> Mapping[str, Any]:
    """Verify an already-decoded receipt and return it unchanged.

    API uploads and notebooks use this same verifier as filesystem loads; there is no weaker
    presentation parser.
    """
    _verify_receipt_record(record)
    return record


def load_evaluation_run_receipt(
    path: Union[str, os.PathLike[str]],
) -> Mapping[str, Any]:
    """Load a saved receipt and verify content identity plus cross-section lineage."""
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read evaluation run receipt: %s" % exc) from exc
    return verify_evaluation_run_receipt(record)


def run_external_evaluation(
    *,
    forecast_artifact_path: Union[str, os.PathLike[str]],
    forecast_request: ExternalForecastRun,
    forecast_result: ExternalForecastResult,
    era5_crop: CropSpec,
    era5_cache_dir: Union[str, os.PathLike[str]],
    config: EvaluationRunConfig,
    receipt_path: Union[str, os.PathLike[str]],
) -> EvaluationRunReceipt:
    """Run T5.6b -> T5.6d -> T5.6c and atomically persist the complete receipt.

    Existing-output refusal happens before costly validation. All opened xarray stores are
    closed on success or failure. The function never invokes a network client or FCN3 worker.
    """
    if Path(receipt_path).exists():
        raise FileExistsError(
            "refusing to overwrite evaluation run receipt at %s" % Path(receipt_path))
    if not isinstance(forecast_request, ExternalForecastRun):
        raise ForecastContractError("forecast_request must be an ExternalForecastRun")
    if not isinstance(forecast_result, ExternalForecastResult):
        raise ForecastContractError("forecast_result must be an ExternalForecastResult")
    if not isinstance(era5_crop, CropSpec):
        raise ForecastContractError("era5_crop must be a validated CropSpec")
    if not isinstance(config, EvaluationRunConfig):
        raise ForecastContractError("config must be a validated EvaluationRunConfig")

    request_record = _run_request(config, forecast_request, forecast_result, era5_crop)
    request_sha256 = _canonical_hash(request_record)
    with open_canonical_forecast_cube(
        forecast_artifact_path, forecast_request, forecast_result,
        max_chunk_bytes=config.max_forecast_chunk_bytes,
    ) as global_cube:
        regional = global_cube.crop(config.bounds)
        source, source_manifest = open_cached_lazy(era5_crop, cache_dir=str(era5_cache_dir))
        try:
            matched = build_matched_truth(
                source, source_manifest, regional, split=config.split,
                split_start=config.split_start, split_end=config.split_end,
                evaluation_role=config.evaluation_role, level_hpa=config.level_hpa)
            evaluation: EnsembleEvaluation = evaluate_regional_ensemble(
                regional, matched.truth, matched.initial_state,
                max_tile_bytes=config.max_evaluation_tile_bytes)
            evaluation_record = evaluation.to_mapping()
            receipt = EvaluationRunReceipt(
                schema=EVALUATION_RUN_RECEIPT_SCHEMA,
                run_request=request_record,
                run_request_sha256=request_sha256,
                forecast_validation=global_cube.validation.to_mapping(),
                regional_forecast_provenance=dict(regional.provenance),
                era5_manifest=dict(source_manifest),
                matched_truth_provenance=dict(matched.provenance),
                evaluation=evaluation_record,
                evaluation_sha256=evaluation.fingerprint(),
                claim_boundary=(
                    "This receipt proves deterministic execution of the authenticated, exact-"
                    "match, bounded-memory evaluation pipeline for the supplied artifacts. It "
                    "does not by itself establish independence, uncertainty, significance, "
                    "calibration, generalisation or scientific forecast skill."),
            )
            save_evaluation_run_receipt(receipt_path, receipt)
            return receipt
        finally:
            source.close()
