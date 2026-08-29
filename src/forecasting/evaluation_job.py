"""Portable, offline job bundles for reproducible external evaluation (T5.6f).

The job is machine-independent and content addressed. Local artifact, cache and receipt paths
live in a separate bindings file, so the identical scientific job can run on a laptop or HPC
filesystem without changing identity. This module never downloads data, runs FCN3 or submits a
scheduler job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from src.core.errors import SpectralEarthError
from src.core.publication import publish_new_bytes
from src.data_layer.zarr_source import (
    CropSpec,
    cache_path,
    is_cached,
    manifest_path,
    open_cached_lazy,
)
from src.forecasting.adapter import ForecastContractError
from src.forecasting.evaluation_run import (
    EvaluationRunConfig,
    EvaluationRunReceipt,
    load_evaluation_run_receipt,
    run_external_evaluation,
)
from src.forecasting.external_fcn3 import (
    ExternalForecastResult,
    ExternalForecastRun,
    load_external_forecast_result,
    load_external_forecast_run,
    verify_external_forecast_result,
)


EVALUATION_JOB_SCHEMA = "external-evaluation-job/fcn3-era5-v1"
EVALUATION_PATH_BINDINGS_SCHEMA = "external-evaluation-path-bindings/v1"
EVALUATION_PREFLIGHT_SCHEMA = "external-evaluation-preflight/v1"
EVALUATION_JOB_CLAIM_BOUNDARY = (
    "Portable instructions and identities only. This job does not prove that the forecast "
    "artifact, ERA5 cache or evaluation exists, and it establishes no forecast skill."
)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForecastContractError("evaluation job must be finite JSON: %s" % exc) from exc


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _exact(value: Any, fields: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForecastContractError("%s must be a JSON object" % label)
    supplied = set(value)
    if supplied != fields:
        raise ForecastContractError(
            "%s fields differ: missing=%r unknown=%r" %
            (label, sorted(fields - supplied), sorted(supplied - fields)))
    return value


def _read_json(path: Union[str, os.PathLike[str]], label: str) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ForecastContractError("cannot read %s: %s" % (label, exc)) from exc


def _atomic_write_new(path: Union[str, os.PathLike[str]], payload: bytes, label: str) -> None:
    target = Path(path)
    try:
        publish_new_bytes(target, payload, label)
    except FileExistsError:
        raise
    except OSError as exc:
        raise ForecastContractError(str(exc)) from exc


@dataclass(frozen=True)
class EvaluationJob:
    """Complete machine-independent inputs to one accepted T5.6e evaluation."""

    schema: str
    forecast_request: ExternalForecastRun
    forecast_result: ExternalForecastResult
    era5_crop: CropSpec
    evaluation_config: EvaluationRunConfig
    claim_boundary: str = EVALUATION_JOB_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if self.schema != EVALUATION_JOB_SCHEMA:
            raise ForecastContractError("unsupported external evaluation job schema")
        if not isinstance(self.forecast_request, ExternalForecastRun):
            raise ForecastContractError("forecast_request must be an ExternalForecastRun")
        if not isinstance(self.forecast_result, ExternalForecastResult):
            raise ForecastContractError("forecast_result must be an ExternalForecastResult")
        if not isinstance(self.era5_crop, CropSpec):
            raise ForecastContractError("era5_crop must be a CropSpec")
        if not isinstance(self.evaluation_config, EvaluationRunConfig):
            raise ForecastContractError("evaluation_config must be an EvaluationRunConfig")
        if self.claim_boundary != EVALUATION_JOB_CLAIM_BOUNDARY:
            raise ForecastContractError("evaluation job claim boundary differs from the schema")
        verify_external_forecast_result(self.forecast_request, self.forecast_result)

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "forecast_request": self.forecast_request.to_mapping(),
            "forecast_result": self.forecast_result.to_mapping(),
            "era5_crop": self.era5_crop.to_provenance(),
            "evaluation_config": self.evaluation_config.to_mapping(),
            "claim_boundary": self.claim_boundary,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.to_mapping())

    def __hash__(self) -> int:
        return int(self.fingerprint()[:16], 16)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationJob":
        fields = set(cls.__dataclass_fields__)
        record = dict(_exact(value, fields, "evaluation job"))
        request = ExternalForecastRun.from_mapping(record["forecast_request"])
        result = ExternalForecastResult.from_mapping(record["forecast_result"])
        crop_record = record["era5_crop"]
        if not isinstance(crop_record, Mapping):
            raise ForecastContractError("evaluation job era5_crop must be a JSON object")
        crop = CropSpec.from_provenance(dict(crop_record))
        if dict(crop_record) != crop.to_provenance():
            raise ForecastContractError("evaluation job ERA5 crop provenance is not canonical")
        config = EvaluationRunConfig.from_mapping(record["evaluation_config"])
        return cls(
            schema=record["schema"], forecast_request=request, forecast_result=result,
            era5_crop=crop, evaluation_config=config, claim_boundary=record["claim_boundary"])


def save_evaluation_job(
    path: Union[str, os.PathLike[str]], job: EvaluationJob,
) -> str:
    if not isinstance(job, EvaluationJob):
        raise ForecastContractError("job must be an EvaluationJob")
    envelope = {"job": job.to_mapping(), "job_sha256": job.fingerprint()}
    _atomic_write_new(path, _canonical_json(envelope) + b"\n", "evaluation job")
    return job.fingerprint()


def load_evaluation_job(path: Union[str, os.PathLike[str]]) -> EvaluationJob:
    envelope = _exact(
        _read_json(path, "evaluation job"), {"job", "job_sha256"}, "evaluation job envelope")
    job = EvaluationJob.from_mapping(envelope["job"])
    if envelope["job_sha256"] != job.fingerprint():
        raise ForecastContractError("evaluation job SHA-256 mismatch")
    return job


def _path_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ForecastContractError("%s must be a non-empty filesystem path" % label)
    return value


@dataclass(frozen=True)
class EvaluationPathBindings:
    """Machine-local paths, deliberately excluded from scientific job identity."""

    schema: str
    job_sha256: str
    forecast_artifact_path: str
    era5_cache_dir: str
    receipt_path: str

    def __post_init__(self) -> None:
        if self.schema != EVALUATION_PATH_BINDINGS_SCHEMA:
            raise ForecastContractError("unsupported evaluation path bindings schema")
        if (not isinstance(self.job_sha256, str) or len(self.job_sha256) != 64
                or any(char not in "0123456789abcdef" for char in self.job_sha256)):
            raise ForecastContractError("bindings job_sha256 must be a lowercase SHA-256")
        for name in ("forecast_artifact_path", "era5_cache_dir", "receipt_path"):
            _path_text(getattr(self, name), name)

    def to_mapping(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return _fingerprint(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationPathBindings":
        fields = set(cls.__dataclass_fields__)
        record = dict(_exact(value, fields, "evaluation path bindings"))
        try:
            return cls(**record)
        except TypeError as exc:
            raise ForecastContractError("invalid evaluation path bindings: %s" % exc) from exc


def save_evaluation_path_bindings(
    path: Union[str, os.PathLike[str]], bindings: EvaluationPathBindings,
) -> str:
    if not isinstance(bindings, EvaluationPathBindings):
        raise ForecastContractError("bindings must be EvaluationPathBindings")
    envelope = {
        "bindings": bindings.to_mapping(),
        "bindings_sha256": bindings.fingerprint(),
    }
    _atomic_write_new(path, _canonical_json(envelope) + b"\n", "evaluation path bindings")
    return bindings.fingerprint()


def load_evaluation_path_bindings(
    path: Union[str, os.PathLike[str]],
) -> EvaluationPathBindings:
    envelope = _exact(
        _read_json(path, "evaluation path bindings"),
        {"bindings", "bindings_sha256"}, "evaluation path bindings envelope")
    bindings = EvaluationPathBindings.from_mapping(envelope["bindings"])
    if envelope["bindings_sha256"] != bindings.fingerprint():
        raise ForecastContractError("evaluation path bindings SHA-256 mismatch")
    return bindings


@dataclass(frozen=True)
class ResolvedEvaluationPaths:
    forecast_artifact_path: Path
    era5_cache_dir: Path
    receipt_path: Path

    def to_mapping(self) -> Dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


def resolve_evaluation_paths(
    bindings: EvaluationPathBindings,
    *,
    base_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> ResolvedEvaluationPaths:
    if not isinstance(bindings, EvaluationPathBindings):
        raise ForecastContractError("bindings must be EvaluationPathBindings")
    root = Path.cwd() if base_dir is None else Path(base_dir)

    def resolve(value: str) -> Path:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve(strict=False)

    return ResolvedEvaluationPaths(
        forecast_artifact_path=resolve(bindings.forecast_artifact_path),
        era5_cache_dir=resolve(bindings.era5_cache_dir),
        receipt_path=resolve(bindings.receipt_path),
    )


def _check_binding_job(job: EvaluationJob, bindings: EvaluationPathBindings) -> None:
    if bindings.job_sha256 != job.fingerprint():
        raise ForecastContractError("path bindings refer to a different evaluation job")


def preflight_evaluation_job(
    job: EvaluationJob,
    bindings: EvaluationPathBindings,
    *,
    base_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> Mapping[str, Any]:
    """Authenticate local inputs without network access, value evaluation or output writes."""
    if not isinstance(job, EvaluationJob):
        raise ForecastContractError("job must be an EvaluationJob")
    _check_binding_job(job, bindings)
    paths = resolve_evaluation_paths(bindings, base_dir=base_dir)
    if paths.receipt_path.exists():
        raise FileExistsError(
            "refusing to overwrite evaluation run receipt at %s" % paths.receipt_path)
    verify_external_forecast_result(
        job.forecast_request, job.forecast_result, paths.forecast_artifact_path)
    if not is_cached(job.era5_crop, str(paths.era5_cache_dir)):
        raise ForecastContractError(
            "ERA5 crop %s is not present in the bound local cache" %
            job.era5_crop.content_key())
    source, manifest = open_cached_lazy(job.era5_crop, str(paths.era5_cache_dir))
    try:
        manifest_crop = CropSpec.from_provenance(manifest.get("spec", {}))
        if manifest_crop.content_key() != job.era5_crop.content_key():
            raise ForecastContractError("ERA5 cache manifest refers to a different crop")
    finally:
        source.close()
    return {
        "schema": EVALUATION_PREFLIGHT_SCHEMA,
        "ready": True,
        "job_sha256": job.fingerprint(),
        "paths": paths.to_mapping(),
        "forecast_artifact_sha256": job.forecast_result.artifact_sha256,
        "era5_crop_content_key": job.era5_crop.content_key(),
        "era5_cache_path": cache_path(job.era5_crop, str(paths.era5_cache_dir)),
        "era5_manifest_path": manifest_path(job.era5_crop, str(paths.era5_cache_dir)),
        "checks": [
            "job_contract_valid", "binding_job_identity_valid",
            "forecast_request_result_valid", "forecast_artifact_authenticated",
            "era5_cache_local_and_openable", "era5_manifest_crop_identity_valid",
            "receipt_target_absent",
        ],
        "network_used": False,
        "model_inference_used": False,
        "claim_boundary": (
            "Ready means that bound local inputs exist and pass identity/open checks. It does "
            "not mean the evaluation has run or that the forecast has scientific skill."),
    }


def verify_evaluation_receipt_for_job(
    job: EvaluationJob, receipt: Mapping[str, Any],
) -> None:
    """Prove that a verified T5.6e receipt corresponds to this portable job."""
    request = receipt.get("run_request", {})
    expected = {
        "config": job.evaluation_config.to_mapping(),
        "config_sha256": job.evaluation_config.fingerprint(),
        "forecast_request_sha256": job.forecast_request.fingerprint(),
        "forecast_result_sha256": job.forecast_result.result_sha256,
        "forecast_artifact_sha256": job.forecast_result.artifact_sha256,
        "era5_crop_content_key": job.era5_crop.content_key(),
    }
    if any(request.get(name) != value for name, value in expected.items()):
        raise ForecastContractError("evaluation receipt does not correspond to the portable job")


def run_evaluation_job(
    job: EvaluationJob,
    bindings: EvaluationPathBindings,
    *,
    base_dir: Optional[Union[str, os.PathLike[str]]] = None,
) -> EvaluationRunReceipt:
    """Execute the bound offline evaluation and verify its receipt against the job."""
    if not isinstance(job, EvaluationJob):
        raise ForecastContractError("job must be an EvaluationJob")
    _check_binding_job(job, bindings)
    paths = resolve_evaluation_paths(bindings, base_dir=base_dir)
    receipt = run_external_evaluation(
        forecast_artifact_path=paths.forecast_artifact_path,
        forecast_request=job.forecast_request,
        forecast_result=job.forecast_result,
        era5_crop=job.era5_crop,
        era5_cache_dir=paths.era5_cache_dir,
        config=job.evaluation_config,
        receipt_path=paths.receipt_path,
    )
    saved = load_evaluation_run_receipt(paths.receipt_path)
    verify_evaluation_receipt_for_job(job, saved)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.forecasting.evaluation_job",
        description="Preflight or run an offline, portable FCN3-to-ERA5 evaluation job",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--forecast-request", required=True)
    create.add_argument("--forecast-result", required=True)
    create.add_argument(
        "--era5-crop", required=True,
        help="CropSpec provenance JSON or an ERA5 cache manifest containing 'spec'")
    create.add_argument(
        "--evaluation-config", required=True,
        help="exact EvaluationRunConfig JSON mapping")
    create.add_argument("--output", required=True, help="new portable job JSON")

    bind = subparsers.add_parser("bind")
    bind.add_argument("--job", required=True)
    bind.add_argument("--forecast-artifact", required=True)
    bind.add_argument("--era5-cache-dir", required=True)
    bind.add_argument("--receipt", required=True)
    bind.add_argument("--output", required=True, help="new machine-local bindings JSON")

    for command in ("preflight", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("--job", required=True, help="portable evaluation job JSON")
        child.add_argument(
            "--bindings", required=True,
            help="machine-local path bindings JSON; relative paths use its directory")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            request = load_external_forecast_run(args.forecast_request)
            result = load_external_forecast_result(args.forecast_result)
            crop_record = _read_json(args.era5_crop, "ERA5 crop provenance")
            if isinstance(crop_record, Mapping) and "spec" in crop_record:
                crop_record = crop_record["spec"]
            if not isinstance(crop_record, Mapping):
                raise ForecastContractError("ERA5 crop provenance must be a JSON object")
            crop = CropSpec.from_provenance(dict(crop_record))
            if dict(crop_record) != crop.to_provenance():
                raise ForecastContractError("ERA5 crop provenance is not canonical")
            config = EvaluationRunConfig.from_mapping(
                _read_json(args.evaluation_config, "evaluation run config"))
            job = EvaluationJob(
                schema=EVALUATION_JOB_SCHEMA, forecast_request=request,
                forecast_result=result, era5_crop=crop, evaluation_config=config)
            save_evaluation_job(args.output, job)
            output = {
                "schema": "external-evaluation-job-created/v1",
                "job_path": str(Path(args.output).resolve()),
                "job_sha256": job.fingerprint(),
                "network_used": False,
                "model_inference_used": False,
                "claim_boundary": job.claim_boundary,
            }
            print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
            return 0

        job = load_evaluation_job(args.job)
        if args.command == "bind":
            bindings = EvaluationPathBindings(
                schema=EVALUATION_PATH_BINDINGS_SCHEMA,
                job_sha256=job.fingerprint(),
                forecast_artifact_path=args.forecast_artifact,
                era5_cache_dir=args.era5_cache_dir,
                receipt_path=args.receipt,
            )
            save_evaluation_path_bindings(args.output, bindings)
            output = {
                "schema": "external-evaluation-path-bindings-created/v1",
                "bindings_path": str(Path(args.output).resolve()),
                "bindings_sha256": bindings.fingerprint(),
                "job_sha256": job.fingerprint(),
                "network_used": False,
            }
            print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
            return 0

        bindings = load_evaluation_path_bindings(args.bindings)
        base_dir = Path(args.bindings).resolve().parent
        if args.command == "preflight":
            output = preflight_evaluation_job(job, bindings, base_dir=base_dir)
        else:
            receipt = run_evaluation_job(job, bindings, base_dir=base_dir)
            paths = resolve_evaluation_paths(bindings, base_dir=base_dir)
            output = {
                "schema": "external-evaluation-job-completion/v1",
                "job_sha256": job.fingerprint(),
                "receipt_path": str(paths.receipt_path),
                "receipt_sha256": receipt.fingerprint(),
                "evaluation_sha256": receipt.evaluation_sha256,
                "network_used": False,
                "model_inference_used": False,
                "claim_boundary": receipt.claim_boundary,
            }
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ForecastContractError, SpectralEarthError, FileExistsError, OSError, ValueError) as exc:
        print("SpectralEarth evaluation job: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
