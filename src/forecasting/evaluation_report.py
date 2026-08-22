"""Verified, content-addressed evaluation reports for the workbench (T5.6g)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

from src.data_layer.zarr_source import CATALOGUE
from src.forecasting.adapter import ForecastContractError
from src.forecasting.evaluation_run import verify_evaluation_run_receipt

EVALUATION_REPORT_SCHEMA = "evaluation-report/fcn3-era5-v1"
DEFAULT_RECEIPT_DIR = os.path.join("data", "evaluation_receipts")
MAX_RECEIPT_BYTES = 10 * 1024 * 1024


def _official_era5_source(receipt: Mapping[str, Any]) -> bool:
    manifest = receipt.get("era5_manifest", {})
    spec = manifest.get("spec", {}) if isinstance(manifest, Mapping) else {}
    entry = CATALOGUE.get(spec.get("store"))
    route = str(manifest.get("source_route", "")).lower()
    return bool(entry and spec.get("uri") == entry.get("uri")
                and "synthetic" not in route and "fixture" not in route)


def _forecast_identity(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    request = receipt["run_request"]
    regional = receipt["regional_forecast_provenance"]
    embedded = request.get("forecast_request", {})
    model = embedded.get("model", {}) if isinstance(embedded, Mapping) else {}
    software = embedded.get("software", {}) if isinstance(embedded, Mapping) else {}
    return {
        "request_sha256": request["forecast_request_sha256"],
        "result_sha256": request["forecast_result_sha256"],
        "artifact_sha256": request["forecast_artifact_sha256"],
        "validation_sha256": regional.get("validation_sha256"),
        "provider": model.get("provider"), "model": model.get("name"),
        "model_version": model.get("model_version"),
        "checkpoint_sha256": model.get("checkpoint_sha256"),
        "runner": software.get("runner"),
        "identity_detail": ("embedded" if model else
                            "hash-bound; full request manifest not embedded in this receipt version"),
    }


def build_evaluation_report(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the only payload the UI is allowed to render from a receipt."""
    verified = verify_evaluation_run_receipt(receipt)
    evaluation = verified["evaluation"]
    config = verified["run_request"]["config"]
    real_source = _official_era5_source(verified)
    metrics: List[Dict[str, Any]] = []
    ranks: List[Dict[str, Any]] = []
    for lead_key in sorted(evaluation["metrics"], key=float):
        for variable in evaluation["variables"]:
            value = evaluation["metrics"][lead_key][variable]
            metrics.append({
                "lead_hours": float(lead_key), "variable": variable, "unit": value["unit"],
                "ensemble_mean": value["ensemble_mean"], "persistence": value["persistence"],
                "crps": value["crps"], "spread_rms": value["spread_rms"],
                "spread_skill_ratio": value["spread_skill_ratio"],
                "member_errors": value["member_errors"],
                "initialization_count": value["initialization_count"],
                "gridpoint_count": value["gridpoint_count"],
            })
            histogram = value["rank_histogram"]
            ranks.append({
                "lead_hours": float(lead_key), "variable": variable,
                "area_weighted_frequency": histogram["area_weighted_frequency"],
                "fractional_tie_counts": histogram["fractional_tie_counts"],
                "bin_definition": histogram["bin_definition"],
                "tie_policy": histogram["tie_policy"], "diagnostic_only": True,
            })
    role = config["evaluation_role"]
    return {
        "schema": EVALUATION_REPORT_SCHEMA, "report_id": verified["receipt_sha256"],
        "readiness": {
            "receipt_integrity": "VERIFIED",
            "forecast_artifact": "AUTHENTICATED_AND_VALIDATED",
            "truth_source": ("DECLARED_OFFICIAL_ERA5" if real_source else "NOT_ACCEPTED_AS_REAL"),
            "display_eligible": real_source,
            "independent_holdout": role == "fresh_post_2019_holdout",
            "scientific_skill": "NOT_ESTABLISHED",
            "reason": ("Verified receipt over a declared official ERA5 catalogue source."
                       if real_source else
                       "Receipt integrity is valid, but its truth source is not an accepted official ERA5 declaration."),
        },
        "scope": {
            "split": config["split"], "split_start": config["split_start"],
            "split_end": config["split_end"], "evaluation_role": role,
            "level_hpa": config["level_hpa"], "bounds": config["bounds"],
            "variables": evaluation["variables"],
            "lead_durations_hours": evaluation["lead_durations_hours"],
            "initialization_count": evaluation["initialization_count"],
            "ensemble_members": evaluation["ensemble_members"],
            "grid_shape": evaluation["grid_shape"], "area_weighting": evaluation["area_weighting"],
        },
        "metrics": metrics, "rank_diagnostics": ranks,
        "provenance": {
            "receipt_sha256": verified["receipt_sha256"],
            "evaluation_sha256": verified["evaluation_sha256"],
            "run_request_sha256": verified["run_request_sha256"],
            "forecast": _forecast_identity(verified),
            "truth_source_sha256": verified["matched_truth_provenance"].get("source_sha256"),
            "truth_content_hash": verified["matched_truth_provenance"].get("source_content_hash"),
            "grid_coordinates_sha256": evaluation["grid_coordinates_sha256"],
            "forecast_values_sha256": evaluation["forecast_values_sha256"],
            "truth_values_sha256": evaluation["truth_values_sha256"],
            "initial_state_values_sha256": evaluation["initial_state_values_sha256"],
        },
        "claim_boundaries": [verified["claim_boundary"], evaluation["claim_boundary"]],
    }


def decode_receipt(payload: bytes) -> Mapping[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise ForecastContractError("evaluation receipt upload must contain JSON bytes")
    if len(payload) > MAX_RECEIPT_BYTES:
        raise ForecastContractError("evaluation receipt exceeds the 10 MiB upload limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForecastContractError("evaluation receipt is not valid UTF-8 JSON: %s" % exc) from exc
    return verify_evaluation_run_receipt(value)


class EvaluationReceiptStore:
    """Content-addressed JSON store; only display-eligible receipts are admitted."""

    def __init__(self, directory: Union[str, os.PathLike[str], None] = None) -> None:
        self.directory = Path(directory or DEFAULT_RECEIPT_DIR)

    def import_bytes(self, payload: bytes) -> Dict[str, Any]:
        receipt = decode_receipt(payload)
        report = build_evaluation_report(receipt)
        if not report["readiness"]["display_eligible"]:
            raise ForecastContractError(report["readiness"]["reason"])
        target = self.directory / (report["report_id"] + ".json")
        self.directory.mkdir(parents=True, exist_ok=True)
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
        if target.exists():
            existing = decode_receipt(target.read_bytes())
            if existing["receipt_sha256"] != report["report_id"]:
                raise ForecastContractError("content-addressed receipt path contains different bytes")
        else:
            try:
                with target.open("xb") as handle:
                    handle.write(canonical)
            except FileExistsError:
                return self.get(report["report_id"])
        return report

    def get(self, report_id: str) -> Dict[str, Any]:
        if (not isinstance(report_id, str) or len(report_id) != 64
                or any(ch not in "0123456789abcdef" for ch in report_id.lower())):
            raise ForecastContractError("report id must be a 64-character hexadecimal SHA-256")
        target = self.directory / (report_id.lower() + ".json")
        if not target.is_file():
            raise FileNotFoundError("evaluation report not found")
        report = build_evaluation_report(decode_receipt(target.read_bytes()))
        if not report["readiness"]["display_eligible"]:
            raise ForecastContractError("stored receipt is no longer display-eligible")
        return report

    def list(self) -> List[Dict[str, Any]]:
        if not self.directory.is_dir():
            return []
        reports = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                reports.append(self.get(path.stem))
            except (ForecastContractError, FileNotFoundError):
                continue
        return reports
