"""Candidate marginal assessment for acquired, period-qualified TESS records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.benchmarks.pool_calibration import CALIBRATION_CONTRACT
from src.core.errors import UserInputError
from src.core.partner_pool import RecordProfile
from src.core.real_pool_method_review import load_adopted_method_review
from src.data_layer.lightcurves import LightCurveProduct, LightCurveSpec
from src.data_layer.tess_source import parse_spoc_fits


SCHEMA = "g17-tess-pool-assessment/v1"
MERGED_ACQUISITION_SCHEMA = "spectral.tess-pool-acquisition-collection/v1"
QUALIFIED_ACQUISITION_SCHEMA = "spectral.tess-qualified-pool-acquisition/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD_REVIEW = REPO_ROOT / "data" / "studies" / "g17_tess_marginal_method_review.json"


def merge_tess_acquisitions(receipt_paths: List[Path]) -> Dict[str, Any]:
    """Merge immutable period-qualified receipts without collapsing provenance."""
    if not receipt_paths:
        raise UserInputError("at least one TESS acquisition receipt is required")
    targets = []
    sources = []
    seen = set()
    total = 0
    for path in receipt_paths:
        raw = path.read_bytes()
        try:
            body = json.loads(raw)
            if body.get("schema") != "spectral.tess-pool-acquisition/v1" \
                    or body.get("discovery_schema") != \
                    "spectral.tess-period-product-discovery/v1":
                raise ValueError("expected a period-qualified acquisition receipt")
            if body.get("target_count") != len(body["targets"]):
                raise ValueError("target_count does not match targets")
        except (KeyError, TypeError, ValueError) as error:
            raise UserInputError("invalid TESS acquisition receipt %s: %s" % (path, error)) \
                from error
        for target in body["targets"]:
            target_id = str(target.get("tic_id", ""))
            if target_id in seen:
                raise UserInputError("TESS acquisition target identities must be unique")
            seen.add(target_id)
            targets.append(target)
        total += int(body["downloaded_bytes"])
        sources.append({"path": path.as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "target_count": body["target_count"]})
    return {
        "schema": MERGED_ACQUISITION_SCHEMA,
        "source_receipts": sources,
        "target_count": len(targets),
        "downloaded_bytes": total,
        "targets": targets,
        "claim_boundary": (
            "A lossless inventory of acquired period-qualified products. Merging does not "
            "establish profile validity, admission or exchangeability."),
    }


def qualify_tess_acquisition(receipt_path: Path, raw_root: Path) -> Dict[str, Any]:
    """Retain only records whose corrected metadata passes profile qualification."""
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
        if receipt.get("schema") != MERGED_ACQUISITION_SCHEMA:
            raise ValueError("expected a merged period-qualified acquisition")
        targets = receipt["targets"]
    except (KeyError, TypeError, ValueError) as error:
        raise UserInputError("invalid merged TESS acquisition: %s" % error) from error
    admitted = []
    refused = []
    for target in targets:
        try:
            _profile(target, raw_root)
            admitted.append(target)
        except (OSError, UserInputError) as error:
            refused.append({"tic_id": str(target.get("tic_id", "")), "reason": str(error)})
    return {
        "schema": QUALIFIED_ACQUISITION_SCHEMA,
        "source_receipt": {
            "path": receipt_path.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
            "target_count": len(targets),
        },
        "target_count": len(admitted),
        "refused_target_count": len(refused),
        "targets": admitted,
        "refused_targets": refused,
        "claim_boundary": (
            "Records pass the adopted single-record profile calculations, including eight "
            "declared native cycles. Pool admission and exchangeability remain unassessed."),
    }


def _profile(target: Dict[str, Any], raw_root: Path) -> RecordProfile:
    product_body = target["products"][0]
    product = LightCurveProduct(**{
        name: product_body[name] for name in (
            "observation_id", "data_uri", "filename", "sector", "size_bytes",
            "provenance_name")})
    path = raw_root / product_body["cache_file"]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != product_body["content_sha256"]:
        raise UserInputError("cached TESS product does not match its acquisition digest")
    spec = LightCurveSpec(
        target_id=target["tic_id"], sectors=(product.sector,), max_products=1,
        max_download_bytes=product.size_bytes)
    parsed = parse_spoc_fits(payload, spec, product)
    valid = np.isfinite(parsed["time"]) & np.isfinite(parsed["flux"]) \
        & np.isfinite(parsed["flux_error"]) & (parsed["quality"] == 0)
    times = np.asarray(parsed["time"])[valid] * 86400.0
    flux = np.asarray(parsed["flux"])[valid]
    errors = np.asarray(parsed["flux_error"])[valid]
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise UserInputError("TESS candidate lacks two strictly increasing admitted samples")
    cadence = float(np.median(np.diff(times)))
    span = float(times[-1] - times[0])
    native = float(target["orbital_period_seconds"])
    if span / native < 8.0:
        raise UserInputError("TESS candidate spans fewer than eight declared orbital cycles")
    centered = flux - float(np.mean(flux))
    denominator = float(np.dot(centered, centered))
    rho = 0.0 if denominator == 0.0 else float(
        np.dot(centered[:-1], centered[1:]) / denominator)
    rho = min(max(rho, -0.999999), 0.999999)
    effective = min(float(times.size), max(
        1.0, float(times.size) * (1.0 - rho) / (1.0 + rho)))
    robust_scale = 1.4826 * float(np.median(np.abs(flux - np.median(flux))))
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise UserInputError("TESS candidate has no positive robust flux scale")
    noise = float(np.median(errors)) / robust_scale
    return RecordProfile(
        record_id="tic-%s" % target["tic_id"],
        provenance_key="MAST-TESS-SPOC/TIC-%s" % target["tic_id"],
        n_samples=int(times.size), effective_sample_size=effective,
        native_seconds=native, cadence_seconds=cadence,
        coverage_fraction=min(1.0, (times.size - 1) * cadence / span),
        noise_floor=noise)


def assess_tess_pool_acquisition(
    receipt_path: Path, raw_root: Path,
    method_review_path: Path = DEFAULT_METHOD_REVIEW) -> Dict[str, Any]:
    """Measure candidate marginals and pool yields without creating G17 profiles."""
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
        if receipt.get("schema") == "spectral.tess-pool-acquisition/v1" \
            and receipt.get("discovery_schema") != \
            "spectral.tess-period-product-discovery/v1":
            raise ValueError("expected a period-qualified TESS acquisition")
        if receipt.get("schema") not in (
                "spectral.tess-pool-acquisition/v1", MERGED_ACQUISITION_SCHEMA,
                QUALIFIED_ACQUISITION_SCHEMA):
            raise ValueError("expected a period-qualified TESS acquisition")
        profiles = [_profile(target, raw_root) for target in receipt["targets"]]
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise UserInputError("G17 TESS acquisition cannot be assessed: %s" % error) from error

    sizes = []
    refusal_counts = {name: 0 for name in CALIBRATION_CONTRACT.ratio_bands}
    refusal_counts["minimum_coverage"] = 0
    for reference in profiles:
        admitted = 0
        for candidate in profiles:
            if candidate.record_id == reference.record_id:
                continue
            refused = False
            if candidate.coverage_fraction < CALIBRATION_CONTRACT.minimum_coverage:
                refusal_counts["minimum_coverage"] += 1
                refused = True
            for name, band in CALIBRATION_CONTRACT.ratio_bands.items():
                ratio = candidate.marginal(name) / reference.marginal(name)
                if ratio > band or ratio < 1.0 / band:
                    refusal_counts[name] += 1
                    refused = True
            if not refused:
                admitted += 1
        sizes.append({"record_id": reference.record_id, "admitted_alternatives": admitted})
    counts = sorted(row["admitted_alternatives"] for row in sizes)
    adopted_review = load_adopted_method_review(method_review_path)
    proposed = {
        "native_seconds_basis": (
            "NASA Exoplanet Archive TOI pl_orbper converted from days to seconds; not "
            "estimated from SPOC flux"),
        "effective_sample_size_method": (
            "quality-zero finite PDCSAP flux; clipped AR(1) n * (1-rho1) / (1+rho1)"),
        "noise_floor_method": (
            "median(PDCSAP_FLUX_ERR) / (1.4826 * MAD(PDCSAP_FLUX)) on quality-zero "
            "finite rows"),
    }
    if any(getattr(adopted_review.review, name) != value for name, value in proposed.items()):
        raise UserInputError("adopted TESS marginal methods do not match the assessment methods")
    return {
        "schema": SCHEMA,
        "acquisition_file": receipt_path.as_posix(),
        "acquisition_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_method_status": "ADOPTED",
        "method_review": adopted_review.binding,
        "candidate_count": len(profiles),
        "minimum_required": 48,
        "records_reaching_minimum": sum(value >= 48 for value in counts),
        "pool_size": {
            "minimum": counts[0], "median": float(np.median(counts)), "maximum": counts[-1]},
        "admission_by_record": sizes,
        "refusal_counts": refusal_counts,
        "methods": {
            "sample_selection": "finite TIME, PDCSAP_FLUX and PDCSAP_FLUX_ERR with QUALITY == 0",
            **proposed,
        },
        "claim_boundary": (
            "An adopted-method inventory feasibility assessment. No G17 profile is emitted, "
            "and pool exchangeability is not assessed."),
    }


__all__ = [
    "MERGED_ACQUISITION_SCHEMA", "QUALIFIED_ACQUISITION_SCHEMA", "SCHEMA",
    "assess_tess_pool_acquisition", "merge_tess_acquisitions", "qualify_tess_acquisition",
]