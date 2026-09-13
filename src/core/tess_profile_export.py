"""Materialize adopted-method G17 profiles from acquired periodic TESS records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.core.errors import UserInputError
from src.core.publication import publish_new_bytes
from src.core.real_pool_ingress import RecordProfileDeclaration, build_source_bound_profile
from src.core.real_pool_method_review import load_adopted_method_review
from src.core.tess_pool_assessment import (
    DEFAULT_METHOD_REVIEW, QUALIFIED_ACQUISITION_SCHEMA, _profile)
from src.data_layer.lightcurves import LightCurveProduct, LightCurveSpec
from src.data_layer.tess_source import parse_spoc_fits


def _canonical_record(target: Dict[str, Any], raw_root: Path) -> bytes:
    product_body = target["products"][0]
    product = LightCurveProduct(**{
        name: product_body[name] for name in (
            "observation_id", "data_uri", "filename", "sector", "size_bytes",
            "provenance_name")})
    payload = (raw_root / product_body["cache_file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != product_body["content_sha256"]:
        raise UserInputError("cached TESS product does not match its acquisition digest")
    parsed = parse_spoc_fits(payload, LightCurveSpec(
        target_id=target["tic_id"], sectors=(product.sector,), max_products=1,
        max_download_bytes=product.size_bytes), product)
    valid = np.isfinite(parsed["time"]) & np.isfinite(parsed["flux"]) \
        & np.isfinite(parsed["flux_error"]) & (parsed["quality"] == 0)
    times = np.asarray(parsed["time"])[valid]
    flux = np.asarray(parsed["flux"])[valid]
    if times.size < 2:
        raise UserInputError("TESS candidate lacks two admitted samples")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("elapsed_seconds", "pdcsap_flux"))
    origin = float(times[0])
    for time_value, flux_value in zip(times, flux):
        writer.writerow((format((float(time_value) - origin) * 86400.0, ".17g"),
                         format(float(flux_value), ".17g")))
    return output.getvalue().encode("utf-8")


def export_tess_partner_profiles(
        acquisition_path: Path, *, raw_root: Path, record_root: Path, profile_root: Path,
        method_review_path: Path = DEFAULT_METHOD_REVIEW) -> Dict[str, Any]:
    """Publish canonical records and source-bound profiles for a complete acquisition."""
    raw = acquisition_path.read_bytes()
    try:
        acquisition = json.loads(raw)
        if acquisition.get("schema") not in (
                "spectral.tess-pool-acquisition/v1",
            "spectral.tess-pool-acquisition-collection/v1",
            QUALIFIED_ACQUISITION_SCHEMA):
            raise ValueError("expected a period-qualified acquisition")
        targets = acquisition["targets"]
    except (KeyError, TypeError, ValueError) as error:
        raise UserInputError("invalid TESS acquisition: %s" % error) from error
    review = load_adopted_method_review(method_review_path)
    planned = []
    for target in targets:
        profile = _profile(target, raw_root)
        payload = _canonical_record(target, raw_root)
        rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
        window_seconds = float(rows[-1]["elapsed_seconds"])
        record_path = record_root / ("tic-%s.csv" % target["tic_id"])
        profile_path = profile_root / ("tic-%s.partner-profile.json" % target["tic_id"])
        if record_path.exists() or profile_path.exists():
            raise UserInputError("TESS G17 records and profiles are immutable")
        declaration = RecordProfileDeclaration(
            record_id=profile.record_id,
            provenance_key=("%s/FITS-%s" % (
                profile.provenance_key, target["products"][0]["content_sha256"])),
            time_column="elapsed_seconds", value_column="pdcsap_flux", time_units="seconds",
            window_seconds=window_seconds,
            native_seconds=profile.native_seconds,
            native_seconds_basis=review.review.native_seconds_basis,
            effective_sample_size=profile.effective_sample_size,
            effective_sample_size_method=review.review.effective_sample_size_method,
            noise_floor=profile.noise_floor,
            noise_floor_method=review.review.noise_floor_method)
        envelope = build_source_bound_profile(
            payload, filename=record_path.name, delimiter=",", declaration=declaration,
            method_review=review)
        planned.append((record_path, payload, profile_path,
                        (json.dumps(envelope, indent=2) + "\n").encode("utf-8")))

    published: List[Path] = []
    try:
        for record_path, payload, profile_path, profile_payload in planned:
            publish_new_bytes(record_path, payload, "canonical TESS G17 record")
            published.append(record_path)
            publish_new_bytes(profile_path, profile_payload, "TESS G17 partner profile")
            published.append(profile_path)
    except (OSError, FileExistsError) as error:
        for path in published:
            path.unlink(missing_ok=True)
        raise UserInputError("TESS G17 profile batch could not be published: %s" % error) from error
    return {
        "schema": "g17-tess-profile-export/v1",
        "acquisition_sha256": hashlib.sha256(raw).hexdigest(),
        "profile_count": len(planned),
        "method_review": review.binding,
        "record_root": record_root.as_posix(),
        "profile_root": profile_root.as_posix(),
        "claim_boundary": (
            "Profiles bind adopted single-record marginals. Their existence does not establish "
            "admission to any particular pool or exchangeability."),
    }


__all__ = ["export_tess_partner_profiles"]