"""Repository-local readiness audit for the G17 real partner-pool obligation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.core.errors import UserInputError
from src.core.partner_pool import RecordProfile
from src.core.real_pool_ingress import PROFILE_SCHEMA
from src.core.scale_shape_successor import ScaleShapeSuccessorDeclaration


SCHEMA = "g17-real-pool-readiness/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_FILE = REPO_ROOT / "measurements" / "g17_real_pool_readiness.json"
DEFAULT_PROFILE_ROOTS = (
    REPO_ROOT / "data" / "profile_collections",
    REPO_ROOT / "data" / "channels",
)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _profile_documents(roots: Sequence[Path]) -> Iterable[Dict[str, Any]]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.partner-profile.json")):
            raw = path.read_bytes()
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
                if body.get("schema") != PROFILE_SCHEMA \
                        or set(body) != {
                            "schema", "source", "derivation", "method_review", "profile"}:
                    raise ValueError("expected a source-bound %s envelope" % PROFILE_SCHEMA)
                source = body["source"]
                if not isinstance(source, dict) or set(source) != {
                        "filename", "content_sha256", "format", "time_column", "value_column",
                        "time_units", "window_seconds"}:
                    raise ValueError("invalid source binding")
                for name in ("filename", "time_column", "value_column"):
                    if not isinstance(source[name], str) or not source[name].strip():
                        raise ValueError("invalid source %s" % name)
                if source["format"] != "delimited_text" or source["time_units"] != "seconds":
                    raise ValueError("invalid source format or time units")
                window = source["window_seconds"]
                if isinstance(window, bool) or not isinstance(window, (int, float)) \
                        or not math.isfinite(float(window)) or float(window) <= 0.0:
                    raise ValueError("invalid source window_seconds")
                digest = source["content_sha256"]
                if not isinstance(digest, str) or len(digest) != 64 \
                        or any(value not in "0123456789abcdef" for value in digest):
                    raise ValueError("invalid source content_sha256")
                derivation = body["derivation"]
                if not isinstance(derivation, dict) or set(derivation) != {
                        "n_samples", "cadence_seconds", "coverage_fraction", "native_seconds",
                        "effective_sample_size", "noise_floor"} \
                        or any(not isinstance(value, str) or not value.strip()
                               for value in derivation.values()):
                    raise ValueError("invalid marginal derivation record")
                method_review = body["method_review"]
                if not isinstance(method_review, dict) or set(method_review) != {
                        "schema", "declaration_file", "declaration_sha256", "adoption_file",
                        "adoption_sha256", "adopted_by", "adopted_on"}:
                    raise ValueError("invalid marginal method review binding")
                for name in ("declaration_file", "adoption_file", "adopted_by", "adopted_on"):
                    if not isinstance(method_review[name], str) or not method_review[name].strip():
                        raise ValueError("invalid marginal method review %s" % name)
                for name in ("declaration_sha256", "adoption_sha256"):
                    digest = method_review[name]
                    if not isinstance(digest, str) or len(digest) != 64 \
                            or any(value not in "0123456789abcdef" for value in digest):
                        raise ValueError("invalid marginal method review %s" % name)
                profile = RecordProfile(**body["profile"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise UserInputError(
                    "%s is not a valid G17 partner profile: %s" % (_relative(path), error)
                ) from error
            yield {
                "path": _relative(path),
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "source": source,
                "derivation": derivation,
                "method_review": method_review,
                "profile": profile.describe(),
            }


def audit_real_pool_readiness(
        declaration: ScaleShapeSuccessorDeclaration,
        *, roots: Sequence[Path] = DEFAULT_PROFILE_ROOTS) -> Dict[str, Any]:
    """Count explicit profile documents; never infer or award exchangeability."""
    documents = list(_profile_documents(roots))
    record_ids = [row["profile"]["record_id"] for row in documents]
    if len(record_ids) != len(set(record_ids)):
        raise UserInputError("G17 partner profile record identities must be unique")
    provenance_keys = [row["profile"]["provenance_key"] for row in documents]
    if len(provenance_keys) != len(set(provenance_keys)):
        raise UserInputError("G17 partner profile provenance identities must be unique")

    required = declaration.minimum_pool_size_per_correspondence
    count = len(documents)
    status = (
        "NO_INVENTORY" if count == 0 else
        "INSUFFICIENT" if count < required else
        "READY_FOR_CURATION_REVIEW"
    )
    body: Dict[str, Any] = {
        "schema": SCHEMA,
        "study_id": declaration.study_id,
        "successor_declaration_sha256": declaration.digest,
        "searched_roots": [_relative(root) for root in roots],
        "profile_file_pattern": "*.partner-profile.json",
        "required_profiles_per_correspondence": required,
        "profile_count": count,
        "shortfall": max(0, required - count),
        "status": status,
        "profiles": documents,
        "exchangeability": "NOT_ASSESSED",
        "claim_boundary": (
            "This audit establishes only whether explicit single-record profiles exist in enough "
            "quantity to begin curation. It does not establish admission to any correspondence "
            "pool or exchangeability. An empty inventory is absence of evidence, not evidence "
            "that real records are non-exchangeable."
        ),
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["assessment_sha256"] = hashlib.sha256(encoded).hexdigest()
    return body


__all__ = [
    "DEFAULT_PROFILE_ROOTS",
    "MEASUREMENT_FILE",
    "PROFILE_SCHEMA",
    "SCHEMA",
    "audit_real_pool_readiness",
]