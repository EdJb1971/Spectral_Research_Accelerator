"""Source-bound construction of one G17 partner-record profile."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict

from src.core.errors import InvalidParameterError, UserInputError
from src.core.partner_pool import RecordProfile
from src.core.real_pool_method_review import AdoptedMarginalMethodReview


PROFILE_SCHEMA = "correspondence-record-profile/v3"


def _nonempty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise InvalidParameterError(name, value, "a non-empty declaration")
    return text


@dataclass(frozen=True)
class RecordProfileDeclaration:
    """Semantics that cannot be inferred from delimited record bytes."""

    record_id: str
    provenance_key: str
    time_column: str
    value_column: str
    time_units: str
    window_seconds: float
    native_seconds: float
    native_seconds_basis: str
    effective_sample_size: float
    effective_sample_size_method: str
    noise_floor: float
    noise_floor_method: str

    def __post_init__(self) -> None:
        for name in (
                "record_id", "provenance_key", "time_column", "value_column", "time_units",
                "native_seconds_basis", "effective_sample_size_method", "noise_floor_method"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.time_units != "seconds":
            raise InvalidParameterError(
                "time_units", self.time_units,
                "'seconds'; conversion must be explicit before cadence_seconds is derived")


def build_source_bound_profile(
        payload: bytes, *, filename: str, delimiter: str,
        declaration: RecordProfileDeclaration,
        method_review: AdoptedMarginalMethodReview) -> Dict[str, Any]:
    """Derive storage facts from exact bytes and retain declared scientific marginals."""
    reviewed = method_review.review
    declared_methods = {
        "native_seconds_basis": declaration.native_seconds_basis,
        "effective_sample_size_method": declaration.effective_sample_size_method,
        "noise_floor_method": declaration.noise_floor_method,
    }
    reviewed_methods = {name: getattr(reviewed, name) for name in declared_methods}
    if declared_methods != reviewed_methods:
        raise UserInputError(
            "G17 profile methods do not exactly match the adopted marginal method review")
    if delimiter not in (",", "\t", ";", "|"):
        raise InvalidParameterError("delimiter", delimiter, "one of comma, tab, semicolon or pipe")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise UserInputError("G17 record bytes must be UTF-8 delimited text") from error

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise UserInputError("G17 record column names must be unique")
    required = {declaration.time_column, declaration.value_column}
    if not required.issubset(columns):
        raise UserInputError(
            "G17 record is missing declared columns: %s" % sorted(required - set(columns)))

    times = []
    for row_number, row in enumerate(reader, start=2):
        try:
            time = float(row[declaration.time_column])
            value = float(row[declaration.value_column])
        except (TypeError, ValueError) as error:
            raise UserInputError(
                "G17 record row %d has a non-numeric declared time or value" % row_number
            ) from error
        if not math.isfinite(time) or not math.isfinite(value):
            raise UserInputError(
                "G17 record row %d has a non-finite declared time or value" % row_number)
        times.append(time)

    if len(times) < 2:
        raise UserInputError("G17 record needs at least two finite observations")
    differences = [right - left for left, right in zip(times, times[1:])]
    if any(value <= 0.0 for value in differences):
        raise UserInputError("G17 record times must be strictly increasing")
    cadence = float(statistics.median(differences))
    span = times[-1] - times[0]
    window = float(declaration.window_seconds)
    if not math.isfinite(window) or window <= 0.0 or span > window:
        raise InvalidParameterError(
            "window_seconds", declaration.window_seconds,
            "a positive finite declared window at least as long as the observed time span")
    coverage = min(1.0, ((len(times) - 1) * cadence) / window)

    profile = RecordProfile(
        record_id=declaration.record_id,
        provenance_key=declaration.provenance_key,
        n_samples=len(times),
        effective_sample_size=declaration.effective_sample_size,
        native_seconds=declaration.native_seconds,
        cadence_seconds=cadence,
        coverage_fraction=coverage,
        noise_floor=declaration.noise_floor,
    )
    return {
        "schema": PROFILE_SCHEMA,
        "source": {
            "filename": _nonempty("filename", filename),
            "content_sha256": hashlib.sha256(payload).hexdigest(),
            "format": "delimited_text",
            "time_column": declaration.time_column,
            "value_column": declaration.value_column,
            "time_units": declaration.time_units,
            "window_seconds": window,
        },
        "derivation": {
            "n_samples": "finite declared time/value rows",
            "cadence_seconds": "median positive adjacent time difference",
            "coverage_fraction": "min(1, (n_samples - 1) * cadence / declared_window)",
            "native_seconds": declaration.native_seconds_basis,
            "effective_sample_size": declaration.effective_sample_size_method,
            "noise_floor": declaration.noise_floor_method,
        },
        "method_review": method_review.binding,
        "profile": profile.describe(),
    }


__all__ = ["PROFILE_SCHEMA", "RecordProfileDeclaration", "build_source_bound_profile"]