"""TG6.1: hashed, append-only scientific evidence bundles.

An evidence bundle is an immutable snapshot of one hypothesis and the evidence accumulated
around it.  Evidence is appended by returning a new bundle; existing entries cannot be edited
or removed through this API.  Every entry binds the previous entry digest, and the outer bundle
binds the complete ordered chain and exposes contradictory evidence and failure states as
ordinary first-class scientific fields.

This module deliberately does not accept LLM commentary as evidence.  TG7 may attach recorded
commentary beside this structure, but commentary must never enter the chain read by TG6 gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.core.errors import InvalidParameterError


EVIDENCE_BUNDLE_SCHEMA = "evidence-bundle/v1"

EVIDENCE_FIELDS = (
    "observations",
    "effect_sizes",
    "uncertainty",
    "null_results",
    "replication_results",
    "holdout_performance",
    "provenance",
    "confounders",
    "contradictory_evidence",
    "failure_states",
)
EVIDENCE_STATUSES = ("PASS", "FAIL", "INVALID", "INCONCLUSIVE", "NOT_APPLICABLE")

_ENTRY_FIELDS = (
    "sequence", "category", "label", "status", "summary", "recorded_at", "payload",
    "source_sha256s", "previous_sha256", "entry_sha256",
)
_HYPOTHESIS_FIELDS = ("identifier", "statement", "prediction", "registered_at", "provenance")
_TOP_LEVEL_FIELDS = (
    "schema", "study_id", "created_at", "hypothesis", "hypothesis_sha256",
    *EVIDENCE_FIELDS, "revision", "head_sha256", "bundle_sha256",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(
            "evidence value", type(value).__name__,
            "finite, canonical JSON data without executable or process-local objects") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _moment(name: str, value: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidParameterError(name, value, "a timezone-bearing ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise InvalidParameterError(name, value,
                                    "a timezone-bearing ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise InvalidParameterError(name, value, "a timestamp with an explicit UTC offset")
    return parsed


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidParameterError(name, value, "a non-empty string")
    return value.strip()


def _sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise InvalidParameterError(name, value, "a lowercase 64-character SHA-256 digest")
    return value


def _exact(value: Any, expected: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidParameterError(name, type(value).__name__, "a JSON object")
    missing = sorted(set(expected) - set(value))
    extra = sorted(set(value) - set(expected))
    if missing or extra:
        raise InvalidParameterError(name, {"missing": missing, "extra": extra},
                                    "exactly the fields in the versioned TG6.1 schema")
    return value


def _finite_json(value: Any, name: str) -> Any:
    """Copy JSON-shaped data and reject NaN, infinity and non-string mapping keys."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise InvalidParameterError(name, list(value), "a JSON object with string keys")
        return {key: _finite_json(item, "%s.%s" % (name, key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, name) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise InvalidParameterError(name, value, "finite JSON data")


@dataclass(frozen=True)
class Hypothesis:
    """The testable statement whose evidence is being accumulated."""

    identifier: str
    statement: str
    prediction: str
    registered_at: str
    provenance: Mapping[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifier", _nonempty("hypothesis.identifier", self.identifier))
        object.__setattr__(self, "statement", _nonempty("hypothesis.statement", self.statement))
        object.__setattr__(self, "prediction", _nonempty("hypothesis.prediction", self.prediction))
        _moment("hypothesis.registered_at", self.registered_at)
        provenance = _finite_json(self.provenance, "hypothesis.provenance")
        object.__setattr__(self, "provenance", _freeze(provenance))

    def to_mapping(self) -> Dict[str, Any]:
        return {"identifier": self.identifier, "statement": self.statement,
                "prediction": self.prediction, "registered_at": self.registered_at,
                "provenance": _thaw(self.provenance)}

    def digest(self) -> str:
        return _digest(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Hypothesis":
        record = _exact(value, _HYPOTHESIS_FIELDS, "hypothesis")
        return cls(identifier=record["identifier"], statement=record["statement"],
                   prediction=record["prediction"], registered_at=record["registered_at"],
                   provenance=record["provenance"])


@dataclass(frozen=True)
class EvidenceEntry:
    """One ordered observation in the bundle's tamper-evident evidence chain."""

    sequence: int
    category: str
    label: str
    status: str
    summary: str
    recorded_at: str
    payload: Mapping[str, Any]
    source_sha256s: Tuple[str, ...]
    previous_sha256: str
    entry_sha256: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) \
                or self.sequence < 1:
            raise InvalidParameterError("evidence.sequence", self.sequence, "a positive integer")
        if self.category not in EVIDENCE_FIELDS:
            raise InvalidParameterError(
                "evidence.category", self.category,
                "one of %s. Commentary is not scientific evidence" % ", ".join(EVIDENCE_FIELDS))
        object.__setattr__(self, "label", _nonempty("evidence.label", self.label))
        if self.status not in EVIDENCE_STATUSES:
            raise InvalidParameterError("evidence.status", self.status,
                                        "one of %s" % ", ".join(EVIDENCE_STATUSES))
        object.__setattr__(self, "summary", _nonempty("evidence.summary", self.summary))
        _moment("evidence.recorded_at", self.recorded_at)
        payload = _finite_json(self.payload, "evidence.payload")
        if not isinstance(payload, dict) or not payload:
            raise InvalidParameterError("evidence.payload", payload,
                                        "a non-empty structured JSON object")
        object.__setattr__(self, "payload", _freeze(payload))
        sources = tuple(_sha("evidence.source_sha256s", value)
                        for value in self.source_sha256s)
        if len(set(sources)) != len(sources):
            raise InvalidParameterError("evidence.source_sha256s", sources,
                                        "unique source-artifact digests")
        object.__setattr__(self, "source_sha256s", sources)
        _sha("evidence.previous_sha256", self.previous_sha256)
        computed = _digest(self.body())
        if not self.entry_sha256:
            object.__setattr__(self, "entry_sha256", computed)
        elif self.entry_sha256 != computed:
            raise InvalidParameterError("evidence.entry_sha256", self.entry_sha256,
                                        "the digest recomputed from this evidence entry")

    def body(self) -> Dict[str, Any]:
        return {"sequence": self.sequence, "category": self.category, "label": self.label,
                "status": self.status, "summary": self.summary,
                "recorded_at": self.recorded_at, "payload": _thaw(self.payload),
                "source_sha256s": list(self.source_sha256s),
                "previous_sha256": self.previous_sha256}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), entry_sha256=self.entry_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceEntry":
        record = _exact(value, _ENTRY_FIELDS, "evidence entry")
        return cls(sequence=record["sequence"], category=record["category"],
                   label=record["label"], status=record["status"],
                   summary=record["summary"], recorded_at=record["recorded_at"],
                   payload=record["payload"], source_sha256s=tuple(record["source_sha256s"]),
                   previous_sha256=record["previous_sha256"],
                   entry_sha256=record["entry_sha256"])


@dataclass(frozen=True)
class EvidenceBundle:
    """An immutable hypothesis plus an ordered, category-visible evidence chain."""

    study_id: str
    created_at: str
    hypothesis: Hypothesis
    entries: Tuple[EvidenceEntry, ...] = ()
    bundle_sha256: str = ""
    schema: str = dc_field(default=EVIDENCE_BUNDLE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _nonempty("study_id", self.study_id))
        created = _moment("created_at", self.created_at)
        if not isinstance(self.hypothesis, Hypothesis):
            raise InvalidParameterError("hypothesis", type(self.hypothesis).__name__,
                                        "a Hypothesis")
        if _moment("hypothesis.registered_at", self.hypothesis.registered_at) > created:
            raise InvalidParameterError("created_at", self.created_at,
                                        "at or after hypothesis registration")
        entries = tuple(self.entries)
        previous = self._anchor_sha256()
        last_time = created
        for sequence, entry in enumerate(entries, 1):
            if not isinstance(entry, EvidenceEntry):
                raise InvalidParameterError("entries", type(entry).__name__,
                                            "EvidenceEntry records")
            if entry.sequence != sequence:
                raise InvalidParameterError("evidence.sequence", entry.sequence,
                                            "the uninterrupted append order %d" % sequence)
            if entry.previous_sha256 != previous:
                raise InvalidParameterError("evidence.previous_sha256", entry.previous_sha256,
                                            "the prior entry digest %s" % previous)
            recorded = _moment("evidence.recorded_at", entry.recorded_at)
            if recorded < last_time:
                raise InvalidParameterError("evidence.recorded_at", entry.recorded_at,
                                            "a non-decreasing append chronology")
            previous = entry.entry_sha256
            last_time = recorded
        object.__setattr__(self, "entries", entries)
        computed = _digest(self.body())
        if not self.bundle_sha256:
            object.__setattr__(self, "bundle_sha256", computed)
        elif self.bundle_sha256 != computed:
            raise InvalidParameterError("bundle_sha256", self.bundle_sha256,
                                        "the digest recomputed from the complete evidence bundle")

    def _anchor_sha256(self) -> str:
        return _digest({"schema": self.schema, "study_id": self.study_id,
                        "created_at": self.created_at,
                        "hypothesis_sha256": self.hypothesis.digest()})

    @property
    def revision(self) -> int:
        return len(self.entries)

    @property
    def head_sha256(self) -> str:
        return self.entries[-1].entry_sha256 if self.entries else self._anchor_sha256()

    def evidence(self, category: str) -> Tuple[EvidenceEntry, ...]:
        if category not in EVIDENCE_FIELDS:
            raise InvalidParameterError("evidence.category", category,
                                        "one of %s" % ", ".join(EVIDENCE_FIELDS))
        return tuple(entry for entry in self.entries if entry.category == category)

    @property
    def contradictory_evidence(self) -> Tuple[EvidenceEntry, ...]:
        return self.evidence("contradictory_evidence")

    @property
    def failure_states(self) -> Tuple[EvidenceEntry, ...]:
        return self.evidence("failure_states")

    def append(self, category: str, *, label: str, status: str, summary: str,
               recorded_at: str, payload: Mapping[str, Any],
               source_sha256s: Sequence[str] = ()) -> "EvidenceBundle":
        """Return the next immutable snapshot; the receiver remains byte-for-byte unchanged."""
        entry = EvidenceEntry(
            sequence=self.revision + 1, category=category, label=label, status=status,
            summary=summary, recorded_at=recorded_at, payload=payload,
            source_sha256s=tuple(source_sha256s), previous_sha256=self.head_sha256)
        return EvidenceBundle(study_id=self.study_id, created_at=self.created_at,
                              hypothesis=self.hypothesis, entries=self.entries + (entry,))

    def body(self) -> Dict[str, Any]:
        grouped = {category: [entry.to_mapping() for entry in self.entries
                              if entry.category == category]
                   for category in EVIDENCE_FIELDS}
        return {"schema": self.schema, "study_id": self.study_id,
                "created_at": self.created_at,
                "hypothesis": self.hypothesis.to_mapping(),
                "hypothesis_sha256": self.hypothesis.digest(), **grouped,
                "revision": self.revision, "head_sha256": self.head_sha256}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), bundle_sha256=self.bundle_sha256)

    def verify_published(self, published_sha256: str) -> None:
        computed = _digest(self.body())
        if self.bundle_sha256 != computed or str(published_sha256).lower() != computed:
            raise InvalidParameterError("published_bundle_sha256", published_sha256,
                                        "the unchanged evidence bundle digest %s" % computed)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceBundle":
        record = _exact(value, _TOP_LEVEL_FIELDS, "evidence bundle")
        if record["schema"] != EVIDENCE_BUNDLE_SCHEMA:
            raise InvalidParameterError("schema", record["schema"], EVIDENCE_BUNDLE_SCHEMA)
        hypothesis = Hypothesis.from_mapping(record["hypothesis"])
        if record["hypothesis_sha256"] != hypothesis.digest():
            raise InvalidParameterError("hypothesis_sha256", record["hypothesis_sha256"],
                                        "the digest recomputed from the hypothesis")
        entries = []
        for category in EVIDENCE_FIELDS:
            values = record[category]
            if not isinstance(values, list):
                raise InvalidParameterError(category, type(values).__name__, "a JSON array")
            for value_item in values:
                entry = EvidenceEntry.from_mapping(value_item)
                if entry.category != category:
                    raise InvalidParameterError("evidence.category", entry.category,
                                                "its enclosing first-class field %s" % category)
                entries.append(entry)
        entries.sort(key=lambda item: item.sequence)
        bundle = cls(study_id=record["study_id"], created_at=record["created_at"],
                     hypothesis=hypothesis, entries=tuple(entries),
                     bundle_sha256=record["bundle_sha256"])
        if record["revision"] != bundle.revision or record["head_sha256"] != bundle.head_sha256:
            raise InvalidParameterError("evidence chain", record["head_sha256"],
                                        "the recomputed revision and chain head")
        return bundle


def create_evidence_bundle(*, study_id: str, hypothesis_id: str, statement: str,
                           prediction: str, registered_at: str, created_at: str,
                           provenance: Optional[Mapping[str, Any]] = None) -> EvidenceBundle:
    """Create revision zero before any observation has been attached."""
    return EvidenceBundle(
        study_id=study_id, created_at=created_at,
        hypothesis=Hypothesis(identifier=hypothesis_id, statement=statement,
                              prediction=prediction, registered_at=registered_at,
                              provenance=dict(provenance or {})))


def save_evidence_bundle(path: Union[str, os.PathLike[str]], bundle: EvidenceBundle) -> str:
    """Publish one canonical snapshot exclusively; existing evidence is never overwritten."""
    if not isinstance(bundle, EvidenceBundle):
        raise InvalidParameterError("bundle", type(bundle).__name__, "an EvidenceBundle")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(bundle.to_mapping()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("refusing to overwrite evidence bundle at %s" % target) from None
    return bundle.bundle_sha256


def load_evidence_bundle(path: Union[str, os.PathLike[str]], *,
                         published_sha256: Optional[str] = None) -> EvidenceBundle:
    target = Path(path)
    try:
        raw = target.read_bytes()
        record = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidParameterError("evidence bundle path", str(target),
                                    "readable UTF-8 JSON: %s" % exc) from exc
    bundle = EvidenceBundle.from_mapping(record)
    if raw != _canonical(bundle.to_mapping()) + b"\n":
        raise InvalidParameterError("evidence bundle bytes", str(target), "canonical JSON")
    if published_sha256 is not None:
        bundle.verify_published(published_sha256)
    return bundle


__all__ = [
    "EVIDENCE_BUNDLE_SCHEMA", "EVIDENCE_FIELDS", "EVIDENCE_STATUSES", "Hypothesis",
    "EvidenceEntry", "EvidenceBundle", "create_evidence_bundle", "save_evidence_bundle",
    "load_evidence_bundle",
]
