"""TG7.1: the recorded-call boundary.

Phase G7 puts an adversarial review layer *above* the G6 gates.  This module is the only door
through which that layer may speak, and it is built so the layer cannot reach the gates through
it.

Three things are enforced here.

**Every call is recorded verbatim (R23).**  A `RecordedCall` carries the exact request that was
sent, the exact response that came back, the exact model id, the effort setting, the API request
id and both timestamps, because an LLM output *cannot be regenerated*.  Sampling parameters are
refused rather than recorded: pinning `temperature` or a `seed` would suggest a reproducibility
that does not exist.  The record says `recorded-not-reproducible` in its own body, and
`render()` states that in words wherever commentary is displayed.

**Every response is constrained to a declared schema.**  A `ResponseSchema` is sent as the
request's `output_config.format`, and the recorded response must parse as JSON matching it
exactly.  Free text arriving where a schema was declared is a refusal, not a string for later
code to parse hopefully.

**No recorded output is an input to any claim level (R22).**  Commentary lives in a
`ReviewRecord` that binds the bundle's digest from outside; it is stored in its own file beside
the bundle and is never appended to the evidence chain.  A `ReviewedBundle` pairs the two, and
every claim-bearing property it exposes is computed from `self.bundle` alone.  Deleting the
whole review is `strip_review`, which returns that same bundle unchanged --- and
`verify_claim_independence` checks the claim state both ways round, plus that no recorded
response has found its way into the bundle's bytes.

This module opens no socket.  The transport is injected, so the client that eventually talks to
the API is a caller's concern and the boundary stays testable without one.
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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

from src.core.claim_ladder import ClaimLadderAssessment, assess_claim_ladder
from src.core.errors import InvalidParameterError
from src.core.evidence import EvidenceBundle
from src.core.five_outputs import FiveOutputs, summarise_evidence


RECORDED_CALL_SCHEMA = "recorded-call/v1"
REVIEW_RECORD_SCHEMA = "review-record/v1"

#: The TG7.2 round-robin's roles, fixed here so a call cannot invent an authority for itself.
REVIEW_ROLES = (
    "candidate_synthesis",
    "statistical_challenge",
    "confounder_challenge",
    "domain_plausibility_challenge",
    "provenance_challenge",
    "response_and_revision",
    "independent_reassessment",
    "final_synthesis",
)

EFFORTS = ("low", "medium", "high")

#: Removed on current Claude models and rejected with a 400.  Refused here for the same reason
#: they were removed: they imply a reproducibility this layer does not have (R23).
FORBIDDEN_SAMPLING_PARAMETERS = ("temperature", "top_p", "top_k", "seed")

DETERMINISM = "recorded-not-reproducible"

R23_DECLARATION = (
    "Recorded output of a non-deterministic process. It cannot be regenerated, and it is "
    "commentary beside the evidence, never evidence: no line below moved any claim level "
    "(R22, R23)."
)

SCHEMA_KINDS = ("string", "number", "boolean", "string_list")

_TRANSPORT_FIELDS = ("raw", "structured", "model_id", "api_request_id", "responded_at", "usage")
_REQUEST_FIELDS = ("role", "model_id", "effort", "system", "instruction", "context",
                   "response_schema", "bundle_sha256", "bundle_revision", "request_sha256")
_RESPONSE_FIELDS = ("raw", "structured", "model_id", "api_request_id", "responded_at", "usage",
                    "response_sha256")
_CALL_FIELDS = ("schema", "sequence", "request", "response", "requested_at", "determinism",
                "previous_sha256", "call_sha256")
_RECORD_FIELDS = ("schema", "study_id", "bundle_sha256", "bundle_revision", "calls", "revision",
                  "head_sha256", "record_sha256")
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
        raise InvalidParameterError("recorded value", type(value).__name__,
                                    "finite, canonical JSON data") from exc


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
                                    "exactly the fields in the versioned TG7.1 schema")
    return value


def _finite_json(value: Any, name: str) -> Any:
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


def _no_sampling_parameters(name: str, value: Mapping[str, Any]) -> None:
    """Refuse an attempt to pin sampling, at any depth: R23 is not negotiable per call."""
    for key, item in value.items():
        if key in FORBIDDEN_SAMPLING_PARAMETERS:
            raise InvalidParameterError(
                "%s.%s" % (name, key), item,
                "no sampling parameter at all. These are rejected by the API and would imply a "
                "reproducibility this layer does not have (R23)")
        if isinstance(item, Mapping):
            _no_sampling_parameters("%s.%s" % (name, key), item)


#: A recorded phrase shorter than this could coincide with legitimate wording, so the smuggling
#: check ignores it.  Anything long enough to be an argument is long enough to be caught.
SMUGGLING_FLOOR = 24


def _phrases(value: Any) -> Tuple[str, ...]:
    """Every recorded string long enough that finding it in a bundle could not be coincidence."""
    found: list = []
    if isinstance(value, Mapping):
        for item in value.values():
            found.extend(_phrases(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_phrases(item))
    elif isinstance(value, str) and len(value.strip()) >= SMUGGLING_FLOOR:
        found.append(value.strip())
    return tuple(found)


# --------------------------------------------------------------------------------------------
# Structured outputs: commentary arrives in a declared shape or it does not arrive
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """One field of a declared response schema, optionally restricted to an enumeration."""

    kind: str
    enum: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in SCHEMA_KINDS:
            raise InvalidParameterError("field.kind", self.kind,
                                        "one of %s" % ", ".join(SCHEMA_KINDS))
        values = tuple(self.enum)
        if values and self.kind != "string":
            raise InvalidParameterError("field.enum", values,
                                        "an enumeration only on a string field")
        for value in values:
            _nonempty("field.enum", value)
        if len(set(values)) != len(values):
            raise InvalidParameterError("field.enum", values, "unique enumeration values")
        object.__setattr__(self, "enum", values)

    def to_mapping(self) -> Dict[str, Any]:
        if self.kind == "string_list":
            shape: Dict[str, Any] = {"type": "array", "items": {"type": "string"}}
        elif self.kind == "number":
            shape = {"type": "number"}
        elif self.kind == "boolean":
            shape = {"type": "boolean"}
        else:
            shape = {"type": "string"}
            if self.enum:
                shape["enum"] = list(self.enum)
        return shape

    def accepts(self, value: Any) -> bool:
        if self.kind == "string":
            return isinstance(value, str) and (not self.enum or value in self.enum)
        if self.kind == "number":
            return (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(float(value)))
        if self.kind == "boolean":
            return isinstance(value, bool)
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FieldSpec":
        if not isinstance(value, Mapping):
            raise InvalidParameterError("field", type(value).__name__, "a JSON object")
        shape = value.get("type")
        if shape == "array":
            return cls(kind="string_list")
        if shape == "number":
            return cls(kind="number")
        if shape == "boolean":
            return cls(kind="boolean")
        if shape == "string":
            return cls(kind="string", enum=tuple(value.get("enum", ())))
        raise InvalidParameterError("field.type", shape, "one of array, boolean, number, string")


@dataclass(frozen=True)
class ResponseSchema:
    """The shape a response must take.  Sent as `output_config.format`, checked on the way back."""

    name: str
    fields: Mapping[str, FieldSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("response_schema.name", self.name))
        if not isinstance(self.fields, Mapping) or not self.fields:
            raise InvalidParameterError("response_schema.fields", self.fields,
                                        "a non-empty mapping of field name to FieldSpec")
        for key, spec in self.fields.items():
            _nonempty("response_schema.field", key)
            if not isinstance(spec, FieldSpec):
                raise InvalidParameterError("response_schema.fields.%s" % key,
                                            type(spec).__name__, "a FieldSpec")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))

    def to_mapping(self) -> Dict[str, Any]:
        """The `output_config.format` block, verbatim as it is sent."""
        return {
            "type": "json_schema",
            "name": self.name,
            "schema": {
                "type": "object",
                "properties": {key: spec.to_mapping()
                               for key, spec in sorted(self.fields.items())},
                "required": sorted(self.fields),
                "additionalProperties": False,
            },
        }

    def digest(self) -> str:
        return _digest(self.to_mapping())

    def validate(self, value: Any) -> Dict[str, Any]:
        """Return the response as structured data, or refuse it."""
        if not isinstance(value, Mapping):
            raise InvalidParameterError(
                "response.structured", type(value).__name__,
                "a JSON object matching the declared schema %r. Free text where a schema was "
                "declared is refused rather than parsed hopefully" % self.name)
        record = _exact(value, tuple(self.fields), "response against schema %r" % self.name)
        for key, spec in sorted(self.fields.items()):
            if not spec.accepts(record[key]):
                raise InvalidParameterError(
                    "response.%s" % key, record[key],
                    "a %s%s as declared by schema %r"
                    % (spec.kind, " from %s" % ", ".join(spec.enum) if spec.enum else "",
                       self.name))
        return _finite_json(dict(record), "response")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResponseSchema":
        if not isinstance(value, Mapping) or value.get("type") != "json_schema":
            raise InvalidParameterError("response_schema", value, "a json_schema output format")
        shape = value.get("schema")
        if not isinstance(shape, Mapping) or not isinstance(shape.get("properties"), Mapping):
            raise InvalidParameterError("response_schema.schema", shape,
                                        "an object schema with properties")
        fields = {key: FieldSpec.from_mapping(spec)
                  for key, spec in shape["properties"].items()}
        return cls(name=value.get("name", ""), fields=fields)


# --------------------------------------------------------------------------------------------
# The call itself
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CallRequest:
    """Exactly what was sent, recorded because the answer to it cannot be regenerated."""

    role: str
    model_id: str
    effort: str
    system: str
    instruction: str
    context: Mapping[str, Any]
    response_schema: ResponseSchema
    bundle_sha256: str
    bundle_revision: int
    request_sha256: str = ""

    def __post_init__(self) -> None:
        if self.role not in REVIEW_ROLES:
            raise InvalidParameterError("request.role", self.role,
                                        "one of %s" % ", ".join(REVIEW_ROLES))
        object.__setattr__(self, "model_id", _nonempty("request.model_id", self.model_id))
        if self.effort not in EFFORTS:
            raise InvalidParameterError("request.effort", self.effort,
                                        "one of %s" % ", ".join(EFFORTS))
        object.__setattr__(self, "system", _nonempty("request.system", self.system))
        object.__setattr__(self, "instruction", _nonempty("request.instruction",
                                                          self.instruction))
        if not isinstance(self.response_schema, ResponseSchema):
            raise InvalidParameterError("request.response_schema",
                                        type(self.response_schema).__name__, "a ResponseSchema")
        context = _finite_json(self.context, "request.context")
        if not isinstance(context, dict):
            raise InvalidParameterError("request.context", type(self.context).__name__,
                                        "a JSON object")
        _no_sampling_parameters("request.context", context)
        object.__setattr__(self, "context", _freeze(context))
        _sha("request.bundle_sha256", self.bundle_sha256)
        if isinstance(self.bundle_revision, bool) or not isinstance(self.bundle_revision, int) \
                or self.bundle_revision < 0:
            raise InvalidParameterError("request.bundle_revision", self.bundle_revision,
                                        "a non-negative integer")
        computed = _digest(self.body())
        if not self.request_sha256:
            object.__setattr__(self, "request_sha256", computed)
        elif self.request_sha256 != computed:
            raise InvalidParameterError("request.request_sha256", self.request_sha256,
                                        "the digest recomputed from the verbatim request")

    def body(self) -> Dict[str, Any]:
        return {"role": self.role, "model_id": self.model_id, "effort": self.effort,
                "system": self.system, "instruction": self.instruction,
                "context": _thaw(self.context),
                "response_schema": self.response_schema.to_mapping(),
                "bundle_sha256": self.bundle_sha256, "bundle_revision": self.bundle_revision}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), request_sha256=self.request_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CallRequest":
        record = _exact(value, _REQUEST_FIELDS, "call request")
        return cls(role=record["role"], model_id=record["model_id"], effort=record["effort"],
                   system=record["system"], instruction=record["instruction"],
                   context=record["context"],
                   response_schema=ResponseSchema.from_mapping(record["response_schema"]),
                   bundle_sha256=record["bundle_sha256"],
                   bundle_revision=record["bundle_revision"],
                   request_sha256=record["request_sha256"])


@dataclass(frozen=True)
class CallResponse:
    """Exactly what came back, including the bytes, so the parse can always be re-checked."""

    raw: str
    structured: Mapping[str, Any]
    model_id: str
    api_request_id: str
    responded_at: str
    usage: Mapping[str, Any] = dc_field(default_factory=dict)
    response_sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.raw, str) or not self.raw.strip():
            raise InvalidParameterError("response.raw", self.raw,
                                        "the verbatim response text as returned")
        structured = _finite_json(self.structured, "response.structured")
        if not isinstance(structured, dict):
            raise InvalidParameterError("response.structured", type(self.structured).__name__,
                                        "a JSON object")
        object.__setattr__(self, "structured", _freeze(structured))
        object.__setattr__(self, "model_id", _nonempty("response.model_id", self.model_id))
        object.__setattr__(self, "api_request_id",
                           _nonempty("response.api_request_id", self.api_request_id))
        _moment("response.responded_at", self.responded_at)
        usage = _finite_json(self.usage, "response.usage")
        if not isinstance(usage, dict):
            raise InvalidParameterError("response.usage", type(self.usage).__name__,
                                        "a JSON object")
        object.__setattr__(self, "usage", _freeze(usage))
        computed = _digest(self.body())
        if not self.response_sha256:
            object.__setattr__(self, "response_sha256", computed)
        elif self.response_sha256 != computed:
            raise InvalidParameterError("response.response_sha256", self.response_sha256,
                                        "the digest recomputed from the verbatim response")

    def body(self) -> Dict[str, Any]:
        return {"raw": self.raw, "structured": _thaw(self.structured),
                "model_id": self.model_id, "api_request_id": self.api_request_id,
                "responded_at": self.responded_at, "usage": _thaw(self.usage)}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), response_sha256=self.response_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CallResponse":
        record = _exact(value, _RESPONSE_FIELDS, "call response")
        return cls(raw=record["raw"], structured=record["structured"],
                   model_id=record["model_id"], api_request_id=record["api_request_id"],
                   responded_at=record["responded_at"], usage=record["usage"],
                   response_sha256=record["response_sha256"])


@dataclass(frozen=True)
class RecordedCall:
    """One request and its answer, chained to the previous call and checked against its schema."""

    sequence: int
    request: CallRequest
    response: CallResponse
    requested_at: str
    previous_sha256: str
    call_sha256: str = ""
    schema: str = dc_field(default=RECORDED_CALL_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) \
                or self.sequence < 1:
            raise InvalidParameterError("call.sequence", self.sequence, "a positive integer")
        if not isinstance(self.request, CallRequest):
            raise InvalidParameterError("call.request", type(self.request).__name__,
                                        "a CallRequest")
        if not isinstance(self.response, CallResponse):
            raise InvalidParameterError("call.response", type(self.response).__name__,
                                        "a CallResponse")
        sent = _moment("call.requested_at", self.requested_at)
        if _moment("response.responded_at", self.response.responded_at) < sent:
            raise InvalidParameterError("response.responded_at", self.response.responded_at,
                                        "at or after the request was sent")
        if self.response.model_id != self.request.model_id:
            raise InvalidParameterError(
                "response.model_id", self.response.model_id,
                "the model that was asked, %r. A bundle reviewed by a different model is "
                "different evidence" % self.request.model_id)
        # The parse must agree with the bytes, or the structured record is a hopeful reading of
        # something else.  Checked here rather than at record time, so it survives a round-trip.
        try:
            parsed = json.loads(self.response.raw)
        except json.JSONDecodeError as exc:
            raise InvalidParameterError(
                "response.raw", self.response.raw[:120],
                "JSON matching the declared schema %r; free text where a schema was declared is "
                "refused (%s)" % (self.request.response_schema.name, exc)) from None
        if parsed != _thaw(self.response.structured):
            raise InvalidParameterError("response.structured", _thaw(self.response.structured),
                                        "exactly the parse of the verbatim response bytes")
        self.request.response_schema.validate(parsed)
        _sha("call.previous_sha256", self.previous_sha256)
        computed = _digest(self.body())
        if not self.call_sha256:
            object.__setattr__(self, "call_sha256", computed)
        elif self.call_sha256 != computed:
            raise InvalidParameterError("call.call_sha256", self.call_sha256,
                                        "the digest recomputed from this recorded call")

    @property
    def determinism(self) -> str:
        return DETERMINISM

    def body(self) -> Dict[str, Any]:
        return {"schema": self.schema, "sequence": self.sequence,
                "request": self.request.to_mapping(), "response": self.response.to_mapping(),
                "requested_at": self.requested_at, "determinism": DETERMINISM,
                "previous_sha256": self.previous_sha256}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), call_sha256=self.call_sha256)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RecordedCall":
        record = _exact(value, _CALL_FIELDS, "recorded call")
        if record["schema"] != RECORDED_CALL_SCHEMA:
            raise InvalidParameterError("schema", record["schema"], RECORDED_CALL_SCHEMA)
        if record["determinism"] != DETERMINISM:
            raise InvalidParameterError(
                "determinism", record["determinism"],
                "%r. An LLM call is a recorded observation, never a reproducible computation "
                "(R23)" % DETERMINISM)
        return cls(sequence=record["sequence"],
                   request=CallRequest.from_mapping(record["request"]),
                   response=CallResponse.from_mapping(record["response"]),
                   requested_at=record["requested_at"],
                   previous_sha256=record["previous_sha256"], call_sha256=record["call_sha256"])


# --------------------------------------------------------------------------------------------
# The review record: commentary beside the bundle, bound to it from outside
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewRecord:
    """An append-only chain of recorded calls, bound to one exact bundle revision."""

    study_id: str
    bundle_sha256: str
    bundle_revision: int
    calls: Tuple[RecordedCall, ...] = ()
    record_sha256: str = ""
    schema: str = dc_field(default=REVIEW_RECORD_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _nonempty("study_id", self.study_id))
        _sha("bundle_sha256", self.bundle_sha256)
        if isinstance(self.bundle_revision, bool) or not isinstance(self.bundle_revision, int) \
                or self.bundle_revision < 0:
            raise InvalidParameterError("bundle_revision", self.bundle_revision,
                                        "a non-negative integer")
        calls = tuple(self.calls)
        previous = self._anchor_sha256()
        last_time: Optional[datetime] = None
        for sequence, call in enumerate(calls, 1):
            if not isinstance(call, RecordedCall):
                raise InvalidParameterError("calls", type(call).__name__, "RecordedCall records")
            if call.sequence != sequence:
                raise InvalidParameterError("call.sequence", call.sequence,
                                            "the uninterrupted append order %d" % sequence)
            if call.previous_sha256 != previous:
                raise InvalidParameterError("call.previous_sha256", call.previous_sha256,
                                            "the prior call digest %s" % previous)
            if call.request.bundle_sha256 != self.bundle_sha256 \
                    or call.request.bundle_revision != self.bundle_revision:
                raise InvalidParameterError(
                    "call.request.bundle_sha256", call.request.bundle_sha256,
                    "the reviewed bundle %s at revision %d. Commentary on one revision is not "
                    "commentary on another" % (self.bundle_sha256, self.bundle_revision))
            sent = _moment("call.requested_at", call.requested_at)
            if last_time is not None and sent < last_time:
                raise InvalidParameterError("call.requested_at", call.requested_at,
                                            "a non-decreasing append chronology")
            previous = call.call_sha256
            last_time = sent
        object.__setattr__(self, "calls", calls)
        computed = _digest(self.body())
        if not self.record_sha256:
            object.__setattr__(self, "record_sha256", computed)
        elif self.record_sha256 != computed:
            raise InvalidParameterError("record_sha256", self.record_sha256,
                                        "the digest recomputed from the complete review record")

    def _anchor_sha256(self) -> str:
        return _digest({"schema": self.schema, "study_id": self.study_id,
                        "bundle_sha256": self.bundle_sha256,
                        "bundle_revision": self.bundle_revision})

    @property
    def revision(self) -> int:
        return len(self.calls)

    @property
    def head_sha256(self) -> str:
        return self.calls[-1].call_sha256 if self.calls else self._anchor_sha256()

    def by_role(self, role: str) -> Tuple[RecordedCall, ...]:
        if role not in REVIEW_ROLES:
            raise InvalidParameterError("role", role, "one of %s" % ", ".join(REVIEW_ROLES))
        return tuple(call for call in self.calls if call.request.role == role)

    def append(self, call: RecordedCall) -> "ReviewRecord":
        """Return the next immutable record; the receiver stays byte-for-byte unchanged."""
        return ReviewRecord(study_id=self.study_id, bundle_sha256=self.bundle_sha256,
                            bundle_revision=self.bundle_revision, calls=self.calls + (call,))

    def body(self) -> Dict[str, Any]:
        return {"schema": self.schema, "study_id": self.study_id,
                "bundle_sha256": self.bundle_sha256, "bundle_revision": self.bundle_revision,
                "calls": [call.to_mapping() for call in self.calls],
                "revision": self.revision, "head_sha256": self.head_sha256}

    def to_mapping(self) -> Dict[str, Any]:
        return dict(self.body(), record_sha256=self.record_sha256)

    def render(self) -> str:
        """Deterministic plain text, with the R23 declaration wherever commentary is shown."""
        lines = [R23_DECLARATION, "",
                 "review of bundle %s at revision %d (study %s)"
                 % (self.bundle_sha256, self.bundle_revision, self.study_id)]
        if not self.calls:
            lines.append("   no calls recorded; that is not the same as none having been made")
            return "\n".join(lines)
        for call in self.calls:
            lines.append("")
            lines.append("%d. %s  [%s, effort %s, request %s, %s]"
                         % (call.sequence, call.request.role, call.request.model_id,
                            call.request.effort, call.response.api_request_id,
                            call.response.responded_at))
            for key in sorted(call.response.structured):
                value = _thaw(call.response.structured[key])
                shown = "; ".join(str(item) for item in value) if isinstance(value, list) \
                    else str(value)
                lines.append("   %s: %s" % (key, shown))
        return "\n".join(lines)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReviewRecord":
        record = _exact(value, _RECORD_FIELDS, "review record")
        if record["schema"] != REVIEW_RECORD_SCHEMA:
            raise InvalidParameterError("schema", record["schema"], REVIEW_RECORD_SCHEMA)
        if not isinstance(record["calls"], list):
            raise InvalidParameterError("calls", type(record["calls"]).__name__, "a JSON array")
        result = cls(study_id=record["study_id"], bundle_sha256=record["bundle_sha256"],
                     bundle_revision=record["bundle_revision"],
                     calls=tuple(RecordedCall.from_mapping(item) for item in record["calls"]),
                     record_sha256=record["record_sha256"])
        if record["revision"] != result.revision or record["head_sha256"] != result.head_sha256:
            raise InvalidParameterError("review chain", record["head_sha256"],
                                        "the recomputed revision and chain head")
        return result


@dataclass(frozen=True)
class ReviewedBundle:
    """A bundle and the commentary recorded about it, held side by side and never merged.

    Every claim-bearing property here is computed from `self.bundle` and nothing else.  That is
    not a convention to be respected by later code; it is the whole content of R22 at this
    boundary, and `verify_claim_independence` checks it.
    """

    bundle: EvidenceBundle
    review: ReviewRecord

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, EvidenceBundle):
            raise InvalidParameterError("bundle", type(self.bundle).__name__, "an EvidenceBundle")
        if not isinstance(self.review, ReviewRecord):
            raise InvalidParameterError("review", type(self.review).__name__, "a ReviewRecord")
        if self.review.study_id != self.bundle.study_id:
            raise InvalidParameterError("review.study_id", self.review.study_id,
                                        "the reviewed study %r" % self.bundle.study_id)
        if self.review.bundle_sha256 != self.bundle.bundle_sha256 \
                or self.review.bundle_revision != self.bundle.revision:
            raise InvalidParameterError(
                "review.bundle_sha256", self.review.bundle_sha256,
                "the bundle actually held, %s at revision %d. Review of an earlier revision is "
                "not review of this one" % (self.bundle.bundle_sha256, self.bundle.revision))

    @property
    def assessment(self) -> ClaimLadderAssessment:
        return assess_claim_ladder(self.bundle)

    @property
    def outputs(self) -> FiveOutputs:
        return summarise_evidence(self.bundle)

    @property
    def rung(self) -> str:
        return self.outputs.rung

    @property
    def claim_sha256(self) -> str:
        """The digest of everything claimable about this study.  A function of the bundle only."""
        return self.outputs.summary_sha256

    def without_review(self) -> "ReviewedBundle":
        """The same study with every recorded call deleted."""
        return ReviewedBundle(bundle=self.bundle, review=ReviewRecord(
            study_id=self.bundle.study_id, bundle_sha256=self.bundle.bundle_sha256,
            bundle_revision=self.bundle.revision))


# --------------------------------------------------------------------------------------------
# Making a call, and proving it changed nothing
# --------------------------------------------------------------------------------------------


def attach_review(bundle: EvidenceBundle) -> ReviewedBundle:
    """Open an empty review beside a bundle, bound to this exact revision."""
    if not isinstance(bundle, EvidenceBundle):
        raise InvalidParameterError("bundle", type(bundle).__name__, "an EvidenceBundle")
    return ReviewedBundle(bundle=bundle, review=ReviewRecord(
        study_id=bundle.study_id, bundle_sha256=bundle.bundle_sha256,
        bundle_revision=bundle.revision))


def record_call(reviewed: ReviewedBundle, *, role: str, model_id: str, effort: str,
                system: str, instruction: str, context: Mapping[str, Any],
                response_schema: ResponseSchema, transport: Callable[[Mapping[str, Any]], Any],
                requested_at: str) -> ReviewedBundle:
    """Send one call through the injected transport and record it verbatim beside the bundle.

    The transport receives a plain copy of the request and must return a mapping carrying the
    verbatim response text, its structured parse, the model that answered, the API request id,
    the response timestamp and the usage block.  Nothing it returns can reach the evidence
    chain: the bundle handed back is the same object that came in.
    """
    if not isinstance(reviewed, ReviewedBundle):
        raise InvalidParameterError("reviewed", type(reviewed).__name__, "a ReviewedBundle")
    if not callable(transport):
        raise InvalidParameterError("transport", type(transport).__name__,
                                    "a callable taking the request mapping")
    request = CallRequest(role=role, model_id=model_id, effort=effort, system=system,
                          instruction=instruction, context=context,
                          response_schema=response_schema,
                          bundle_sha256=reviewed.bundle.bundle_sha256,
                          bundle_revision=reviewed.bundle.revision)
    answer = transport(request.to_mapping())
    if not isinstance(answer, Mapping):
        raise InvalidParameterError("transport result", type(answer).__name__,
                                    "a mapping with %s" % ", ".join(_TRANSPORT_FIELDS))
    record = _exact(answer, _TRANSPORT_FIELDS, "transport result")
    response = CallResponse(raw=record["raw"], structured=record["structured"],
                            model_id=record["model_id"],
                            api_request_id=record["api_request_id"],
                            responded_at=record["responded_at"], usage=record["usage"])
    call = RecordedCall(sequence=reviewed.review.revision + 1, request=request,
                        response=response, requested_at=requested_at,
                        previous_sha256=reviewed.review.head_sha256)
    return ReviewedBundle(bundle=reviewed.bundle, review=reviewed.review.append(call))


def strip_review(reviewed: ReviewedBundle) -> EvidenceBundle:
    """Delete every LLM output.  What is left is the bundle, unchanged, because it always was."""
    if not isinstance(reviewed, ReviewedBundle):
        raise InvalidParameterError("reviewed", type(reviewed).__name__, "a ReviewedBundle")
    return reviewed.bundle


def verify_claim_independence(reviewed: ReviewedBundle) -> str:
    """R22 made executable: refuse unless deleting the whole review changes no claim level.

    Three things are checked, and the third is the one that could actually fail.  The claim
    state is recomputed with the review present and with it deleted; the bundle is rebuilt from
    its own canonical bytes and must give the same state again; and no wording from a recorded
    response may appear anywhere in those bytes, which is the only way commentary could have
    been smuggled into a field the gates read.  Returns the claim digest.
    """
    if not isinstance(reviewed, ReviewedBundle):
        raise InvalidParameterError("reviewed", type(reviewed).__name__, "a ReviewedBundle")
    with_review = reviewed.claim_sha256
    without = summarise_evidence(strip_review(reviewed)).summary_sha256
    if with_review != without:
        raise InvalidParameterError(
            "claim_sha256", with_review,
            "unchanged by deleting every recorded call. An LLM output reached a claim level, "
            "which means the review layer is implemented wrongly (R22)")
    rebuilt = summarise_evidence(
        EvidenceBundle.from_mapping(reviewed.bundle.to_mapping())).summary_sha256
    if rebuilt != with_review:
        raise InvalidParameterError("claim_sha256", rebuilt,
                                    "the claim state recomputed from the bundle's own bytes")
    published = _canonical(reviewed.bundle.to_mapping()).decode("utf-8")
    for call in reviewed.review.calls:
        for phrase in _phrases(call.response.structured):
            if phrase in published:
                raise InvalidParameterError(
                    "bundle bytes", phrase,
                    "free of recorded commentary. Wording from the response to call %d is inside "
                    "the evidence bundle, where no LLM output may ever be (R22)" % call.sequence)
    return with_review


def save_review_record(path: Union[str, os.PathLike[str]], record: ReviewRecord, *,
                       bundle_path: Optional[Union[str, os.PathLike[str]]] = None) -> str:
    """Publish the commentary in its own file, exclusively, and never over a bundle."""
    if not isinstance(record, ReviewRecord):
        raise InvalidParameterError("record", type(record).__name__, "a ReviewRecord")
    target = Path(path)
    if bundle_path is not None and target.resolve() == Path(bundle_path).resolve():
        raise InvalidParameterError("path", str(target),
                                    "a file of its own. Commentary is stored beside the evidence "
                                    "bundle, never inside it (R22)")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(_canonical(record.to_mapping()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        raise FileExistsError("refusing to overwrite review record at %s" % target) from None
    return record.record_sha256


def load_review_record(path: Union[str, os.PathLike[str]], *,
                       published_sha256: Optional[str] = None) -> ReviewRecord:
    target = Path(path)
    try:
        raw = target.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidParameterError("review record path", str(target),
                                    "readable UTF-8 JSON: %s" % exc) from exc
    record = ReviewRecord.from_mapping(value)
    if raw != _canonical(record.to_mapping()) + b"\n":
        raise InvalidParameterError("review record bytes", str(target), "canonical JSON")
    if published_sha256 is not None and str(published_sha256).lower() != record.record_sha256:
        raise InvalidParameterError("published_record_sha256", published_sha256,
                                    "the unchanged review record digest %s" % record.record_sha256)
    return record


__all__ = [
    "RECORDED_CALL_SCHEMA", "REVIEW_RECORD_SCHEMA", "REVIEW_ROLES", "EFFORTS",
    "FORBIDDEN_SAMPLING_PARAMETERS", "DETERMINISM", "R23_DECLARATION", "SCHEMA_KINDS",
    "SMUGGLING_FLOOR",
    "FieldSpec", "ResponseSchema", "CallRequest", "CallResponse", "RecordedCall",
    "ReviewRecord", "ReviewedBundle", "attach_review", "record_call", "strip_review",
    "verify_claim_independence", "save_review_record", "load_review_record",
]
